# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

# CUSTOMIZATION START
import sys
sys.path.extend(["./", "./src/models/stylegan3/", "./src/models/stylegan3/dnnlib", "./src/models/stylegan3/torch_utils"])
# CUSTOMIZATION END

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import pickle

import dnnlib
import legacy

def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)
    # CUSTOMIZATION START
    if target.shape[0] == 1:
        target = target.repeat([3, 1, 1])  # make a three channel tensor
    # CUSTOMIZATION END

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim) # sample w_avg_samples noise vector Z
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None) # map the noise vector Z to W using the mapping netowrk: [N, L, C] N --> w_avg_samples, L--> number of layer in G, C--> dimension of W
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32) # take only the W vector for the first layer L: [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True) # [1, 1, C] # to find the starting point
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5 # to find the scale of W

    # Setup noise inputs. (noisy input to generator network)
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32) # add a dimension (batch dimension)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')
        # CUSTOMIZATION START
        if synth_images.shape[1] == 1:
            synth_images = synth_images.repeat([1, 3, 1, 1])  # make a three channel tensor
        # CUSTOMIZATION END

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    num_steps: int,
    seed: int,
    save_video: bool
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python3 ./genlib/projector.py
      --network="/home/lorenzo/Gan-track/models/claro_retrospettivo/00000-stylegan2--gpus2-batch32-gamma0.4096/network-snapshot-005000.pkl"
      --target="/home/lorenzo/Gan-track/data/interim/claro_retrospettivo/stylegan2-ada/9309_98.png"
        --num-steps=1000
        --seed=303
        --save-video=True
        --outdir="/home/lorenzo/Gan-track/reports/claro_retrospettivo/projector/00000-stylegan2--gpus2-batch32-gamma0.4096/network-snapshot-005000/"
    """
    np.random.seed(seed) # set seed
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    # CUSTOMIZING START
    with open(network_pkl, 'rb') as f:
        G = pickle.load(f)['G_ema'].requires_grad_(False).to(device) # torch.nn.Module
    # CUSTOMIZING END

    # Load target image.
    # CUSTOMIZATION START
    target_pil = PIL.Image.open(target_fname).convert('L') # convert in grayscale
    # CUSTOMIZATION END
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)) # crop the image if is not square
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS) # resize the image at 256 x 256 resolution (resolution of generator)
    # CUSTOMIZATION START
    target = np.array(target_pil, dtype=np.uint8) # convert the image to uint8
    # CUSTOMIZATION END

    # CUSTOMIZATION START - add a dimension if needed --> from [(G_resolution, G_resolution)] to [(G_resolution, G_resolution, 1)]
    if len(target.shape) < 3:
        target = np.expand_dims(target, axis=-1)
    # CUSTOMIZATION END

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=torch.tensor(target.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    # CUSTOMIZATION START
    PIL.Image.fromarray(synth_image.squeeze(-1)).save(f'{outdir}/proj.png')
    # CUSTOMIZATION END
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

#----------------------------------------------------------------------------

if __name__ == "__main__":
    # CUSTOMIZING START
    # FOR DEBUG REASON:
    # RuntimeError:Ninja is required to load C++ extension #167
    # Solutions: The subprocess does not include the lib path of conda environments. So manually set the environments in the script (https://github.com/zhanghang1989/PyTorch-Encoding/issues/167)
    my_env = os.environ.copy()
    my_env["PATH"] = "/home/lorenzo/miniconda3/envs/stylegan3/bin:" + my_env["PATH"]
    os.environ.update(my_env)
    # CUSTOMIZING END
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------