
import dnnlib
import os
import numpy as np
import PIL
from PIL import Image
import imageio
import pickle
import torch
import torch.nn.functional as F
import time
from time import perf_counter

import copy

# ----------------------------------------------------------------------------

def project(
        G,
        target: torch.Tensor,  # [1, C, H, W], dynamic range [0.0, 255.0], dtype float32 W & H must match G output resolution
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate= 1, # 0.1,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,  ## time that lr taks to go 0 again the "initial_learning_rate"
        lr_rampup_length=0.05, # time that lr taks to reach the "initial_learning_rate" starting from 0
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        verbose=False,
        device: torch.device,
        modalities: list
):
    assert target.shape == (1, G.img_channels, G.img_resolution, G.img_resolution)

    if target.shape[1] == 1: # Make a three channel tensor.
        target = {
            modalities[0]: target.repeat([1, 3, 1, 1])
        }
    elif target.shape[1] == 3:  # already in the correct format
        target = {
            modalities[0]: target
        }
        pass
    else: # Multimodal input.
        target = {
            mode: target[:, idx_mode, :, :].unsqueeze(dim=1).repeat([1, 3, 1, 1]) for idx_mode, mode in enumerate(modalities)
        }

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)  # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C] # to find the starting point
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5  # to find the scale of W

    # Setup noise inputs. (noisy input to generator network)
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    # Load feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_features = {}
    for mode in modalities:
        target_images = target[mode].to(device).to(torch.float32)
        if target_images.shape[2] > 256:
            target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        target_features[mode] = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)  # pylint: disable=not-callable
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
        # Normalize from [-1 1] range to [0.0, 255.0] the synthetic image
        synth_images = (synth_images + 1) * (255 / 2) # normalize on the entire multichannel stack

        if synth_images.shape[1] == 1:  # Make a three channel tensor.
            synth_images = {
                modalities[0]: synth_images.repeat([1, 3, 1, 1])
            }
        elif synth_images.shape[1] == 3:  # already in the correct format
            synth_images = {
                modalities[0]: synth_images
            }
            pass
        else:  # Multimodal input.
            synth_images = {
                mode: synth_images[:, idx_mode, :, :].unsqueeze(dim=1).repeat([1, 3, 1, 1]) for idx_mode, mode in enumerate(modalities)
            }
        synth_features = {}
        dist = {}
        for mode in modalities:
            target_synth_images = synth_images[mode]
            # Downsample image to 256x256 if it's larger than that. VGG was built for 256x256 images.
            # target_synth_images = (target_synth_images + 1) * (255 / 2) # to normalize in [0 255] per channel
            if target_synth_images.shape[2] > 256:
                target_synth_images = F.interpolate(target_synth_images, size=(256, 256), mode='area')
            # Features for synth images.
            synth_features[mode] = vgg16(target_synth_images, resize_images=False, return_lpips=True)

            # Compute distance.
            dist[mode] = (target_features[mode] - synth_features[mode]).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = reg_loss * regularize_noise_weight
        for mode in modalities:
            loss += dist[mode]

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        desc = ""
        for mode in modalities:
            desc += f"dist_{mode} {dist[mode]:<4.2f} "

        logprint(f'step {step + 1:>4d}/{num_steps}, lr: {lr}: {desc}loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]
        # todo save loss
        # todo save distances

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])

def run_projection(
        img_tensor,
        outdir: str,
        outdir_model: str,
        device: torch.device,
        modalities: list,
        dtype: str,
        num_steps: int,
        save_video: bool,
        **kwargs
):
    """Project given image to the latent space of pretrained network pickle."""
    # Load networks.
    print('Loading networks from "%s"...' % outdir_model)
    with open(outdir_model, 'rb') as f:
        G = pickle.load(f)['G_ema'].requires_grad_(False).to(device)  # torch.nn.Module

    assert img_tensor.detach().cpu().numpy().dtype == dtype
    assert img_tensor.min().item() >= 0.0
    assert img_tensor.max().item() <= 255.0
    assert img_tensor.shape[2] == G.img_resolution # check over resolution
    assert img_tensor.shape[1] == G.img_channels # check over the number of channel

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G=G,
        target=img_tensor,  # pylint: disable=not-callable
        num_steps=num_steps,
        verbose=True,
        device=device,
        modalities=modalities
    )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video: #todo save video per mode
        synth_image_list = []
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image_list.append((synth_image + 1) * (255 / 2))

        for idx_mode, mode in enumerate(modalities):
            video = imageio.get_writer(f'{outdir}/proj_{mode}.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
            print(f'Saving optimization progress video "{outdir}/proj_{mode}.mp4"')

            for synth_image in synth_image_list:
                synth_image = synth_image[:, idx_mode, :, :].unsqueeze(0)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                target_image =  img_tensor[:, idx_mode, :, :].unsqueeze(0).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                video.append_data(np.concatenate([target_image, synth_image], axis=1))

            video.close()

    # Save final projected frame and W vector.
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255 / 2)
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

    for idx_mode, mode in enumerate(modalities):
        target_image = img_tensor[:, idx_mode, :, :].unsqueeze(0).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        target_pil = PIL.Image.fromarray(target_image.squeeze(-1))
        target_pil.save(f'{outdir}/target_{mode}.png')

        synth_image_pil = synth_image[:, idx_mode, :, :].unsqueeze(0).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        PIL.Image.fromarray(synth_image_pil.squeeze(-1)).save(f'{outdir}/proj_{mode}.png')

    return projected_w_steps

def projection_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    projector_kwargs        = {},       # Optiond for Projector algorithm
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 1,        # Total batch size for one training iteration.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
):

    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank) # device = torch.device('cuda:1')
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.

    # Load training set.
    print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_iterator = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size//num_gpus)
    print()
    print('Num images: ', len(training_set))
    print('Image shape:', training_set.image_shape)
    print('Label shape:', training_set.label_shape)
    print()

    # todo  todo .pkl file for latent position of training dataset
    # todo add a MSE based loss function on x vs x_tilde
    # todo add parameter search over lambda
    projected_w = np.zeros(shape=(len(training_set), 14, 512), dtype=float) # todo parametrize 14 and 512
    idx = 0
    for phase_real_img, _ in training_set_iterator:

        phase_real_img = phase_real_img.to(device).to(torch.float32)

        projected_w[idx, :, :] = run_projection(img_tensor=phase_real_img, outdir=run_dir, device=device, modalities=training_set_kwargs.modalities, dtype=training_set_kwargs.dtype, **projector_kwargs)
        idx += 1
