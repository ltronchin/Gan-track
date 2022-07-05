
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
from torch.autograd import Variable


import copy

import matplotlib.pyplot as plt
# Set plot parameters.
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 16

import pandas as pd

from src.utils import util_general

#----------------------------------------------------------------------------
# Reports.

def intersection(l1, l2):
    l = [value for value in l1 if value in l2]
    return l

def get_cmap(n: int, name: str ='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)

def plot_training(history, plot_training_dir, columns_to_plot=None, **plot_args):

    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame.from_dict(history, orient='index').transpose()

    columns_in_history = list(history.keys())
    if columns_to_plot is not None:
        columns_to_plot = intersection(columns_in_history, columns_to_plot)
    else:
        columns_to_plot = columns_in_history

    cmap = get_cmap(n=len(columns_to_plot)+1, name='hsv')
    plt.figure(figsize=(8, 6))
    for idx, key in enumerate(columns_to_plot):
        plt.plot(history[key], label=key, c=cmap(idx))

    plt.title(plot_args['title'])
    plt.xlabel(plot_args['xlab'])
    plt.ylabel(plot_args['ylab'])
    plt.legend()
    plt.savefig(os.path.join(plot_training_dir, f"{plot_args['title']}.png"), dpi=400, format='png')
    plt.show()

# ----------------------------------------------------------------------------

def project(
        G,
        target: torch.Tensor,  # [1, C, H, W], dynamic range [0.0, 255.0], dtype float32 W & H must match G output resolution
        *,
        num_steps =1000,
        w_lpips: float = 1.0,
        w_pix: float = 1e-6,
        w_avg_samples = 10000,
        initial_learning_rate =  0.1,
        initial_noise_factor = 0.05,
        lr_rampdown_length = 0.25,  # time that lr taks to go 0 again the "initial_learning_rate"
        lr_rampup_length = 0.05, # time that lr taks to reach the "initial_learning_rate" starting from 0
        noise_ramp_length = 0.75,
        regularize_noise_weight =  1e5,
        verbose = False,
        device: torch.device,
        modalities: list
):
    assert target.shape == (1, G.img_channels, G.img_resolution, G.img_resolution)
    # Create a modalities dictionary and take a three channel tensor.
    if target.shape[1] == 1:
        target_modatilies = {
            modalities[0]: target.repeat([1, 3, 1, 1])
        }
    elif target.shape[1] == 3:  # Already in the correct format
        target_modatilies = {
            modalities[0]:  target
        }
        pass
    else: # Multimodal input.
        target_modatilies = {
            mode: target[:, idx_mode, :, :].unsqueeze(dim=1).repeat([1, 3, 1, 1]) for idx_mode, mode in enumerate(modalities) # todo clone?
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

    # Compute the target LPIPS features for the real images using VGG16.
    target_features = {}
    for mode in modalities:
        target_mode = target_modatilies[mode].clone()
        target_mode = target_mode.to(device).to(torch.float32)
        if target_mode.shape[2] > 256:
            target_mode = F.interpolate(target_mode, size=(256, 256), mode='area')
        target_features[mode] = vgg16(target_mode, resize_images=False, return_lpips=True)  # todo why the operation change target_mode?

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)  # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    history = util_general.list_dict()
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
        synth = G.synthesis(ws, noise_mode='const')

       # Rescale the synthetic image  from [-1 1] to [0.0, 255.0]
        synth = (synth + 1) * (255 / 2)

        # Create a modalities dictionary and take a three channel tensor.
        if synth.shape[1] == 1:
            synth_modalities = {
                modalities[0]: synth.repeat([1, 3, 1, 1])
            }
        elif synth.shape[1] == 3:  # already in the correct format
            synth_modalities = {
                modalities[0]:synth
            }
            pass
        else:  # Multimodal input.
            synth_modalities = {
                mode: synth[:, idx_mode, :, :].unsqueeze(dim=1).repeat([1, 3, 1, 1]) for idx_mode, mode in enumerate(modalities)
            } # todo define a new variable for each channel here?
            

        # Pixel-based loss.
        pix_loss = {}
        for mode in modalities:
            synth_mode = synth_modalities[mode].clone()
            target_mode = target_modatilies[mode].clone()
            target_mode = target_mode.to(device).to(torch.float32)
            pix_loss[mode] = w_pix * (torch.mean((target_mode.float() - synth_mode.float()) ** 2))

        # Compute the synthetic LPIPS features for the synthetic images using VGG16.
        synth_features = {}
        dist = {}
        for mode in modalities:
            synth_mode = synth_modalities[mode].clone()
            # Downsample image to 256x256 if it's larger than that. VGG was built for 256x256 images.
            if synth_mode.shape[2] > 256:
                synth_mode = F.interpolate(synth_mode, size=(256, 256), mode='area')
            # Features for synth images.
            synth_features[mode] = vgg16(synth_mode, resize_images=False, return_lpips=True) # todo why the operation change synth_mode?

            # Compute distance.
            dist[mode] = w_lpips * ((target_features[mode] - synth_features[mode]).square().sum())

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

        # Compute the total loss
        loss = reg_loss * regularize_noise_weight
        history['reg_loss'].append(reg_loss.item() * regularize_noise_weight)
        for mode in modalities:
            loss = loss + dist[mode] + pix_loss[mode]
            history[f'{mode}_lpips_loss'].append(dist[mode].item())
            history[f'{mode}_pix_loss'].append(pix_loss[mode].item())

        history['tot_loss'].append(loss.item())

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        desc = ""
        for mode in modalities:
            desc += f"dist_{mode} {dist[mode]:<4.2f}--pix_loss_{mode} {pix_loss[mode]:<4.2f} "

        logprint(f'step {step + 1:>4d}/{num_steps}, lr: {lr}: {desc} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()
    return w_out.repeat([1, G.mapping.num_ws, 1]), history

def run_projection(
        img_idx: int,
        img_tensor: torch.Tensor,
        outdir: str,
        outdir_model: str,
        device: torch.device,
        modalities: list,
        dtype: str,
        num_steps: int,
        w_lpips: float,
        w_pix: float,
        save_video: bool,
        save_final_projection: bool,
        save_optimization_history: bool,
        **kwargs
):
    """Project given image to the latent space of pretrained network pickle."""
    # Load networks.
    print('Loading networks from "%s"...' % outdir_model)
    with open(outdir_model, 'rb') as f:
        G = pickle.load(f)['G_ema'].requires_grad_(False).to(device)  # torch.nn.Module

    assert img_tensor.detach().cpu().numpy().dtype == dtype
    assert img_tensor.min().item() >= 0.0
    assert img_tensor.max().item() > 1.0
    assert img_tensor.max().item() <= 255.0
    assert img_tensor.shape[2] == G.img_resolution # check over resolution
    assert img_tensor.shape[1] == G.img_channels # check over the number of channel

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps, history = project(
        G=G,
        target=img_tensor,  # pylint: disable=not-callable
        num_steps=num_steps,
        w_lpips=w_lpips,
        w_pix=w_pix,
        verbose=True,
        device=device,
        modalities=modalities
    )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    os.makedirs(outdir, exist_ok=True)
    if save_optimization_history:
        if 'claro' in outdir:
            plot_training(
                history=history, plot_training_dir=outdir, columns_to_plot=['CT_lpips_loss'], title= 'Lpips Loss', xlab='Step', ylab='Loss'
            )
            plot_training(
                history=history, plot_training_dir=outdir, columns_to_plot=['CT_pix_loss'], title= 'Pix Loss', xlab='Step', ylab='Loss'
            )
        else:
            plot_training(
                history=history, plot_training_dir=outdir, columns_to_plot=['MR_nonrigid_CT_lpips_loss', 'MR_MR_T2_lpips_loss'], title= 'Lpips Loss modalities', xlab='Step', ylab='Loss'
            )
            plot_training(
                history=history, plot_training_dir=outdir, columns_to_plot=['MR_nonrigid_CT_pix_loss', 'MR_MR_T2_pix_loss'], title= 'Pix Loss modalities', xlab='Step', ylab='Loss'
            )
    history.to_csv(os.path.join(outdir, 'optimization_loss.csv'))

    # Render debug output: optional video and projected image and W vector.
    if save_video:
        synth_image_list = []
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image_list.append((synth_image + 1) * (255 / 2))  # normalize per stack

        for idx_mode, mode in enumerate(modalities):
            video = imageio.get_writer(f'{outdir}/proj_{img_idx}_{mode}.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
            print(f'Saving optimization progress video "{outdir}/proj_{img_idx}_{mode}.mp4')

            for synth_image in synth_image_list:
                synth_image = synth_image[:, idx_mode, :, :].unsqueeze(0)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                target_image =  img_tensor[:, idx_mode, :, :].unsqueeze(0).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                video.append_data(np.concatenate([target_image, synth_image], axis=1))

            video.close()

    # Save final projected frame and W vector.
    if save_final_projection:
        projected_w = projected_w_steps[-1]
        synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
        synth_image = (synth_image + 1) * (255 / 2)  # normalize per stack
        np.savez(f'{outdir}/projected_w_{img_idx}.npz', w=projected_w.unsqueeze(0).cpu().numpy())

        for idx_mode, mode in enumerate(modalities):
            target_image = img_tensor[:, idx_mode, :, :].unsqueeze(0).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            target_pil = PIL.Image.fromarray(target_image.squeeze(-1))
            target_pil.save(f'{outdir}/target_{img_idx}_{mode}.png')

            synth_image_pil = synth_image[:, idx_mode, :, :].unsqueeze(0).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            PIL.Image.fromarray(synth_image_pil.squeeze(-1)).save(f'{outdir}/proj_{img_idx}_{mode}.png')

    return projected_w_steps

def projection_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    projector_kwargs        = {},       # Optiond for Projector algorithm
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 1,        # Total batch size for one training iteration.
    image_snapshot_ticks    =50,  # How often to save image snapshots? None = disable.
    network_snapshot_ticks  =50,  # How often to save network snapshots? None = disable.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
):

    # Initialize.
    start_time = time.time()
    device =  torch.device('cuda', rank) #torch.device('cuda', rank) # device = torch.device('cuda:1')
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.

    # Load training set.
    print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Datasetx
    training_set_iterator = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size//num_gpus)
    print()
    print('Num images: ', len(training_set))
    print('Image shape:', training_set.image_shape)
    print('Label shape:', training_set.label_shape)
    print()

    # todo parameter tuning
    # todo add a MSE based loss function on x vs x_tilde
    # todo add parameter search over lambda
    projected_w = np.zeros(shape=(len(training_set), 14, 512), dtype=float)
    idx = 0
    for phase_real_img, _ in training_set_iterator:

        phase_real_img = phase_real_img.to(device).to(torch.float32)
        projected_w_tensor = run_projection(
            img_idx=idx, img_tensor=phase_real_img, outdir=run_dir, device=device, modalities=training_set_kwargs.modalities, dtype=training_set_kwargs.dtype, **projector_kwargs
        )
        projected_w[idx, :, :] = projected_w_tensor[-1].detach().cpu().numpy()
        idx += 1

    with open(os.path.join(run_dir,'projected_w'), 'wb') as handle:
        pickle.dump(projected_w, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ----------------------------------------------------------------------------

def projection_test(
        target_fname=           '.',    # Path to image to invert
        run_dir=                '.',    # Output directory.
        training_set_kwargs=    {},     # Options for training set.
        projector_kwargs=       {},     # Optiond for Projector algorithm
        random_seed=            0,      # Global random seed.
        num_gpus=               1,      # Number of GPUs participating in the training.
        rank=                   0,      # Rank of the current process in [0, num_gpus[.
        cudnn_benchmark=        True,   # Enable torch.backends.cudnn.benchmark?
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)  # torch.device('cuda', rank) # device = torch.device('cuda:1')
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False  # Improves numerical accuracy.

    # Load single Target Image.
    print('Loading Test Image...')

    target_pil = Image.open(target_fname)
    if target_pil.format != 'TIFF':
        target_pil = Image.open(target_fname).convert('L')  # convert in grayscale
    target_numpy = np.array(target_pil)
    if target_numpy.max() <= 1:
        print(f"Unsupported max value : Expected values in [0, 255] but got [{target_numpy.min()}, {target_numpy.max()}]!")
        target_numpy = target_numpy * 255.0
    target = torch.Tensor(target_numpy)  # convert to Torch Tensor
    if target.ndim != 4:
        print(f"Unsupported input dimension: Expected 4D (batched) input but got input of size: {target.shape}! Let's add dimension!")
        for _ in range(4 - target.ndim):
            target = torch.unsqueeze(target, dim=0)
        target = target.view([1, 1, 256, 256])
    if target.dtype != torch.float32:
        print( f"Unsupported input type: Expected float32 but got {target.dtype}! Let's change Tensor dtype!")
        target = target.to(torch.float32)
    print()
    print('Image shape:', target.shape)
    print('Image dtype:', target.dtype)
    print('Image min:', target.min().item())
    print('Image max:', target.max().item())
    print()

    _ = run_projection(
        img_idx=0,
        img_tensor=target,
        outdir=run_dir,
        device=device,
        modalities=training_set_kwargs['modalities'],
        dtype=training_set_kwargs['dtype'],
        **projector_kwargs
    )

    print("May be the force with you!")

if __name__ == "__main__":
    # For debug.
    my_env = os.environ.copy()
    my_env["PATH"] = "/home/lorenzo/miniconda3/envs/stylegan3/bin:" + my_env["PATH"]
    os.environ.update(my_env)

    '''
    Options:
        For claro retrospettivo
            outdir_model="/home/lorenzo/Gan tracker/reports/claro_retrospettivo/training-runs/claro_retrospettivo/CT/00000-stylegan2--gpus2-batch32-gamma0.4096/network-snapshot-005000.pkl",
            run_dir="/home/lorenzo/Gan tracker/reports/claro_retrospettivo/projection-runs/claro_retrospettivo/CT/00000-stylegan2--gpus2-batch32-gamma0.4096_network-snapshot-005000.pkl/"
            target_fname="/home/lorenzo/Gan tracker/data/interim/claro_retrospettivo/stylegan2-ada/100151470_103.png"
        
        For claro retrospettivo no casting
            outdir_model="/home/lorenzo/Gan tracker/reports/claro_retrospettivo_no_casting/training-runs/claro_retrospettivo_no_casting/CT/00000-stylegan2-stylegan2-ada-gpus2-batch32-gamma0.4096/network-snapshot-004400.pkl",
            run_dir="/home/lorenzo/Gan tracker/reports/claro_retrospettivo_no_casting/projection-runs/claro_retrospettivo_no_casting/CT/00000-stylegan2-stylegan2-ada-gpus2-batch32-gamma0.4096-network-snapshot-004400.pkl/"
            target_fname="/home/lorenzo/Gan tracker/data/interim/claro_retrospettivo_no_casting/stylegan2-ada/100151470_103.png"
    '''

    w_pix_list = [0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    w_lips = 1.0
    for idx_exp, w_pix in enumerate(w_pix_list):

        outdir_model="/home/lorenzo/Gan tracker/reports/claro_retrospettivo_no_casting/training-runs/claro_retrospettivo_no_casting/CT/00000-stylegan2-stylegan2-ada-gpus2-batch32-gamma0.4096/network-snapshot-004400.pkl"
        run_dir=f"/home/lorenzo/Gan tracker/reports/claro_retrospettivo_no_casting/projection-runs/claro_retrospettivo_no_casting/CT/0000{idx_exp}-stylegan2-w_lips_{w_lips}-w_pix_{w_pix}-network-snapshot-004400.pkl/"
        target_fname="/home/lorenzo/Gan tracker/data/interim/claro_retrospettivo_no_casting/stylegan2-ada/100151470_103.tif"

        training_set_kwargs = {
            'modalities': ['CT'],
            'dtype': 'float32'
        }
        projector_kwargs = {
            'outdir_model': outdir_model,
            'num_steps': 1000,
            'w_lpips': w_lips,
            'w_pix': w_pix,
            'save_video': True,
            'save_final_projection': True,
            'save_optimization_history': True
        }

        projection_test(
            target_fname=target_fname,
            run_dir = run_dir,
            training_set_kwargs = training_set_kwargs,
            projector_kwargs = projector_kwargs
        ) # pylint: disable=no-value-for-parameter