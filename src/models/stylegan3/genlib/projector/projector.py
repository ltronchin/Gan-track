import pickle
import os
import time
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import copy

import matplotlib.pyplot as plt
# Set plot parameters.
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 16
import PIL
import imageio

from src.models.stylegan3.genlib.utils import util_general
import dnnlib

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

class StyleGAN2Projector(torch.nn.Module):
    def __init__(
        self,
        outdir:                         str,
        device:                         torch.device,
        modalities:                     list,
        dtype:                          str,
        outdir_model: str,
        #num_steps:                      int = 500,
        #early_stopping:                 int = 50,
        w_lpips:                        float = 1.0,
        w_pix:                          float = 1e-4,
        w_avg_samples:                  int = 10000,
        initial_learning_rate:          float = 0.1,
        initial_noise_factor:           float = 0.05,
        lr_rampdown_length:             float = 0.25,  # time that lr takes to go 0 again the "initial_learning_rate"
        lr_rampup_length:               float = 0.05,  # time that lr takes to reach the "initial_learning_rate" starting from 0
        noise_ramp_length:              float = 0.75,
        regularize_noise_weight:        float = 1e5,
        save_final_projection: bool = True,
        snap_video:                     bool = False,
        snap_optimization_history:      bool = False,
        **kwargs
    ):
        super().__init__()
        self.outdir                     = outdir
        self.outdir_model               = outdir_model
        self.device                     = device
        self.dtype                      = dtype
        self.modalities                 = modalities

        #self.num_steps                  = num_steps
        #self.early_stopping             = early_stopping
        self.w_lpips                    = w_lpips
        self.w_pix                      = w_pix
        self.w_avg_samples              = w_avg_samples
        self.initial_learning_rate      = initial_learning_rate
        self.initial_noise_factor       = initial_noise_factor
        self.lr_rampdown_length         = lr_rampdown_length
        self.lr_rampup_length           = lr_rampup_length
        self.noise_ramp_length          = noise_ramp_length
        self.regularize_noise_weight    = regularize_noise_weight
        self.save_final_projection       = save_final_projection
        self.snap_video                 = snap_video
        self.snap_optimization_history  = snap_optimization_history

        # Load networks.
        print('Loading networks from "%s"...' % self.outdir_model)
        with open(self.outdir_model, 'rb') as f:
            G = pickle.load(f)['G_ema'].requires_grad_(False).to(self.device)  # torch.nn.Module
        self.G = G

    def forward(
            self,
            id_patient:         str,
            id_slice:           int,
            target:             torch.Tensor, # [1, C, H, W], dynamic range [0.0, 255.0], dtype float32 W & H must match G output resolution
            num_steps:          int,
            early_stopping:     int,
            w_init:             torch.Tensor = None, # todo add smooth init
            verbose:            bool = False,
            verbose_logger:     bool = True
    ):
        """Project given image to the latent space of pretrained network pickle."""

        def logprint(*args):
            if verbose_logger:
                print(*args)

        assert target.detach().cpu().numpy().dtype ==  self.dtype
        assert target.min().item() >= 0.0
        assert target.max().item() > 1.0
        assert target.max().item() <= 255.0
        assert target.shape[2] == self.G.img_resolution # check over resolution
        assert target.shape[1] == self.G.img_channels # check over the number of channel

        # Optimize projection.
        since = time.time()
        assert target.shape == (1, self.G.img_channels, self.G.img_resolution, self.G.img_resolution)
        # Create a modalities dictionary and take a three channel tensor.
        if target.shape[1] == 1:
            target_modatilies = {
                self.modalities[0]: target.repeat([1, 3, 1, 1])
            }
        elif target.shape[1] == 3:  # Already in the correct format
            target_modatilies = {
                self.modalities[0]: target
            }
            pass
        else:  # Multimodal input.
            target_modatilies = {
                mode: target[:, idx_mode, :, :].unsqueeze(dim=1).repeat([1, 3, 1, 1]) for idx_mode, mode in  enumerate(self.modalities)  # target is a leaf variable without gradient, no need for .clone()
            }

        G = copy.deepcopy(self.G).eval().requires_grad_(False).to(self.device)  # type: ignore

        # Compute w stats.
        if w_init is not None: # todo we consider w_avg as w_init itself and we computed w_std on w_init elements instead on a popolution of random samples
            w_sample = torch.unsqueeze(w_init, dim=0) # [14, 512]-> [1, 14, 512]
            w_avg    = w_sample[:, :1, :] # [1, 14, 512] -> [1, 1, 512] # todo change to numpy
            w_std    = torch.std(w_avg).item() # float32 # todo change to numpy
        else:
            logprint(f'Computing W midpoint and stddev using {self.w_avg_samples} samples...')
            z_samples = np.random.RandomState(123).randn(self.w_avg_samples, G.z_dim)
            w_samples = G.mapping(torch.from_numpy(z_samples).to(self.device), None)  # [N, L, C]
            w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
            w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C] # to find the starting point
            w_std = (np.sum((w_samples - w_avg) ** 2) / self.w_avg_samples) ** 0.5  # to find the scale of W

        # Setup noise inputs. (noisy input to generator network)
        noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

        # Load feature detector.
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(self.device)

        # Compute the target LPIPS features for the real images using VGG16.
        target_features = {}
        for mode in self.modalities:
            target_mode = target_modatilies[mode].clone()
            target_mode = target_mode.to(self.device).to(torch.float32)
            if target_mode.shape[2] > 256:
                target_mode = F.interpolate(target_mode, size=(256, 256), mode='area')
            target_features[mode] = vgg16(target_mode, resize_images=False,
                                          return_lpips=True)  # NOTE: the values of input tensor are normalized from an internal operation of vgg16->inplace operation on target_mode.
            # For this reason we need to use .clone() on tensor target_modatilies[mode] to work on a different memory location and propagate the gradient.
            # We do not use retain_grad on the variable "target_mode" as we do not use its gradient directly: it is only used as an identity function that allows the inplace ops without modifying the original Tensor
            # and propagate the gradient to the original tensor itself.

        w_opt = torch.tensor(w_avg, dtype=torch.float32, device=self.device, requires_grad=True)  # pylint: disable=not-callable # todo  UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor). w_opt = torch.tensor(w_avg, dtype=torch.float32, device=self.device, requires_grad=True)  # pylint: disable=not-callable
        w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=self.initial_learning_rate)

        # Init noise.
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

        history = util_general.list_dict()
        # Early stopping parameters
        steps_no_improve = 0
        best_loss = np.Inf
        best_step = num_steps
        print(f"Early stopping set to {early_stopping}")
        for step in range(num_steps):
            # Learning rate schedule.
            t = step / num_steps
            w_noise_scale = w_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2 # todo check here for w_init
            lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
            lr = self.initial_learning_rate * lr_ramp
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
                    self.modalities[0]: synth.repeat([1, 3, 1, 1])
                }
            elif synth.shape[1] == 3:  # already in the correct format
                synth_modalities = {
                    self.modalities[0]: synth
                }
                pass
            else:  # Multimodal input.
                synth_modalities = {
                    mode: synth[:, idx_mode, :, :].unsqueeze(dim=1).repeat([1, 3, 1, 1]) for idx_mode, mode in enumerate(self.modalities)
                }  # synth[:, idx_mode, :, :] is a leaf variable with gradient, we do not use .clone() as we do not perform inplace operation directly on synth[:, idx_mode, :, :]

            # Pixel-based loss.
            pix_loss = {}
            for mode in self.modalities:
                synth_mode = synth_modalities[mode].clone()
                target_mode = target_modatilies[mode].clone()
                target_mode = target_mode.to(self.device).to(torch.float32)
                pix_loss[mode] = self.w_pix * (torch.mean((target_mode.float() - synth_mode.float()) ** 2))

            # Compute the synthetic LPIPS features for the synthetic images using VGG16.
            synth_features = {}
            dist = {}
            for mode in self.modalities:
                synth_mode = synth_modalities[mode].clone()  # .clone() can be seen as an identity function
                # Downsample image to 256x256 if it's larger than that. VGG was built for 256x256 images.
                if synth_mode.shape[2] > 256:
                    synth_mode = F.interpolate(synth_mode, size=(256, 256), mode='area')
                # Features for synth images.
                synth_features[mode] = vgg16(synth_mode, resize_images=False,
                                             return_lpips=True)  # NOTE: the values of input tensor are normalized from an internal operation of vgg16->inplace operation on synth_mode.
                # For this reason we need to use .clone() on tensor synth_modalities[mode] to work on a different memory location and propagate the gradient.
                # We do not use retain_grad on the variable "synth_mode" as we do not use its gradient directly: it is only used as an identity function that allows the inplace ops without modifying the original Tensor
                # and propagate the gradient to the original tensor itself.

                # Compute distance.
                dist[mode] = self.w_lpips * ((target_features[mode] - synth_features[mode]).square().sum())

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
            loss = reg_loss * self.regularize_noise_weight
            history['reg_loss'].append(reg_loss.item() * self.regularize_noise_weight)
            for mode in self.modalities:
                loss = loss + dist[mode] + pix_loss[mode]
                history[f'{mode}_lpips_loss'].append(dist[mode].item())
                history[f'{mode}_pix_loss'].append(pix_loss[mode].item())

            history['tot_loss'].append(loss.item())

            # Step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            desc = ""
            for mode in self.modalities:
                desc += f"dist_{mode} {dist[mode]:<4.2f}--pix_loss_{mode} {pix_loss[mode]:<4.2f} "

            logprint(f'step {step + 1:>4d}/{num_steps}, lr: {lr}: {desc} loss {float(loss):<5.2f}')

            # Save projected W for each optimization step.
            w_out[step] = w_opt.detach()[0]

            # Normalize noise.
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

            # Early stopping criteria
            running_loss = loss.item() # todo activate early stopping only after n_iterations
            if running_loss < best_loss:
                best_step = step
                best_loss = running_loss
                steps_no_improve = 0
            else:
                steps_no_improve += 1
                # Trigger early stopping
                if steps_no_improve >= early_stopping:
                    print(f'\nEarly Stopping! Total steps: {step}')
                    break
        time_elapsed = time.time() - since
        print('Optimization completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best step: {:0f}'.format(best_step))
        print('Best loss: {:4f}'.format(best_loss))

        projected_w_steps = w_out.repeat([1, G.mapping.num_ws, 1])

        # Save final projected frame and W vector.
        if self.save_final_projection:
            log_dir =  os.path.join(self.outdir, id_patient, 'projections')
            util_general.create_dir(log_dir)
            projected_w = projected_w_steps[best_step]  # projected_w_steps[-1]
            np.savez(os.path.join(log_dir, f'w_{id_slice:05d}-best_step_{best_step}.npz'), w=projected_w.unsqueeze(0).cpu().numpy())

            synth_image = self.G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image = (synth_image + 1) * (255 / 2)  # normalize per stack

            for idx_mode, mode in enumerate(self.modalities):
                log_dir = os.path.join(self.outdir, id_patient, mode, 'image_log')
                util_general.create_dir(log_dir)
                target_mode = target[:, idx_mode, :, :].unsqueeze(0).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                synth_mode = synth_image[:, idx_mode, :, :].unsqueeze(0).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

                log_image = np.concatenate([target_mode, synth_mode], axis=1)
                log_image_pil = PIL.Image.fromarray(log_image.squeeze(-1))
                log_image_pil.save(os.path.join(log_dir, f'img_{id_slice:05d}-best_step_{best_step}.png'))

        # Format history
        history = pd.DataFrame.from_dict(history, orient='index').transpose()

        # Logs
        if verbose:
            self.logs(id_patient=id_patient, id_slice=id_slice, target=target, history=history, projected_w_steps=projected_w_steps)

        return projected_w_steps, best_step

    def logs(self, id_patient, id_slice, target, history, projected_w_steps):

        log_dir = os.path.join(self.outdir, id_patient, 'loss')
        util_general.create_dir(log_dir)
        if id_slice % self.snap_optimization_history == 0:
            if 'claro' in self.outdir:
                plot_training(
                    history=history, plot_training_dir=log_dir, columns_to_plot=['CT_lpips_loss'], title=f'lpips_loss_{id_slice:05d}', xlab='Step', ylab='Loss'
                )
                plot_training(
                    history=history, plot_training_dir=log_dir, columns_to_plot=['CT_pix_loss'], title=f'pix_loss_{id_slice:05d}', xlab='Step', ylab='Loss'
                )
            else:
                plot_training(
                    history=history, plot_training_dir=log_dir, columns_to_plot=['MR_nonrigid_CT_lpips_loss', 'MR_MR_T2_lpips_loss'], title=f'lpips_loss_modalities_{id_slice:05d}',  xlab='Step', ylab='Loss'
                )
                plot_training(
                    history=history, plot_training_dir=log_dir, columns_to_plot=['MR_nonrigid_CT_pix_loss', 'MR_MR_T2_pix_loss'], title=f'pix_loss_modalities_{id_slice:05d}',  xlab='Step', ylab='Loss'
                )
            history.to_csv(os.path.join(log_dir, f'opt_loss_{id_slice:05d}.csv'))

        # Render debug output: optional video and projected image and W vector.
        if id_slice % self.snap_video == 0:
            synth_image_list = []
            for projected_w in projected_w_steps:
                synth_image = self.G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
                synth_image_list.append((synth_image + 1) * (255 / 2))  # normalize per stack

            for idx_mode, mode in enumerate(self.modalities):
                log_dir = os.path.join(self.outdir, id_patient, mode, 'video_log')
                util_general.create_dir(log_dir)
                video = imageio.get_writer(os.path.join(log_dir, f'movie_{id_slice:05d}.mp4'), mode='I', fps=10, codec='libx264',   bitrate='16M')
                print(f'Saving optimization progress video for slice {id_slice:05d}.mp4 of patient {id_patient}')

                for synth_image in synth_image_list:
                    synth_image = synth_image[:, idx_mode, :, :].unsqueeze(0)
                    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    target_image = target[:, idx_mode, :, :].unsqueeze(0).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    video.append_data(np.concatenate([target_image, synth_image], axis=1))

                video.close()