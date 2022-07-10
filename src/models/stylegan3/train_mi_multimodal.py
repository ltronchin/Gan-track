# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

import os
import click
import re
import json
import tempfile
import torch
import sys

import dnnlib
from training import training_loop_mi_multimodal
from metrics import metric_main_mi_multimodal
from torch_utils import training_stats
from torch_utils import custom_ops
import requests

# CUSTOMIZATION START
#----------------------------------------------------------------------------
def notification_ifttt(info):
    private_key = "isnY23hWBGyL-mF7F18BUAC-bGAN6dx1UAPoqnfntUa"
    url = "https://maker.ifttt.com/trigger/Notification/json/with/key/" + private_key
    requests.post(url, json={'Info': str(info)})
#----------------------------------------------------------------------------
# CUSTOMIZATION END

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    # CUSTOMIZING START
    training_loop_mi_multimodal.training_loop(rank=rank, **c)
    # CUSTOMIZING END

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    # CUSTOMIZING START
    print(f'Dataset dtype:       {c.training_set_kwargs.dtype}')
    print(f'Split:               {c.training_set_kwargs.split}')
    print(f'Modalities:          {c.training_set_kwargs.modalities}')
    print(f'Metric Cache:        {c.metrics_cache}')
    # CUSTOMIZING END
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:      {c.training_set_kwargs.xflip}')
    # CUSTOMIZING START
    print(f'ADA adjustment speed:{c.ada_kimg}')
    # CUSTOMIZING END
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir) # subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

# CUSTOMIZING START
def init_dataset_mi_multimodal_kwargs(data, dtype, split, modalities):
    # Convert string to list.
    modalities = (modalities.replace(" ", "").split(","))
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset_mi_multimodal.CustomImageFolderDataset', path=data, dtype=dtype, use_labels=True, max_size=None, xflip=False, split=split, modalities=modalities) # create attributes for the Dataset class
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
# CUSTOMIZING END


#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

# CUSTOMIZING START
def parse_separated_list_comma(l):
    if isinstance(l, str):
        return l
    if len(l) == 0:
        return ''
    return ','.join(l)
# CUSTOMIZING STOP
#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--cfg',          help='Base configuration',                                       type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
# CUSTOMIZING START-ADDING NEW PARAMETERs
@click.option('--dtype',        help='Dynamic range of images',                                 type=str, default='float32')
@click.option('--modalities',   help="Modalities for StyleGAN",   metavar="STRING",             type=str, default="MR_nonrigid_CT,MR_MR_T2",  required=True)
@click.option('--dataset',      help="Dataset name",   metavar="STRING",                        type=str, default="Pelvis_2.1",   required=True)
@click.option('--split',        help="Validation split",   metavar="STRING",                    type=str, default="train",   required=True)
@click.option('--metrics_cache',help="Use cache to upload precomputed features for real images",type=bool,default=False, required=True)
# CUSTOMIZING END
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), required=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                   type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='ada', show_default=True)
# CUSTOMIZING START-ADDING NEW PARAMETERs
@click.option('--ada_kimg',     help='ADA adjustment speed',                                    type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--aug_opts',     help='Augmentation transformation option to enable',            type=parse_comma_separated_list, default='xflip,rotate90,xint,scale,rotate,aniso,xfrac', show_default=True)
# CUSTOMIZING END
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                  type=click.IntRange(min=0), default=0, show_default=True)

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

def main(**kwargs):
    """Train a GAN using the techniques described in the paper
    "Alias-Free Generative Adversarial Networks".

    Examples:
    \b Train StyleGAN2 for Pelvis_2.1 with 375 patients at 256x256 resolution using 2 GPUs
    python train_mi_multimodal.py
        --outdir=/home/lorenzo/Gan\ tracker/reports --cfg=stylegan2 \\
        --data=/home/lorenzo/data_m2/data/interim/Pelvis_2.1/Pelvis_2.1-num-375_train-0.70_val-0.20_test-0.10.zip \\
        --dataset=Pelvis_2.1 --dtype=float32 --modalities=MR_nonrigid_CT,MR_MR_T2  --split='train' --metrics_cache=True \\
        --gpus=2 --batch=32 --gamma=0.4096 --mirror=1 --kimg=5000 --glr=0.0025 --dlr=0.0025 --snap=10 --cbase=16384
    """

    # CUSTOMIZING START
    import shutil
    cache_dir = dnnlib.util.take_cache_dir_path()
    cache_dir_metric = os.path.join(cache_dir, 'gan-metrics')
    if os.path.isdir(cache_dir_metric):
        #user_input = input(f"Hi! 'gan-metrics' directory finded in {cache_dir_metric}. Do you want to remove it? \nY/N ")
        user_input = 'N'
        if user_input == 'Y':
            shutil.rmtree(cache_dir_metric)
            print('Deleted.')
    # CUSTOMIZING END

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    # CUSTOMIZING START
    c.training_set_kwargs, dataset_name = init_dataset_mi_multimodal_kwargs(data=opts.data, dtype=opts.dtype, split=opts.split, modalities=opts.modalities)
    # CUSTOMIZING END

    # CUSTOMIZING START
    c.metrics_cache = opts.metrics_cache
    # CUSTOMIZING END

    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = (8 if opts.cfg == 'stylegan2' else 2) if opts.map_depth is None else opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.loss_kwargs.r1_gamma = opts.gamma
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    # CUSTOMIZING START
    if any(not metric_main_mi_multimodal.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main_mi_multimodal.list_valid_metrics()))
    # CUSTOMIZING END

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'stylegan2':
        c.G_kwargs.class_name = 'training.networks_stylegan2.Generator'
        c.loss_kwargs.style_mixing_prob = 0.9 # Enable style mixing regularization.
        c.loss_kwargs.pl_weight = 2 # Enable path length regularization.
        c.G_reg_interval = 4 # Enable lazy regularization for G.
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
        c.loss_kwargs.pl_no_weight_grad = True # Speed up path length regularization by skipping gradient computation wrt. conv2d weights.
    else:
        c.G_kwargs.class_name = 'training.networks_stylegan3.Generator'
        c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
        if opts.cfg == 'stylegan3-r':
            c.G_kwargs.conv_kernel = 1 # Use 1x1 convolutions.
            c.G_kwargs.channel_base *= 2 # Double the number of feature maps.
            c.G_kwargs.channel_max *= 2
            c.G_kwargs.use_radial_filters = True # Use radially symmetric downsampling filters.
            c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
            c.loss_kwargs.blur_fade_kimg = c.batch_size * 200 / 32 # Fade out the blur during the first N kimg.

    # Augmentation.
    if opts.aug != 'noaug':
        # CUSTOMIZING START -- added options to turn on some augmentation
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **{aug: 1 for aug in opts.aug_opts})
        if opts.aug == 'ada':
            c.ada_target = opts.target
            c.ada_kimg = opts.ada_kimg # added
        if opts.aug == 'fixed':
            c.augment_p = opts.p
        # CUSTOMIZING END

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # CUSTOMIZING START
    s_modalities = opts.modalities.replace(" ", "").split(",")
    s_modalities = ",".join(s_modalities)

    if opts.aug != 'noaug':
        s_aug_opts = parse_separated_list_comma(opts.aug_opts)
    else:
        s_aug_opts = 'noaug'
        c.ada_kimg = opts.ada_kimg = 'noaug'

    # Update output directory.
    opts.outdir = os.path.join(opts.outdir, opts.dataset, "training-runs", f"{dataset_name:s}", f"{s_modalities:s}")
    # Description string.
    desc = f"{dataset_name:s}-{opts.cfg:s}-gpus_{c.num_gpus:d}-batch_{c.batch_size:d}-gamma_{c.loss_kwargs.r1_gamma:g}-dtype_{opts.dtype}-split_{opts.split}-modalities_{s_modalities:s}--aug_{opts.aug}-ada_kimg_{c.ada_kimg}-aug_opts_{s_aug_opts}"
    if opts.desc is not None:
        desc += f'-{opts.desc}'
    # CUSTOMIZING END

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    # CUSTOMIZING START
    # FOR DEBUG REASON:
    # RuntimeError:Ninja is required to load C++ extension #167
    # Solutions: The subprocess does not include the lib path of conda environments. So manually set the environments in the script (https://github.com/zhanghang1989/PyTorch-Encoding/issues/167)
    my_env = os.environ.copy()
    my_env["PATH"] = "/home/lorenzo/miniconda3/envs/stylegan3/bin:" + my_env["PATH"]
    os.environ.update(my_env)

    try:
        main() # pylint: disable=no-value-for-parameter
    except OSError as err:
         fstr = "OS error: {0}".format(err)
         print(fstr)
         notification_ifttt(fstr)
    except ImportError as err:
         fstr = f"Import error: {err}"
         print(fstr)
         notification_ifttt(fstr)
    except MemoryError as err:
         fstr = f"Memory error: {err}"
         print(fstr)
         notification_ifttt(fstr)
    except BaseException as err: # to consider exception not listed above. Use it with
        fstr = f"Unexpected {err=}, {type(err)=}"
        notification_ifttt(fstr)
        raise # to raise the error
    # CUSTOMIZING END

#----------------------------------------------------------------------------
