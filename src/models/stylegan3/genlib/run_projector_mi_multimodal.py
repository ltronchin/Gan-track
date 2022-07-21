import sys
sys.path.extend(["./", "./src/models/stylegan3/", "./src/models/stylegan3/dnnlib", "./src/models/stylegan3/torch_utils"])

import dnnlib
import click
import os
import json
import matplotlib.pyplot as plt
import re

from projector import projection_loop
from genlib.metric import metric_utils

#----------------------------------------------------------------------------

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

def launch_projection(c, desc, outdir, outdir_model):
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
    print(f'Output directory:                   {c.run_dir}')
    print(f'Number of GPUs:                     {c.num_gpus}')
    print(f'Batch size:                         {c.batch_size} images')
    print(f'Dataset path:                       {c.training_set_kwargs.path}')

    print(f'Dataset dtype:                      {c.training_set_kwargs.dtype}')
    print(f'Split:                              {c.training_set_kwargs.split}')
    print(f'Modalities:                         {c.training_set_kwargs.modalities}')

    print(f'Dataset size:                       {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:                 {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:                     {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:                     {c.training_set_kwargs.xflip}')

    print(f'Experiment id for StyleGAN2-ADA     {c.projector_kwargs.experiment}')
    print(f'Selection step for inversion        {c.projector_kwargs.step_patient_slice}')
    print(f'Num steps:                          {c.projector_kwargs.num_steps}')
    print(f'Patience for early stopping:        {c.projector_kwargs.early_stopping}')
    print(f'w_lpips:                            {c.projector_kwargs.w_lpips}')
    print(f'w_pix:                              {c.projector_kwargs.w_pix}')
    print()

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    c.projector_kwargs.outdir_model = metric_utils.get_outdir_model_path(
        outdir=c.run_dir, outdir_model=outdir_model,  experiment=c.projector_kwargs.experiment, metric_name= c.projector_kwargs.network_pkl, modalities=c.training_set_kwargs.modalities, snap= c.network_snapshot_ticks,  projector_kwargs=c.projector_kwargs,
    )

    # Launch processes.
    print('Launching processes...')
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)
    projection_loop.projection_loop(rank=0, **c)

#----------------------------------------------------------------------------

@click.command()
# Data parameters
# Required
@click.option('--data',                         help='Target image dataset to project to', metavar='[ZIP|DIR]',     type=str, required=True)
@click.option('--outdir',                       help='Where to save the results', metavar='DIR',                required=True)

@click.option('--modalities',                   help="Modalities for StyleGAN",   metavar="STRING",             type=str, default="MR_nonrigid_CT,MR_MR_T2")
@click.option('--dataset',                      help="Dataset name",   metavar="STRING",                        type=str, default="Pelvis_2.1")
@click.option('--split',                        help="Validation split",   metavar="STRING",                    type=str, default="train")
@click.option('--dtype',                        help='Dynamic range of images',                                 type=str, default='float32')
@click.option('--desc',                         help='String to include in result dir name', metavar='STR',     type=str)

@click.option('--gpus',                         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--seed',                         help='Random seed', type=int, default=303, show_default=True)
@click.option('--snap',                         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--workers',                      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)

# Projector parameters.
# Required
@click.option('--experiment',                   help='Experiment run id to take from the results', type=str, required=True)
@click.option('--network_pkl',                  help='Network pickle filename or Metric filename', type=str, required=True)
@click.option('--step_patient_slice',           help='Selection step for slices to invert in each volume', type=click.IntRange(min=1, max=20), default=2, required=True)

@click.option('--target_fname',                 help='Path to image for test experiment',  metavar='DIR', default=None, show_default=True)
@click.option('--num-steps',                    help='Number of optimization steps', type=int, default=500, show_default=True)
@click.option('--early_stopping',               help='Patience for early stopping',  type=click.IntRange(min=1), default=50, show_default=True)

@click.option('--w_lpips',                      help='Weight of lpips loss', type=float, default=1.0, show_default=True)
@click.option('--w_pix',                        help='Weight of recontruction loss', type=float, default=1e-4, show_default=True)

@click.option('--save_final_projection',         help='Save the final results of each projection', type=bool, default=True, show_default=True)
@click.option('--snap_video',                   help='How often to save video snapshots in the patient stack', metavar='TICKS', type=click.IntRange(min=1), default=60, show_default=True)
@click.option('--snap_optimization_history',    help='How often to save loss snapshots in the patient stack', metavar='TICKS', type=click.IntRange(min=1), default=60, show_default=True)
@click.option('--verbose',                      help='Save logs of optimization process (mp4 video and loss history)', type=bool, default=False, show_default=True)

def main(**kwargs):

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.projector_kwargs = dnnlib.EasyDict()

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_mi_multimodal_kwargs(data=opts.data, dtype=opts.dtype, split=opts.split, modalities=opts.modalities)

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = 1 # opts.batch
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed

    # Projector arguments.
    # Update output directory.
    s_modalities = opts.modalities.replace(" ", "").split(",")
    s_modalities = ",".join(s_modalities)
    c.projector_kwargs.experiment = opts.experiment
    c.projector_kwargs.network_pkl = opts.network_pkl
    c.projector_kwargs.step_patient_slice = opts.step_patient_slice
    c.projector_kwargs.target_fname = opts.target_fname
    c.projector_kwargs.num_steps = opts.num_steps
    c.projector_kwargs.early_stopping = opts.early_stopping
    c.projector_kwargs.w_lpips = opts.w_lpips
    c.projector_kwargs.w_pix = opts.w_pix
    c.projector_kwargs.save_final_projection = opts.save_final_projection
    c.projector_kwargs.snap_video = opts.snap_video
    c.projector_kwargs.snap_optimization_history = opts.snap_optimization_history
    c.projector_kwargs.verbose = opts.verbose

    # Save in outdir_model the directory that contains the results for StyleGAN2-ADA
    opts.outdir_model =  os.path.join(opts.outdir, opts.dataset, "training-runs", f"{dataset_name:s}", f"{s_modalities:s}")

    # Update output directory.
    opts.outdir = os.path.join(opts.outdir, opts.dataset, "projection-runs", f"{dataset_name:s}", f"{s_modalities:s}")
    # Description string.
    desc = f"{dataset_name:s}-split_{opts.split}-stylegan_exp_{opts.experiment}-network_filename_{opts.network_pkl}-w_lips_{opts.w_lpips}-w_pix_{opts.w_pix}"
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    launch_projection(c=c, desc=desc, outdir=opts.outdir, outdir_model=opts.outdir_model)

if __name__ == "__main__":
    # For debug.
    my_env = os.environ.copy()
    my_env["PATH"] = "/home/lorenzo/miniconda3/envs/stylegan3/bin:" + my_env["PATH"]
    os.environ.update(my_env)

    main() # pylint: disable=no-value-for-parameter
