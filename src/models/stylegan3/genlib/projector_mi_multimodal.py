import sys
sys.path.extend(["./", "./src/models/stylegan3/", "./src/models/stylegan3/dnnlib", "./src/models/stylegan3/torch_utils"])

import dnnlib
import click
import os
import json
import numpy as np
from typing import Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import re

from projector import projection_loop

# Set plot parameters.
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 16

#----------------------------------------------------------------------------

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
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Dataset path:        {c.training_set_kwargs.path}')

    print(f'Dataset dtype:       {c.training_set_kwargs.dtype}')
    print(f'Split:               {c.training_set_kwargs.split}')
    print(f'Modalities:          {c.training_set_kwargs.modalities}')

    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:      {c.training_set_kwargs.xflip}')

    print(f'Experiment id for StyleGAN2-ADA {c.projector_kwargs.experiment}')
    print(f'Num steps:                  {c.projector_kwargs.num_steps}')
    print(f'Normalize per channel:      {c.projector_kwargs.save_final_projection}')
    print()


    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    c.projector_kwargs.outdir_model = ger_outdir_model_path(c.run_dir, outdir_model, c=c)

    # Launch processes.
    print('Launching processes...')
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)
    projection_loop.projection_loop(rank=0, **c)


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

#----------------------------------------------------------------------------
# Reports.

def get_cmap(n: int, name: str ='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)

def plot_metric(report_dir, metric_dir, metric_name, snap_interval):
    cmap = get_cmap(n=len(metric_dir), name='winter')
    kimg = np.arange(start=0, stop= len(list(metric_dir.values())[0]) * snap_interval, step=snap_interval) * 4 # 4 is the kimg_per_tick interval settled at 4 by default in the sytlegan3 implementation
    for idx, key in enumerate(list(metric_dir.keys())):
        plt.plot(kimg, metric_dir[key], linewidth=1, label=key, c=np.reshape([cmap(idx)], newshape=(-1, 4)))
    plt.xlabel('kimg')
    plt.ylabel('Metrics')
    plt.legend()
    plt.savefig(os.path.join(report_dir, f"{metric_name}_vs_kimg.png"), dpi=400, format='png')
    plt.show()

#----------------------------------------------------------------------------
# Utils.

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

def parse_separated_list_comma(l):
    if isinstance(l, str):
        return l
    if len(l) == 0:
        return ''
    return ','.join(l)

def ger_outdir_model_path(outdir, outdir_model, c):
    net = c.projector_kwargs.network_pkl
    modalities = c.training_set_kwargs.modalities
    snap = c.network_snapshot_ticks

    # Add runid filename to the path.
    runname = [x for x in os.listdir(outdir_model) if c.projector_kwargs.experiment in x]
    assert len(runname) == 1

    # Upload checkpoint.
    if net == "fid50k_full":
        print(f"Select model checkpoint using {net}")
        fid_report = {
            mode: [json.loads(line) for line in
                   open(os.path.join(os.path.join(outdir_model, runname[0]), f"metric-{mode}-{net}.jsonl"), 'r')] for mode in
            c.training_set_kwargs.modalities
        }

        fid = {
            mode: [fid_report[mode][snap_idx]['results'][net] for snap_idx in
                   range(len(fid_report[mode]))] for mode in modalities
        }

        # Plot metrics.
        plot_metric(report_dir=outdir, metric_dir=fid, metric_name=net, snap_interval=snap)

        # Find the network with minimum fid and save the results in Json file.
        fid_min = {
            mode: {
                'min': fid_report[mode][np.argsort(fid[mode])[0]]['results'],
                'snapshot_pkl': fid_report[mode][np.argsort(fid[mode])[0]]['snapshot_pkl']
            } for mode in modalities
        }
        with open(os.path.join(outdir, f"{net}_analysis.json"), 'w') as f:
            json.dump(fid_min, f, ensure_ascii=False, indent=4)  # save as json

        # Upload the checkpoint.
        gettrace = getattr(sys, 'gettrace', None)
        if gettrace():
            print('Hmm, Big Debugger is watching me')
            user_input = 'MR_MR_T2'
        else:
            print('No sys.gettrace')
            user_input = input(f"To select network-snapshot choose the modality from {[*fid_min]}")
        net = fid_min[user_input]['snapshot_pkl']
        print(f'Modality chosed:             {user_input}')
        print(f"Fid score:                   {fid_min[user_input]['min']}")

    c.projector_kwargs.network_pkl = net
    c.projector_kwargs.outdir_model = os.path.join(os.path.join(outdir_model, runname[0]), c.projector_kwargs.network_pkl)
    print(f'Snapshot selected:           {c.projector_kwargs.network_pkl}')
    print(f'Outdir model:                {c.projector_kwargs.outdir_model}')

    return c.projector_kwargs.outdir_model

#----------------------------------------------------------------------------

@click.command()
# Data parameters
# Required
@click.option('--data',         help='Target image dataset to project to', metavar='[ZIP|DIR]',     type=str, required=True)
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--modalities',   help="Modalities for StyleGAN",   metavar="STRING",             type=str, default="MR_nonrigid_CT,MR_MR_T2")
@click.option('--dataset',      help="Dataset name",   metavar="STRING",                        type=str, default="Pelvis_2.1")
@click.option('--split',        help="Validation split",   metavar="STRING",                    type=str, default="train")
@click.option('--dtype',        help='Dynamic range of images',                                 type=str, default='float32')
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)

#@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
#@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--seed',         help='Random seed', type=int, default=303, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)

# Projector parameters.
# Required
@click.option('--experiment',   help='Experiment run id to take from the results', type=str, required=True)
@click.option('--network_pkl',  help='Network pickle filename or Metric filename', type=str, required=True)
@click.option('--target_fname', help='Path to image for test experiment',  metavar='DIR', default=None, show_default=True)
@click.option('--num-steps',    help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--save_video',   help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--save_final_projection', help='Save the final results of projection', type=bool, default=True, show_default=True)
def main(**kwargs):

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.projector_kwargs = dnnlib.EasyDict()

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_mi_multimodal_kwargs(data=opts.data, dtype=opts.dtype, split=opts.split, modalities=opts.modalities)

    # Hyperparameters & settings.
    c.num_gpus = 1 #opts.gpus
    c.batch_size = 1 # opts.batch
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed

    # Projector arguments
    # Update output directory.
    s_modalities = opts.modalities.replace(" ", "").split(",")
    s_modalities = ",".join(s_modalities)
    c.projector_kwargs.experiment = opts.experiment
    c.projector_kwargs.network_pkl = opts.network_pkl
    c.projector_kwargs.target_fname = opts.target_fname
    c.projector_kwargs.num_steps = opts.num_steps
    c.projector_kwargs.save_video = opts.save_video
    c.projector_kwargs.save_final_projection = opts.save_final_projection

    # Save in outdir_model the directory that contains the results for StyleGAN2-ADA
    opts.outdir_model =  os.path.join(opts.outdir, opts.dataset, "training-runs", f"{dataset_name:s}", f"{s_modalities:s}")

    # Update output directory.
    opts.outdir = os.path.join(opts.outdir, opts.dataset, "projection-runs", f"{dataset_name:s}", f"{s_modalities:s}")
    # Description string.
    desc = f"{dataset_name:s}-gpus_{c.num_gpus:d}-batch_{c.batch_size:d}-dtype_{opts.dtype}-split_{opts.split}-modalities_{s_modalities:s}" # todo add projector parameters
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    launch_projection(c=c, desc=desc, outdir=opts.outdir, outdir_model=opts.outdir_model)

if __name__ == "__main__":
    # For debug.
    my_env = os.environ.copy()
    my_env["PATH"] = "/home/lorenzo/miniconda3/envs/stylegan3/bin:" + my_env["PATH"]
    os.environ.update(my_env)
    main() # pylint: disable=no-value-for-parameter
