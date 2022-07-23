import sys
sys.path.extend(["./", "./src/models/stylegan3/", "./src/models/stylegan3/dnnlib", "./src/models/stylegan3/torch_utils"])

import dnnlib
import click
import os
import matplotlib.pyplot as plt
import re
import copy
import json

from genlib.metric import metric_utils
from genlib.utils import util_general

@click.command()
# Data parameters
# Required
@click.option('--outdir',                       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--modalities',                   help="Modalities for StyleGAN",   metavar="STRING",             type=str, default="MR_nonrigid_CT,MR_MR_T2")
@click.option('--dataset',                      help="Raw Dataset name",   metavar="STRING",                    type=str, default="Pelvis_2.1")
@click.option('--dataset_logname',              help="Dataset name used for log files",   metavar="STRING",      type=str, default="Pelvis_2.1-num-375_train-0.70_val-0.20_test-0.10")
@click.option('--split',                        help="Validation split",   metavar="STRING",                    type=str, default="train")
@click.option('--network_pkl',                  help='Network pickle filename or Metric filename',                type=str, required=True)
@click.option('--snap',                         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--repo_no_mask',                 help='Load the no_mask experiments', metavar='BOOL',            type=bool, default=False, show_default=True)
# Projector parameters.
# Required
@click.option('--experiment',                   help='Experiment run id to take from the results',              type=str, required=True)

def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)  # Command line arguments.

    if opts.repo_no_mask:
        print('Load no-mask results.')
        # Update output directory for model results (where to find the trained models).
        outdir_model = os.path.join(opts.outdir, opts.dataset, "repo_no_mask", "training-runs", f"{opts.dataset_logname:s}", f"{opts.modalities:s}")
        # Update output directory to save results.
        opts.outdir = os.path.join(opts.outdir, opts.dataset, "repo_no_mask", "metric-runs", f"{opts.dataset_logname:s}", f"{opts.modalities:s}")
    else:
        # Update output directory for model results (where to find the trained models).
        outdir_model = os.path.join(opts.outdir, opts.dataset, "training-runs", f"{opts.dataset_logname:s}", f"{opts.modalities:s}")
        # Update output directory to save results.
        opts.outdir = os.path.join(opts.outdir, opts.dataset, "metric-runs", f"{opts.dataset_logname:s}", f"{opts.modalities:s}")

    # Initialize the logger.
    dnnlib.util.Logger(should_flush=True)
    desc = f"{opts.dataset_logname:s}-split_{opts.split}-stylegan_exp_{opts.experiment}-metric_name_{opts.network_pkl}"

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(opts.outdir):
        prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(run_dir)

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(run_dir)

    # Launch processes.
    print('Launching processes...')
    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
    _ = metric_utils.get_outdir_model_path(
        outdir=run_dir, outdir_model=outdir_model,  experiment=opts.experiment, metric_name=opts.network_pkl, modalities=opts.modalities.replace(" ", "").split(","), snap=opts.snap, logplot=True,
    )
    print('May be the force with you.')

if __name__ == "__main__":
    # For debug.
    my_env = os.environ.copy()
    my_env["PATH"] = "/home/lorenzo/miniconda3/envs/stylegan3/bin:" + my_env["PATH"]
    os.environ.update(my_env)

    main() # pylint: disable=no-value-for-parameter


