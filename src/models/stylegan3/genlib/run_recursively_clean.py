import sys
sys.path.extend(["./", "./src/models/stylegan3/", "./src/models/stylegan3/dnnlib", "./src/models/stylegan3/torch_utils"])

import dnnlib
import click
import os
import re
import copy
import json

from genlib.utils import util_general

# Recursively clean parameters.
@click.command()
@click.option('--outdir_model',                       help='Network output dir', metavar='DIR',                required=True)
@click.option('--outdir_metric',                      help='Metric logs dir', metavar='DIR',                required=True)
@click.option('--metric_name',                        help='Network pickle filename or Metric filename',                type=str, required=True)
@click.option('--recursively_clean',                  help='Delete all snaps (network-snapshot and fakes images).',   type=bool, default=False, show_default=True)
@click.option('--external_exluded_files',              help='Exluded filename from cleaning procedure.',                type=str, default="network-snapshot-005000.pkl,fakes005000.png")

def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)  # Command line arguments.

    # Recursively clean.
    if opts.recursively_clean:
        base_exluded = [
            "log.txt",
            "metric-MR_MR_T2-fid50k_full.jsonl",
            "metric-MR_nonrigid_CT-fid50k_full.jsonl",
            "metric-MR_nonrigid_CT-fid50k_full_check.jsonl",
            "reals.png",
            "fakes_init.png",
            "stats.jsonl",
            "training_options.json"
        ]

        exluded_files = base_exluded + util_general.parse_comma_separated_list(opts.external_exluded_files)

        with open(os.path.join(opts.outdir_metric, f"{opts.metric_name}_analysis.json")) as f:
            fid_min = json.load(f)
        exluded_files = exluded_files + [fid_min[mode]['snapshot_pkl'] for mode in fid_min.keys()]
        exluded_files = exluded_files + [f"fakes{fid_min[mode]['snapshot_pkl'].split(sep='-')[-1].split('.')[0]}.png" for mode in fid_min.keys()]

        recu_files = util_general.list_dir_recursively_with_ignore(dir_path=opts.outdir_model,  ignores=exluded_files)
        for f in recu_files:
            util_general.delete_file(f[0])
            print(f"{f[1]} deleted")
    print('May be the force with you.')


if __name__ == "__main__":
    # For debug.
    my_env = os.environ.copy()
    my_env["PATH"] = "/home/lorenzo/miniconda3/envs/stylegan3/bin:" + my_env["PATH"]
    os.environ.update(my_env)

    main() # pylint: disable=no-value-for-parameter
