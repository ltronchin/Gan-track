import os
import click
import json
import dnnlib
import glob
import shutil
import numpy as np
import pandas as pd
import engine.utils.path_utils as path_utils


def get_network(path):
    if "_stylegan2_" in path:
        return "stylegan2"
    elif "_condstylegan2_" in path:
        return "condstylegan2"
    else:
        return "condstylegan2-v"


def analyze_jsonl_stylegan(path):
    c = dnnlib.EasyDict()  # Command line arguments.
    path = os.path.join(path, "metric-fid50k_full.jsonl")

    with open(path) as f:
        data = [json.loads(line) for line in f]

    c.best_score = np.inf
    c.best_model = ""
    c.num_lines = len(data)
    c.network = get_network(path)
    c.last_model = data[-1]["snapshot_pkl"].split(".")[0].split("-")[-1]
    c.experiment = path_utils.get_filename(path_utils.get_parent_dir(path))
    c.run = c.experiment.split("-")[0]
    c.experiment = '-'.join(c.experiment.split("-")[1:])
    c.dataset = c.experiment.split("-")[0]

    for line in data:
        s = line["results"]["fid50k_full"]
        m = line["snapshot_pkl"]
        if s < c.best_score:
            c.best_score = s
            c.best_model = m

    print()
    print(f"Dataset   : {c.dataset}")
    print(f"Run       : {c.run}")
    print(f"Network   : {c.network}")
    print(f"Experiment: {c.experiment}")
    print(f"Num lines : {c.num_lines}")
    print(f"Last model: {c.last_model}")
    print(f"Best model: {c.best_model}")
    print(f"Best score: {c.best_score}")

    print()

    return {
        "Dataset": c.dataset,
        "Run": c.run,
        "Network": c.network,
        "Exp": c.experiment,
        "Num lines": c.num_lines,
        "Last model": c.last_model,
        "Best model": c.best_model,
        "Best score": c.best_score,
    }


def get_dir_list(root, key_word="metric-fid50k_full.jsonl"):
    l = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if key_word in name and "best_models" not in path:
                l.append(os.path.join(path, name))

    return l


@click.command()
# Required.
@click.option(
    "--dir",
    required=True,
    default="_database",
)
@click.option(
    "--is_write_gan",
    required=True,
    default=0,
)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)  # Command line arguments.

    # stylegan
    folders = get_dir_list(opts.dir, key_word="metric-fid50k_full.jsonl")
    folders = sorted(folders)
    rows_list = []
    for folder in folders:
        path = path_utils.get_parent_dir(folder)
        d = analyze_jsonl_stylegan(path)
        rows_list.append(d)

    df = pd.DataFrame(rows_list)
    df.to_csv(os.path.join(opts.dir, "gan_analysis.csv"), index=False)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
