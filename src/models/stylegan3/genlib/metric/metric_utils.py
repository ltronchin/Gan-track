import sys
sys.path.extend(["./", "./src/models/stylegan3/", "./src/models/stylegan3/dnnlib", "./src/models/stylegan3/torch_utils"])

import dnnlib
import os
import json
import numpy as np
import copy
import matplotlib.pyplot as plt
# Set plot parameters.
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 16

# ----------------------------------------------------------------------------
# Reports.

def get_cmap(n: int, name: str = 'hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def plot_metric(report_dir, metric_dir, metric_name, snap_interval):
    cmap = get_cmap(n=len(metric_dir), name='winter')
    kimg = np.arange(start=0, stop=len(list(metric_dir.values())[0]) * snap_interval,
                     step=snap_interval) * 4  # 4 is the kimg_per_tick interval settled at 4 by default in the sytlegan3 implementation
    for idx, key in enumerate(list(metric_dir.keys())):
        plt.plot(kimg, metric_dir[key], linewidth=1, label=key, c=np.reshape([cmap(idx)], newshape=(-1, 4)))
    plt.xlabel('kimg')
    plt.ylabel('Metrics')
    plt.legend()
    plt.savefig(os.path.join(report_dir, f"{metric_name}_vs_kimg.png"), dpi=400, format='png')
    plt.show()


def get_outdir_model_path(outdir, outdir_model, experiment, metric_name="fid50k_full", modalities=None, snap=10, projector_kwargs=None, logplot=False):
    # Add runid filename to the path.
    if modalities is None:
        modalities = ['MR_nonrigid_CT', 'MR_MR_T2']
    runname = [x for x in os.listdir(outdir_model) if experiment in x]
    assert len(runname) == 1

    # Upload checkpoint.
    if metric_name == "fid50k_full":
        print(f"Select model checkpoint using {metric_name}")
        fid_report = {
            mode: [json.loads(line) for line in
                   open(os.path.join(os.path.join(outdir_model, runname[0]), f"metric-{mode}-{metric_name}.jsonl"), 'r')] for mode in  modalities
        }

        fid = {
            mode: [fid_report[mode][snap_idx]['results'][metric_name] for snap_idx in range(len(fid_report[mode]))] for mode in modalities
        }

        # Plot metrics.
        if logplot:
            plot_metric(report_dir=outdir, metric_dir=fid, metric_name=metric_name, snap_interval=snap)

        # Find the network with minimum fid and save the results in Json file.
        fid_min = {
            mode: {
                'min': fid_report[mode][np.argsort(fid[mode])[0]]['results'],
                'snapshot_pkl': fid_report[mode][np.argsort(fid[mode])[0]]['snapshot_pkl']
            } for mode in modalities
        }
        with open(os.path.join(outdir, f"{metric_name}_analysis.json"), 'w') as f:
            json.dump(fid_min, f, ensure_ascii=False, indent=4)  # save as json

        # Upload the checkpoint.
        gettrace = getattr(sys, 'gettrace', None)
        if gettrace():
            print('Hmm, Big Debugger is watching me')
            user_input = 'MR_MR_T2'  # 'MR_nonrigid_CT'
        else:
            print('No sys.gettrace')
            user_input = input(f"To select network-snapshot choose the modality from {[*fid_min]}")
        metric_name = fid_min[user_input]['snapshot_pkl']
        print(f'Modality chosed:             {user_input}')
        print(f"Fid score:                   {fid_min[user_input]['min']}")

    network_pkl = copy.deepcopy(metric_name)
    if projector_kwargs is not None:
        projector_kwargs.network_pkl         = network_pkl
        projector_kwargs.outdir_model        = os.path.join(os.path.join(outdir_model, runname[0]), projector_kwargs.network_pkl)
        print(f'Snapshot selected:           {projector_kwargs.network_pkl}')
        print(f'Outdir model:                {projector_kwargs.outdir_model}')
        return projector_kwargs.outdir_model
    else:
        outdir_model = os.path.join(os.path.join(outdir_model, runname[0]), network_pkl)
        print(f'Snapshot selected:           {network_pkl}')
        print(f'Outdir model:                {outdir_model}')
        return outdir_model