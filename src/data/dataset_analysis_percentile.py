# Author: Lorenzo Tronchin
# Data: Claro Retrospettivo 512x512
# Script to prepare medical data for StyleGAN

# Input:
# Output: folder claro_retrospettivo_tif

#!/usr/bin/

#  Libraries
print('Import the library')
import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

from tqdm import tqdm
import pandas as pd
import argparse
import numpy as np
import os
import yaml
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 12

import torch

from src.models.stylegan3.genlib.utils import util_general

# Argument function


# Configuration file
print("Upload configuration file")
with open('./configs/pelvis_preprocessing.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
worker = cfg['device']['worker']
source_dataset_name = cfg['data']['source_dataset']
res = cfg['data']['image_size']
modes = cfg['data']['modes']

# Submit run:
print("Submit run")
# Get new id_exp
util_general.create_dir(os.path.join('log_run', source_dataset_name))
log_dir = os.path.join('log_run', source_dataset_name, cfg['network']['model_name'])
util_general.create_dir(log_dir)
# Save the configuration file
with open(os.path.join(log_dir, 'configuration.yaml'), 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
# Initialize Logger
logger = util_general.Logger(file_name=os.path.join(log_dir, 'log.txt'), file_mode="w", should_flush=True)
# Copy the code in log_dir
files = util_general.list_dir_recursively_with_ignore('src', ignores=['.DS_Store'], add_base_to_relative=True)
files = [(f[0], os.path.join(log_dir, f[1])) for f in files]
util_general.copy_files_and_create_dirs(files)

# Welcome
from datetime import datetime
now = datetime.now()
date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
print("Hello!",date_time)

# Seed everything
print("Seed all")
util_general.seed_all(cfg['seed'])

# Parameters

# Useful print
print(f"Source dataset: {source_dataset_name}")
print(f"Modality {modes}")

# Register and history
print("Initialize history")
overall_time = util_general.nested_dict()
overall_history = {}
start = time.time()

# Device
print("Select device")
device = torch.device(worker if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print(f'device: {device}')

# Files and Directories
print('Create file and directory')
data_dir = os.path.join(cfg['data']['data_dir'], source_dataset_name)
interim_dir = os.path.join(cfg['data']['interim_dir'], source_dataset_name)
reports_dir =  os.path.join(cfg['data']['reports_dir'], source_dataset_name)
util_general.create_dir(reports_dir)

frames = []
for fname in os.listdir(interim_dir):
    if 'xlsx' in fname:
        print(fname)
        frames.append(pd.read_excel(os.path.join(interim_dir, fname), index_col=0))
df = pd.concat(frames)

print('\nStart analysis to determine the maximum values for rescaling step')
per_list = [key for key in df.keys() if 'per' in key]
per_list = [per_list[p] for p in range(0, len(per_list), 10)] + [per_list[-1]]
for mode in modes:
    std_dict = {}
    mean_dict = {}
    min_dict = {}
    max_dict = {}

    df_mode = df[df['modality'] == mode]
    mmin=df_mode['min_intensity'].min()
    mmax=df_mode['max_intensity'].max()

    fig = plt.figure()
    for per in per_list:
        std_dict[per] = df_mode[per].std()
        mean_dict[per] = df_mode[per].mean()
        min_dict[per] = df_mode[per].min()
        max_dict[per] = df_mode[per].max()
        df_mode[per].plot(kind='kde', xlim=[mmin, mmax], linewidth=0.5)
    plt.xlabel(f'{mode} values')
    plt.suptitle('Kernel Density Estimation across Percentiles')
    fig.savefig(os.path.join(reports_dir, f"kde_percentiles_{mode}.png"), dpi=400, format='png')
    plt.show()

    min_std_idx = np.argsort(list(std_dict.values()))
    min_std_perc_value = list(std_dict.values())[min_std_idx[0]]
    min_std_perc_name = list(std_dict.keys())[min_std_idx[0]]
    print(f"Mode: {mode}")
    print(f"Min value per max intensity: {df_mode['max_intensity'].min()}")
    print(f"Max value per max intensity: {df_mode['max_intensity'].max()}")
    print(f"Minimum std values finded: {min_std_perc_value}, percentile: {min_std_perc_name}")
    print(f"Using the {min_std_perc_name} the values for the upper limit are ranging between:"
          f"[{list(min_dict.values())[min_std_idx[0]]}, {list(max_dict.values())[min_std_idx[0]]}]"
          f" with a mean value: {list(mean_dict.values())[min_std_idx[0]]}")

    print('--diff/slope analysis--')
    thres = 0.01
    abs_diff = np.gradient(list(std_dict.values())) # np.abs(np.diff(np.asarray(list(std_dict.values()))))
    lo, hi = abs_diff.min(), abs_diff.max()
    abs_diff_norm = (abs_diff - lo) / (hi - lo)
    abs_diff_norm_bool = np.asarray(abs_diff_norm < thres, dtype='uint8')
    best_perc_idx = np.argwhere(abs_diff_norm_bool == 1)[-1]
    best_std_perc_value = list(std_dict.values())[best_perc_idx[0]]
    best_std_perc_name = list(std_dict.keys())[best_perc_idx[0]]
    print(f"Std values finded thres {thres}: {best_std_perc_value}, percentile: {best_std_perc_name}")
    print(f"Using the {best_std_perc_name} the values for the upper limit are ranging between:"
          f"[{list(min_dict.values())[best_perc_idx[0]]}, {list(max_dict.values())[best_perc_idx[0]]}]"
          f" with a mean value: {list(mean_dict.values())[best_perc_idx[0]]}")
    print('\n')

    # Std
    fig, ax = plt.subplots()
    plt.plot(std_dict.values(), marker='.')
    plt.xticks(np.arange(len(per_list)), [p.split('_')[-1] for p in per_list], rotation=-90)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    plt.xlabel('Perc')
    plt.ylabel('Standard Deviation')
    plt.title(mode)
    fig.savefig(os.path.join(reports_dir, f"std_{mode}.png"), dpi=400, format='png')
    plt.show()

    # Std + slope
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(std_dict.values(), color='blue', marker='.')
    ax2.plot(np.gradient(list(std_dict.values())), color='black', linestyle='--')
    ax1.set_xlabel('Perc')
    ax1.set_ylabel('Standard Deviation', color='blue')
    ax2.set_ylabel('Slope', color='black')
    plt.title(mode)
    fig.savefig(os.path.join(reports_dir, f"std_slope_{mode}.png"), dpi=400, format='png')
    plt.show()

    # Std differences
    fig, ax = plt.subplots()
    plt.plot(np.abs(np.diff(np.asarray(list(std_dict.values()))[:, np.newaxis], axis=0)), marker='.')
    plt.xticks(np.arange(len(per_list)), [p.split('_')[-1] for p in per_list], rotation=-90)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    plt.xlabel('Perc')
    plt.ylabel('Differences between i and i-1 standard deviation values in module')
    plt.title(mode)
    plt.savefig(os.path.join(reports_dir, f"abs_diff_{mode}.png"), dpi=400, format='png')
    fig.show()

    # Mean
    fig, ax = plt.subplots()
    plt.plot(mean_dict.values(), marker='.')
    plt.xticks(np.arange(len(per_list)), [p.split('_')[-1] for p in per_list], rotation=-90)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    plt.xlabel('Perc')
    plt.ylabel('Mean percentile values')
    plt.title(mode)
    fig.savefig(os.path.join(reports_dir, f"mean_{mode}.png"), dpi=400, format='png')
    plt.show()

    # Mean +/- std
    fig, ax = plt.subplots()
    plt.errorbar(np.arange(len(mean_dict)), mean_dict.values(), yerr=std_dict.values(), ecolor='tomato', label='+/-std', linewidth=2, marker='.')
    plt.xticks(np.arange(len(per_list)), [p.split('_')[-1] for p in per_list], rotation=-90)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    plt.xlabel('Perc')
    plt.ylabel('Mean percentile value + Standard Deviation')
    plt.title(mode)
    plt.legend()
    fig.savefig(os.path.join(reports_dir, f"mean_std_{mode}.png"), dpi=400, format='png')
    plt.show()

print("May the force be with you!")
