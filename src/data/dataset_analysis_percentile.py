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

import torch

from src.utils import util_general

# Argument function


# Configuration file
print("Upload configuration file")
with open('./configs/pelvic_preprocessing.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
id_exp = cfg['id_exp']
worker = cfg['device']['worker']
source_dataset_name = cfg['data']['source_dataset']
res = cfg['data']['image_size']
modatility = cfg['data']['modatility']

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
files = util_general.list_dir_recursively_with_ignore('src', ignores=['.DS_Store', 'models'], add_base_to_relative=True)
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
print(f"Modatility {modatility}")

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
    print(fname)
    frames.append(pd.read_excel(os.path.join(interim_dir, fname), index_col=0))
df = pd.concat(frames)

per_list = [key for key in df.keys() if 'per' in key]
for mode in modatility:
    std_dict = {}
    mean_dict = {}
    df_mode = df[df['modality'] == mode]
    mmin=df_mode['min_intensity'].min()
    mmax=df_mode['max_intensity'].max()

    fig = plt.figure()
    for per in per_list:
        std_dict[per] = df_mode[per].std()
        mean_dict[per] = df_mode[per].mean()
        df_mode[per].plot(kind='kde', xlim=[mmin, mmax], linewidth=0.5)
    plt.xlabel(f'{mode} values')
    plt.suptitle('Kernel Density Estimation across Percentiles')
    fig.savefig(os.path.join(reports_dir, f"kde_percentiles_{mode}.png"), dpi=400, format='png')
    plt.show()

    # Std
    plt.plot(std_dict.values(), marker='.')
    plt.xlabel('Perc from 99.00 to 99.99 with step 0.01')
    plt.ylabel('Standard Deviation')
    plt.title(mode)
    plt.savefig(os.path.join(reports_dir, f"std_{mode}.png"), dpi=400, format='png')
    plt.show()

    # Std + slope
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(std_dict.values(), color='blue', marker='.')
    ax2.plot(np.gradient(list(std_dict.values())), color='black', linestyle='--')
    ax1.set_xlabel('Perc from 99.00 to 99.99 with step 0.01')
    ax1.set_ylabel('Standard Deviation', color='blue')
    ax2.set_ylabel('Slope', color='black')
    plt.title(mode)
    fig.savefig(os.path.join(reports_dir, f"std_{mode}.png"), dpi=400, format='png')
    plt.show()

    # Std differences
    plt.plot(np.abs(np.diff(np.asarray(list(std_dict.values()))[:, np.newaxis], axis=0)), marker='.')
    plt.xlabel('Perc from 99.00 to 99.99 with step 0.01')
    plt.ylabel('Differences between i and i-1 values in module')
    plt.title(mode)
    plt.savefig(os.path.join(reports_dir, f"abs_diff_{mode}.png"), dpi=400, format='png')
    plt.show()

    # Mean
    plt.plot(mean_dict.values(), marker='.')
    plt.xlabel('Perc from 99.00 to 99.99 with step 0.01')
    plt.ylabel('Mean')
    plt.title(mode)
    plt.savefig(os.path.join(reports_dir, f"mean_{mode}.png"), dpi=400, format='png')
    plt.show()

    # Mean +/- std
    plt.errorbar(np.arange(len(mean_dict)), mean_dict.values(), yerr=std_dict.values(), ecolor='tomato', label='+/-std', linewidth=2, marker='.')
    plt.xlabel('Perc from 99.00 to 99.99 with step 0.01')
    plt.ylabel('Mean + Standard Deviation')
    plt.title(mode)
    plt.legend()
    plt.savefig(os.path.join(reports_dir, f"mean_std_{mode}.png"), dpi=400, format='png')
    plt.show()




