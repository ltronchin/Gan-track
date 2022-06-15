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

import torch
from torch.utils.data import DataLoader

from src.utils import util_general
from src.utils import util_medical_data

from PIL import Image

# Argument function
def get_args():
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("-f", "--cfg_file", help="Path of Configuration File", type=str)
    parser.add_argument("-i", "--id_exp", help="Id of experiment to load models and datas", type=int, default=1)
    parser.add_argument("-g", "--gpu", help="CUDA device", type=str, default="cuda:0")
    parser.add_argument("--source_dataset", help="Name of dataset to upload", type=str, default="mnist")
    parser.add_argument("--processed_dataset", help="Name of dataset to create", type=str, default="mnist")
    parser.add_argument("--resolution", help="Target resolution for images", type=int, default=512)

    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=53667)
    return parser.parse_args()

# Configuration file
print("Upload configuration file")
debug = 'develop'
debug = input(f"Enter '{debug}' to run a debug session press enter otherwise")
if debug == 'develop':
    with open('./configs/claro_preprocessing.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = cfg['id_exp']
    worker = cfg['device']['worker']
    source_dataset_name = cfg['data']['source_dataset']
    processed_dataset_name = cfg['data']['processed_dataset']
    res = cfg['data']['image_size']
else:
    args = get_args()
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = args.id_exp
    worker = args.gpu
    source_dataset_name = args.source_dataset
    processed_dataset_name = args.processed_dataset
    res =  args.resolution

# Submit run:
print("Submit run")
# Get new id_exp
util_general.create_dir(os.path.join('log_run', processed_dataset_name))
log_dir = os.path.join('log_run', processed_dataset_name, cfg['network']['model_name'])
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
print("Parameters")
channel = cfg['data']['channel']
convert_to_uint8 = cfg['data']['convert_to_uint8']

iid_class = cfg['data']['iid_classes']
ood_class  = cfg['data']['ood_classes']

# Useful print
print(f"Source dataset: {source_dataset_name}")
print(f"Processed dataset: {processed_dataset_name}")

print(f"iid classes:{iid_class}")
print(f"ood classes:{ood_class}")

print(f"image size: {res}")
print(f"channel: {channel}")
print(f"Convert to uint8: {convert_to_uint8}")

print(f"Model: {cfg['network']['model_name']}")

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

interim_dir_runtime = os.path.join(cfg['data']['interim_dir'], processed_dataset_name, cfg['network']['model_name'])
util_general.create_dir(interim_dir_runtime)

# Data loaders
data_raw = pd.read_excel(os.path.join(interim_dir, f'patients_info_{source_dataset_name}.xlsx'), index_col=0)
id_patients_slice = pd.Series([row.split(os.path.sep)[1].split('.tif')[0] for row in data_raw['image']])
id_patients = pd.Series([idp.split('_')[0] for idp in id_patients_slice.iloc])

box_data = pd.read_excel(cfg["data"]["box_file"])
id_patients_slice_box = box_data['img ID']

id_patients_slice_lung = pd.Series(np.intersect1d(id_patients_slice, id_patients_slice_box))

print(f'Number of images: {len(id_patients_slice_lung)}')
print(f'Number of patients: {len(np.unique(id_patients))}')
print('Create dataloader')
dataset = util_medical_data.ImgDatasetPreparation(data=id_patients_slice_lung, cfg_data=cfg['data'], data_dir=data_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

with tqdm(total=len(dataloader.dataset),  unit='img') as pbar:
    for x, idp, ids in dataloader:  # Iterate over the batches of the dataset
        img = x.detach().cpu().numpy()[0][0]
        idp = idp[0]
        ids = ids[0]

        image = Image.fromarray(img)
        filename = idp + '_' + ids

        if cfg['data']['convert_to_uint8'] is not None:
            assert image.mode == 'L'
            image.save(os.path.join(interim_dir_runtime, f'{filename}.png'), compress_level=0, optimize=False)
        else:
            image.save(os.path.join(interim_dir_runtime, f'{filename}.tif' ), 'tiff', compress_level=0, optimize=False)
        pbar.update(x.shape[0])

print("May be the force with you!")