## Author: Lorenzo Tronchin
# Script useful to create a bash file to run the "calc_metrics.py" script from stylegan3 folder

#!/usr/bin/
#  Libraries
print('Import the library')
import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import argparse
import os

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

# Argument function
def get_args():
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT')
    parser.add_argument('--metrics', help='Quality metrics',  type=str,  default='fid50k_full,kid50k_full,pr50k3_full,ppl2_wend')
    parser.add_argument("--network_folder", help="PATH to the folder that contains all the reports from StyleGAN training", metavar='PATH', required=True) # /home/lorenzo/GANReverseEngineer/reports/claro_retrospettivo/stylegan2-ada/00000-stylegan2--gpus2-batch32-gamma0.4096/
    parser.add_argument("--bash_folder", help="PATH to save the BASH script", metavar='PATH',  required=True)  # /home/lorenzo/GANReverseEngineer/src/bash/

    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=53667)
    return parser.parse_args()

args = get_args()
gpus = args.gpus
metrics = args.metrics
network_folder =  args.network_folder
bash_folder = args.bash_folder

with open(os.path.join(bash_folder, f'calc_metrics_{os.path.basename(os.path.normpath(network_folder))}.sh'), "w") as f:
    f.write('#!/bin/bash')
    f.write('\n')
    list_snapshot = os.listdir(network_folder)
    list_snapshot.sort()
    list_snapshot = [name for name in list_snapshot if 'network-snapshot' in name]
    for network_snapshot in list_snapshot:
        network_pkl = os.path.join(network_folder, network_snapshot)
        f.write(f'CUDA_VISIBLE_DEVICES=1,2 python3 /home/lorenzo/GANReverseEngineer/src/models/stylegan3/calc_metrics.py --gpus={gpus} --metrics={metrics} --network={network_pkl}')
        f.write('\n')
        f.write('sleep 1m')
        f.write('\n')

print("May be the force with you!")