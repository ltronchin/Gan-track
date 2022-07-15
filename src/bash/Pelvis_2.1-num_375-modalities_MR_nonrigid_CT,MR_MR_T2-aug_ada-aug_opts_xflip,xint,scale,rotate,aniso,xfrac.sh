#!/bin/bash
#SBATCH -A SNIC2021-5-452
#SBATCH --exclusive
#SBATCH --time=07-00:00:00
#SBATCH --error=%J_error.out
#SBATCH --output=%J_output.out
#SBATCH -n 1
#SBATCH --gres=gpu:v100:1

ml GCC/10.3.0 OpenMPI/4.1.1 Python/3.9.5 PyTorch/1.10.0-CUDA-11.3.1 torchvision/0.11.1-CUDA-11.3.1
ml Pillow/8.2.0 matplotlib/3.4.2 SciPy-bundle/2021.05 tqdm/4.61.2 Ninja/1.10.2

source $HOME/Public/stylegan3/bin/activate

cd /pfs/proj/nobackup/fs/projnb10/snic2020-6-234/lotr/Gan-track/src/models/stylegan3/
export PYTHONPATH=$PYTHONPATH:/pfs/stor10/users/home/l/lotr0009/Public/stylegan3/lib/python3.9/site-packages
export PYTHONPATH=${PWD}:$PYTHONPATH
echo "Load done"

export command="python train_mi_multimodal.py --outdir=/pfs/proj/nobackup/fs/projnb10/snic2020-6-234/lotr/Gan-track/reports --data=/pfs/proj/nobackup/fs/projnb10/snic2020-6-234/lotr/Gan-track/data/interim/Pelvis_2.1/Pelvis_2.1-num-375_train-0.70_val-0.20_test-0.10.zip --dataset=Pelvis_2.1 --split=train --modalities=MR_nonrigid_CT,MR_MR_T2 --dtype=float32 --cfg=stylegan2 --batch=16  --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384 --kimg=10000 --gpus=1 --workers=3 --gamma=0.8192 --snap=10 --mirror=1  --aug=ada  --ada_kimg=500 --aug_opts=xflip,xint,scale,rotate,aniso,xfrac --xint_max=0.05 --rotate_max=3 --xfrac_std=0.05 --scale_std=0.05 --aniso_std=0.05 --target=0.6 --metrics=fid50k_full --metrics_cache=True"

echo "$command"
srun $command

wait