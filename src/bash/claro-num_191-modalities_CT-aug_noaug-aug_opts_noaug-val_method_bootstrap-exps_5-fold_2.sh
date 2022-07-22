#!/usr/bin/env bash
#SBATCH -A PROJECTID  -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -t 0-24:00:00
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=email

module load CUDA/11.3.1
module load Python/3.8.6-GCCcore-10.2.0

cd /mimer/NOBACKUP/groups/snic2022-5-277/ltronchin/Gan-track/envs/stylegan3
source bin/activate

cd /mimer/NOBACKUP/groups/snic2022-5-277/ltronchin/Gan-track/src/models/stylegan3/

export command="python train_mi_multimodal.py --outdir=/cephyr/users/tronchin/Alvis/ltronchin/Gan-track/reports --cond=True --data=/cephyr/users/tronchin/Alvis/ltronchin/data/interim/claro/claro-num-191_val-bootstrap_exps-5_fold-2_train-0.80_val-0.10_test-0.10.zip --dataset=claro --split=train --modalities=CT --dtype=float32 --cfg=stylegan2 --batch=16  --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384 --kimg=10000 --gpus=1 --workers=3 --gamma=0.8192 --snap=10 --mirror=1  --aug=noaug  --ada_kimg=500 --aug_opts=noaug --xint_max=0 --rotate_max=0 --xfrac_std=0 --scale_std=0 --aniso_std=0 --target=0 --metrics=fid50k_full --metrics_cache=True"

echo "$command"
srun $command

wait

deactivate