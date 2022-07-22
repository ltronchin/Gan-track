#!/usr/bin/env bash
#SBATCH -A snic2022-5-277 -p alvis
#SBATCH -N 1 --gpus-per-node=V100:1
#SBATCH -t 0-24:00:00
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=l.tronchin@unicampus.it

module load CUDA/11.3.1
module load Python/3.8.6-GCCcore-10.2.0
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1

cd /cephyr/users/tronchin/Alvis/ltronchin/envs/stylegan3
source bin/activate

cd /cephyr/users/tronchin/Alvis/ltronchin/Gan-track/src/models/stylegan3/
export command="python train_mi_multimodal.py --outdir=/cephyr/users/tronchin/Alvis/ltronchin/Gan-track/reports --cond=True --data=/cephyr/users/tronchin/Alvis/ltronchin/data/interim/claro/claro-num-191_val-bootstrap_exps-5_fold-3_train-0.80_val-0.10_test-0.10.zip --dataset=claro --split=train --modalities=CT --dtype=float32 --cfg=stylegan2 --batch=16  --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384 --kimg=10000 --gpus=1 --workers=3 --gamma=0.8192 --snap=10 --mirror=1  --aug=ada  --ada_kimg=500 --aug_opts=xflip,xint,scale,rotate,aniso,xfrac --xint_max=0.05 --rotate_max=3 --xfrac_std=0.05 --scale_std=0.05 --aniso_std=0.05 --target=0.6 --metrics=fid50k_full --metrics_cache=True"

echo "$command"
srun $command

wait

deactivate