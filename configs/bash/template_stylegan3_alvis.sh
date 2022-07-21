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

python3 ....py
deactivate