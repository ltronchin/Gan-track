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
cd /pfs/proj/nobackup/fs/projnb10/snic2020-6-234/minhvu/minhvu/stylegan3/
export PYTHONPATH=$PYTHONPATH:/pfs/stor10/users/home/m/minhvu/Public/stylegan3/lib/python3.9/site-packages
export PYTHONPATH=${PWD}:$PYTHONPATH
echo "Load done"

export command="python scripts/train_medical.py --dataset=<dataset> --outdir=database --cfg=<model> --data=database/<dataset>/<dataset>-num-<num_patients>_train-0.60_val-0.20_test-0.20.zip --gpus=1 --gamma=2 --mirror=<mirror> --aug=<aug> --workers=1 --snap=25 --tick=4 --cmax=512 --cbase=16384 --batch=24 --in_modal=<in_modal> --out_modal=<out_modal>"

echo "$command"
srun $command

wait