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

export command="python train_mi_multimodal.py --outdir=<outdir> --data=<source_path>/<dataset>/<dataset>-num-<num_patients>_train-0.70_val-0.20_test-0.10.zip --dataset=<dataset> --split=<split> --modalities=<modalities> --dtype=<dtype> --cfg=<model> --batch=<batch>  --map-depth=<map_depth> --glr=<glr> --dlr=<dlr> --cbase=<cbase> --kimg=<kimg> --gpus=<gpus> --workers=<workers> --gamma=<gamma> --snap=<snap> --mirror=<mirror>  --aug=<aug>  --ada_kimg=<ada_kimg> --aug_opts=<aug_opts> --xint_max=<xint_max> --rotate_max=<rotate_max> --xfrac_std=<xfrac_std> --scale_std=<scale_std> --aniso_std=<aniso_std> --target=<target> --metrics=<metrics> --metrics_cache=<metrics_cache>"

echo "$command"
srun $command

wait