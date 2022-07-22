#!/usr/bin/env bash
#SBATCH -A snic2022-5-277 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -t 0-24:00:00
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=l.tronchin@unicampus.it

module load CUDA/11.3.1
module load Python/3.8.6-GCCcore-10.2.0

cd /cephyr/users/tronchin/Alvis/ltronchin/envs/stylegan3
source bin/activate

cd /cephyr/users/tronchin/Alvis/ltronchin/Gan-track/src/models/stylegan3/
export command="python train_mi_multimodal.py --outdir=<outdir> --cond=True --data=<source_path>/<dataset>/<dataset>-num-<num_patients>_val-<validation_method>_exps-<n_exps>_fold-<fold>_train-0.80_val-0.10_test-0.10.zip --dataset=<dataset> --split=<split> --modalities=<modalities> --dtype=<dtype> --cfg=<model> --batch=<batch>  --map-depth=<map_depth> --glr=<glr> --dlr=<dlr> --cbase=<cbase> --kimg=<kimg> --gpus=<gpus> --workers=<workers> --gamma=<gamma> --snap=<snap> --mirror=<mirror>  --aug=<aug>  --ada_kimg=<ada_kimg> --aug_opts=<aug_opts> --xint_max=<xint_max> --rotate_max=<rotate_max> --xfrac_std=<xfrac_std> --scale_std=<scale_std> --aniso_std=<aniso_std> --target=<target> --metrics=<metrics> --metrics_cache=<metrics_cache>"

echo "$command"
srun $command

wait

deactivate