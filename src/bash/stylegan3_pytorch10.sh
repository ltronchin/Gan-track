#!/bin/bash
ml GCC/10.3.0 OpenMPI/4.1.1 Python/3.9.5 PyTorch/1.10.0-CUDA-11.3.1 torchvision/0.11.1-CUDA-11.3.1 Pillow/8.2.0 matplotlib/3.4.2 SciPy-bundle/2021.05 tqdm/4.61.2 Ninja/1.10.2 && source $HOME/Public/stylegan3/bin/activate && cd /pfs/proj/nobackup/fs/projnb10/snic2020-6-234/lotr/Gan-track/src/models/stylegan3/ && export PYTHONPATH=$PYTHONPATH:/pfs/stor10/users/home/l/lotr0009/Public/stylegan3/lib/python3.9/site-packages && export PYTHONPATH=${PWD}:$PYTHONPATH && echo "Done"