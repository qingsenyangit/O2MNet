#!/bin/bash
#SBATCH --gpus=1

module load anaconda/2020.11
source activate hdr

data
tar -xf ../data/H5file.tar -C /dev/shm
data

python train.py