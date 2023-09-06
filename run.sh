#!/bin/bash
#sbatch --gpus=1

module load anaconda/2020.11
source activate hdr

data
tar -xf ../../data/dataset_siggraph17.tar -C /dev/shm
data

python train.py
