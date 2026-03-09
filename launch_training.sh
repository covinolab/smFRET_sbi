#!/bin/bash -x
#SBATCH --job-name=fret_train
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
    
torchrun --nproc_per_node=8 src/train_density_estimator.py --config-name highnodes.yaml