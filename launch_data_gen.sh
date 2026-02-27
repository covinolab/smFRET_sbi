#!/bin/bash -x
#SBATCH --job-name=fret_data_gen
#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --partition=general1
#SBATCH --mail-type=ALL

python src/generate_simulations.py --config conf/simulator/default.yaml --format h5