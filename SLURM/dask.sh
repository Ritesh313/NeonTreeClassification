#!/bin/bash
#SBATCH --job-name=dask_master
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riteshchowdhry@ufl.edu
#SBATCH --account=azare
#SBATCH --partition=gpu
#SBATCH --output=/home/riteshchowdhry/logs/macrosystems/dask_deepforest/master_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=ai
#SBATCH --gpus=1

date; hostname
module load conda
conda activate dfor_311
pwd


srun -u python dask_deepforest_slurm.py

date