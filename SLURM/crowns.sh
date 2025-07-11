#!/bin/bash
#SBATCH --job-name=crop_crowns
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riteshchowdhry@ufl.edu
#SBATCH --account=azare
#SBATCH --output=/home/riteshchowdhry/logs/macrosystems/crop_crowns_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=ai
#SBATCH --gpus=1

date; hostname
module load conda
conda activate dfor_311
pwd


srun -u python crop_crowns.py

date