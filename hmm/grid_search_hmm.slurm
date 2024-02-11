#!/bin/bash
#SBATCH --export=ALL,WANDB_API_KEY
#SBATCH --job-name=yuxi_grid_search_hmm
#SBATCH --ntasks=1
#SBATCH --time=96:00
#SBATCH --mem=8gb

# Set project folder
projectfolder=/nas/ucb/yuxiliu1995/scaling-law-research/hmm

# Navigate to the project folder
cd $projectfolder

# Activate the virtual environment
source .venv/bin/activate

# Run your Python script
wandb agent yuxiliu1995/english_hmm/qx4420zo

# Deactivate the environment
deactivate
