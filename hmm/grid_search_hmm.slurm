#!/bin/bash
#SBATCH --export=ALL,WANDB_API_KEY
#SBATCH --job-name=yuxi_grid_search_hmm
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=20  # Adjust based on how many CPUs you want to use per task
#SBATCH --time=96:00:00
#SBATCH --mem=64gb

# Set project folder
projectfolder=/nas/ucb/yuxiliu1995/scaling-law-research/hmm

# Navigate to the project folder
cd $projectfolder

# Activate the virtual environment
source .venv/bin/activate

# Define how many agents you want to run in parallel
NUM_AGENTS=8

# Start wandb agents in parallel
for (( i=0; i<$NUM_AGENTS; i++ ))
do
   srun --ntasks=1 wandb agent yuxiliu1995/english_hmm/qx4420zo &
done

# Wait for all background processes to finish
wait

# Deactivate the environment
deactivate
