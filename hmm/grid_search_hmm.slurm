#!/bin/bash
#SBATCH --export=ALL,WANDB_API_KEY
#SBATCH --job-name=yuxi_grid_search_hmm
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4  # Adjust based on how many CPUs you want to use per task
#SBATCH --time=96:00:00
#SBATCH --mem=64gb

NUM_AGENTS=8

projectfolder=/nas/ucb/yuxiliu1995/scaling-law-research/hmm
wandbid=yuxiliu1995/english_hmm/tbde4807

cd $projectfolder
source .venv/bin/activate

# Start wandb agents in parallel
for (( i=0; i<$NUM_AGENTS; i++ ))
do
   srun --ntasks=1 wandb agent $wandbid &
done

wait
deactivate
