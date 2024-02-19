#!/bin/bash
#SBATCH --export=ALL,WANDB_API_KEY
#SBATCH --job-name=yuxi_llm_wiki_arithmetic_coding
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem=8gb
#SBATCH --gres=gpu:1

# for model_id in gpt2 gpt2-medium gpt2-large gpt2-xl; do sbatch --export=ALL,WANDB_API_KEY,MODEL_ID=$model_id llm_wiki_arithmetic_coding.slurm; done

projectfolder=/nas/ucb/yuxiliu1995/scaling-law-research/gpt
scriptname=llm_wiki_arithmetic_coding.py
modelid=${MODEL_ID:-gpt2}

cd $projectfolder

eval "$(/nas/ucb/yuxiliu1995/miniconda/bin/conda shell.bash hook)"
conda activate math

srun python $scriptname --model_id $modelid --dataset_split test --stride 1 &

wait
deactivate
