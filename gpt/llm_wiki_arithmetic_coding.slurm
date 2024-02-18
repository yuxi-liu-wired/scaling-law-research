#!/bin/bash
#SBATCH --export=ALL,WANDB_API_KEY
#SBATCH --job-name=yuxi_llm_wiki_arithmetic_coding
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem=8gb
#SBATCH --gres=gpu:1

projectfolder=/nas/ucb/yuxiliu1995/scaling-law-research/gpt
scriptname=llm_wiki_arithmetic_coding.py
modelid=gpt2-large
cd $projectfolder

eval "$(/nas/ucb/yuxiliu1995/miniconda/bin/conda shell.bash hook)"
conda activate math

srun python $scriptname --model_id $modelid --dataset_split test --stride 1 &

wait
deactivate
