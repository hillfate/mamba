#!/bin/bash
#SBATCH --job-name=train_bigram_mamba
#SBATCH --nodelist=dgx19
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00                  # Job time limit (7 days)
#SBATCH --cpus-per-task=128
#SBATCH --mem=320GB
#SBATCH --output=mamba_eval_%j.out
#SBATCH --error=mamba_eval_%j.err

# Load the conda setup script to enable `conda activate`
source /network/rit/home/yd291146/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate sam_mamba

# Change directory to the project folder
cd ~/projects/mamba/samba

# Run the training script
# ./run_train.sh
python -u slimpj_ppl_eval.py