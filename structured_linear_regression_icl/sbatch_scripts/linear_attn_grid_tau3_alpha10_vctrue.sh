#!/usr/bin/env bash
#SBATCH --job-name=linear_attn_grid_tau3_alpha10_vctrue
#SBATCH --account=kempner_grads
#SBATCH -p kempner_h100
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=24G
#SBATCH -o log_files/linear_attn_grid_tau3_alpha10_vctrue.out
#SBATCH -e log_files/linear_attn_grid_tau3_alpha10_vctrue.err
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

module load cuda/12.2
source ~/.bashrc
conda activate mytorchenv

# ---- Weights & Biases: force a descriptive run name (and stable id) ----
export WANDB_NAME="tau3_alpha10_vctrue"
export WANDB_RUN_ID="tau3_alpha10_vctrue"   # optional but useful for stable resuming
# export WANDB_PROJECT="linear-attn-demo"  # optional if you prefer env var over --project

python run.py --project context_study_yaml --steps 2000 --d 32 --tau 3 --alpha 10 --variablecontext
