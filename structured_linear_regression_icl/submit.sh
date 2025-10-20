#!/usr/bin/env bash
# asymetrictcontextlength_linearattention_test
#
#SBATCH --job-name=asymetrictcontextlength_linearattention_test
#SBATCH --account kempner_grads
#SBATCH -p kempner_h100
#SBATCH --gres=gpu:1
#SBATCH -n 1                
#SBATCH -N 1               
#SBATCH -t 01:00:00   
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=24G
#SBATCH -o log_files/asymetrictcontextlength_linearattention_test.out 
#SBATCH -e log_files/asymetrictcontextlength_linearattention_test.err  
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

module load cuda/12.2  
source ~/.bashrc
conda activate mytorchenv  

#python run.py --project linear-attn-demo --run_name "variablecontext" --steps 2000 --d 32 --alpha 1 --tau 3 --variablecontext True
python run.py --project linear-attn-demo --run_name "fixedcontext_CORRECT" --steps 2000 --d 32 --alpha 1 --tau 3 --no-variablecontext