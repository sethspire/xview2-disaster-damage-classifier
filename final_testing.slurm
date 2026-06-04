#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH -A sds_ag
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --mem=80G
#SBATCH --job-name=full_img_final
#SBATCH --output=sbatch_output/fp_nb_%j.out
#SBATCH --error=sbatch_output/fp_nb_%j.err

# Load environment
module purge
module load miniforge
conda activate /home/kyn5du/.conda/envs/nn_env

# Move to working directory
cd /home/kyn5du/ML3/final

# help with printing to out file
export PYTHONUNBUFFERED=1

# Run notebook
# jupyter nbconvert \
#     --to notebook \
#     --execute Final_Project_Testing.ipynb \
#     --output sbatch_output/FP_Test--OUT--${SLURM_JOB_ID}.ipynb \
#     --ExecutePreprocessor.timeout=-1
papermill Final_Project_Testing.ipynb \
          sbatch_output/FP_Test--OUT--${SLURM_JOB_ID}.ipynb \
          --log-output