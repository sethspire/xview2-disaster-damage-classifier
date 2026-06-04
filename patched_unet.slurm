#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH -A sds_ag
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --mem=80G
#SBATCH --job-name=patched_ps512_final2
#SBATCH --output=patched/sbatch_output/fp_nb_%j.out
#SBATCH --error=patched/sbatch_output/fp_nb_%j.err

# Load environment
module purge
module load miniforge

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/kyn5du/.conda/envs/nn_env

# Move to working directory
cd /home/kyn5du/ML3/final

CUDA_LAUNCH_BLOCKING=1

# run file
/home/kyn5du/.conda/envs/nn_env/bin/python /home/kyn5du/ML3/final/patched_unet.py \
    --patch_size 512 \
    --batch_size 80 \
    --lr 1e-4