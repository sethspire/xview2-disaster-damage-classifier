#!/bin/bash
#SBATCH --job-name=unet_sweep
#SBATCH -A sds_ag
#SBATCH -p gpu
#SBATCH --output=patched/sbatch_output/%A_%a.out
#SBATCH --error=patched/sbatch_output/%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11

# Load environment
module purge
module load miniforge

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/kyn5du/.conda/envs/nn_env

# Move to working directory
cd /home/kyn5du/ML3/final

# -------------------------
# PARAM GRID
# -------------------------

PATCH_SIZES=(256 512)
BATCH_SIZES=(128 256)
LRS=(1e-4 1e-5)

# Compute indices
ps_idx=$((SLURM_ARRAY_TASK_ID % 2))
bs_idx=$(((SLURM_ARRAY_TASK_ID / 2) % 2))
lr_idx=$(((SLURM_ARRAY_TASK_ID / 6) % 2))

PATCH_SIZE=${PATCH_SIZES[$ps_idx]}
BATCH_SIZE=${BATCH_SIZES[$bs_idx]}
LR=${LRS[$lr_idx]}

echo "Running with:"
echo "PATCH_SIZE=$PATCH_SIZE"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "LR=$LR"

/home/kyn5du/.conda/envs/nn_env/bin/python /home/kyn5du/ML3/final/patched_unet.py \
    --patch_size $PATCH_SIZE \
    --batch_size $BATCH_SIZE \
    --lr $LR