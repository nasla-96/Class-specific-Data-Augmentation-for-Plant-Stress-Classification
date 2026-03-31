#!/bin/bash
#SBATCH --job-name=plant_aug_ga
#SBATCH --output=logs/plant_aug_ga_%j.out
#SBATCH --error=logs/plant_aug_ga_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

REPO_DIR=/work/mech-ai-scratch/nasla/Class-specific-Data-Augmentation-for-Plant-Stress-Classification
ENV_NAME=plant_aug

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

source ~/.bashrc
conda activate "${ENV_NAME}"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

python scripts/run_ga_augmentation.py \
  --config configs/default_config.yaml \
  --output_dir outputs
