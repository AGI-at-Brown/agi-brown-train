#!/usr/bin/env bash
#SBATCH --job-name=qwen-flash-next
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export JAX_PLATFORMS=cuda,cpu
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

uv run --with 'jax[cuda12]' python -m experiments.pretraining.qwen_flash_next \
  --num-devices 4 --output "artifacts/qwen_flash_next/${SLURM_JOB_ID}.msgpack" "$@"
