#!/usr/bin/env bash
#SBATCH --job-name=basic-job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export JAX_PLATFORMS=cuda

uv run --with 'jax[cuda12]' python -m experiments.baselines.transformer_addition "$@"
