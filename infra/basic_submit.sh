#!/usr/bin/env bash
#SBATCH --job-name=basic-job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

export JAX_PLATFORMS=cuda,cpu

uv run --with 'jax[cuda12]' python -m experiments.baselines.transformer_addition "$@"

