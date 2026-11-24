#!/usr/bin/env bash
#SBATCH --job-name=qwen-flash-next
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

printf '[%s] Starting Slurm job %s on %s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${HOSTNAME}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export JAX_PLATFORMS=cuda,cpu
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
if [[ -f infra/hf-token.env ]]; then
  set -a
  source infra/hf-token.env
  set +a
fi

uv_dependencies=(--with 'jax[cuda12]')
tracking_args=()
if [[ -f wandb.env ]]; then
  set -a
  source wandb.env
  set +a
fi
if [[ -n ${WANDB_API_KEY:-} ]]; then
  export WANDB_PROJECT="${WANDB_PROJECT:-qwen-flash-next}"
  export WANDB_NAME="${WANDB_NAME:-slurm-${SLURM_JOB_ID}}"
  uv_dependencies+=(--with wandb)
  tracking_args+=(--wandb)
fi

printf '[%s] Launching Qwen Flash Next training\n' "$(date -Is)"
exec uv run "${uv_dependencies[@]}" python -u -m experiments.pretraining.qwen_flash_next \
  --num-devices 4 --output "artifacts/qwen_flash_next/${SLURM_JOB_ID}.msgpack" \
  "${tracking_args[@]}" "$@" --common-crawl
