#!/bin/bash
# Week 1 GPU checks 3, 4 and 5 on one 40 GB A100 (see scripts/gpu_checks.py).
# Cache the questions on a login node first, then submit from the repo root:
#
#   "$PB_ENV/bin/python" -c "from parabeam.eval.datasets import load_subset; load_subset('math500', 'pilot')"
#   sbatch --account=<project_id> slurm/checks.sbatch
#
#SBATCH --job-name=pb-checks
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-40gb
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
source configs/carc.env
export PATH="$PB_ENV/bin:$PATH"  # vllm, and ninja for kernel builds
export TOKENIZERS_PARALLELISM=false

echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -u +%FT%TZ), commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PB_ENV/bin/python" scripts/gpu_checks.py run
