#!/bin/bash
# Week 1 checks 1 and 2 on one A100, plus the environment. Submit from the repo
# root after scripts/setup_env.sh, which creates logs/:
#
#   sbatch --account=<project_id> slurm/preflight.sbatch
#
#SBATCH --job-name=pb-preflight
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-40gb
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

# Slurm runs a spooled copy of this script, so locate the repo from the submit dir.
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
source configs/carc.env

failures=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; failures=$((failures + 1)); }
note() { echo "NOTE  $*"; }

echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -u +%FT%TZ)"

# shellcheck disable=SC2086  # the module list is meant to split
pb_modules $PB_MPI_MODULES || fail "could not load modules: $PB_MPI_MODULES"

gpu_query="$(nvidia-smi --query-gpu=name,memory.total,compute_mode,driver_version \
    --format=csv,noheader,nounits)" || fail "nvidia-smi failed"
echo "gpu: $gpu_query"
IFS=',' read -r gpu_name gpu_mem gpu_mode gpu_driver <<< "$gpu_query"
gpu_mem="${gpu_mem// /}"
gpu_mode="${gpu_mode// /}"
gpu_driver="${gpu_driver// /}"

# Check 1: every run must be on a 40 GB A100.
if [[ "$gpu_name" == *A100* ]] && ((${gpu_mem:-0} >= 39000 && ${gpu_mem:-0} <= 41500)); then
    pass "check 1: 40 GB A100 ($gpu_name, $gpu_mem MiB)"
else
    fail "check 1: not a 40 GB A100 ($gpu_name, ${gpu_mem:-?} MiB)"
fi

# Check 2: exclusive-process mode blocks the second vLLM server unless MPS runs.
if [[ "$gpu_mode" == "Default" ]]; then
    pass "check 2: compute mode is Default"
else
    fail "check 2: compute mode is ${gpu_mode:-unknown}; start MPS inside the job"
fi

# The driver decides whether a CUDA 13 vLLM (0.33 or later) was ever an option.
note "driver $gpu_driver; vLLM 0.33+ needs R580 or newer, the cu129 pin does not"

if "$PB_ENV/bin/python" scripts/check_env.py --gpu; then
    pass "environment: pins match and torch can use the GPU"
else
    fail "environment: see errors above"
fi

if srun --mpi=pmix_v5 -n 1 "$PB_ENV/bin/python" -c \
    'from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size(), "rank via srun")'; then
    pass "mpi4py starts under srun"
else
    fail "mpi4py does not start under srun"
fi

# Downloads may still be running when this job starts, so this only reports.
while IFS=$'\t' read -r name _ _ role; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    if compgen -G "$PB_MODELS/$name/*.safetensors" >/dev/null; then
        note "model $name ($role): present"
    else
        note "model $name ($role): not downloaded yet"
    fi
done < "$PB_MODEL_LIST"

if ((failures > 0)); then
    echo "PREFLIGHT FAILED: $failures check(s)"
    exit 1
fi
echo "PREFLIGHT PASSED"
