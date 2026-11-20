#!/bin/bash
# One rank, started by srun from slurm/levelsync.sbatch: start this task's two
# vLLM servers on its own GPU, then run the rank with the arguments given, which
# go to `python -m parabeam.eval.run $PB_MODE` (levelsync or async). MPI starts when
# Python imports mpi4py, after the servers are up, so ranks whose servers start
# quickly wait there for the slow ones.

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
# shellcheck source=configs/carc.env
source "${PB_ENV_FILE:-configs/carc.env}"
export PATH="$PB_ENV/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
source slurm/launch_servers.sh

job="${SLURM_JOB_ID:-local}"
rank="${SLURM_PROCID:-0}"
devices="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$devices" || "$devices" == *,* ]]; then
    echo "FAIL  rank $rank needs exactly one GPU, has CUDA_VISIBLE_DEVICES='$devices'"
    exit 1
fi
# The index means the same GPU to nvidia-smi whether or not cgroups hide the
# others, so this names the physical GPU; rank 0 refuses two ranks on one.
PB_GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$devices")"
export PB_GPU_UUID
echo "rank $rank on $(hostname), GPU $PB_GPU_UUID"

trap pb_stop_servers EXIT
pb_start_servers "logs/pb-levelsync-$job-r$rank" || exit 1
python -m parabeam.eval.run "${PB_MODE:-levelsync}" "$@"
status=$?
echo "rank $rank finished at $(date -u +%FT%TZ) with status $status"
exit "$status"
