#!/bin/bash
# Level-sync or async search on P 40 GB A100s, one MPI rank per GPU, each rank
# with its own generator and PRM servers. P is --ntasks. Submit from the repo root:
#
#   sbatch --account=<project_id> --ntasks=4 slurm/levelsync.sbatch
#   sbatch --account=<project_id> --ntasks=4 --export=ALL,PB_MODE=async,PB_CAP=64 \
#       slurm/levelsync.sbatch
#
# Slurm may spread the ranks over any nodes (each has two GPUs); add
# --ntasks-per-node=2 to pack them. Settings, through --export=ALL,NAME=value:
# PB_MODE (levelsync or async), PB_SUBSET (weak), PB_LIMIT (20), PB_N (256),
# PB_PLACEMENT (keep_local, level-sync only), PB_RUNS (~/parabeam/runs), and for
# async PB_CAP (none), PB_EARLY (1.0), PB_STEAL (1), PB_SCORE (each or level).
# Resubmitting the same
# command resumes.
#
#SBATCH --job-name=pb-levelsync
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-40gb
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
# shellcheck source=configs/carc.env
source "${PB_ENV_FILE:-configs/carc.env}"
pb_modules $PB_MPI_MODULES || { echo "FAIL  could not load $PB_MPI_MODULES"; exit 1; }

job="${SLURM_JOB_ID:-local}"
P="${SLURM_NTASKS:-1}"
runs="${PB_RUNS:-$HOME/parabeam/runs}"
subset="${PB_SUBSET:-weak}"
limit="${PB_LIMIT:-20}"
n="${PB_N:-256}"
mode="${PB_MODE:-levelsync}"
export PB_MODE="$mode"
if [[ "$mode" == async ]]; then
    cap="${PB_CAP:-}"
    early="${PB_EARLY:-1.0}"
    score="${PB_SCORE:-each}"
    extra=(--early "$early" --score "$score")
    [[ -n "$cap" ]] && extra+=(--cap "$cap")
    steal=""
    if [[ "${PB_STEAL:-1}" == 0 ]]; then
        extra+=(--no-steal)
        steal="-nosteal"
    fi
    out="$runs/async-$subset$limit-n$n-p$P-cap${cap:-none}-early$early-score$score$steal"
elif [[ "$mode" == levelsync ]]; then
    placement="${PB_PLACEMENT:-keep_local}"
    extra=(--placement "$placement")
    out="$runs/levelsync-$subset$limit-n$n-p$P-$placement"
else
    echo "FAIL  PB_MODE must be levelsync or async, not '$mode'"
    exit 1
fi
mkdir -p logs "$runs"
echo "job $job: P=$P on $(scontrol show hostnames "${SLURM_JOB_NODELIST:-}" | paste -sd,)"
echo "started at $(date -u +%FT%TZ), writing $out"
cat SNAPSHOT.json 2>/dev/null || git rev-parse HEAD

srun --mpi=pmix_v5 --kill-on-bad-exit=1 bash slurm/levelsync_rank.sh \
    --subset "$subset" --limit "$limit" --N "$n" --M 4 "${extra[@]}" --out "$out"
status=$?
echo "finished at $(date -u +%FT%TZ) with status $status"
exit "$status"
