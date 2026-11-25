#!/bin/bash
# Submit the rest of the run matrix, about 80 GPU-hours, held until the jobs
# given have finished (whether or not they succeeded). From the repo root:
#
#   bash slurm/submit_matrix.sh <account> [job_id ...]
#   bash slurm/submit_matrix.sh prasanna_933 14721459 14721460 ...
#
# Every job is resumable. Runs too long for one job's time limit (all 500
# MATH500 problems) are split into parts that each resume the last, chained one
# after another. Prints one line per job: its ID and what it runs.
#
#   strong scaling   MATH500 timing (100), 3 runs: serial, level-sync and async at P = 1, 2, 4
#   weak scaling     MATH500 weak (50), 3 runs: level-sync P = 1 at N = 256 as the base,
#                    level-sync and async at N = 512 (P = 2) and N = 1,024 (P = 4)
#   accuracy         all 500 MATH500 problems, once: serial, and async at P = 4
#   width, variance  100 problems, once: serial, level-sync and async P = 4 at N = 64,
#                    and on GSM8K (timing) at N = 256

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

if (($# < 1)); then
    echo "usage: bash slurm/submit_matrix.sh <account> [job_id ...]" >&2
    exit 2
fi
account=$1
shift
held=""
(($#)) && held="$(IFS=:; echo "$*")"
runs="PB_RUNS=${PB_RUNS:-$HOME/parabeam/runs/aio}"
async="PB_MODE=async,PB_SCORE=level,PB_STEAL=0,PB_EARLY=1.0"
count=0

# submit TASKS SCRIPT EXPORTS [AFTER_ID]: one job, after the held jobs and AFTER_ID.
submit() {
    local tasks=$1 script=$2 exports=$3 after=${4:-} deps=$held id
    [[ -n "$after" ]] && deps="${deps:+$deps:}$after"
    id=$(sbatch --parsable --account="$account" --ntasks="$tasks" \
        ${deps:+--dependency=afterany:$deps} --export="ALL,$runs,$exports" "slurm/$script")
    id=${id%%;*}
    count=$((count + 1))
    printf '%s  P=%s  %s  %s\n' "$id" "$tasks" "${script%.sbatch}" "$exports" >&2
    echo "$id"
}

# chain PARTS TASKS SCRIPT EXPORTS: the same run PARTS times, each part resuming the last.
chain() {
    local parts=$1 tasks=$2 script=$3 exports=$4 last=""
    for _ in $(seq "$parts"); do
        last=$(submit "$tasks" "$script" "$exports" "$last")
        count=$((count + 1))  # submit ran in a subshell, so its count was lost
    done
}

strong="PB_SUBSET=timing,PB_LIMIT=100"
weak="PB_SUBSET=weak,PB_LIMIT=50"
for k in 1 2 3; do
    submit 1 serial.sbatch "PB_SUBSET=timing,PB_LIMIT=100,PB_REPEAT=$k" >/dev/null
    for P in 1 2 4; do
        submit "$P" levelsync.sbatch "$strong,PB_REPEAT=$k" >/dev/null
        submit "$P" levelsync.sbatch "$async,$strong,PB_REPEAT=$k" >/dev/null
    done
    submit 1 levelsync.sbatch "$weak,PB_REPEAT=$k" >/dev/null
    submit 2 levelsync.sbatch "$weak,PB_N=512,PB_REPEAT=$k" >/dev/null
    submit 2 levelsync.sbatch "$async,$weak,PB_N=512,PB_REPEAT=$k" >/dev/null
    submit 4 levelsync.sbatch "$weak,PB_N=1024,PB_REPEAT=$k" >/dev/null
    submit 4 levelsync.sbatch "$async,$weak,PB_N=1024,PB_REPEAT=$k" >/dev/null
done

# About 5.8 h of serial and 3.2 h of async at P = 4, against limits of 3 h and 2.5 h.
chain 3 1 serial.sbatch "PB_SUBSET=all,PB_LIMIT=500"
chain 2 4 levelsync.sbatch "$async,PB_SUBSET=all,PB_LIMIT=500"

submit 1 serial.sbatch "PB_SUBSET=timing,PB_LIMIT=100,PB_N=64" >/dev/null
submit 4 levelsync.sbatch "$strong,PB_N=64" >/dev/null
submit 4 levelsync.sbatch "$async,$strong,PB_N=64" >/dev/null
submit 1 serial.sbatch "PB_DATASET=gsm8k,PB_LIMIT=100" >/dev/null
submit 4 levelsync.sbatch "PB_DATASET=gsm8k,PB_LIMIT=100" >/dev/null
submit 4 levelsync.sbatch "$async,PB_DATASET=gsm8k,PB_LIMIT=100" >/dev/null

echo "submitted $count jobs on $account${held:+, held until $held finish}" >&2
