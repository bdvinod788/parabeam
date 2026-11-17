#!/bin/bash
# Where does a level's time go? Runs parabeam/tools/frontend_bench.py against
# the generator under several server settings, restarting it for each. Only the
# generator runs; the PRM is not involved. Submit from the repo root:
#
#   sbatch --account=<project_id> slurm/frontend_bench.sbatch
#
# Settings, as label:cpus:extra vllm flags. The server and the client share the
# CPUs, as they do in a level-sync rank (8 CPUs per task).
#
#SBATCH --job-name=pb-bench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-40gb
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
# shellcheck source=configs/carc.env
source "${PB_ENV_FILE:-configs/carc.env}"
export PATH="$PB_ENV/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
source slurm/launch_servers.sh  # pb_wait_healthy and the keep-alive setting

SETTINGS=(
    "c1-cpus8-default:8:"
    "c2-cpus8-nolog:8:--disable-uvicorn-access-log"
    "c3-cpus8-api4:8:--disable-uvicorn-access-log --api-server-count 4"
    "c4-cpus16-default:16:"
    "c5-cpus16-api4:16:--disable-uvicorn-access-log --api-server-count 4"
)

job="${SLURM_JOB_ID:-local}"
out="${PB_RUNS:-$HOME/parabeam/runs}/frontend-bench-$job"
mkdir -p logs "$out"
echo "job $job on $(hostname) at $(date -u +%FT%TZ)"
git rev-parse HEAD
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cpus=()
while read -r cpu; do cpus+=("$cpu"); done < <(python -c 'import os
get = getattr(os, "sched_getaffinity", None)
print(*(sorted(get(0)) if get else range(os.cpu_count())), sep="\n")')
echo "allowed CPUs: ${cpus[*]}"

port=$((20000 + (${job//[^0-9]/0} % 1000) * 8))
export PB_GEN_URL="http://127.0.0.1:$port"
status=0
for setting in "${SETTINGS[@]}"; do
    IFS=: read -r label ncpu flags <<< "$setting"
    if ((ncpu > ${#cpus[@]})); then
        echo "SKIP  $label needs $ncpu CPUs, the job has ${#cpus[@]}"
        continue
    fi
    cpu_list="$(IFS=,; echo "${cpus[*]:0:ncpu}")"
    log="logs/pb-bench-$job-$label-generator.log"
    echo "=== $label on CPUs $cpu_list, flags: ${flags:-none}"
    # shellcheck disable=SC2086  # flags are meant to split
    taskset -c "$cpu_list" vllm serve "$PB_MODELS/generator" --served-model-name generator \
        --port "$port" --gpu-memory-utilization "$PB_GEN_FRACTION" \
        --max-num-seqs "$PB_MAX_SEQS" $flags > "$log" 2>&1 &
    pid=$!
    if pb_wait_healthy "generator ($label)" "$PB_GEN_URL" "$pid" "$log"; then
        taskset -c "$cpu_list" python -m parabeam.tools.frontend_bench --label "$label" \
            --server-pid "$pid" --out "$out/bench.jsonl" || status=1
    else
        status=1
    fi
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    for _ in $(seq 60); do  # the next start sizes itself from free memory
        [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break
        sleep 1
    done
done
echo "finished at $(date -u +%FT%TZ) with status $status; results in $out/bench.jsonl"
exit "$status"
