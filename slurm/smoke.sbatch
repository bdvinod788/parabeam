#!/bin/bash
# Start both vLLM servers on one 40 GB A100 and get a completion from the
# generator. Submit from the repo root:
#
#   sbatch --account=<project_id> slurm/smoke.sbatch
#
#SBATCH --job-name=pb-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-40gb
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
source configs/carc.env
export PATH="$PB_ENV/bin:$PATH"  # vLLM's kernel builds look for ninja
export HF_HUB_OFFLINE=1          # the models are local paths

GEN_FRACTION=0.25
PRM_FRACTION=0.6
START_TIMEOUT=420
job="${SLURM_JOB_ID:-0}"
gen_port=$((20000 + job % 10000))
prm_port=$((30000 + job % 10000))
gen_log="logs/pb-smoke-$job-generator.log"
prm_log="logs/pb-smoke-$job-prm.log"

failures=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; failures=$((failures + 1)); }
note() { echo "NOTE  $*"; }

pids=()
trap 'kill "${pids[@]}" 2>/dev/null' EXIT

# Each server sizes itself from the free memory, so the second waits for the first.
wait_healthy() {  # name port pid log
    local name="$1" port="$2" pid="$3" log="$4" waited=0
    until curl -sf "http://127.0.0.1:$port/health" >/dev/null; do
        if ! kill -0 "$pid" 2>/dev/null; then
            fail "$name server exited after ${waited}s; last lines of $log:"
            tail -n 15 "$log"
            return 1
        fi
        if ((waited >= START_TIMEOUT)); then
            fail "$name server not healthy after ${waited}s; last lines of $log:"
            tail -n 15 "$log"
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    pass "$name server healthy after about ${waited}s"
}

echo "job $job on $(hostname) at $(date -u +%FT%TZ)"

# Kernels are optional, so failing here only slows server startup.
if vllm download-kernels; then
    pass "kernels installed"
else
    note "vllm download-kernels failed; servers will build or skip them"
fi

vllm serve "$PB_MODELS/generator" --served-model-name generator --port "$gen_port" \
    --gpu-memory-utilization "$GEN_FRACTION" > "$gen_log" 2>&1 &
gen_pid=$!
pids+=("$gen_pid")
gen_up=0
wait_healthy generator "$gen_port" "$gen_pid" "$gen_log" && gen_up=1

if ((gen_up)); then
    reply="$(curl -s --max-time 120 "http://127.0.0.1:$gen_port/v1/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model": "generator", "prompt": "What is 2 + 3? Answer:", "max_tokens": 16, "temperature": 0}')"
    echo "reply: $reply"
    if [[ "$reply" == *'"text"'* ]]; then
        pass "generator returned a completion"
    else
        fail "generator did not return a completion"
    fi

    vllm serve "$PB_MODELS/prm-7b" --served-model-name prm --port "$prm_port" \
        --gpu-memory-utilization "$PRM_FRACTION" > "$prm_log" 2>&1 &
    prm_pid=$!
    pids+=("$prm_pid")
    wait_healthy prm "$prm_port" "$prm_pid" "$prm_log"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
fi

if ((failures > 0)); then
    echo "SMOKE FAILED: $failures check(s)"
    exit 1
fi
echo "SMOKE PASSED"
