# Start one rank's two vLLM servers on its GPU. Source this, then call
#   pb_start_servers LOG_PREFIX   (exports PB_GEN_URL and PB_PRM_URL)
#   pb_stop_servers
# The generator starts first, since each server sizes itself from free memory.
# The PRM is served through parabeam's LAST-pooling plugin so that its prefix
# cache works (parabeam/backends/vllm_prm.py).

PB_GEN_FRACTION="${PB_GEN_FRACTION:-0.25}"
PB_PRM_FRACTION="${PB_PRM_FRACTION:-0.6}"
PB_MAX_SEQS="${PB_MAX_SEQS:-256}"  # N = 256 per GPU, set explicitly
PB_START_TIMEOUT="${PB_START_TIMEOUT:-900}"  # a cold generator start took 492 s
PB_SERVER_PIDS=()
# vLLM drops connections idle for 5 s by default, as long as httpx keeps them,
# so a reused connection could be closed mid-request (job 13956488). Keep this
# above CLIENT_KEEP_ALIVE_S in parabeam/backends/vllm_http.py.
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE="${VLLM_HTTP_TIMEOUT_KEEP_ALIVE:-300}"

pb_wait_healthy() {  # name url pid log
    local name="$1" url="$2" pid="$3" log="$4" waited=0
    until curl -sf "$url/health" >/dev/null; do
        if ! kill -0 "$pid" 2>/dev/null || ((waited >= PB_START_TIMEOUT)); then
            echo "FAIL  $name server did not start (${waited}s); last lines of $log:"
            tail -n 20 "$log"
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "$name server healthy after about ${waited}s"
}

pb_start_servers() {
    local prefix="$1" job="${SLURM_JOB_ID:-0}"
    local gen_port=$((20000 + job % 10000)) prm_port=$((30000 + job % 10000))
    export PB_GEN_URL="http://127.0.0.1:$gen_port" PB_PRM_URL="http://127.0.0.1:$prm_port"

    vllm serve "$PB_MODELS/generator" --served-model-name generator --port "$gen_port" \
        --gpu-memory-utilization "$PB_GEN_FRACTION" --max-num-seqs "$PB_MAX_SEQS" \
        > "$prefix-generator.log" 2>&1 &
    PB_SERVER_PIDS+=("$!")
    pb_wait_healthy generator "$PB_GEN_URL" "$!" "$prefix-generator.log" || return 1

    vllm serve "$PB_MODELS/prm-7b" --served-model-name prm --port "$prm_port" \
        --gpu-memory-utilization "$PB_PRM_FRACTION" --max-num-seqs "$PB_MAX_SEQS" \
        --hf-overrides '{"architectures": ["ParabeamQwen2PRMLast"]}' --enable-prefix-caching \
        > "$prefix-prm.log" 2>&1 &
    PB_SERVER_PIDS+=("$!")
    pb_wait_healthy prm "$PB_PRM_URL" "$!" "$prefix-prm.log" || return 1
    grep -q "ParabeamQwen2PRMLast" "$prefix-prm.log" \
        || { echo "FAIL  the PRM server did not load the plugin"; return 1; }
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
}

pb_stop_servers() {
    ((${#PB_SERVER_PIDS[@]})) || return 0
    kill "${PB_SERVER_PIDS[@]}" 2>/dev/null
    wait "${PB_SERVER_PIDS[@]}" 2>/dev/null
    PB_SERVER_PIDS=()
}
