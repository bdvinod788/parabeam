"""Find where a level's time goes when the GPU is idle: vLLM's front end or our client.

    python -m parabeam.tools.frontend_bench --gen-url URL --label c1 --out bench.jsonl

The level-sync runs (jobs 14245459 and 14245461) finished requests at a steady
~30 a second per server, whatever their length, while the generator's engine had
about 7 requests running and none waiting. So each request spends about 30 ms
somewhere before or after the engine, one at a time.

Each batch here is a level's shape: BATCH requests at once, in groups of GROUP
sharing a cached prefix of a given length. Each request generates at most
TOKENS tokens (a random prefix rarely stops early), so the GPU work is about the
same in every variant and only overhead changes. For each batch it records the
client's wall time and latencies, vLLM's own per-request timers (queue, prefill,
decode, end to end), and the CPU used by this process and by each vLLM process.
Variants:

    real     the body the search sends: token IDs, a stop string, token IDs back
    minimal  no stop string and no token IDs back
    grouped  one request per group with n = GROUP, BATCH / GROUP requests in all

Server settings (CPUs, --api-server-count, access log) are varied by
slurm/frontend_bench.sbatch, which restarts the generator for each.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from pathlib import Path

BATCH, GROUP, TOKENS = 256, 4, 32
LENGTHS = (256, 1024, 2560)
REPEATS = 2
VOCAB = 150_000  # below Qwen's special tokens
HISTOGRAMS = {
    "e2e": "vllm:e2e_request_latency_seconds",
    "queue": "vllm:request_queue_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "decode": "vllm:request_decode_time_seconds",
    "ttft": "vllm:time_to_first_token_seconds",
}
METRICS_WAIT_S = 10


def prefixes(length: int, rng: random.Random) -> list[list[int]]:
    """One random prefix per group."""
    return [[rng.randrange(VOCAB) for _ in range(length)] for _ in range(BATCH // GROUP)]


def body(prompt: list[int], seed: int, variant: str) -> dict:
    out = {"model": "generator", "prompt": prompt, "max_tokens": TOKENS,
           "temperature": 0.8, "top_p": 1.0, "seed": seed}
    if variant == "minimal":
        out["min_tokens"] = TOKENS  # exactly TOKENS each, with nothing to check per token
    else:
        # As vllm_http sends it. A random prefix rarely produces a blank line, so
        # nearly every request still runs to TOKENS; the server's decode timer
        # shows any difference in GPU work.
        out.update({"stop": ["\n\n"], "include_stop_str_in_output": True,
                    "return_token_ids": True, "skip_special_tokens": True})
    if variant == "grouped":
        out["n"] = GROUP
    return out


def requests_for(groups: list[list[int]], variant: str) -> list[dict]:
    if variant == "grouped":
        return [body(prefix, g, variant) for g, prefix in enumerate(groups)]
    return [body(prefix, i, variant) for prefix in groups for i in range(GROUP)]


def histogram_sums(text: str) -> dict[str, float | None]:
    from parabeam.eval.metrics import prometheus_value

    sums = {key: prometheus_value(text, f"{name}_sum") for key, name in HISTOGRAMS.items()}
    sums["count"] = prometheus_value(text, f"{HISTOGRAMS['e2e']}_count")
    return sums


class CpuWatch:
    """CPU seconds used by this process and by a server and all its children."""

    def __init__(self, server_pid: int | None) -> None:
        import psutil

        self._psutil = psutil
        self._server = psutil.Process(server_pid) if server_pid else None

    def processes(self) -> dict[str, object]:
        procs = {"client": self._psutil.Process(os.getpid())}
        if self._server is not None:
            for proc in [self._server, *self._server.children(recursive=True)]:
                try:
                    name = proc.name() if proc is not self._server else "server"
                    procs[f"{name}:{proc.pid}"] = proc
                except self._psutil.Error:
                    pass
        return procs

    def snapshot(self) -> dict[str, float]:
        times = {}
        for name, proc in self.processes().items():
            try:
                t = proc.cpu_times()
                times[name] = t.user + t.system
            except self._psutil.Error:
                pass
        return times


async def run_batch(client, gen_url: str, requests: list[dict], expected: int,
                    cpu: CpuWatch | None) -> dict:
    from parabeam.backends.vllm_http import post_json

    async def metrics() -> dict:
        reply = await client.get(f"{gen_url}/metrics")
        reply.raise_for_status()
        return histogram_sums(reply.text)

    latencies: list[float] = []
    tokens = 0

    async def one(req: dict) -> None:
        nonlocal tokens
        began = time.perf_counter()
        reply = await post_json(client, f"{gen_url}/v1/completions", req)
        latencies.append(time.perf_counter() - began)
        tokens += reply["usage"]["completion_tokens"]

    before = await metrics()
    cpu_before = cpu.snapshot() if cpu else {}
    start = time.perf_counter()
    await asyncio.gather(*(one(req) for req in requests))
    wall = time.perf_counter() - start
    cpu_after = cpu.snapshot() if cpu else {}
    deadline = time.perf_counter() + METRICS_WAIT_S
    after = await metrics()
    while (after["count"] or 0) - (before["count"] or 0) < expected:
        if after["count"] is None or time.perf_counter() > deadline:
            break  # metrics missing or late; report what there is
        await asyncio.sleep(0.05)
        after = await metrics()
    done = (after["count"] or 0) - (before["count"] or 0)
    server = {key: ((after[key] - before[key]) / done if done and after[key] is not None
                    and before[key] is not None else None)
              for key in HISTOGRAMS}
    cpu_s = {name: round(cpu_after[name] - cpu_before[name], 3)
             for name in cpu_after if name in cpu_before}
    latencies.sort()
    return {
        "wall_s": wall,
        "requests": len(requests),
        "tokens": tokens,
        "client_mean_s": statistics.fmean(latencies),
        "client_p50_s": latencies[len(latencies) // 2],
        "client_max_s": latencies[-1],
        "server_mean_s": server,  # vLLM's own timers, per request
        "server_requests": done,
        "cpu_s": cpu_s,
        "cpu_util": {name: round(s / wall, 2) for name, s in cpu_s.items()},
    }


def encode_cost(groups: list[list[int]]) -> float:
    """Seconds to build and JSON-encode a level's bodies, the client's part before sending."""
    start = time.perf_counter()
    for req in requests_for(groups, "real"):
        json.dumps(req).encode()
    return time.perf_counter() - start


async def bench(gen_url: str, label: str, out: Path, server_pid: int | None,
                lengths=LENGTHS, repeats: int = REPEATS, transport=None) -> list[dict]:
    from parabeam.backends.vllm_http import http_client

    cpu = CpuWatch(server_pid) if transport is None else None
    rows = []
    async with http_client(1800, transport) as client:
        for length in lengths:
            groups = prefixes(length, random.Random(length))
            # Untimed: fills the prefix cache, as a real level's parents already are.
            await run_batch(client, gen_url, requests_for(groups, "real"), BATCH, None)
            plan = [("real", r) for r in range(repeats)] + [("minimal", 0), ("grouped", 0)]
            for variant, repeat in plan:
                reqs = requests_for(groups, variant)
                row = await run_batch(client, gen_url, reqs, len(reqs), cpu)
                row.update({"label": label, "length": length, "variant": variant,
                            "repeat": repeat, "encode_s": encode_cost(groups),
                            "cpus": len(os.sched_getaffinity(0))
                            if hasattr(os, "sched_getaffinity") else os.cpu_count()})
                rows.append(row)
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                busiest = sorted(row["cpu_util"].items(), key=lambda kv: -kv[1])[:3]
                server = row["server_mean_s"]
                print(f"{label} L={length:>4} {variant:<7} wall {row['wall_s']:6.2f} s, "
                      f"client mean {row['client_mean_s']:5.2f} s, server e2e "
                      f"{_fmt(server['e2e'])}, queue {_fmt(server['queue'])}, decode "
                      f"{_fmt(server['decode'])}, CPU {busiest}", flush=True)
    return rows


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.2f} s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gen-url", default=os.environ.get("PB_GEN_URL"), required=False)
    parser.add_argument("--label", required=True, help="the server setting, e.g. c1-cpus8")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--server-pid", type=int, default=os.environ.get("PB_GEN_PID"))
    args = parser.parse_args()
    if not args.gen_url:
        parser.error("--gen-url or PB_GEN_URL is required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(bench(args.gen_url, args.label, args.out, args.server_pid))


if __name__ == "__main__":
    main()
