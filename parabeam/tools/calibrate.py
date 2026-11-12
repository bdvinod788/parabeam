"""Measure the cost model on the GPU (check 8) and write cost_model.json.

    python -m parabeam.tools.calibrate --gen-url URL --prm-url URL --out cost_model.json

Decode time t(b) is one decode step for a batch of b requests at a context
length. It comes from the generator's own timers, not the client's clock:
vLLM's request_decode_time_seconds runs from a request's first token to its
last, so it leaves out HTTP, queueing and prefill. Job 13956487 timed batches on
the client and got nonsense once a batch took longer to send than to decode
(0.39 ms per step at b = 128, a tenth of b = 1). Each batch runs SHORT and LONG
tokens long with every prompt cached, and the extra decode time over
b * (LONG - SHORT) is the step alone; requests that start before the rest of
their batch arrives do so in both runs. Prompts come in groups of GROUP sharing
a prefix, as a survivor's children do, which also keeps b = 256 at 2,000 tokens
inside the KV cache. Prefill is vLLM's request_prefill_time_seconds for a fresh
prompt. The PRM cost is the time to score fresh tokens.

A step that is faster at a larger batch or a longer context than its neighbour
cannot be right, so those are listed as suspect.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path
from typing import NamedTuple

CONTEXTS = (500, 1000, 2000)
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SHORT, LONG = 32, 160
GROUP = 4
REPEATS = 3
VOCAB = 150_000  # below Qwen's special tokens
PRM_INPUTS, PRM_TOKENS = 8, 1000
DECODE = "vllm:request_decode_time_seconds"
PREFILL = "vllm:request_prefill_time_seconds"
TOLERANCE = 0.85  # a neighbour may be this much faster before it is suspect
METRICS_WAIT_S = 10


def decode_step(short_s: float, long_s: float, batch: int) -> float:
    """Return one step's time from the batch's total decode time at SHORT and LONG tokens."""
    return (long_s - short_s) / (batch * (LONG - SHORT))


def suspect(decode: list[list[float]]) -> list[tuple[int, int]]:
    """Return (context, batch) cells faster than a smaller batch or a shorter context."""
    bad = []
    for c, context in enumerate(CONTEXTS):
        for b, batch in enumerate(BATCHES):
            t = decode[c][b]
            if (b and t < TOLERANCE * decode[c][b - 1]) or (c and t < TOLERANCE * decode[c - 1][b]):
                bad.append((context, batch))
    return bad


def prompts_for(batch: int, context: int, rng: random.Random) -> list[list[int]]:
    """Return batch prompts in groups of GROUP that share one random prefix."""
    groups = [[rng.randrange(VOCAB) for _ in range(context)]
              for _ in range(-(-batch // GROUP))]
    return [groups[i // GROUP] for i in range(batch)]


class Timers(NamedTuple):
    decode: float  # summed over requests, from vLLM's histograms
    prefill: float
    requests: float


class Run(NamedTuple):
    wall: float
    decode: float  # this run's requests' decode time, summed
    prefill: float


class Client:
    def __init__(self, gen_url: str, prm_url: str, transport=None) -> None:
        from parabeam.backends.vllm_http import http_client

        self.gen_url, self.prm_url = gen_url, prm_url
        self.http = http_client(1800, transport)

    async def _metrics(self) -> str:
        reply = await self.http.get(f"{self.gen_url}/metrics")
        reply.raise_for_status()
        return reply.text

    async def timers(self) -> Timers:
        from parabeam.eval.metrics import prometheus_value

        text = await self._metrics()
        values = [prometheus_value(text, name) for name in
                  (f"{DECODE}_sum", f"{PREFILL}_sum", f"{PREFILL}_count")]
        if any(value is None for value in values):
            raise RuntimeError(f"the generator does not export {DECODE} and {PREFILL}")
        return Timers(*values)

    async def complete(self, prompts: list[list[int]], tokens: int) -> Run:
        """Generate exactly tokens per prompt, all at once; return the run's times."""
        from parabeam.backends.vllm_http import post_json

        async def one(i: int, prompt: list[int]) -> None:
            reply = await post_json(self.http, f"{self.gen_url}/v1/completions", {
                "model": "generator", "prompt": prompt, "max_tokens": tokens,
                "min_tokens": tokens, "ignore_eos": True, "temperature": 0.8, "seed": i,
            })
            got = reply["usage"]["completion_tokens"]
            if got != tokens:
                raise RuntimeError(f"asked for {tokens} tokens, got {got}")

        before = await self.timers()
        start = time.perf_counter()
        await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
        wall = time.perf_counter() - start
        # vLLM can record a request's timers just after replying, so wait for all.
        deadline = time.perf_counter() + METRICS_WAIT_S
        while (after := await self.timers()).requests - before.requests < len(prompts):
            if time.perf_counter() > deadline:
                raise RuntimeError("the generator did not record every request's timers")
            await asyncio.sleep(0.05)
        return Run(wall, after.decode - before.decode, after.prefill - before.prefill)

    async def preemptions(self) -> float:
        from parabeam.eval.metrics import prometheus_value

        return prometheus_value(await self._metrics(), "vllm:num_preemptions") or 0.0

    async def score(self, inputs: list[list[int]]) -> float:
        from parabeam.backends.vllm_http import post_json

        start = time.perf_counter()
        await post_json(self.http, f"{self.prm_url}/pooling",
                        {"model": "prm", "input": inputs, "task": "classify"})
        return time.perf_counter() - start


async def measure(gen_url: str, prm_url: str, seed: int = 0, transport=None) -> dict:
    client = Client(gen_url, prm_url, transport)
    rng = random.Random(seed)
    decode = [[0.0] * len(BATCHES) for _ in CONTEXTS]
    decode_wall = [[0.0] * len(BATCHES) for _ in CONTEXTS]  # the old method, for comparison
    preempted = [[False] * len(BATCHES) for _ in CONTEXTS]
    try:
        await client.complete([[1, 2, 3]], 1)  # first request pays one-off costs
        for c, context in enumerate(CONTEXTS):
            for b, batch in enumerate(BATCHES):
                prompts = prompts_for(batch, context, rng)
                before = await client.preemptions()
                await client.complete(prompts[::GROUP], 1)  # cache each shared prefix
                steps, walls = [], []
                for _ in range(REPEATS):
                    short = await client.complete(prompts, SHORT)
                    long = await client.complete(prompts, LONG)
                    steps.append(decode_step(short.decode, long.decode, batch))
                    walls.append((long.wall - short.wall) / (LONG - SHORT))
                decode[c][b] = statistics.median(steps)
                decode_wall[c][b] = statistics.median(walls)
                preempted[c][b] = await client.preemptions() > before
                print(f"decode  context {context:>4}  batch {batch:>3}: "
                      f"{decode[c][b] * 1000:7.2f} ms/step  (client clock "
                      f"{decode_wall[c][b] * 1000:7.2f})"
                      f"{'  PREEMPTED' if preempted[c][b] else ''}", flush=True)

        prefill = []
        for context in CONTEXTS:
            times = [(await client.complete(prompts_for(1, context, rng), 1)).prefill
                     for _ in range(REPEATS)]
            prefill.append(statistics.median(times))
            print(f"prefill context {context:>4}: {prefill[-1] * 1000:7.2f} ms", flush=True)

        per_token = []
        for _ in range(REPEATS):
            inputs = [[rng.randrange(VOCAB) for _ in range(PRM_TOKENS)]
                      for _ in range(PRM_INPUTS)]
            per_token.append(await client.score(inputs) / (PRM_INPUTS * PRM_TOKENS))
        prm_s_per_token = statistics.median(per_token)
        print(f"PRM: {prm_s_per_token * 1e6:.2f} us per uncached token", flush=True)
    finally:
        await client.http.aclose()

    return {
        "contexts": list(CONTEXTS),
        "batches": list(BATCHES),
        "decode_s": decode,
        "prefill_s": prefill,
        "prm_s_per_token": prm_s_per_token,
        "preempted": preempted,
        "suspect": suspect(decode),
        "decode_wall_s": decode_wall,
        "method": {"timers": "vllm request_decode_time and request_prefill_time",
                   "short": SHORT, "long": LONG, "group": GROUP, "repeats": REPEATS,
                   "prm_inputs": PRM_INPUTS, "prm_tokens": PRM_TOKENS},
    }


def main() -> None:
    import os

    from parabeam.dist.costmodel import CostModel

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gen-url", default=os.environ.get("PB_GEN_URL"))
    parser.add_argument("--prm-url", default=os.environ.get("PB_PRM_URL"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (args.gen_url and args.prm_url):
        parser.error("--gen-url and --prm-url (or PB_GEN_URL and PB_PRM_URL) are required")
    data = asyncio.run(measure(args.gen_url, args.prm_url))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1) + "\n")
    CostModel.load(args.out)  # what the mock and placement will read
    if any(any(row) for row in data["preempted"]):
        print("WARNING: some batches were preempted; their decode times are too high")
    if data["suspect"]:
        print(f"WARNING: suspect decode times at (context, batch) {data['suspect']}")
    print(f"cost model written to {args.out}")


if __name__ == "__main__":
    main()
