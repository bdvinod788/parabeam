#!/usr/bin/env python
"""Week 1 GPU checks 3, 4 and 5 on one 40 GB A100. Run by slurm/checks.sbatch.

Phases run in order, each with the GPU to itself unless stated:
  1. The generator writes solutions to pilot MATH500 problems.
  2. Check 4 reference: Transformers scores them with the PRM.
  3. Check 4: vLLM's PRM, as configured by default (STEP pooling, no prefix
     caching), must match the reference.
  4. Check 5: the PRM through parabeam's LAST-pooling plugin, scored level by
     level as the search would, must give the same newest-step rewards while
     reading most of each level from the prefix cache.
  5. Check 3: both servers together, the generator under N = 256 requests.

Everything measured goes to logs/pb-checks-<job>.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from parabeam.backends.prm_ids import (  # noqa: E402
    POSITIVE,
    STEP_TAG,
    SYSTEM,
    newest_reward,
    probabilities,
    solution_ids,
    tag_rewards,
)
from parabeam.backends.vllm_plugin import PRM_LAST_ARCH  # noqa: E402

MODELS = Path(os.environ.get("PB_MODELS", "models"))
GENERATOR = MODELS / "generator"
PRM = MODELS / "prm-7b"
CARD_EXAMPLE = REPO / "scripts" / "prm_card_example.json"
CARD_REWARDS = [1.0, 0.1904296875, 0.9765625, 1.0]  # printed in the model card

GEN_FRACTION = 0.25
PRM_FRACTION = 0.6
PROBLEMS = 8
SAMPLES = 4  # solutions per problem
SOLUTION_TOKENS = 1536
MAX_STEPS = 40
MAX_PRM_TOKENS = 4000  # both models stop at 4,096
LOAD_BEAMS = 256
LOAD_TOKENS = 2048
START_TIMEOUT = 900  # a cold generator start took 492 s
FP32_ITEMS = 4  # solutions rescored in fp32, to show how much bf16 alone moves rewards

# The plan asks for agreement "to about 0.01". bf16 kernels differ between
# Transformers and vLLM, so most rewards should agree within 0.01 and none by
# much more; the numbers are reported either way.
P95_TOLERANCE = 0.01
MAX_TOLERANCE = 0.05
CACHED_SHARE = 0.5  # with caching, deeper levels must compute under half their tokens

# vLLM never reads the cache for token_classify requests, so the stock PRM
# cannot cache. The plugin serves it for classify requests with LAST pooling.
CACHED_PRM = ("--hf-overrides", json.dumps({"architectures": [PRM_LAST_ARCH]}),
              "--enable-prefix-caching")
CACHED_TASK = "classify"


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.results: dict = {}

    def passed(self, message: str) -> None:
        print(f"PASS  {message}", flush=True)

    def failed(self, message: str) -> None:
        print(f"FAIL  {message}", flush=True)
        self.failures += 1

    def note(self, message: str) -> None:
        print(f"NOTE  {message}", flush=True)

    def check(self, ok: bool, message: str) -> None:
        (self.passed if ok else self.failed)(message)


def metric(text: str, name: str) -> float | None:
    """Sum a Prometheus metric over its labels; counters may carry _total."""
    total = None
    for line in text.splitlines():
        for full in (name, f"{name}_total"):
            if line.startswith(full) and line[len(full) : len(full) + 1] in (" ", "{"):
                total = (total or 0.0) + float(line.rsplit(" ", 1)[1])
    return total


def split_steps(text: str, limit: int = MAX_STEPS) -> list[str]:
    return [step for step in text.split("\n\n") if step.strip()][:limit]


def level_prefixes(ids: list[int], tag_id: int) -> list[list[int]]:
    """Return the input through each tag, as the search would score it level by level."""
    return [ids[: i + 1] for i, token in enumerate(ids) if token == tag_id]


def differences(a: list[float], b: list[float]) -> dict:
    if len(a) != len(b) or not a:
        raise ValueError(f"cannot compare {len(a)} values with {len(b)}")
    diffs = sorted(abs(x - y) for x, y in zip(a, b))
    return {
        "count": len(diffs),
        "max": diffs[-1],
        "p95": diffs[min(len(diffs) - 1, int(0.95 * len(diffs)))],
        "mean": statistics.fmean(diffs),
        "over_0.01": sum(d > 0.01 for d in diffs),
    }


def within_tolerance(stats: dict) -> bool:
    return stats["p95"] <= P95_TOLERANCE and stats["max"] <= MAX_TOLERANCE


def gpu_apps() -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return [line for line in out.stdout.splitlines() if line.strip()]


def wait_for_free_gpu(timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while gpu_apps() and time.monotonic() < deadline:
        time.sleep(2)


class Server:
    """A vLLM server in its own process group, stopped on exit.

    label names the log; requests use the served name, generator or prm.
    """

    def __init__(self, label: str, served: str, model: Path, port: int, fraction: float,
                 log_dir: Path, job: str, extra: tuple[str, ...] = (),
                 wait_free: bool = True) -> None:
        self.name, self.port, self.wait_free = label, port, wait_free
        self.url = f"http://127.0.0.1:{port}"
        self.log = log_dir / f"pb-checks-{job}-{label}.log"
        self.args = [
            "vllm", "serve", str(model), "--served-model-name", served, "--port", str(port),
            "--gpu-memory-utilization", str(fraction), *extra,
        ]

    def __enter__(self) -> Server:
        import httpx

        print(f"starting {self.name}: {' '.join(self.args)}", flush=True)
        self.proc = subprocess.Popen(
            self.args, stdout=self.log.open("w"), stderr=subprocess.STDOUT, start_new_session=True
        )
        start = time.monotonic()
        while time.monotonic() - start < START_TIMEOUT:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} exited with {self.proc.returncode}; see {self.log}")
            try:
                if httpx.get(f"{self.url}/health", timeout=5).status_code == 200:
                    self.startup_s = time.monotonic() - start
                    print(f"{self.name} healthy after {self.startup_s:.0f} s", flush=True)
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(3)
        self.__exit__(None, None, None)
        raise RuntimeError(f"{self.name} not healthy after {START_TIMEOUT} s; see {self.log}")

    def __exit__(self, *exc) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        if self.wait_free:
            wait_for_free_gpu()

    def log_says(self, text: str) -> bool:
        return text in self.log.read_text(errors="replace")


# Phase 1 ---------------------------------------------------------------------


async def generate(url: str, questions: list[str], samples: int, max_tokens: int,
                   seed: int = 0) -> list[dict]:
    import httpx

    # httpx allows 100 connections by default, which would cap the load test at 100 beams.
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    transport = httpx.AsyncHTTPTransport(retries=3, limits=limits)
    async with httpx.AsyncClient(timeout=1800, transport=transport) as client:
        async def one(i: int, question: str) -> dict:
            reply = await client.post(f"{url}/v1/chat/completions", json={
                "model": "generator",
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": question}],
                "max_tokens": max_tokens, "temperature": 0.8, "top_p": 1.0, "seed": seed + i,
            })
            reply.raise_for_status()
            body = reply.json()
            return {"question": question, "text": body["choices"][0]["message"]["content"],
                    "tokens": body["usage"]["completion_tokens"]}

        jobs = [one(i, q) for i, q in enumerate(q for q in questions for _ in range(samples))]
        return await asyncio.gather(*jobs)


# Phase 2 ---------------------------------------------------------------------


def reference(items_path: Path, out_path: Path) -> None:
    """Score with Transformers, in its own process so the GPU is freed after."""
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer, Qwen2Model

    tokenizer = AutoTokenizer.from_pretrained(PRM)
    tag_id = tokenizer.convert_tokens_to_ids(STEP_TAG)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731

    # The model card loads remote code written for Transformers 4, so build the
    # same network from Transformers' own Qwen2 and the checkpoint's score head.
    model, info = Qwen2Model.from_pretrained(
        PRM, dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True
    )
    if info["missing_keys"]:
        raise RuntimeError(f"weights missing from the backbone: {info['missing_keys'][:5]}")
    model = model.cuda().eval()
    weight_map = json.loads((PRM / "model.safetensors.index.json").read_text())["weight_map"]
    head = torch.nn.Sequential(
        torch.nn.Linear(model.config.hidden_size, model.config.hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(model.config.hidden_size, 2),
    )
    state = {}
    for key in ("0.weight", "0.bias", "2.weight", "2.bias"):
        with safe_open(PRM / weight_map[f"score.{key}"], "pt") as shard:
            state[key] = shard.get_tensor(f"score.{key}")
    head.load_state_dict(state)
    head = head.to(torch.bfloat16).cuda().eval()

    @torch.no_grad()
    def rewards(ids: list[int]) -> list[float]:
        tokens = torch.tensor([ids], device="cuda")
        logits = head(model(input_ids=tokens).last_hidden_state[0])
        probs = torch.softmax(logits.float(), dim=-1)[:, 1]
        return probs[tokens[0] == tag_id].tolist()

    def card_ids(system: str, question: str, steps: list[str]) -> list[int]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": STEP_TAG.join(steps) + STEP_TAG},
        ]
        return tokenizer.encode(tokenizer.apply_chat_template(messages, tokenize=False))

    card = json.loads(CARD_EXAMPLE.read_text())
    out = {"tag_id": tag_id, "card": {}, "items": []}
    full = card_ids(card["system"], card["query"], card["response"])
    out["card"]["ids"] = full
    out["card"]["rewards"] = rewards(full)

    for item in json.loads(items_path.read_text()):
        steps = [step.removesuffix("\n\n") for step in item["steps"]]
        ids = solution_ids(encode, item["question"], steps, tag_id)
        if len(ids) > MAX_PRM_TOKENS:
            continue
        full = card_ids(SYSTEM, item["question"], steps)
        out["items"].append({
            "ids": ids,
            "rewards": rewards(ids),
            "matches_card": full[: len(ids)] == ids,
            "card_rewards": rewards(full),
        })

    # The card example and a few solutions again in fp32 (about 30 GB), to show
    # how far bf16 rounding alone moves a reward. rewards() sees the new model.
    del model
    torch.cuda.empty_cache()
    model = Qwen2Model.from_pretrained(PRM, dtype=torch.float32, attn_implementation="sdpa")
    model = model.cuda().eval()
    head = head.float()
    out["fp32"] = {
        "card": rewards(out["card"]["ids"]),
        "items": [rewards(item["ids"]) for item in out["items"][:FP32_ITEMS]],
    }
    out_path.write_text(json.dumps(out))


# Phases 3 and 4 --------------------------------------------------------------


async def pool(url: str, inputs: list[list[int]], task: str = "token_classify") -> tuple[list, float]:
    import httpx

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=900) as client:
        reply = await client.post(f"{url}/pooling", json={
            "model": "prm", "input": inputs, "task": task,
        })
    elapsed = time.perf_counter() - start
    if reply.status_code != 200:
        raise RuntimeError(f"/pooling returned {reply.status_code}: {reply.text[:500]}")
    data = sorted(reply.json()["data"], key=lambda item: item["index"])
    return [item["data"] for item in data], elapsed


def newest(data: list) -> float:
    """Return the newest step's reward from one input's pooling output.

    token_classify returns rows, whose last is the newest tag; classify returns
    the last token's single row.
    """
    if data and isinstance(data[0], list):
        return newest_reward(data)
    return probabilities(data)[POSITIVE]


async def cache_counters(url: str) -> tuple[float, float]:
    import httpx

    async with httpx.AsyncClient(timeout=10) as client:
        text = (await client.get(f"{url}/metrics")).text
    return (metric(text, "vllm:prefix_cache_hits") or 0.0,
            metric(text, "vllm:prefix_cache_queries") or 0.0)


async def score_by_level(url: str, items: list[dict], tag_id: int,
                         task: str = "token_classify") -> dict:
    """Score each level's prefixes in one batch, deepest level last.

    Each level records the prefix-cache tokens queried and hit while it ran.
    """
    prefixes = [level_prefixes(item["ids"], tag_id) for item in items]
    depth = max(map(len, prefixes))
    levels = []
    hits, queries = await cache_counters(url)
    for level in range(depth):
        members = [i for i, p in enumerate(prefixes) if len(p) > level]
        inputs = [prefixes[i][level] for i in members]
        data, elapsed = await pool(url, inputs, task)
        new_hits, new_queries = await cache_counters(url)
        levels.append({
            "level": level + 1,
            "members": members,
            "seconds": elapsed,
            "tokens": sum(map(len, inputs)),
            "cache_hits": new_hits - hits,
            "cache_queries": new_queries - queries,
            "rewards": [newest(d) for d in data],
        })
        hits, queries = new_hits, new_queries
    return {"levels": levels}


def cached_share(levels: list[dict]) -> tuple[float, float]:
    """Return the share of tokens read from the cache at levels 2+, and the tokens queried."""
    deep = [level for level in levels if level["level"] >= 2]
    queried = sum(level["cache_queries"] for level in deep)
    hit = sum(level["cache_hits"] for level in deep)
    return (hit / queried if queried else 0.0), queried


def newest_reference(items: list[dict], levels: list[dict]) -> tuple[list[float], list[float]]:
    got, want = [], []
    for level in levels:
        for i, reward in zip(level["members"], level["rewards"]):
            got.append(reward)
            want.append(items[i]["rewards"][level["level"] - 1])
    return got, want


# Phase 5 ---------------------------------------------------------------------


async def load_test(gen_url: str, prm_url: str, questions: list[str], items: list[dict],
                    tag_id: int) -> dict:
    import httpx

    peak = {"kv": 0.0, "running": 0.0, "waiting": 0.0}
    done = asyncio.Event()

    async def watch() -> None:
        async with httpx.AsyncClient(timeout=10) as client:
            while not done.is_set():
                try:
                    text = (await client.get(f"{gen_url}/metrics")).text
                    for key, name in (("kv", "vllm:kv_cache_usage_perc"),
                                      ("running", "vllm:num_requests_running"),
                                      ("waiting", "vllm:num_requests_waiting")):
                        value = metric(text, name)
                        if value is not None:
                            peak[key] = max(peak[key], value)
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)

    async def preemptions() -> float | None:
        async with httpx.AsyncClient(timeout=10) as client:
            return metric((await client.get(f"{gen_url}/metrics")).text, "vllm:num_preemptions")

    before = await preemptions()
    watcher = asyncio.create_task(watch())
    start = time.perf_counter()
    prm_alone = None
    try:
        prompts = [questions[i % len(questions)] for i in range(LOAD_BEAMS)]
        generation = asyncio.create_task(
            generate(gen_url, prompts, 1, LOAD_TOKENS, seed=10_000)
        )
        await asyncio.sleep(20)  # let the batch fill before the PRM joins
        prm_busy = await score_by_level(prm_url, items, tag_id, CACHED_TASK)
        outs = await generation
    finally:
        done.set()
        await watcher
    wall = time.perf_counter() - start
    prm_alone = await score_by_level(prm_url, items, tag_id, CACHED_TASK)
    after = await preemptions()
    tokens = sum(out["tokens"] for out in outs)
    return {
        "beams": len(outs),
        "wall_s": wall,
        "completion_tokens": tokens,
        "tokens_per_s": tokens / wall,
        "peak_kv_usage": peak["kv"],
        "peak_running": peak["running"],
        "peak_waiting": peak["waiting"],
        "preemptions": None if before is None or after is None else after - before,
        "prm_s_during_load": sum(level["seconds"] for level in prm_busy["levels"]),
        "prm_s_alone": sum(level["seconds"] for level in prm_alone["levels"]),
        "gpu_apps": gpu_apps(),
    }


# Main ------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--log-dir", type=Path, default=REPO / "logs")
    run.add_argument("--job", default=os.environ.get("SLURM_JOB_ID", "local"))
    ref = sub.add_parser("reference")
    ref.add_argument("items", type=Path)
    ref.add_argument("out", type=Path)
    args = parser.parse_args()

    if args.command == "reference":
        reference(args.items, args.out)
        return 0

    from parabeam.eval.datasets import load_subset

    report = Report()
    results = report.results
    log_dir, job = args.log_dir, args.job
    offset = int(job) % 10000 if job.isdigit() else 0
    gen_port, prm_port = 20000 + offset, 30000 + offset
    questions = [example.question for example in load_subset("math500", "pilot")[:PROBLEMS]]

    # Phase 1: solutions to score.
    with Server("generator", "generator", GENERATOR, gen_port, GEN_FRACTION, log_dir, job) as gen:
        results["generator_startup_s"] = gen.startup_s
        outs = asyncio.run(generate(gen.url, questions, SAMPLES, SOLUTION_TOKENS))
    items = [{"question": o["question"], "steps": split_steps(o["text"])} for o in outs]
    items = [item for item in items if item["steps"]]
    report.note(f"generator wrote {len(items)} solutions, "
                f"{sum(len(i['steps']) for i in items)} steps")

    # Phase 2: the Transformers reference.
    items_path = log_dir / f"pb-checks-{job}-items.json"
    ref_path = log_dir / f"pb-checks-{job}-reference.json"
    items_path.write_text(json.dumps(items))
    ref = None
    done = subprocess.run([sys.executable, __file__, "reference", str(items_path), str(ref_path)])
    wait_for_free_gpu()
    if done.returncode == 0:
        ref = json.loads(ref_path.read_text())
        # The card's printed rewards come from its own run, whose kernels we do
        # not know, so they are reported beside bf16 and fp32 rather than gated.
        card = differences(ref["card"]["rewards"], CARD_REWARDS)
        results["reference_vs_card"] = card
        fp32 = ref["fp32"]
        results["card_rewards"] = {"printed": CARD_REWARDS, "bf16": ref["card"]["rewards"],
                                   "fp32": fp32["card"]}
        report.note("card rewards, printed / bf16 / fp32: " + ", ".join(
            f"{a:.4f}/{b:.4f}/{c:.4f}" for a, b, c in
            zip(CARD_REWARDS, ref["card"]["rewards"], fp32["card"])))
        bf16_vs_fp32 = differences([r for i in ref["items"][:FP32_ITEMS] for r in i["rewards"]],
                                   [r for rs in fp32["items"] for r in rs])
        results["bf16_vs_fp32"] = bf16_vs_fp32
        report.note(f"bf16 vs fp32 on {bf16_vs_fp32['count']} steps: "
                    f"p95 {bf16_vs_fp32['p95']:.4f}, max {bf16_vs_fp32['max']:.4f}")
        matches = sum(item["matches_card"] for item in ref["items"])
        results["ids_match_card"] = [matches, len(ref["items"])]
        report.check(matches == len(ref["items"]),
                     f"step-by-step PRM IDs equal the chat template's for {matches}/{len(ref['items'])}")
        own = differences([r for i in ref["items"] for r in i["rewards"]],
                          [r for i in ref["items"] for r in i["card_rewards"]])
        results["own_ids_vs_card_ids"] = own
        report.note(f"rewards on our IDs vs the template's: max diff {own['max']:.4f}")
    else:
        report.failed("check 4: the Transformers reference did not run; see the traceback above")

    # Phase 3: check 4, the default PRM server.
    if ref:
        items_ref = ref["items"]
        tag_id = ref["tag_id"]
        try:
            with Server("prm-step", "prm", PRM, prm_port, PRM_FRACTION, log_dir, job) as prm:
                results["prm_startup_s"] = prm.startup_s
                rows, _ = asyncio.run(pool(prm.url, [ref["card"]["ids"]]))
                card_vllm = tag_rewards(ref["card"]["ids"], rows[0], tag_id)
                results["card_rewards"]["vllm"] = card_vllm
                report.note("card rewards from vLLM: " + ", ".join(f"{r:.4f}" for r in card_vllm))
                rows, _ = asyncio.run(pool(prm.url, [item["ids"] for item in items_ref]))
                got = card_vllm + [r for item, row in zip(items_ref, rows)
                                   for r in tag_rewards(item["ids"], row, tag_id)]
                want = ref["card"]["rewards"] + [r for item in items_ref for r in item["rewards"]]
                stats = differences(got, want)
                results["check4"] = stats
                report.check(within_tolerance(stats),
                             f"check 4: vLLM matches Transformers on {stats['count']} steps "
                             f"(p95 {stats['p95']:.4f}, max {stats['max']:.4f}, "
                             f"{stats['over_0.01']} over 0.01)")
                uncached = asyncio.run(score_by_level(prm.url, items_ref, tag_id))
                results["uncached_levels"] = uncached
        except Exception as error:  # keep going: check 5 can still run
            report.failed(f"check 4: {error}")
            uncached = None

        # Phase 4: check 5, the LAST-pooling plugin with prefix caching.
        try:
            with Server("prm-last", "prm", PRM, prm_port, PRM_FRACTION, log_dir, job,
                        CACHED_PRM) as prm:
                report.check(prm.log_says(PRM_LAST_ARCH),
                             f"check 5: the PRM server loaded the plugin's {PRM_LAST_ARCH}")
                report.check(prm.log_says("enable_prefix_caching=True"),
                             "check 5: the PRM server runs with prefix caching on")
                cached = asyncio.run(score_by_level(prm.url, items_ref, tag_id, CACHED_TASK))
                results["cached_levels"] = cached
                got, want = newest_reference(items_ref, cached["levels"])
                stats = differences(got, want)
                results["check5_rewards"] = stats
                report.check(within_tolerance(stats),
                             f"check 5: cached newest-step rewards match Transformers on "
                             f"{stats['count']} steps (p95 {stats['p95']:.4f}, "
                             f"max {stats['max']:.4f})")
                share, queried = cached_share(cached["levels"])
                results["check5_cached_share"] = share
                report.check(queried > 0 and share > 1 - CACHED_SHARE,
                             f"check 5: levels 2+ read {share:.1%} of {queried:.0f} tokens "
                             f"from the prefix cache")
                if uncached:
                    slow = sum(lv["seconds"] for lv in uncached["levels"])
                    fast = sum(lv["seconds"] for lv in cached["levels"])
                    results["check5_seconds"] = {"uncached": slow, "cached": fast}
                    report.note(f"scoring every level took {slow:.2f} s uncached, "
                                f"{fast:.2f} s cached")
        except Exception as error:
            report.failed(f"check 5: {error}")

    # Phase 5: check 3, both servers under load.
    try:
        # The generator sizes itself first; the PRM then takes its share of what is left.
        with Server("generator-load", "generator", GENERATOR, gen_port, GEN_FRACTION,
                    log_dir, job) as gen, \
                Server("prm-load", "prm", PRM, prm_port, PRM_FRACTION, log_dir, job,
                       CACHED_PRM, wait_free=False) as prm:
            if ref:
                load = asyncio.run(load_test(gen.url, prm.url, questions, ref["items"],
                                             ref["tag_id"]))
                results["check3"] = load
                report.check(load["beams"] == LOAD_BEAMS,
                             f"check 3: both servers served {LOAD_BEAMS} beams together")
                report.note(
                    f"check 3: {load['completion_tokens']} tokens in {load['wall_s']:.0f} s "
                    f"({load['tokens_per_s']:.0f}/s), peak KV {load['peak_kv_usage']:.0%}, "
                    f"peak running {load['peak_running']:.0f}, waiting {load['peak_waiting']:.0f}, "
                    f"preemptions {load['preemptions']}")
                report.note(f"check 3: PRM scoring took {load['prm_s_during_load']:.2f} s during "
                            f"generation, {load['prm_s_alone']:.2f} s alone")
            else:
                report.passed("check 3: both servers started together")
            report.note(f"GPU processes: {'; '.join(gpu_apps())}")
    except Exception as error:
        report.failed(f"check 3: {error}")

    out = log_dir / f"pb-checks-{job}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"results in {out}")
    if report.failures:
        print(f"CHECKS FAILED: {report.failures} check(s)")
        return 1
    print("CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
