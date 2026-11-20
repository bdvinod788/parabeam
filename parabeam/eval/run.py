"""Timed serial runs and the greedy reference, on the mock or on vLLM servers.

    python -m parabeam.eval.run serial --backend mock --subset pilot --N 64 --out runs/demo
    python -m parabeam.eval.run serial --backend vllm --gen-url URL --prm-url URL \\
        --subset weak --N 256 --out runs/serial-n256
    python -m parabeam.eval.run greedy --gen-url URL --prm-url URL --subset weak --out runs/greedy
    python -m parabeam.eval.run levelsync --backend mock --local 4 --subset pilot --out runs/ls
    srun --mpi=pmix_v5 ... python -m parabeam.eval.run levelsync --subset weak --limit 20 \\
        --out runs/levelsync-p4   (slurm/levelsync.sbatch; each rank has its own servers)
    python -m parabeam.eval.run async --backend mock --local 4 --cap 64 --out runs/as
    srun ... python -m parabeam.eval.run async --cap 64 ... (slurm/levelsync.sbatch, PB_MODE=async)

Each run directory gets a manifest.json and one JSON line per problem, written
as the problem finishes, so a job that runs out of time loses nothing and a
rerun with the same arguments skips the problems already done. The timer covers
the search only; voting and grading run after it stops (see eval/grade.py).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from parabeam.core.beam import SearchConfig
from parabeam.core.serial import fingerprint, search
from parabeam.core.vote import weighted_vote
from parabeam.eval.grade import is_correct, last_boxed, same_answer

REPO = Path(__file__).resolve().parents[2]
RESULTS = "results.jsonl"
MANIFEST = "manifest.json"


@dataclass(frozen=True, slots=True)
class Item:
    id: int
    question: str
    gold: str
    level: int | None = None


def load_items(dataset: str, subset: str, mock: bool) -> tuple[list[Item], Item]:
    """Return a subset and its warm-up problem. The mock needs only the IDs."""
    from parabeam.eval.datasets import SUBSETS_PATH, load_subset

    if mock:
        ids = json.loads(SUBSETS_PATH.read_text())[dataset]["subsets"]
        make = lambda i: Item(i, f"mock problem {i}", "0")
        return [make(i) for i in ids[subset]], make(ids["warmup"][0])
    items = [Item(e.id, e.question, e.answer, e.level) for e in load_subset(dataset, subset)]
    warm = load_subset(dataset, "warmup")[0]
    return items, Item(warm.id, warm.question, warm.answer, warm.level)


def code_version() -> dict:
    """Return the commit, or the snapshot a job was submitted from."""
    snapshot = REPO / "SNAPSHOT.json"
    if snapshot.exists():
        return json.loads(snapshot.read_text())
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None}


def gpu_info() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def write_manifest(out: Path, config: dict, extra: dict) -> None:
    """Write the manifest; a run that resumes earlier results is listed under "resumed".

    A manifest whose run wrote no results is replaced, so a job that failed
    before its first problem does not keep its name on the results that follow.
    Results from other settings or other code are never mixed.
    """
    path = out / MANIFEST
    session = {
        "code": code_version(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "gpu": gpu_info(),
        "python": platform.python_version(),
        "argv": sys.argv,
        **extra,
    }
    if path.exists() and done_ids(out):
        old = json.loads(path.read_text())
        if old["config"] != config:
            raise SystemExit(f"{path} was written with other settings: {old['config']}")
        if old["code"] != session["code"]:
            raise SystemExit(f"{path} holds results from other code: {old['code']}; "
                             "pass a new --out")
        old.setdefault("resumed", []).append(session)
        path.write_text(json.dumps(old, indent=1) + "\n")
        return
    path.write_text(json.dumps({"config": config, **session}, indent=1) + "\n")


def done_ids(out: Path) -> set[int]:
    path = out / RESULTS
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}


def append(out: Path, record: dict) -> None:
    with (out / RESULTS).open("a") as results:
        results.write(json.dumps(record) + "\n")


class KvWatch:
    """Polls the generator's /metrics for its peak KV cache use."""

    def __init__(self, url: str | None) -> None:
        self.url, self.peak, self._task = url, 0.0, None

    async def _poll(self) -> None:
        import httpx

        from parabeam.eval.metrics import prometheus_value

        async with httpx.AsyncClient(timeout=10) as client:
            while True:
                try:
                    text = (await client.get(f"{self.url}/metrics")).text
                    value = prometheus_value(text, "vllm:kv_cache_usage_perc")
                    if value is not None:
                        self.peak = max(self.peak, value)
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)

    def start(self) -> None:
        self.peak = 0.0
        if self.url:
            self._task = asyncio.create_task(self._poll())

    async def stop(self) -> float:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        return self.peak


def make_backend(args):
    if args.backend == "mock":
        from parabeam.backends.mock import MockBackend, mock_problem

        backend = MockBackend(clock=getattr(args, "sim_clock", None))
        return backend, lambda item: mock_problem(item.id), backend.now
    from parabeam.backends.vllm_http import VllmBackend

    models = Path(os.environ.get("PB_MODELS", "models"))
    backend = VllmBackend.from_models(args.gen_url, args.prm_url,
                                      models / "generator", models / "prm-7b")
    return backend, lambda item: backend.problem(item.id, item.question), time.perf_counter


async def run_serial(args) -> None:
    cfg = SearchConfig(args.N, args.M, args.max_depth, args.max_step_tokens)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    items, warmup = load_items(args.dataset, args.subset, args.backend == "mock")
    items = items[: args.limit] if args.limit else items
    config = {"mode": "serial", "backend": args.backend, "dataset": args.dataset,
              "subset": args.subset, "N": cfg.N, "M": cfg.M, "max_depth": cfg.max_depth,
              "max_step_tokens": cfg.max_step_tokens, "limit": args.limit}
    extra = {}
    if args.backend == "vllm":
        extra["vllm"] = await server_versions(args)
    write_manifest(out, config, extra)

    backend, to_problem, clock = make_backend(args)
    watch = KvWatch(args.gen_url if args.backend == "vllm" else None)
    done = done_ids(out)
    try:
        if not done and args.backend == "vllm":
            start = time.perf_counter()
            await search(to_problem(warmup), cfg, backend, clock)
            print(f"warm-up problem {warmup.id}: {time.perf_counter() - start:.1f} s", flush=True)
        for item in items:
            if item.id in done:
                continue
            problem = to_problem(item)
            # The backend's counters run across problems; record this one's share.
            truncated, retried = getattr(backend, "truncated", 0), getattr(backend, "retried", 0)
            watch.start()
            result = await search(problem, cfg, backend, clock)
            peak_kv = await watch.stop()
            # The timer has stopped; voting and grading may be slow.
            vote = weighted_vote(result.finished, last_boxed, same_answer)
            answer = vote.answer if vote else None
            record = {
                "id": item.id,
                "level": item.level,
                "gold": item.gold,
                "answer": answer,
                "correct": is_correct(answer, item.gold) if args.backend == "vllm" else None,
                "seconds": result.seconds,
                "levels": len(result.levels),
                "finished": len(result.finished),
                "requests": sum(lv.requests for lv in result.levels),
                "gen_s": sum(lv.gen_s for lv in result.levels),
                "prm_s": sum(lv.prm_s for lv in result.levels),
                "gen_tokens": sum(lv.gen_tokens for lv in result.levels),
                "per_level": [[lv.level, lv.requests, round(lv.gen_s, 4), round(lv.prm_s, 4),
                               lv.gen_tokens] for lv in result.levels],
                "peak_kv": peak_kv if args.backend == "vllm" else None,
                "vote_weight": vote.weight if vote else None,
                "voters": vote.voters if vote else 0,
                "fingerprint": fingerprint(result),
                "prm_truncated": getattr(backend, "truncated", 0) - truncated,
                "retried": getattr(backend, "retried", 0) - retried,
            }
            append(out, record)
            print(f"problem {item.id:>3}: {record['seconds']:7.1f} s, {record['levels']:>2} levels, "
                  f"answer {answer!r}, correct {record['correct']}, peak KV {peak_kv:.0%}",
                  flush=True)
    finally:
        if hasattr(backend, "aclose"):
            await backend.aclose()
    summarize(out)


async def levelsync_rank(args, comm) -> int:
    """One rank of a level-sync or async run; returns the exit status, the same on every rank.

    Rank 0 writes the manifest, results.jsonl and summary.json. Every rank
    appends its own ranks/rank-<r>.jsonl: one line per problem and level, with
    phase times measured from a barrier at the problem's start, and each
    request's token count, finish reason and duration. Level-sync lines have
    five marks (move, generate, score, allgather, done); async lines have four
    (move, work, work done, allgather done), since its scoring overlaps generation.
    """
    from parabeam.dist import async_sched, levelsync

    is_async = args.mode == "async"
    opts = (async_sched.AsyncOptions(args.cap, not args.no_steal, args.early, args.early_after,
                                     args.score) if is_async else None)

    cfg = SearchConfig(args.N, args.M, args.max_depth, args.max_step_tokens)
    vllm = args.backend == "vllm"
    items, warmup = load_items(args.dataset, args.subset, not vllm)
    items = items[: args.limit] if args.limit else items
    config = {"mode": args.mode, "backend": args.backend, "dataset": args.dataset,
              "subset": args.subset, "N": cfg.N, "M": cfg.M, "max_depth": cfg.max_depth,
              "max_step_tokens": cfg.max_step_tokens, "limit": args.limit, "P": comm.size,
              "placement": args.placement}
    if is_async:
        config.update({"cap": opts.cap, "steal": opts.steal, "early": opts.early,
                       "early_after": opts.early_after, "score": opts.score})
    backend, to_problem, clock = make_backend(args)
    sync = getattr(backend, "advance_to", None) if not vllm else None
    me = {"rank": comm.rank, "host": socket.gethostname(),
          "gpu": gpu_info() if vllm else None, "gpu_uuid": os.environ.get("PB_GPU_UUID"),
          "gen_url": args.gen_url, "prm_url": args.prm_url,
          "vllm": await server_versions(args) if vllm else None}
    ranks = await comm.allgather(me)

    error, done = None, set()
    if comm.rank == 0:
        gpus = [(r["host"], r["gpu_uuid"]) for r in ranks if r["gpu_uuid"]]
        if len(gpus) != len(set(gpus)):
            error = f"ranks share a GPU: {gpus}"
        else:
            args.out.mkdir(parents=True, exist_ok=True)
            try:
                write_manifest(args.out, config, {"ranks": ranks})
                done = done_ids(args.out)
            except SystemExit as exit_:
                error = str(exit_)
    error, done = (await comm.allgather((error, sorted(done))))[0]
    if error:
        if comm.rank == 0:
            print(error, file=sys.stderr, flush=True)
        return 1
    log_dir = args.out / "ranks"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"rank-{comm.rank}.jsonl"
    job = os.environ.get("SLURM_JOB_ID")
    watch = KvWatch(args.gen_url if vllm else None)

    async def one(problem):
        """Start every rank together, then search; return the result and this rank's t0."""
        starts = await comm.allgather(clock())
        if sync is not None:
            sync(max(starts))
        t0 = clock()
        if is_async:
            return await async_sched.search(problem, cfg, backend, comm, opts, clock, sync), t0
        return await levelsync.search(problem, cfg, backend, comm, args.placement, clock,
                                      sync), t0

    try:
        if vllm:
            began = time.perf_counter()
            await one(to_problem(warmup))
            if comm.rank == 0:
                print(f"warm-up problem {warmup.id}: {time.perf_counter() - began:.1f} s",
                      flush=True)
        for item in items:
            if item.id in done:
                continue
            truncated, retried = getattr(backend, "truncated", 0), getattr(backend, "retried", 0)
            watch.start()
            run, t0 = await one(to_problem(item))
            peak_kv = await watch.stop()
            with log_path.open("a") as log:
                for s in run.ranks:
                    if s.rank == comm.rank:
                        line = {
                            "id": item.id, "job": job, "rank": s.rank, "level": s.level,
                            "requests": s.requests, "gen_tokens": s.gen_tokens,
                            "moved_in": s.moved_in, "sent_bytes": s.sent_bytes,
                            "marks": [round(m - t0, 4) for m in s.marks],
                            "steps": [[st.tokens, st.finish, round(st.seconds, 4)]
                                      for st in s.steps],
                        }
                        if is_async:
                            line.update({"stolen": s.stolen, "given": s.given,
                                         "early_started": s.early_started,
                                         "early_kept": s.early_kept})
                        log.write(json.dumps(line) + "\n")
            mine = (peak_kv if vllm else None, getattr(backend, "truncated", 0) - truncated,
                    getattr(backend, "retried", 0) - retried)
            every = await comm.allgather(mine)
            if comm.rank != 0:
                continue
            result = run.result
            vote = weighted_vote(result.finished, last_boxed, same_answer)
            answer = vote.answer if vote else None
            kv = [k for k, _, _ in every]
            record = {
                "id": item.id,
                "level": item.level,
                "gold": item.gold,
                "answer": answer,
                "correct": is_correct(answer, item.gold) if vllm else None,
                "seconds": result.seconds,
                "levels": len(result.levels),
                "finished": len(result.finished),
                "requests": sum(lv.requests for lv in result.levels),
                # A level's generate and score times are its slowest rank's.
                "gen_s": sum(lv.gen_s for lv in result.levels),
                "prm_s": sum(lv.prm_s for lv in result.levels),
                "gen_tokens": sum(lv.gen_tokens for lv in result.levels),
                "wait_s_mean": sum(s.wait_s for s in run.ranks) / comm.size,
                "move_s_mean": sum(s.marks[1] - s.marks[0] for s in run.ranks) / comm.size,
                "moved": sum(s.moved_in for s in run.ranks),
                "sent_bytes": sum(s.sent_bytes for s in run.ranks),
                "per_level": [[lv.level, lv.requests, round(lv.gen_s, 4), round(lv.prm_s, 4),
                               lv.gen_tokens] for lv in result.levels],
                "peak_kv": max(kv) if vllm else None,
                "peak_kv_ranks": kv if vllm else None,
                "vote_weight": vote.weight if vote else None,
                "voters": vote.voters if vote else 0,
                "fingerprint": fingerprint(result),
                "prm_truncated": sum(t for _, t, _ in every),
                "retried": sum(r for _, _, r in every),
                "job": job,
            }
            if is_async:
                record.update({key: sum(getattr(s, key) for s in run.ranks)
                               for key in ("stolen", "early_started", "early_kept")})
            append(args.out, record)
            print(f"problem {item.id:>3}: {record['seconds']:7.1f} s, {record['levels']:>2} levels, "
                  f"wait {record['wait_s_mean']:6.1f} s, moved {record['moved']:>4}, "
                  f"answer {answer!r}, correct {record['correct']}", flush=True)
    finally:
        if hasattr(backend, "aclose"):
            await backend.aclose()
    if comm.rank == 0:
        summarize(args.out)
    return 0


def run_levelsync(args) -> int:
    if args.local:
        from parabeam.backends.mock import SimClock
        from parabeam.dist.runtime import run_local

        args.sim_clock = SimClock()  # the ranks share a process, so they share time
        statuses = asyncio.run(run_local(args.local, lambda comm: levelsync_rank(args, comm)))
        return max(statuses)
    from parabeam.dist.runtime import MpiComm, abort_on_error

    async def over_mpi() -> int:
        comm = MpiComm()
        try:
            return await levelsync_rank(args, comm)
        finally:
            await comm.close()

    with abort_on_error():
        return asyncio.run(over_mpi())


async def server_versions(args) -> dict:
    import httpx

    versions = {}
    async with httpx.AsyncClient(timeout=30) as client:
        for name, url in (("generator", args.gen_url), ("prm", args.prm_url)):
            versions[name] = (await client.get(f"{url}/version")).json().get("version")
    return versions


async def run_greedy(args) -> None:
    """Greedy decoding with no search: the accuracy check 9 must beat."""
    from parabeam.backends.vllm_http import MAX_MODEL_LEN, VllmBackend

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    items, _ = load_items(args.dataset, args.subset, False)
    items = items[: args.limit] if args.limit else items
    config = {"mode": "greedy", "dataset": args.dataset, "subset": args.subset,
              "limit": args.limit}
    write_manifest(out, config, {"vllm": await server_versions(args)})
    models = Path(os.environ.get("PB_MODELS", "models"))
    backend = VllmBackend.from_models(args.gen_url, args.prm_url,
                                      models / "generator", models / "prm-7b")
    done = done_ids(out)
    todo = [item for item in items if item.id not in done]

    async def one(item: Item) -> tuple[Item, str]:
        prompt = list(backend.problem(item.id, item.question).gen_prompt_ids)
        body = {"model": "generator", "prompt": prompt, "temperature": 0.0,
                "max_tokens": MAX_MODEL_LEN - len(prompt)}
        reply = await backend._post(f"{backend.gen_url}/v1/completions", body)
        return item, reply["choices"][0]["text"]

    try:
        outputs = await asyncio.gather(*(one(item) for item in todo))
    finally:
        await backend.aclose()
    for item, text in outputs:
        answer = last_boxed(text)
        append(out, {"id": item.id, "level": item.level, "gold": item.gold, "answer": answer,
                     "correct": is_correct(answer, item.gold)})
    summarize(out)


def summarize(out: Path) -> dict:
    records = [json.loads(line) for line in (out / RESULTS).read_text().splitlines() if line]
    summary = {"problems": len(records)}
    graded = [r["correct"] for r in records if r.get("correct") is not None]
    if graded:
        summary["accuracy"] = sum(graded) / len(graded)
    seconds = [r["seconds"] for r in records if "seconds" in r]
    if seconds:
        summary.update({
            "seconds_mean": statistics.fmean(seconds),
            "seconds_median": statistics.median(seconds),
            "seconds_max": max(seconds),
            "gen_share": sum(r["gen_s"] for r in records) / sum(seconds),
            "prm_share": sum(r["prm_s"] for r in records) / sum(seconds),
            "levels_mean": statistics.fmean(r["levels"] for r in records),
            "tokens_per_s": sum(r["gen_tokens"] for r in records) / sum(seconds),
        })
    kv = [r["peak_kv"] for r in records if r.get("peak_kv") is not None]
    if kv:
        summary["peak_kv"] = max(kv)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("summary: " + json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("serial", "greedy", "levelsync", "async"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--dataset", default="math500", choices=["math500", "gsm8k"])
        p.add_argument("--subset", default="weak")
        p.add_argument("--limit", type=int, help="run only the first problems of the subset")
        p.add_argument("--gen-url", default=os.environ.get("PB_GEN_URL"))
        p.add_argument("--prm-url", default=os.environ.get("PB_PRM_URL"))
    for name in ("serial", "levelsync", "async"):
        p = sub.choices[name]
        p.add_argument("--backend", default="vllm", choices=["vllm", "mock"])
        p.add_argument("--N", type=int, default=256)
        p.add_argument("--M", type=int, default=4)
        p.add_argument("--max-depth", type=int, default=40)
        p.add_argument("--max-step-tokens", type=int, default=2048)
    from parabeam.dist.placement import PLACEMENTS

    for name in ("levelsync", "async"):
        p = sub.choices[name]
        p.add_argument("--local", type=int, metavar="P",
                       help="mock only: run P ranks in this process instead of over MPI")
    sub.choices["levelsync"].add_argument("--placement", default="keep_local",
                                          choices=sorted(PLACEMENTS))
    p = sub.choices["async"]
    p.set_defaults(placement="keep_local")
    p.add_argument("--cap", type=int, help="requests generating at once per rank (default: none)")
    p.add_argument("--no-steal", action="store_true")
    p.add_argument("--early", type=float, default=1.0,
                   help="parents a rank may start early, in shares of N / (M * P); 0 for none")
    p.add_argument("--early-after", type=float, default=0.5)
    p.add_argument("--score", default="each", choices=["each", "level"],
                   help="score each child as it finishes, or a rank's whole level at once")
    args = parser.parse_args()
    needs_servers = args.mode == "greedy" or args.backend == "vllm"
    if needs_servers and not (args.gen_url and args.prm_url):
        parser.error("--gen-url and --prm-url (or PB_GEN_URL and PB_PRM_URL) are required")
    if args.mode in ("levelsync", "async"):
        if args.local and args.backend != "mock":
            parser.error("--local runs every rank in one process, so it needs --backend mock")
        sys.exit(run_levelsync(args))
    asyncio.run(run_serial(args) if args.mode == "serial" else run_greedy(args))


if __name__ == "__main__":
    main()
