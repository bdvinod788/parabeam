"""Compare schedulers on the mock, with the A100's measured costs and the GPU's step lengths.

    python -m parabeam.tools.simulate --out runs/sim.jsonl
    python -m parabeam.tools.simulate --ranks 4 --variants levelsync async --limit 5

Every variant runs every problem at every P, all ranks on one simulated clock,
and must keep exactly the serial search's beams; a mismatch stops the run. One
JSON line per (variant, P, problem) goes to --out; the table at the end gives
each variant's mean seconds per problem and per level, and its speedup over the
serial search. On Nov 19 the mock's level-sync matched the GPU's seconds per
level within 4% at P = 1, 2 and 4; on Nov 20 the generator/PRM overlap slowdown
was fitted to the first async GPU run (see dist/cost_model_a100.json).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from parabeam.core.beam import SearchConfig
from parabeam.dist.async_sched import AsyncOptions

VARIANTS: dict[str, AsyncOptions | str] = {
    "levelsync": "keep_local",
    "levelsync-rr": "round_robin",
    # Scoring each child as it finishes, as in the first GPU pilot (job 14398026).
    "each": AsyncOptions(steal=False, early=0),
    "each+early-cap64": AsyncOptions(cap=64, early=1.0),
    # Scoring a rank's whole level at once, so the PRM never shares the GPU with generation.
    "level": AsyncOptions(steal=False, early=0, score="level"),
    "level+steal-cap64": AsyncOptions(cap=64, early=0, score="level"),
    "level+early": AsyncOptions(steal=False, early=1.0, score="level"),
    "level+early-cap64": AsyncOptions(cap=64, steal=False, early=1.0, score="level"),
    "level+early+steal-cap64": AsyncOptions(cap=64, early=1.0, score="level"),
    "level+early2": AsyncOptions(steal=False, early=2.0, score="level"),
}


def run_one(variant: str, ranks: int, problem_id: int, cfg: SearchConfig) -> dict:
    """One problem under one variant, checked against the serial search."""
    from parabeam.backends.mock import MockBackend, SimClock, mock_problem
    from parabeam.core.serial import fingerprint
    from parabeam.core.serial import search as serial_search
    from parabeam.dist import async_sched, levelsync
    from parabeam.dist.runtime import run_local

    spec = VARIANTS[variant]
    clock = SimClock()

    async def rank_main(comm):
        backend = MockBackend(clock=clock)
        if isinstance(spec, str):
            return await levelsync.search(mock_problem(problem_id), cfg, backend, comm, spec,
                                          backend.now, backend.advance_to)
        return await async_sched.search(mock_problem(problem_id), cfg, backend, comm, spec,
                                        backend.now, backend.advance_to)

    run = asyncio.run(run_local(ranks, rank_main))[0]
    serial_backend = MockBackend()
    serial = asyncio.run(serial_search(mock_problem(problem_id), cfg, serial_backend,
                                       serial_backend.now))
    if fingerprint(run.result) != fingerprint(serial):
        raise RuntimeError(f"{variant} at P={ranks} kept other beams than serial on {problem_id}")
    stats = run.ranks
    row = {"variant": variant, "P": ranks, "id": problem_id, "N": cfg.N,
           "seconds": run.result.seconds, "serial_seconds": serial.seconds,
           "levels": len(run.result.levels),
           "wait_s_mean": sum(s.wait_s for s in stats) / ranks}
    if not isinstance(spec, str):
        row.update({
            "stolen": sum(s.stolen for s in stats),
            "early_started": sum(s.early_started for s in stats),
            "early_kept": sum(s.early_kept for s in stats),
        })
    return row


def summarize(rows: list[dict]) -> list[dict]:
    table = []
    for variant in VARIANTS:
        for ranks in sorted({r["P"] for r in rows}):
            group = [r for r in rows if r["variant"] == variant and r["P"] == ranks]
            if not group:
                continue
            seconds = sum(r["seconds"] for r in group)
            entry = {
                "variant": variant, "P": ranks, "problems": len(group),
                "s_per_problem": seconds / len(group),
                "s_per_level": seconds / sum(r["levels"] for r in group),
                "speedup": sum(r["serial_seconds"] for r in group) / seconds,
                "wait_share": statistics.fmean(r["wait_s_mean"] / r["seconds"] for r in group),
            }
            started = sum(r.get("early_started", 0) for r in group)
            if started:
                entry["early_kept_share"] = sum(r["early_kept"] for r in group) / started
            table.append(entry)
    return table


def main() -> None:
    from parabeam.eval.datasets import SUBSETS_PATH

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subset", default="weak", choices=["pilot", "weak", "timing"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--jobs", type=int, help="worker processes (default: one per CPU)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    cfg = SearchConfig(args.N, args.M, 40, 2048)
    ids = json.loads(SUBSETS_PATH.read_text())["math500"]["subsets"][args.subset][: args.limit]
    work = [(v, p, i) for v in args.variants for p in args.ranks for i in ids]
    rows: list[dict] = []
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(run_one, v, p, i, cfg) for v, p, i in work]
        for future in futures:
            row = future.result()
            rows.append(row)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                with args.out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
    show: Callable[[float | None], str] = lambda x: "    -" if x is None else f"{x:5.2f}"
    print(f"{'variant':<15} {'P':>2} {'s/problem':>9} {'s/level':>7} {'speedup':>7} "
          f"{'wait':>5} {'early kept':>10}")
    for e in summarize(rows):
        print(f"{e['variant']:<15} {e['P']:>2} {e['s_per_problem']:9.1f} {e['s_per_level']:7.2f} "
              f"{e['speedup']:6.2f}x {e['wait_share']:5.0%} {show(e.get('early_kept_share')):>10}")


if __name__ == "__main__":
    main()
