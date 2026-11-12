"""Level-synchronous search over P ranks, the baseline the async scheduler must beat.

Each level, every rank expands the units placed on it, scores its children on
its own PRM, and shares only (path, score, finished) through one allgather.
Every rank then runs the same selection on the same candidates, so all reach the
same survivors without a leader. Beams stay on the rank that scored them. A
survivor moves, as token IDs, only when placement puts one of its units on
another rank. Finished beams are gathered once, at the end, for the vote.

On the mock, ranks run as tasks in one process (--local P) or as MPI processes;
either way each rank's result is checked against the serial search:

    python -m parabeam.dist.levelsync --local 4 --subset pilot
    mpiexec -n 4 python -m parabeam.dist.levelsync --subset pilot
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from parabeam.backends.base import Backend
from parabeam.core.beam import Beam, Expansion, Problem, SearchConfig, expand, is_finished
from parabeam.core.seeding import Path
from parabeam.core.select import Scored, child_counts, select_top
from parabeam.core.serial import LevelStats, SearchResult, candidate_digest, generate_all
from parabeam.dist.placement import PLACEMENTS, Owner, Unit, check_placement, make_units
from parabeam.dist.runtime import MOVE, Comm


@dataclass(frozen=True, slots=True)
class RankLevel:
    """What one rank did in one level."""

    rank: int
    level: int
    requests: int
    gen_s: float
    prm_s: float
    move_s: float  # receiving the survivors placed here, before generating
    wait_s: float  # in the allgather, mostly waiting for the slowest rank
    gen_tokens: int
    moved_in: int  # survivors received from other ranks
    sent_bytes: int  # everything this rank sent during the level


@dataclass(frozen=True, slots=True)
class LevelSyncResult:
    result: SearchResult  # the same on every rank
    ranks: tuple[RankLevel, ...]  # by level, then rank


class Candidate(NamedTuple):
    path: Path
    score: float
    finished: bool


class _Report(NamedTuple):
    candidates: list[Candidate]
    clock: float  # when this rank reached the allgather


async def _move(
    comm: Comm,
    level: int,
    units: Sequence[Unit],
    where: Sequence[int],
    owners: Sequence[Owner],
    survivors: Sequence[Path],
    held: dict[Path, Beam],
) -> int:
    """Send survivors to the ranks that will expand them; return how many arrived here."""
    outgoing: defaultdict[int, dict[int, None]] = defaultdict(dict)  # dest -> survivors
    senders: set[int] = set()
    for unit, dest in zip(units, where, strict=True):
        owner = owners[unit.survivor]
        if owner is None or owner == dest:
            continue
        if owner == comm.rank:
            outgoing[dest][unit.survivor] = None
        if dest == comm.rank:
            senders.add(owner)

    async def receive() -> int:
        waiting, arrived = set(senders), 0
        while waiting:
            source, (sent_level, beams) = await comm.recv(MOVE)
            if source not in waiting or sent_level != level:
                raise RuntimeError(f"unexpected move from rank {source} for level {sent_level}")
            waiting.remove(source)
            for beam in beams:
                held[beam.path] = beam
            arrived += len(beams)
        return arrived

    async with asyncio.TaskGroup() as group:
        for dest, indices in sorted(outgoing.items()):
            beams = [held[survivors[i]] for i in indices]
            group.create_task(comm.send((level, beams), dest, MOVE))
        arrived = group.create_task(receive())
    return arrived.result()


def _expansions(
    rank: int,
    units: Sequence[Unit],
    where: Sequence[int],
    survivors: Sequence[Path],
    held: dict[Path, Beam],
    cfg: SearchConfig,
) -> list[Expansion]:
    mine = []
    for unit, dest in zip(units, where, strict=True):
        if dest == rank:
            parent = held[survivors[unit.survivor]]
            mine.extend(expand(parent, unit.stop, cfg)[unit.start :])
    return mine


async def search(
    problem: Problem,
    cfg: SearchConfig,
    backend: Backend,
    comm: Comm,
    placement: str = "keep_local",
    clock: Callable[[], float] = time.perf_counter,
    sync_clock: Callable[[float], None] | None = None,
) -> LevelSyncResult:
    """Run one problem on every rank at once. Every rank returns the same result.

    sync_clock, if given, is called with the slowest rank's time whenever the
    ranks meet; the mock needs it because each rank has its own simulated clock.
    """
    place = PLACEMENTS[placement]

    async def gather(payload):
        reports = await comm.allgather((payload, clock()))
        if sync_clock is not None:
            sync_clock(max(at for _, at in reports))
        return [payload for payload, _ in reports]

    start = clock()
    survivors: list[Path] = [()]  # best first, the same on every rank
    owners: list[Owner] = [None]  # every rank can build the root
    held: dict[Path, Beam] = {(): Beam(problem)}  # survivors this rank can expand
    finished: list[Beam] = []  # this rank's
    kept: list[tuple[Path, ...]] = []
    digests: list[str] = []
    mine: list[RankLevel] = []
    while survivors:
        level = len(survivors[0]) + 1
        units = make_units(child_counts(len(survivors), cfg.N), cfg.M)
        where = place(units, owners, comm.size)
        check_placement(units, where, comm.size)
        sent_before, moving = comm.sent_bytes, clock()
        moved_in = await _move(comm, level, units, where, owners, survivors, held)

        begin = clock()
        expansions = _expansions(comm.rank, units, where, survivors, held, cfg)
        outs = await generate_all(backend, expansions) if expansions else []
        generated = clock()
        children = [
            e.parent.child(e.index, out, backend.prm_step_ids(out))
            for e, out in zip(expansions, outs, strict=True)
        ]
        scores = await backend.score([child.prm_ids for child in children]) if children else []
        scored_at = clock()
        scored = [child.scored(s) for child, s in zip(children, scores, strict=True)]

        reports = await gather([Candidate(c.path, c.score, is_finished(c, cfg)) for c in scored])
        candidates = [c for report in reports for c in report]
        if len(candidates) != len(set(c.path for c in candidates)) or len(candidates) != sum(
            unit.size for unit in units
        ):
            raise RuntimeError(f"level {level} lost or repeated beams")
        owner_of = {c.path: rank for rank, report in enumerate(reports) for c in report}
        top = select_top((Scored(c.score, c.path) for c in candidates if not c.finished),
                         cfg.survivors)
        survivors = [c.path for c in top]
        owners = [owner_of[path] for path in survivors]
        keep = set(survivors)
        held = {child.path: child for child in scored if child.path in keep}
        finished.extend(child for child in scored if is_finished(child, cfg))
        kept.append(tuple(survivors))
        digests.append(candidate_digest(Scored(c.score, c.path) for c in candidates))
        mine.append(RankLevel(
            comm.rank, level, len(expansions), generated - begin, scored_at - generated,
            begin - moving, clock() - scored_at, sum(len(out.token_ids) for out in outs),
            moved_in, comm.sent_bytes - sent_before,
        ))

    gathered = await gather((finished, mine))
    every = sorted((s for _, stats in gathered for s in stats), key=lambda s: (s.level, s.rank))
    return LevelSyncResult(
        SearchResult(
            problem.id,
            tuple(sorted((b for beams, _ in gathered for b in beams), key=lambda b: b.path)),
            tuple(kept),
            tuple(digests),
            _per_level(every),
            clock() - start,
        ),
        tuple(every),
    )


def _per_level(stats: Sequence[RankLevel]) -> tuple[LevelStats, ...]:
    """Sum requests and tokens over ranks; a level's times are its slowest rank's."""
    by_level: defaultdict[int, list[RankLevel]] = defaultdict(list)
    for s in stats:
        by_level[s.level].append(s)
    return tuple(
        LevelStats(
            level,
            sum(s.requests for s in group),
            max(s.gen_s for s in group),
            max(s.prm_s for s in group),
            sum(s.gen_tokens for s in group),
        )
        for level, group in sorted(by_level.items())
    )


async def _mock_rank(comm: Comm, problem_id: int, cfg: SearchConfig, placement: str,
                     chaos: int | None) -> LevelSyncResult:
    from parabeam.backends.mock import Chaos, MockBackend, mock_problem

    backend = MockBackend(chaos=None if chaos is None else Chaos(chaos + 1000 * comm.rank))
    return await search(mock_problem(problem_id), cfg, backend, comm, placement,
                        backend.now, backend.advance_to)


async def _serial(problem_id: int, cfg: SearchConfig) -> SearchResult:
    from parabeam.backends.mock import MockBackend, mock_problem
    from parabeam.core.serial import search as serial_search

    backend = MockBackend()
    return await serial_search(mock_problem(problem_id), cfg, backend, backend.now)


def _line(problem_id: int, run: LevelSyncResult, serial: SearchResult) -> tuple[bool, str]:
    from parabeam.core.serial import fingerprint

    same = fingerprint(run.result) == fingerprint(serial)
    moved = sum(s.moved_in for s in run.ranks)
    return same, (
        f"problem {problem_id:>3}: {len(run.result.levels):>2} levels, "
        f"{len(run.result.finished):>4} finished, {run.result.seconds:7.1f} s simulated "
        f"vs {serial.seconds:7.1f} s serial ({serial.seconds / run.result.seconds:4.2f}x), "
        f"{moved:>3} moved, {'match' if same else 'MISMATCH'}"
    )


def main() -> None:
    from parabeam.eval.datasets import SUBSETS_PATH

    parser = argparse.ArgumentParser(description="Level-sync search on the mock backend.")
    parser.add_argument("--local", type=int, metavar="P",
                        help="run P ranks in this process instead of over MPI")
    parser.add_argument("--subset", default="pilot", choices=["pilot", "weak", "timing"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=40)
    parser.add_argument("--max-step-tokens", type=int, default=2048)
    parser.add_argument("--placement", default="keep_local", choices=sorted(PLACEMENTS))
    parser.add_argument("--chaos", type=int, help="seed for chaos timing and delivery")
    args = parser.parse_args()

    cfg = SearchConfig(args.N, args.M, args.max_depth, args.max_step_tokens)
    ids = json.loads(SUBSETS_PATH.read_text())["math500"]["subsets"][args.subset][: args.limit]

    def rank_main(problem_id: int):
        return lambda comm: _mock_rank(comm, problem_id, cfg, args.placement, args.chaos)

    if args.local:
        from parabeam.dist.runtime import run_local

        size = args.local
        matches = 0
        for problem_id in ids:
            run = asyncio.run(run_local(size, rank_main(problem_id), args.chaos))[0]
            same, line = _line(problem_id, run, asyncio.run(_serial(problem_id, cfg)))
            matches += same
            print(line, flush=True)
    else:
        from parabeam.dist.runtime import MpiComm, abort_on_error

        async def over_mpi() -> tuple[int, int, int]:
            comm = MpiComm()
            matches = 0
            for problem_id in ids:
                run = await rank_main(problem_id)(comm)
                if comm.rank == 0:
                    same, line = _line(problem_id, run, await _serial(problem_id, cfg))
                    matches += same
                    print(line, flush=True)
            await comm.close()
            return comm.rank, comm.size, matches

        with abort_on_error():
            rank, size, matches = asyncio.run(over_mpi())
        if rank != 0:
            return
    print(f"{matches} of {len(ids)} match the serial search, P={size}, {args.placement}")
    sys.exit(0 if matches == len(ids) else 1)


if __name__ == "__main__":
    main()
