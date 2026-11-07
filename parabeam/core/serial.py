"""The reference search: one backend, one level at a time.

Every other policy must keep exactly the beams this one keeps. plan_level and
select_level are the two steps they share, and fingerprint is how they are
compared.

    python -m parabeam.core.serial --subset pilot --N 64 --M 4
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from parabeam.backends.base import Backend, StepOut
from parabeam.core.beam import Beam, Expansion, Problem, SearchConfig, expand, is_finished
from parabeam.core.seeding import Path
from parabeam.core.select import Scored, child_counts, select_top


@dataclass(frozen=True, slots=True)
class LevelStats:
    level: int
    requests: int
    gen_s: float
    prm_s: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    problem_id: int
    finished: tuple[Beam, ...]  # sorted by path
    survivors: tuple[tuple[Path, ...], ...]  # per level, best first
    candidates: tuple[str, ...]  # per level, see candidate_digest
    levels: tuple[LevelStats, ...]
    seconds: float


def plan_level(survivors: list[Beam], cfg: SearchConfig) -> list[Expansion]:
    """Return the N expansions of a level, given its survivors best first."""
    counts = child_counts(len(survivors), cfg.N)
    return [e for parent, n in zip(survivors, counts) for e in expand(parent, n, cfg)]


def select_level(children: Iterable[Beam], cfg: SearchConfig) -> tuple[list[Beam], list[Beam]]:
    """Split scored children into finished beams and survivors, best first."""
    finished, active = [], {}
    for child in children:
        if is_finished(child, cfg):
            finished.append(child)
        else:
            active[child.path] = child
    scored = [Scored(beam.score, path) for path, beam in active.items()]
    return finished, [active[c.path] for c in select_top(scored, cfg.survivors)]


def candidate_digest(children: Iterable[Beam]) -> str:
    """Hash every scored child of a level, in any order.

    Survivors alone would not show a lost beam that would have lost selection
    anyway, or a wrong score on one.
    """
    pairs = sorted((child.path, child.score) for child in children)
    return hashlib.sha256(repr(pairs).encode()).hexdigest()


def fingerprint(result: SearchResult) -> str:
    """Hash what the search decided, leaving out all timing."""
    decided = (result.problem_id, result.survivors, result.candidates)
    digest = hashlib.sha256(repr(decided).encode())
    for beam in result.finished:
        digest.update(repr((beam.path, beam.steps)).encode())
    return digest.hexdigest()


async def _generate_all(backend: Backend, expansions: list[Expansion]) -> list[StepOut]:
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(backend.generate(e.req)) for e in expansions]
    return [task.result() for task in tasks]


async def search(
    problem: Problem,
    cfg: SearchConfig,
    backend: Backend,
    clock: Callable[[], float] = time.perf_counter,
) -> SearchResult:
    """Run one problem. Vote after this returns, since voting can be slow."""
    start = clock()
    survivors = [Beam(problem)]
    finished: list[Beam] = []
    kept: list[tuple[Path, ...]] = []
    candidates: list[str] = []
    levels: list[LevelStats] = []
    while survivors:
        level = survivors[0].level + 1
        expansions = plan_level(survivors, cfg)
        begin = clock()
        outs = await _generate_all(backend, expansions)
        generated = clock()
        children = [
            e.parent.child(e.index, out, backend.prm_step_ids(out))
            for e, out in zip(expansions, outs, strict=True)
        ]
        scores = await backend.score([child.prm_ids for child in children])
        scored_at = clock()
        scored = [child.scored(s) for child, s in zip(children, scores, strict=True)]
        done, survivors = select_level(scored, cfg)
        finished.extend(done)
        kept.append(tuple(beam.path for beam in survivors))
        candidates.append(candidate_digest(scored))
        levels.append(LevelStats(level, len(expansions), generated - begin, scored_at - generated))
    return SearchResult(
        problem.id,
        tuple(sorted(finished, key=lambda beam: beam.path)),
        tuple(kept),
        tuple(candidates),
        tuple(levels),
        clock() - start,
    )


def main() -> None:
    from parabeam.backends.mock import Chaos, MockBackend, mock_problem
    from parabeam.core.vote import weighted_vote
    from parabeam.eval.datasets import SUBSETS_PATH
    from parabeam.eval.grade import last_boxed, same_answer

    parser = argparse.ArgumentParser(description="Serial search on the mock backend.")
    parser.add_argument("--subset", default="pilot", choices=["pilot", "weak", "timing"])
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=40)
    parser.add_argument("--max-step-tokens", type=int, default=2048)
    parser.add_argument("--chaos", type=int, help="seed for chaos timing")
    args = parser.parse_args()

    cfg = SearchConfig(args.N, args.M, args.max_depth, args.max_step_tokens)
    ids = json.loads(SUBSETS_PATH.read_text())["math500"]["subsets"][args.subset]
    chaos = None if args.chaos is None else Chaos(args.chaos)
    total = 0.0
    for problem_id in ids:
        backend = MockBackend(chaos=chaos)
        result = asyncio.run(search(mock_problem(problem_id), cfg, backend, backend.now))
        vote = weighted_vote(result.finished, last_boxed, same_answer)
        total += result.seconds
        print(
            f"problem {problem_id:>3}: {len(result.levels):>2} levels, "
            f"{len(result.finished):>4} finished, {result.seconds:7.1f} s simulated, "
            f"answer {vote.answer if vote else None}, {fingerprint(result)[:12]}"
        )
    print(f"{len(ids)} problems, {total / len(ids):.1f} s simulated per problem")


if __name__ == "__main__":
    main()
