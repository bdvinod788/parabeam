"""Every policy must keep exactly the beams the serial search keeps.

The real schedulers replace `scattered` as they are built: level-sync on Nov 12,
async on Nov 16. The sabotaged policies check that this harness can fail.
"""
import asyncio
import random

import pytest

from parabeam.backends.mock import Chaos, MockBackend, mock_problem
from parabeam.core.beam import Beam, SearchConfig
from parabeam.core.serial import (
    SearchResult,
    candidate_digest,
    fingerprint,
    plan_level,
    search,
    select_level,
)

CFG = SearchConfig(N=16, M=4, max_depth=40, max_step_tokens=2048)
PROBLEMS = [0, 17, 462]


async def rank_expand(backend, expansions, rng, order, sabotage):
    """Run one rank's share, scoring children in small batches as they arrive."""
    tasks = {}
    for expansion in expansions:
        req = expansion.req
        task = asyncio.ensure_future(backend.generate(req))
        tasks[task] = expansion

    children = []
    pending = set(tasks)
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        done = sorted(done, key=lambda task: tasks[task].req.beam_id)  # wait returns a set
        rng.shuffle(done)
        while done:
            size = rng.randint(1, len(done))
            batch, done = done[:size], done[size:]
            pairs = [(tasks[t], t.result()) for t in batch]
            order.extend(e.req.beam_id for e, _ in pairs)
            beams = [
                e.parent.child(e.index, out, backend.prm_step_ids(out)) for e, out in pairs
            ]
            scores = await backend.score([beam.prm_ids for beam in beams])
            if sabotage == "swapped_scores":
                scores = scores[::-1]
            children.extend(b.scored(s) for b, s in zip(beams, scores, strict=True))
    return children


async def scattered(problem, cfg, backends, seed, sabotage=None):
    """Spread each level over ranks at random and consume results by arrival."""
    rng = random.Random(seed)
    survivors, finished, kept, digests, order = [Beam(problem)], [], [], [], []
    while survivors:
        expansions = plan_level(survivors, cfg)
        rng.shuffle(expansions)
        if sabotage == "lost_beam" and survivors[0].level == 2:
            expansions.pop()
        if sabotage == "rank_seed":
            expansions = [
                e._replace(req=_with_seed(e.req, e.req.seed ^ (i % len(backends))))
                for i, e in enumerate(expansions)
            ]
        shares = [expansions[rank :: len(backends)] for rank in range(len(backends))]
        gathered = await asyncio.gather(
            *(
                rank_expand(backend, share, rng, order, sabotage)
                for backend, share in zip(backends, shares)
            )
        )
        children = [child for rank in gathered for child in rank]
        rng.shuffle(children)
        if sabotage == "wrong_loser_score":
            worst = min(children, key=lambda beam: beam.score)
            children[children.index(worst)] = worst.scored(worst.score / 2)
        done, survivors = select_level(children, cfg)
        if sabotage == "first_arrivals":
            arrived = {beam.path: i for i, beam in enumerate(rank_order(gathered))}
            active = [c for c in children if c not in done]
            active.sort(key=lambda beam: arrived[beam.path])
            survivors = active[: cfg.survivors]
        finished.extend(done)
        kept.append(tuple(beam.path for beam in survivors))
        digests.append(candidate_digest(children))
    finished.sort(key=lambda beam: beam.path)
    return SearchResult(problem.id, tuple(finished), tuple(kept), tuple(digests), (), 0.0), order


def rank_order(gathered):
    return [child for rank in gathered for child in rank]


def _with_seed(req, seed):
    return type(req)(req.beam_id, req.prefix_ids, seed, req.max_tokens, req.final)


def serial(problem_id, chaos=None):
    backend = MockBackend(chaos=chaos)
    return asyncio.run(search(mock_problem(problem_id), CFG, backend, backend.now))


def spread(problem_id, ranks, chaos_seed, sabotage=None):
    backends = [
        MockBackend(chaos=None if chaos_seed is None else Chaos(chaos_seed + 1000 * rank))
        for rank in range(ranks)
    ]
    return asyncio.run(
        scattered(mock_problem(problem_id), CFG, backends, chaos_seed or 0, sabotage)
    )


@pytest.fixture(scope="module")
def expected():
    return {problem_id: fingerprint(serial(problem_id)) for problem_id in PROBLEMS}


@pytest.mark.parametrize("problem_id", PROBLEMS)
@pytest.mark.parametrize("seed", range(4))
def test_serial_under_chaos(expected, problem_id, seed):
    assert fingerprint(serial(problem_id, Chaos(seed))) == expected[problem_id]


@pytest.mark.parametrize("problem_id", PROBLEMS)
@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_policy_without_chaos(expected, problem_id, ranks):
    assert fingerprint(spread(problem_id, ranks, None)[0]) == expected[problem_id]


@pytest.mark.parametrize("problem_id", PROBLEMS)
@pytest.mark.parametrize("ranks", [1, 2, 4])
@pytest.mark.parametrize("seed", range(4))
def test_policy_under_chaos(expected, problem_id, ranks, seed):
    assert fingerprint(spread(problem_id, ranks, seed)[0]) == expected[problem_id]


def test_chaos_really_changes_arrival_order():
    orders = {tuple(spread(17, 4, seed)[1]) for seed in range(6)}
    assert len(orders) == 6
    assert sorted(orders.pop()) == sorted(spread(17, 4, None)[1])


def test_chaos_failures_replay_from_their_seed():
    assert spread(17, 4, 3)[1] == spread(17, 4, 3)[1]


@pytest.mark.parametrize(
    "sabotage",
    ["first_arrivals", "rank_seed", "lost_beam", "swapped_scores", "wrong_loser_score"],
)
def test_the_harness_catches_broken_policies(expected, sabotage):
    caught = [
        fingerprint(spread(problem_id, 4, seed, sabotage)[0]) != expected[problem_id]
        for problem_id in PROBLEMS
        for seed in range(3)
    ]
    assert all(caught)
