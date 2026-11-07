"""The reference search, end to end on the mock."""
import asyncio
from dataclasses import replace

import pytest

from parabeam.backends.base import StepOut
from parabeam.backends.mock import MockBackend, mock_problem
from parabeam.core.beam import Beam, Problem, SearchConfig
from parabeam.core.serial import (
    candidate_digest,
    fingerprint,
    plan_level,
    search,
    select_level,
)
from parabeam.core.vote import weighted_vote
from parabeam.dist.costmodel import CostModel
from parabeam.eval.grade import last_boxed, same_answer

CFG = SearchConfig(N=16, M=4, max_depth=40, max_step_tokens=2048)


def run(problem_id=17, cfg=CFG, backend=None):
    backend = backend or MockBackend()
    return asyncio.run(search(mock_problem(problem_id), cfg, backend, backend.now))


def test_every_level_spends_exactly_n():
    result = run()
    assert result.levels
    assert all(level.requests == CFG.N for level in result.levels)
    assert [level.level for level in result.levels] == list(range(1, len(result.levels) + 1))


def test_survivors_never_exceed_n_over_m_and_end_empty():
    result = run()
    assert all(len(paths) <= CFG.survivors for paths in result.survivors)
    assert result.survivors[-1] == ()
    assert len(result.survivors) == len(result.levels)


def test_finished_beams_are_sorted_unique_and_scored():
    result = run()
    paths = [beam.path for beam in result.finished]
    assert paths == sorted(paths) and len(set(paths)) == len(paths)
    assert all(step.score is not None for beam in result.finished for step in beam.steps[-1:])


def test_a_beam_descends_from_the_survivors_of_each_level():
    result = run()
    for beam in result.finished:
        for level in range(1, beam.level):
            assert beam.path[:level] in result.survivors[level - 1]


def test_each_step_is_scored_on_its_own_prefix():
    beam = run().finished[0]
    assert len(beam.prm_ids) == len(mock_problem(17).prm_prompt_ids) + sum(
        len(step.prm_ids) for step in beam.steps
    )


def test_depth_cap_ends_the_search_with_final_requests():
    cfg = SearchConfig(N=16, M=4, max_depth=2, max_step_tokens=2048)
    result = run(cfg=cfg)
    assert len(result.levels) == 2
    assert all(beam.level <= 2 for beam in result.finished)
    assert {beam.steps[-1].finish for beam in result.finished if beam.level == 2} <= {
        "eos",
        "length",
    }


def test_a_tiny_token_cap_ends_every_beam_with_length():
    cfg = SearchConfig(N=16, M=4, max_depth=40, max_step_tokens=1)
    result = run(cfg=cfg)
    assert len(result.levels) == 1
    assert {beam.steps[-1].finish for beam in result.finished} == {"length"}


def test_the_vote_runs_on_the_real_grader():
    result = run()
    vote = weighted_vote(result.finished, last_boxed, same_answer)
    assert vote is not None and vote.voters > 0
    assert vote.answer in {"0", "1", "2", "3"}


def test_timing_is_kept_out_of_the_fingerprint():
    slow = MockBackend(chaos=None)
    first = run(backend=slow)
    assert first.seconds > 0
    assert sum(level.gen_s for level in first.levels) > 0
    assert sum(level.prm_s for level in first.levels) > 0
    assert fingerprint(first) == fingerprint(run())


@pytest.mark.parametrize("field", ["survivors", "candidates", "finished"])
def test_the_fingerprint_covers_each_decision(field):
    result = run()
    assert fingerprint(replace(result, **{field: ()})) != fingerprint(result)


def test_time_goes_in_the_right_columns():
    contexts, batches = (1000, 2000), (1, 256)
    decode_only = CostModel(contexts, batches, ((1.0, 1.0), (1.0, 1.0)), (0.0, 0.0), 0.0)
    prm_only = CostModel(contexts, batches, ((0.0, 0.0), (0.0, 0.0)), (0.0, 0.0), 1.0)
    gen = run(backend=MockBackend(cost=decode_only)).levels
    prm = run(backend=MockBackend(cost=prm_only)).levels
    assert all(level.gen_s > 0 and level.prm_s == 0 for level in gen)
    assert all(level.gen_s == 0 and level.prm_s > 0 for level in prm)


def test_seconds_count_from_the_start_of_the_search():
    backend = MockBackend()
    clock = lambda: 1000.0 + backend.now()
    result = asyncio.run(search(mock_problem(17), CFG, backend, clock))
    assert result.seconds == pytest.approx(backend.now())


def test_candidate_digest_ignores_order_but_not_content():
    root = Beam(Problem(1, (1,), (2,)))
    stop = StepOut([1], "x\n\n", "stop", 0.0, 0.0)
    children = [root.child(i, stop, [1]).scored(i / 10) for i in range(4)]
    assert candidate_digest(children) == candidate_digest(children[::-1])
    assert candidate_digest(children) != candidate_digest(children[:-1])
    changed = [*children[:-1], children[-1].scored(0.31)]
    assert candidate_digest(children) != candidate_digest(changed)


def test_different_problems_differ():
    assert fingerprint(run(17)) != fingerprint(run(18))


def test_plan_level_shares_the_budget_among_few_survivors():
    root = Beam(Problem(1, (1,), (2,)))
    expansions = plan_level([root], CFG)
    assert len(expansions) == CFG.N
    assert [e.index for e in expansions] == list(range(CFG.N))


def test_select_level_keeps_finished_beams_out_of_the_competition():
    root = Beam(Problem(1, (1,), (2,)))
    stop = StepOut([1], "x\n\n", "stop", 0.0, 0.0)
    eos = StepOut([1], "x", "eos", 0.0, 0.0)
    best = root.child(0, eos, [1]).scored(0.99)
    others = [root.child(i, stop, [1]).scored(i / 100) for i in range(1, 8)]
    cfg = SearchConfig(N=8, M=4, max_depth=5, max_step_tokens=8)
    finished, survivors = select_level([*others, best], cfg)
    assert finished == [best]
    assert [beam.path for beam in survivors] == [(7,), (6,)]


def test_a_backend_failure_reaches_the_caller():
    class Broken(MockBackend):
        async def generate(self, req):
            if req.beam_id.endswith("/3"):
                raise RuntimeError("server died")
            return await super().generate(req)

    with pytest.raises(ExceptionGroup):
        run(backend=Broken())


def test_a_short_score_list_is_an_error():
    class Short(MockBackend):
        async def score(self, prm_ids):
            return (await super().score(prm_ids))[:-1]

    with pytest.raises(ValueError):
        run(backend=Short())
