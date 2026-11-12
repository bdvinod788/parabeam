"""Level-sync must keep exactly the beams the serial search keeps, at any P."""
import asyncio
from dataclasses import replace

import pytest

from parabeam.backends.mock import Chaos, MockBackend, mock_problem
from parabeam.core.beam import SearchConfig
from parabeam.core.serial import fingerprint
from parabeam.core.serial import search as serial_search
from parabeam.dist import levelsync, placement
from parabeam.dist.runtime import run_local

CFG = SearchConfig(N=16, M=4, max_depth=40, max_step_tokens=2048)
PROBLEMS = [0, 17, 462]


def serial(problem_id, cfg=CFG):
    backend = MockBackend()
    return asyncio.run(serial_search(mock_problem(problem_id), cfg, backend, backend.now))


def spread(problem_id, ranks, where="keep_local", chaos=None, cfg=CFG, wrap=None):
    """Run level-sync on the mock and return every rank's result."""

    async def rank_main(comm):
        backend = MockBackend(chaos=None if chaos is None else Chaos(chaos + 1000 * comm.rank))
        used = backend if wrap is None else wrap(backend, comm.rank)
        return await levelsync.search(mock_problem(problem_id), cfg, used, comm, where,
                                      backend.now, backend.advance_to)

    return asyncio.run(run_local(ranks, rank_main, chaos))


@pytest.fixture(scope="module")
def expected():
    return {problem_id: fingerprint(serial(problem_id)) for problem_id in PROBLEMS}


@pytest.mark.parametrize("problem_id", PROBLEMS)
@pytest.mark.parametrize("ranks", [1, 2, 4])
@pytest.mark.parametrize("where", ["keep_local", "round_robin"])
@pytest.mark.parametrize("chaos", [None, 0, 1, 2])
def test_level_sync_matches_serial(expected, problem_id, ranks, where, chaos):
    for run in spread(problem_id, ranks, where, chaos):
        assert fingerprint(run.result) == expected[problem_id]


@pytest.mark.parametrize("ranks", [3, 4])
def test_level_sync_matches_serial_at_n64(ranks):
    cfg = replace(CFG, N=64)
    run = spread(17, ranks, cfg=cfg, chaos=5)[0]
    assert fingerprint(run.result) == fingerprint(serial(17, cfg))


def test_every_rank_returns_the_same_result():
    runs = spread(17, 4, chaos=1)
    assert len({fingerprint(run.result) for run in runs}) == 1
    assert len({run.result.seconds for run in runs}) == 1
    assert len({run.ranks for run in runs}) == 1


def test_levels_spend_n_split_evenly_over_ranks():
    run = spread(17, 4)[0]
    assert [level.requests for level in run.result.levels] == [
        level.requests for level in serial(17).levels
    ]
    for level in run.result.levels:
        counts = [s.requests for s in run.ranks if s.level == level.level]
        assert len(counts) == 4 and max(counts) - min(counts) <= CFG.M


def test_one_rank_is_the_serial_search_with_the_same_clock():
    run = spread(17, 1)[0]
    reference = serial(17)
    assert run.result.seconds == pytest.approx(reference.seconds)
    assert all(s.moved_in == 0 and s.wait_s == 0 for s in run.ranks)


def test_more_ranks_finish_sooner_in_simulated_time():
    seconds = [spread(17, ranks)[0].result.seconds for ranks in (1, 2, 4)]
    assert seconds[0] > seconds[1] > seconds[2]


def test_keep_local_moves_fewer_beams_than_round_robin():
    def moved(where):
        return sum(s.moved_in for s in spread(17, 4, where)[0].ranks)

    assert 0 < moved("keep_local") < moved("round_robin")


def test_moves_are_counted_in_bytes():
    run = spread(17, 2, "round_robin")[0]
    assert all(s.sent_bytes > 0 for s in run.ranks)
    heavy = [s for s in run.ranks if s.moved_in]
    assert heavy


class Sabotaged:
    """A backend that misbehaves on one rank only."""

    def __init__(self, backend, how):
        self._backend, self._how = backend, how

    def __getattr__(self, name):
        return getattr(self._backend, name)

    async def generate(self, req):
        if self._how == "rank_seed":
            req = replace(req, seed=req.seed ^ 1)
        return await self._backend.generate(req)

    async def score(self, prm_ids):
        scores = await self._backend.score(prm_ids)
        return scores[::-1] if self._how == "swapped_scores" else scores


@pytest.mark.parametrize("how", ["rank_seed", "swapped_scores"])
def test_the_check_catches_one_bad_rank(expected, how):
    def wrap(backend, rank):
        return Sabotaged(backend, how) if rank == 1 else backend

    for problem_id in PROBLEMS:
        assert fingerprint(spread(problem_id, 4, wrap=wrap)[0].result) != expected[problem_id]


def test_a_placement_that_drops_a_unit_is_refused(monkeypatch):
    def drop_last(units, owners, size):
        return placement.round_robin(units, owners, size)[:-1]

    monkeypatch.setitem(levelsync.PLACEMENTS, "keep_local", drop_last)
    with pytest.raises(ExceptionGroup) as caught:
        spread(17, 2)
    assert caught.group_contains(ValueError, match="expected")


def test_a_rank_that_loses_a_beam_is_caught():
    class Forgetful(Sabotaged):
        async def score(self, prm_ids):
            return (await self._backend.score(prm_ids))[:-1]

    with pytest.raises(ExceptionGroup):
        spread(17, 2, wrap=lambda backend, rank: Forgetful(backend, None) if rank else backend)
