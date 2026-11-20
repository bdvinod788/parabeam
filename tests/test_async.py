"""Async must keep exactly the beams the serial search keeps, with every part on or off."""
import asyncio
from dataclasses import replace

import pytest

from parabeam.backends.mock import Chaos, MockBackend, SimClock, mock_problem
from parabeam.core.beam import SearchConfig
from parabeam.core.serial import fingerprint
from parabeam.core.serial import search as serial_search
from parabeam.dist import async_sched
from parabeam.dist.async_sched import AsyncOptions
from parabeam.dist.runtime import run_local

CFG = SearchConfig(N=16, M=4, max_depth=40, max_step_tokens=2048)
PROBLEMS = [0, 17, 462]
OPTIONS = {
    "overlap only": AsyncOptions(steal=False, early=0),
    "steal": AsyncOptions(cap=4, early=0),
    "early": AsyncOptions(steal=False, early=1.0),
    "all, cap 4": AsyncOptions(cap=4, early=1.0),
    "all, no cap, eager": AsyncOptions(early=2.0, early_after=0.0),
    "level scoring": AsyncOptions(steal=False, early=0, score="level"),
    "level scoring, all, cap 4": AsyncOptions(cap=4, early=1.0, score="level"),
}


def serial(problem_id, cfg=CFG):
    backend = MockBackend()
    return asyncio.run(serial_search(mock_problem(problem_id), cfg, backend, backend.now))


def spread(problem_id, ranks, opts, chaos=None, cfg=CFG, wrap=None):
    """Run async on the mock, every rank on one simulated clock; return every rank's result."""
    clock = SimClock()

    async def rank_main(comm):
        backend = MockBackend(chaos=None if chaos is None else Chaos(chaos + 1000 * comm.rank),
                              clock=clock)
        used = backend if wrap is None else wrap(backend, comm.rank)
        return await async_sched.search(mock_problem(problem_id), cfg, used, comm, opts,
                                        backend.now, backend.advance_to)

    return asyncio.run(run_local(ranks, rank_main, chaos))


@pytest.fixture(scope="module")
def expected():
    return {problem_id: fingerprint(serial(problem_id)) for problem_id in PROBLEMS}


@pytest.mark.parametrize("problem_id", PROBLEMS)
@pytest.mark.parametrize("ranks", [1, 2, 4])
@pytest.mark.parametrize("name", sorted(OPTIONS))
@pytest.mark.parametrize("chaos", [None, 0, 1])
def test_async_matches_serial(expected, problem_id, ranks, name, chaos):
    for run in spread(problem_id, ranks, OPTIONS[name], chaos):
        assert fingerprint(run.result) == expected[problem_id]


@pytest.mark.parametrize("ranks", [3, 4])
def test_async_matches_serial_at_n64(ranks):
    cfg = replace(CFG, N=64)
    run = spread(17, ranks, AsyncOptions(cap=8), chaos=5, cfg=cfg)[0]
    assert fingerprint(run.result) == fingerprint(serial(17, cfg))


def test_every_rank_returns_the_same_result():
    runs = spread(17, 4, AsyncOptions(cap=4), chaos=1)
    assert len({fingerprint(run.result) for run in runs}) == 1
    assert len({run.ranks for run in runs}) == 1


class Counting:
    """Wraps a backend to record the most requests generating at once."""

    def __init__(self, backend):
        self._backend = backend
        self.now = self.peak = 0

    async def generate(self, req):
        self.now += 1
        self.peak = max(self.peak, self.now)
        try:
            return await self._backend.generate(req)
        finally:
            self.now -= 1

    def __getattr__(self, name):
        return getattr(self._backend, name)


@pytest.mark.parametrize("cap", [4, 8])
def test_a_rank_never_generates_more_than_its_cap(cap):
    wrapped = {}

    def wrap(backend, rank):
        wrapped[rank] = Counting(backend)
        return wrapped[rank]

    spread(17, 4, AsyncOptions(cap=cap, early=2.0, early_after=0.0), chaos=2,
           cfg=replace(CFG, N=64), wrap=wrap)
    assert max(w.peak for w in wrapped.values()) <= cap


def test_idle_ranks_steal_and_every_unit_is_done_once():
    run = spread(17, 4, AsyncOptions(cap=4, early=0), cfg=replace(CFG, N=64))[0]
    assert sum(s.stolen for s in run.ranks) > 0
    assert sum(s.stolen for s in run.ranks) == sum(s.given for s in run.ranks)
    assert [lv.requests for lv in run.result.levels] == [lv.requests for lv in
                                                         serial(17, replace(CFG, N=64)).levels]


def test_without_stealing_nothing_moves_during_a_level():
    run = spread(17, 4, AsyncOptions(cap=4, steal=False, early=0), cfg=replace(CFG, N=64))[0]
    assert sum(s.stolen + s.given for s in run.ranks) == 0


def test_early_starts_are_used_and_the_rest_aborted():
    run = spread(17, 2, AsyncOptions(early=1.0), cfg=replace(CFG, N=64))[0]
    started = sum(s.early_started for s in run.ranks)
    kept = sum(s.early_kept for s in run.ranks)
    assert 0 < kept <= started


def test_no_early_starts_when_switched_off():
    run = spread(17, 2, AsyncOptions(early=0), cfg=replace(CFG, N=64))[0]
    assert sum(s.early_started for s in run.ranks) == 0


def level_sync_seconds(problem_id, ranks, cfg):
    from parabeam.dist import levelsync

    clock = SimClock()

    async def rank_main(comm):
        backend = MockBackend(clock=clock)
        return await levelsync.search(mock_problem(problem_id), cfg, backend, comm,
                                      "keep_local", backend.now, backend.advance_to)

    return asyncio.run(run_local(ranks, rank_main))[0].result.seconds


def test_scoring_whole_levels_without_early_starts_is_level_sync():
    cfg = replace(CFG, N=64)
    run = spread(17, 4, AsyncOptions(steal=False, early=0, score="level"), cfg=cfg)[0]
    assert run.result.seconds == pytest.approx(level_sync_seconds(17, 4, cfg), rel=1e-6)


def test_early_starts_fill_the_wait_at_the_barrier():
    cfg = replace(CFG, N=64)
    runs = [spread(p, 4, AsyncOptions(steal=False, early=1.0, score="level"), cfg=cfg)[0]
            for p in PROBLEMS]
    assert sum(r.result.seconds for r in runs) < sum(level_sync_seconds(p, 4, cfg)
                                                     for p in PROBLEMS)


def test_scoring_while_generating_costs_time_on_a_shared_gpu():
    cfg = replace(CFG, N=64)
    each = spread(17, 1, AsyncOptions(steal=False, early=0), cfg=cfg)[0].result.seconds
    assert each > level_sync_seconds(17, 1, cfg)


def test_options_are_checked():
    with pytest.raises(ValueError):
        AsyncOptions(cap=0)
    with pytest.raises(ValueError):
        AsyncOptions(early=-1)
    with pytest.raises(ValueError):
        AsyncOptions(early_after=2)
    with pytest.raises(ValueError):
        AsyncOptions(score="sometimes")
