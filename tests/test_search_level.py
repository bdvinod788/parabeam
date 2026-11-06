"""The core pieces composed into whole searches, using a scripted model."""
import hashlib
import random

import pytest

from parabeam.backends.base import StepOut
from parabeam.core.beam import Beam, Problem, SearchConfig, expand, is_finished
from parabeam.core.select import Scored, child_counts, select_top
from parabeam.core.vote import weighted_vote


def draw(seed, salt):
    digest = hashlib.sha256(f"{salt}:{seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def generate(req, stop_chance=0.15):
    if req.final:
        return StepOut([1], f"[{req.beam_id}]", "eos", 0.0, 0.0)
    finish = "eos" if draw(req.seed, "finish") < stop_chance else "stop"
    return StepOut([1, 2], f"[{req.beam_id}]\n\n", finish, 0.0, 0.0)


def run(cfg, shuffle=None, stop_chance=0.15):
    """Run one problem. shuffle reorders results, as a parallel run would."""
    survivors = [Beam(Problem(3, (9,), (8,)))]
    finished, requests = [], []
    while survivors:
        expansions = []
        for parent, count in zip(survivors, child_counts(len(survivors), cfg.N)):
            expansions.extend(expand(parent, count, cfg))
        assert len(expansions) == cfg.N
        requests.append(expansions)
        if shuffle:
            shuffle(expansions)

        active = {}
        for parent, index, req in expansions:
            child = parent.child(index, generate(req, stop_chance), [5]).scored(draw(req.seed, "score"))
            if is_finished(child, cfg):
                finished.append(child)
            else:
                active[child.path] = child

        scored = [Scored(beam.score, path) for path, beam in active.items()]
        if shuffle:
            shuffle(scored)
        survivors = [active[c.path] for c in select_top(scored, cfg.survivors)]
    return finished, requests


def outcome(finished):
    return sorted((beam.path, beam.text, beam.score) for beam in finished)


CFG = SearchConfig(N=64, M=4, max_depth=6, max_step_tokens=64)


def test_every_level_spends_exactly_n():
    run(CFG)  # the assertion is inside run


def test_arrival_order_does_not_change_the_search():
    expected = outcome(run(CFG)[0])
    assert expected
    rng = random.Random(0)
    for _ in range(20):
        assert outcome(run(CFG, rng.shuffle)[0]) == expected


def test_only_the_last_level_is_final():
    _, levels = run(CFG)
    flags = [{e.req.final for e in level} for level in levels]
    assert flags[:-1] == [{False}] * (len(levels) - 1)
    assert flags[-1] == {len(levels) == CFG.max_depth}


def test_paths_are_unique_and_match_their_tokens():
    finished, _ = run(CFG)
    assert len({beam.path for beam in finished}) == len(finished)
    for beam in finished:
        assert beam.level == len(beam.steps)
        assert beam.gen_ids[:1] == [9]


def test_search_always_ends_with_finished_beams():
    for depth in (1, 2, 3):
        finished, _ = run(SearchConfig(N=16, M=2, max_depth=depth, max_step_tokens=8))
        assert finished


def test_search_stops_early_when_every_beam_is_done():
    finished, levels = run(CFG, stop_chance=1.0)
    assert len(levels) == 1
    assert len(finished) == CFG.N


def test_vote_over_a_search_does_not_depend_on_order():
    def extract(text):
        return str(sum(map(ord, text)) % 5)

    same = lambda a, b: a == b
    finished, _ = run(CFG)
    expected = weighted_vote(finished, extract, same)
    rng = random.Random(0)
    for _ in range(20):
        shuffled, _ = run(CFG, rng.shuffle)
        vote = weighted_vote(shuffled, extract, same)
        assert (vote.answer, vote.weight, vote.beam) == (
            expected.answer,
            expected.weight,
            expected.beam,
        )


@pytest.mark.parametrize("N, M", [(16, 4), (32, 2), (64, 8), (8, 8), (8, 1)])
def test_widths(N, M):
    assert run(SearchConfig(N, M, 5, 8))[0]
