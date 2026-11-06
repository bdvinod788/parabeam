import dataclasses
import pickle

import pytest

from parabeam.backends.base import StepOut
from parabeam.core.beam import Beam, Problem, SearchConfig, expand, is_finished
from parabeam.core.seeding import child_seed

PROBLEM = Problem(id=7, gen_prompt_ids=(1, 2), prm_prompt_ids=(3,))
CFG = SearchConfig(N=8, M=2, max_depth=3, max_step_tokens=100)


def out(ids, text="x\n\n", finish="stop"):
    return StepOut(list(ids), text, finish, 0.0, 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"N": 8, "M": 3},
        {"N": 2, "M": 4},
        {"N": 8, "M": 0},
        {"N": 8, "M": 2, "max_depth": 0},
        {"N": 8, "M": 2, "max_step_tokens": 0},
    ],
)
def test_bad_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SearchConfig(**{"N": 8, "M": 2, "max_depth": 3, "max_step_tokens": 9, **kwargs})


def test_survivors():
    assert CFG.survivors == 4


def test_child_extends_tokens_without_touching_the_parent():
    root = Beam(PROBLEM)
    a = root.child(5, out([10, 11], "a\n\n"), [20, 99]).scored(0.5)
    b = a.child(0, out([12], "b"), [21, 99])
    assert (b.path, b.level, b.beam_id) == ((5, 0), 2, "p7/L2/5-0")
    assert b.gen_ids == [1, 2, 10, 11, 12]
    assert b.prm_ids == [3, 20, 99, 21, 99]
    assert b.text == "a\n\nb"
    assert a.gen_ids == [1, 2, 10, 11]
    assert root.steps == ()


def test_scored_only_changes_the_newest_step():
    a = Beam(PROBLEM).child(0, out([1]), [1]).scored(0.25)
    b = a.child(0, out([2]), [2]).scored(0.75)
    assert b.score == 0.75
    assert b.steps[0].score == 0.25
    assert a.score == 0.25


def test_reading_a_missing_score_raises():
    with pytest.raises(ValueError):
        Beam(PROBLEM).score
    with pytest.raises(ValueError):
        Beam(PROBLEM).child(0, out([1]), [1]).score
    with pytest.raises(ValueError):
        Beam(PROBLEM).scored(0.5)


def test_beams_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Beam(PROBLEM).path = (1,)


def test_beam_survives_pickle():
    # mpi4py sends Python objects with pickle.
    a = Beam(PROBLEM).child(3, out([1, 2]), [4]).scored(0.5)
    assert pickle.loads(pickle.dumps(a)) == a


@pytest.mark.parametrize(
    "ids, finish, stopped",
    [
        ([1], "stop", False),
        ([1], "eos", True),
        ([1], "length", True),
        ([], "stop", True),
    ],
)
def test_stopped(ids, finish, stopped):
    assert Beam(PROBLEM).child(0, out(ids, finish=finish), [1]).stopped is stopped


def test_root_is_not_stopped():
    assert not Beam(PROBLEM).stopped


def test_expand_builds_seeded_requests():
    parent = Beam(PROBLEM).child(2, out([9]), [1]).scored(0.5)
    expansions = expand(parent, 3, CFG)
    assert [e.index for e in expansions] == [0, 1, 2]
    req = expansions[1].req
    assert req.beam_id == "p7/L2/2-1"
    assert req.prefix_ids == [1, 2, 9]
    assert req.seed == child_seed(7, (2,), 1)
    assert req.max_tokens == 100
    assert not req.final


def test_a_childs_request_does_not_depend_on_how_many_siblings_it_has():
    parent = Beam(PROBLEM)
    assert expand(parent, 2, CFG)[1].req == expand(parent, 9, CFG)[1].req


def test_only_the_last_level_drops_the_stop():
    beam = Beam(PROBLEM)
    finals = []
    for level in range(CFG.max_depth):
        expansion = expand(beam, 1, CFG)[0]
        finals.append(expansion.req.final)
        beam = beam.child(0, out([1]), [1])
    assert finals == [False, False, True]
    with pytest.raises(ValueError):
        expand(beam, 1, CFG)


def test_is_finished_at_max_depth_even_if_the_backend_says_stop():
    beam = Beam(PROBLEM)
    for level in range(CFG.max_depth):
        assert not is_finished(beam, CFG)
        beam = beam.child(0, out([1]), [1])
    assert is_finished(beam, CFG)
