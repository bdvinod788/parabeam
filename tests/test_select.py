import math
import random

import pytest

from parabeam.core.select import Scored, child_counts, select_top


def test_best_first():
    picked = select_top([Scored(0.1, (0,)), Scored(0.9, (1,)), Scored(0.5, (2,))], 2)
    assert picked == [Scored(0.9, (1,)), Scored(0.5, (2,))]


def test_ties_break_by_path():
    picked = select_top([Scored(0.5, (2,)), Scored(0.5, (0,)), Scored(0.5, (1,))], 2)
    assert [c.path for c in picked] == [(0,), (1,)]


def test_path_ties_compare_numerically_not_as_text():
    picked = select_top([Scored(0.5, (10,)), Scored(0.5, (9,))], 1)
    assert picked[0].path == (9,)


def test_k_larger_than_candidates_keeps_all():
    assert len(select_top([Scored(0.5, (0,))], 4)) == 1


def test_empty():
    assert select_top([], 4) == []


def test_result_does_not_depend_on_arrival_order():
    rng = random.Random(0)
    # Few distinct scores, so ties are common.
    candidates = [
        Scored(rng.choice([0.1, 0.2, 0.3]), (rng.randrange(50), i)) for i in range(400)
    ]
    expected = select_top(candidates, 64)
    for _ in range(50):
        rng.shuffle(candidates)
        assert select_top(candidates, 64) == expected


def test_split_across_ranks_matches_serial():
    rng = random.Random(1)
    candidates = [Scored(rng.random(), (i,)) for i in range(256)]
    expected = select_top(candidates, 64)
    for ranks in (2, 3, 4):
        parts = [candidates[r::ranks] for r in range(ranks)]
        gathered = [c for part in reversed(parts) for c in part]
        assert select_top(gathered, 64) == expected


def test_repeated_path_is_rejected():
    with pytest.raises(ValueError, match="repeated path"):
        select_top([Scored(0.1, (1,)), Scored(0.2, (1,))], 1)


def test_nan_is_rejected():
    with pytest.raises(ValueError, match="NaN"):
        select_top([Scored(math.nan, (1,)), Scored(0.2, (2,))], 1)


def test_full_survivors_get_m_children_each():
    assert child_counts(4, 16) == [4, 4, 4, 4]


def test_root_gets_all_of_n():
    assert child_counts(1, 256) == [256]


def test_shortfall_keeps_the_budget_and_favors_the_best():
    assert child_counts(3, 16) == [6, 5, 5]


@pytest.mark.parametrize("survivors", range(1, 70))
def test_budget_is_always_exact(survivors):
    assert sum(child_counts(survivors, 64)) == 64


def test_no_survivors_is_an_error():
    with pytest.raises(ValueError):
        child_counts(0, 8)
