import random

import pytest

from parabeam.core.select import child_counts
from parabeam.dist.placement import (
    Unit,
    check_placement,
    keep_local,
    make_units,
    round_robin,
)


def loads(units, where, size):
    load = [0] * size
    for unit, rank in zip(units, where, strict=True):
        load[rank] += unit.size
    return load


def test_the_root_splits_into_units_of_m():
    assert make_units([16], 4) == [Unit(0, 0, 4), Unit(0, 4, 8), Unit(0, 8, 12), Unit(0, 12, 16)]


def test_uneven_counts_split_into_consecutive_children():
    units = make_units(child_counts(3, 16), 4)  # counts 6, 5, 5
    assert units == [Unit(0, 0, 4), Unit(0, 4, 6), Unit(1, 0, 4), Unit(1, 4, 5),
                     Unit(2, 0, 4), Unit(2, 4, 5)]


def test_round_robin_deals_in_order():
    assert round_robin(make_units([4] * 6, 4), [0] * 6, 4) == [0, 1, 2, 3, 0, 1]


def test_keep_local_moves_nothing_when_already_balanced():
    units = make_units([4] * 8, 4)
    owners = [0, 1, 2, 3, 3, 2, 1, 0]
    assert keep_local(units, owners, 4) == owners


def test_keep_local_spreads_the_root_evenly():
    units = make_units([256], 4)
    assert loads(units, keep_local(units, [None], 4), 4) == [64, 64, 64, 64]


def test_keep_local_keeps_a_share_and_moves_the_rest():
    units = make_units([4] * 8, 4)
    where = keep_local(units, [0] * 8, 4)  # every survivor on rank 0
    assert where[:2] == [0, 0]  # the best two stay
    assert loads(units, where, 4) == [8, 8, 8, 8]


@pytest.mark.parametrize("seed", range(200))
def test_keep_local_is_balanced_to_within_one_unit(seed):
    rng = random.Random(seed)
    size, M = rng.choice([1, 2, 3, 4, 8]), rng.choice([1, 2, 4, 8])
    survivors = rng.randint(1, 64)
    N = M * rng.randint(survivors // M + 1, 64)
    units = make_units(child_counts(survivors, N), M)
    owners = [rng.choice([None, *range(size)]) for _ in range(survivors)]
    where = keep_local(units, owners, size)
    check_placement(units, where, size)
    share = -(-N // size)
    assert max(loads(units, where, size)) <= share + M
    assert sum(loads(units, where, size)) == N
    assert where == keep_local(units, owners, size)


def test_check_placement_rejects_missing_and_unknown_ranks():
    units = make_units([8], 4)
    with pytest.raises(ValueError, match="expected 2"):
        check_placement(units, [0], 2)
    with pytest.raises(ValueError, match="no rank 2"):
        check_placement(units, [0, 2], 2)
