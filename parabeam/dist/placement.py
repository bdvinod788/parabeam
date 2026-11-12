"""Which rank expands which survivor.

The unit of placement is a survivor and up to M of its children, so its
children share a prefix on one GPU. A survivor with more than M children, such
as the root or any survivor once finished beams have left fewer than N/M, is
split into several units of consecutive children.

Every rank computes the same placement from the same inputs, so no rank has to
announce it. Expanding a survivor on a rank that does not hold it means sending
the beam there first.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

Owner = int | None  # the rank holding a survivor; None if every rank does (the root)


class Unit(NamedTuple):
    survivor: int  # index into the level's survivors, best first
    start: int  # first child index
    stop: int  # one past the last

    @property
    def size(self) -> int:
        return self.stop - self.start


Placement = Callable[[Sequence[Unit], Sequence[Owner], int], list[int]]


def make_units(counts: Sequence[int], M: int) -> list[Unit]:
    """Split each survivor's children, from child_counts, into units of at most M."""
    return [
        Unit(survivor, start, min(start + M, count))
        for survivor, count in enumerate(counts)
        for start in range(0, count, M)
    ]


def round_robin(units: Sequence[Unit], owners: Sequence[Owner], size: int) -> list[int]:
    """Deal units out in order, ignoring where their survivors are."""
    return [i % size for i in range(len(units))]


def keep_local(units: Sequence[Unit], owners: Sequence[Owner], size: int) -> list[int]:
    """Keep units where their survivors are, then even out the load.

    Each rank keeps its own units, best first, up to its share of the level's
    expansions, rounded up. The rest go, in order, to the least-loaded rank.
    """
    share = -(-sum(unit.size for unit in units) // size)
    load = [0] * size
    where: list[int | None] = [None] * len(units)
    for i, unit in enumerate(units):
        owner = owners[unit.survivor]
        if owner is not None and load[owner] + unit.size <= share:
            where[i] = owner
            load[owner] += unit.size
    for i, unit in enumerate(units):
        if where[i] is None:
            rank = min(range(size), key=lambda r: (load[r], r))
            where[i] = rank
            load[rank] += unit.size
    return where  # type: ignore[return-value]


PLACEMENTS: dict[str, Placement] = {"keep_local": keep_local, "round_robin": round_robin}


def check_placement(units: Sequence[Unit], where: Sequence[int], size: int) -> None:
    """Raise unless every unit has exactly one rank, and that rank exists."""
    if len(where) != len(units):
        raise ValueError(f"placed {len(where)} units, expected {len(units)}")
    bad = [rank for rank in where if not 0 <= rank < size]
    if bad:
        raise ValueError(f"no rank {bad[0]} among {size}")
