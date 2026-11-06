"""Global selection, shared by every scheduler.

Every rank runs this on the same gathered scores and must reach the same answer,
so the result depends only on the set of candidates. (-score, path) is a total
order because paths are unique within a level.
"""
from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from typing import NamedTuple

from parabeam.core.seeding import Path


class Scored(NamedTuple):
    score: float
    path: Path


def select_top(candidates: Iterable[Scored], k: int) -> list[Scored]:
    """Return the best k candidates, best first, with ties broken by path.

    Raises ValueError on a NaN score or a repeated path. Either would make the
    order ill-defined, and a repeated path means a beam was duplicated.
    """
    candidates = list(candidates)
    if len({c.path for c in candidates}) != len(candidates):
        raise ValueError("repeated path among candidates")
    if any(math.isnan(c.score) for c in candidates):
        raise ValueError("NaN score among candidates")
    return heapq.nsmallest(k, candidates, key=lambda c: (-c.score, c.path))


def child_counts(n_survivors: int, N: int) -> list[int]:
    """Split N expansions among survivors listed best first.

    With a full N/M survivors each gets M children. If finished beams left fewer
    survivors, the budget stays N and the best survivors take the remainder.
    """
    if n_survivors < 1:
        raise ValueError("no survivors to expand")
    base, extra = divmod(N, n_survivors)
    return [base + 1 if i < extra else base for i in range(n_survivors)]
