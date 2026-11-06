"""The PRM-weighted vote over finished beams.

The vote may run on beams gathered in any order, so nothing here depends on
input order. That includes the sums: plain float addition is order-dependent, so
weights use math.fsum.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from parabeam.core.beam import Beam


@dataclass(frozen=True, slots=True)
class Vote:
    answer: str
    weight: float  # total reward behind the winning answer
    beam: Beam  # best-scoring beam that gave it
    voters: int  # beams that gave any answer


def weighted_vote(
    beams: Iterable[Beam],
    extract: Callable[[str], str | None],
    equivalent: Callable[[str, str], bool],
) -> Vote | None:
    """Return the answer with the largest total reward, or None if no beam gave one.

    Beams with no answer do not vote. Answers are grouped by comparing each
    distinct answer, in sorted order, against the first answer of each existing
    group. Equal weights go to the group whose first answer sorts first.
    """
    by_answer: dict[str, list[Beam]] = {}
    for beam in beams:
        answer = extract(beam.text)
        if answer is not None:
            by_answer.setdefault(answer, []).append(beam)
    if not by_answer:
        return None

    groups: list[list[str]] = []
    for answer in sorted(by_answer):
        for group in groups:
            if equivalent(group[0], answer):
                group.append(answer)
                break
        else:
            groups.append([answer])

    def members(group: list[str]) -> list[Beam]:
        return [beam for answer in group for beam in by_answer[answer]]

    def weight(group: list[str]) -> float:
        return math.fsum(beam.score for beam in members(group))

    winner = min(groups, key=lambda group: (-weight(group), group[0]))
    best = min(members(winner), key=lambda beam: (-beam.score, beam.path))
    return Vote(
        answer=extract(best.text),
        weight=weight(winner),
        beam=best,
        voters=sum(len(found) for found in by_answer.values()),
    )
