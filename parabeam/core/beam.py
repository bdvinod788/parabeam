"""Beams, the search parameters and the rules for extending a beam.

Beams are immutable and share their steps with their parents. They carry token
IDs and are never rebuilt from text, so prefixes match exactly across levels and
GPUs, which is what makes the prefix caches hit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

from parabeam.backends.base import GenReq, StepOut
from parabeam.core.seeding import Path, child_seed, path_key


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """The search budget: N expansions per level, keeping the best N/M."""

    N: int
    M: int
    max_depth: int
    max_step_tokens: int

    def __post_init__(self) -> None:
        if self.M < 1 or self.N < self.M or self.N % self.M != 0:
            raise ValueError(f"N must be a positive multiple of M, got N={self.N}, M={self.M}")
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be positive, got {self.max_depth}")
        if self.max_step_tokens < 1:
            raise ValueError(f"max_step_tokens must be positive, got {self.max_step_tokens}")

    @property
    def survivors(self) -> int:
        return self.N // self.M


@dataclass(frozen=True, slots=True)
class Problem:
    id: int
    gen_prompt_ids: tuple[int, ...]
    prm_prompt_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Step:
    gen_ids: tuple[int, ...]
    prm_ids: tuple[int, ...]  # the step without its blank line, then <extra_0>
    text: str
    finish: str
    score: float | None = None  # PRM reward; None until scored


@dataclass(frozen=True, slots=True)
class Beam:
    problem: Problem
    path: Path = ()
    steps: tuple[Step, ...] = ()

    @property
    def level(self) -> int:
        return len(self.path)

    @property
    def beam_id(self) -> str:
        return path_key(self.problem.id, self.path)

    @property
    def gen_ids(self) -> list[int]:
        ids = list(self.problem.gen_prompt_ids)
        for step in self.steps:
            ids.extend(step.gen_ids)
        return ids

    @property
    def prm_ids(self) -> list[int]:
        ids = list(self.problem.prm_prompt_ids)
        for step in self.steps:
            ids.extend(step.prm_ids)
        return ids

    @property
    def text(self) -> str:
        return "".join(step.text for step in self.steps)

    @property
    def score(self) -> float:
        """The newest step's reward, which ranks beams and weights the vote."""
        if not self.steps or self.steps[-1].score is None:
            raise ValueError(f"{self.beam_id} has no score")
        return self.steps[-1].score

    @property
    def stopped(self) -> bool:
        """True if the model ended this solution, by EOS, the cap or an empty step."""
        step = self.steps[-1] if self.steps else None
        return step is not None and (step.finish != "stop" or not step.gen_ids)

    def child(self, index: int, out: StepOut, prm_ids: list[int]) -> Beam:
        step = Step(tuple(out.token_ids), tuple(prm_ids), out.text, out.finish)
        return Beam(self.problem, (*self.path, index), (*self.steps, step))

    def scored(self, score: float) -> Beam:
        if not self.steps:
            raise ValueError("the root has no step to score")
        step = replace(self.steps[-1], score=score)
        return replace(self, steps=(*self.steps[:-1], step))


class Expansion(NamedTuple):
    parent: Beam
    index: int
    req: GenReq


def expand(parent: Beam, n_children: int, cfg: SearchConfig) -> list[Expansion]:
    """Return the requests for a parent's first n_children children.

    Child i of a parent is the same request however many children it ends up
    with, so an early start stays valid after selection.
    """
    level = parent.level + 1
    if level > cfg.max_depth:
        raise ValueError(f"{parent.beam_id} is already at max_depth")
    prefix = parent.gen_ids
    return [
        Expansion(
            parent,
            index,
            GenReq(
                beam_id=path_key(parent.problem.id, (*parent.path, index)),
                prefix_ids=prefix,
                seed=child_seed(parent.problem.id, parent.path, index),
                max_tokens=cfg.max_step_tokens,
                final=level == cfg.max_depth,
            ),
        )
        for index in range(n_children)
    ]


def is_finished(beam: Beam, cfg: SearchConfig) -> bool:
    """Return True if a beam stops expanding and joins the vote."""
    return beam.stopped or beam.level >= cfg.max_depth
