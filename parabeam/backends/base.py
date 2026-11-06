"""The contract between the search and whatever runs the models.

Search code talks only to `Backend`, so the mock and the vLLM client are
interchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

Finish = Literal["stop", "length", "eos"]


@dataclass(frozen=True, slots=True)
class GenReq:
    """One child expansion."""

    beam_id: str  # path key of the child, e.g. p17/L3/2-0-1
    prefix_ids: list[int]  # generator token IDs so far
    seed: int  # from seeding.child_seed
    max_tokens: int  # per-step cap, same for every variant
    final: bool = False  # last level: generate to EOS, with no blank-line stop


@dataclass(frozen=True, slots=True)
class StepOut:
    token_ids: list[int]
    text: str
    finish: Finish
    t_start: float
    t_end: float


class Backend(Protocol):
    async def generate(self, req: GenReq) -> StepOut: ...

    async def score(self, prm_ids: list[list[int]]) -> list[float]:
        """Return the newest step's reward for each input."""
        ...

    def abort(self, beam_id: str) -> None:
        """Cancel a pruned early start."""
        ...
