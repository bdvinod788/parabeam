"""What each operation costs on a GPU, used by placement and replayed by the mock.

calibrate.py measures these on an A100 and writes them to cost_model.json. Until
then GUESS stands in, so timing from the mock means nothing yet.
"""
from __future__ import annotations

import json
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


def _interp(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Interpolate linearly, flat below xs[0] and following the last segment above xs[-1]."""
    if x <= xs[0]:
        return ys[0]
    i = min(bisect_left(xs, x), len(xs) - 1)
    return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1])


@dataclass(frozen=True, slots=True)
class CostModel:
    contexts: tuple[int, ...]  # context lengths measured, ascending
    batches: tuple[int, ...]  # batch sizes measured, ascending
    decode_s: tuple[tuple[float, ...], ...]  # decode_s[c][b]: one decode step, whole batch
    prefill_s: tuple[float, ...]  # prefill_s[c]: prefilling contexts[c] tokens
    prm_s_per_token: float  # scoring one token the PRM has not cached

    def __post_init__(self) -> None:
        rows = {len(row) for row in self.decode_s}
        if len(self.decode_s) != len(self.contexts) or rows != {len(self.batches)}:
            raise ValueError("decode_s must have one row per context and one column per batch")
        if len(self.prefill_s) != len(self.contexts):
            raise ValueError("prefill_s must have one value per context")

    def decode(self, batch: int, context: float) -> float:
        """Return the time of one decode step for a batch at a mean context length."""
        by_context = [_interp(batch, self.batches, row) for row in self.decode_s]
        return _interp(context, self.contexts, by_context)

    def prefill(self, tokens: int) -> float:
        """Return the time to prefill tokens that are not in the cache."""
        return _interp(tokens, (0, *self.contexts), (0.0, *self.prefill_s))

    @classmethod
    def load(cls, path: str | Path) -> CostModel:
        data = json.loads(Path(path).read_text())
        return cls(
            contexts=tuple(data["contexts"]),
            batches=tuple(data["batches"]),
            decode_s=tuple(tuple(row) for row in data["decode_s"]),
            prefill_s=tuple(data["prefill_s"]),
            prm_s_per_token=data["prm_s_per_token"],
        )


_CONTEXTS = (500, 1000, 2000)
_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)

GUESS = CostModel(
    contexts=_CONTEXTS,
    batches=_BATCHES,
    decode_s=tuple(
        tuple((0.008 + 0.0004 * batch) * (1 + context / 4000) for batch in _BATCHES)
        for context in _CONTEXTS
    ),
    prefill_s=tuple(context * 0.0002 for context in _CONTEXTS),
    # The plan's 10 to 20 s to rescore 256 beams of about 1,000 tokens.
    prm_s_per_token=6e-5,
)
