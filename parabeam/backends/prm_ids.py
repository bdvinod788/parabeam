"""PRM inputs for Qwen2.5-Math-PRM, built one step at a time.

A beam's PRM IDs are the prompt, then for each step its text without the
trailing blank line followed by the <extra_0> tag. Each step is tokenized once
and appended, so every level's input extends the last level's and the PRM's
prefix cache can hit. The PRM is causal, so the reward at a tag does not depend
on what follows, and the model card's closing <|im_end|><|endoftext|> is left off.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
STEP_TAG = "<extra_0>"
POSITIVE = 1  # column of the "step is correct" class

Encode = Callable[[str], list[int]]


def prompt_text(question: str, system: str = SYSTEM) -> str:
    """Return the PRM prompt up to the start of the response.

    The model's chat template has no generation prompt, so the assistant header
    is written out here and checked against the template on the GPU.
    """
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def step_ids(encode: Encode, text: str, tag_id: int) -> list[int]:
    """Return the PRM IDs for one generated step."""
    return [*encode(text.removesuffix("\n\n")), tag_id]


def solution_ids(encode: Encode, question: str, steps: Sequence[str], tag_id: int) -> list[int]:
    ids = encode(prompt_text(question))
    for step in steps:
        ids.extend(step_ids(encode, step, tag_id))
    return ids


def probabilities(row: Sequence[float]) -> list[float]:
    """Return a row as probabilities, applying softmax if the server sent logits."""
    if all(0.0 <= value <= 1.0 for value in row) and abs(math.fsum(row) - 1.0) < 1e-3:
        return list(row)
    top = max(row)
    exps = [math.exp(value - top) for value in row]
    total = math.fsum(exps)
    return [value / total for value in exps]


def tag_rewards(ids: Sequence[int], rows: Sequence[Sequence[float]], tag_id: int) -> list[float]:
    """Return the reward at every tag, given one row per input token."""
    if len(rows) != len(ids):
        raise ValueError(f"expected one row per token, got {len(rows)} rows for {len(ids)} tokens")
    return [probabilities(row)[POSITIVE] for token, row in zip(ids, rows) if token == tag_id]


def newest_reward(rows: Sequence[Sequence[float]]) -> float:
    """Return the reward at the last token, which is the newest step's tag.

    With prefix caching the server returns rows only for the tokens it computed,
    and it always computes the last one.
    """
    if not rows:
        raise ValueError("no rows returned")
    return probabilities(rows[-1])[POSITIVE]
