"""Answer extraction and comparison, built on Math-Verify.

Math-Verify enforces its timeouts with signal.alarm, so call these from the main
thread only. One pathological answer can cost 5 s per comparison, so grade and
vote after the per-problem timer stops. Do not shorten the timeout to compensate:
a comparison that times out under load would make the result depend on speed.
"""
from __future__ import annotations

from functools import lru_cache

from math_verify import parse, verify


def last_boxed(text: str) -> str | None:
    r"""Return the contents of the last complete \boxed{...}, or None.

    Only boxed answers count. A solution cut off mid-way often mentions numbers
    that are not its answer, and a truncated final \boxed{ is skipped for the
    same reason.
    """
    marker = "\\boxed{"
    end = len(text)
    while (start := text.rfind(marker, 0, end)) != -1:
        depth = 0
        for i in range(start + len(marker) - 1, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                contents = text[start + len(marker) : i].strip()
                if contents:
                    return contents
                break
        end = start
    return None


@lru_cache(maxsize=4096)
def _parsed(answer: str) -> tuple:
    # A bare answer such as Evelyn only parses inside \boxed{}.
    return tuple(parse(f"\\boxed{{{answer}}}"))


def same_answer(a: str, b: str) -> bool:
    """Return True if two answer strings are mathematically the same."""
    return a == b or verify(list(_parsed(a)), list(_parsed(b)))


def is_correct(prediction: str | None, gold: str) -> bool:
    return prediction is not None and same_answer(gold, prediction)
