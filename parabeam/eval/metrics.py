"""Reading vLLM's Prometheus metrics."""
from __future__ import annotations


def prometheus_value(text: str, name: str) -> float | None:
    """Sum a metric over its labels. Counters may carry a _total suffix."""
    total = None
    for line in text.splitlines():
        for full in (name, f"{name}_total"):
            if line.startswith(full) and line[len(full) : len(full) + 1] in (" ", "{"):
                total = (total or 0.0) + float(line.rsplit(" ", 1)[1])
    return total
