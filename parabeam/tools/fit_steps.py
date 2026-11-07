"""Fit the mock backend's step statistics from PRM800K.

    python -m parabeam.tools.fit_steps phase2_train.jsonl tokenizer.json

The input is OpenAI's prm800k/data/phase2_train.jsonl and the generator's
tokenizer. Each distinct solution counts once, however many people labelled it.
The result, parabeam/backends/step_stats.json, is committed, so the mock needs
neither file. Requires the tokenizers package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

OUTPUT = Path(__file__).resolve().parents[1] / "backends" / "step_stats.json"
QUANTILES = 1001  # the tail is where the stragglers are, so keep it fine
MIN_SOLUTIONS = 5  # problems with fewer say little about their own spread


def inverse_cdf(values: list[float]) -> list[int]:
    grid = np.linspace(0, 1, QUANTILES)
    return [round(float(v)) for v in np.quantile(values, grid, method="inverted_cdf")]


def fit(step_tokens: list[int], steps_by_problem: dict[str, list[int]]) -> dict:
    """Summarise tokens per step, and steps per solution within and across problems."""
    solved = [steps for steps in steps_by_problem.values() if len(steps) >= MIN_SOLUTIONS]
    within_sd = float(np.mean([np.var(steps) for steps in solved])) ** 0.5
    return {
        "step_tokens": inverse_cdf(step_tokens),
        "problem_steps": inverse_cdf([float(np.mean(steps)) for steps in solved]),
        # Half-width of a uniform distribution with the within-problem spread.
        "stop_spread": round(within_sd * math.sqrt(3)),
        "within_problem_sd": within_sd,
        "problems": len(solved),
        "solutions": sum(map(len, steps_by_problem.values())),
        "steps": len(step_tokens),
    }


def main() -> None:
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", type=Path)
    parser.add_argument("tokenizer", type=Path)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    seen: set[bytes] = set()
    steps: list[str] = []
    steps_by_problem: dict[str, list[int]] = {}
    with args.labels.open() as lines:
        for line in lines:
            question = json.loads(line)["question"]
            solution = question.get("pre_generated_steps")
            if not solution:
                continue
            key = hashlib.sha256(json.dumps([question["problem"], solution]).encode()).digest()
            if key in seen:
                continue
            seen.add(key)
            # Steps are generated up to a blank line, which counts towards their length.
            steps.extend(step.rstrip("\n") + "\n\n" for step in solution)
            steps_by_problem.setdefault(question["problem"], []).append(len(solution))

    encoded = tokenizer.encode_batch(steps, add_special_tokens=False)
    stats = fit([len(e.ids) for e in encoded], steps_by_problem)
    stats["source"] = f"{args.labels.name}, {args.tokenizer.name}"
    OUTPUT.write_text(json.dumps(stats) + "\n")
    print(f"{stats['problems']} problems, {stats['steps']} steps -> {OUTPUT}")


if __name__ == "__main__":
    main()
