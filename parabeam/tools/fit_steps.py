"""Fit the mock backend's step statistics from PRM800K, then step lengths from GPU runs.

    python -m parabeam.tools.fit_steps phase2_train.jsonl tokenizer.json
    python -m parabeam.tools.fit_steps --from-logs RUN_DIR [RUN_DIR ...]

The first form reads OpenAI's prm800k/data/phase2_train.jsonl and the
generator's tokenizer, counting each distinct solution once however many people
labelled it. Requires the tokenizers package.

The second replaces the tokens per step with every step the generator wrote in
level-sync runs (ranks/rank-*.jsonl), and adds level_steps: the distribution of
each logged level on its own, since steps within a level are far more alike
than independent draws. Drawing a level's 256 steps independently from the
pooled distribution gave it a ~400-token straggler; the logged levels' longest
step has a median of 143 tokens. PRM800K's human-checked steps are
shorter and have a far lighter tail than what Qwen2.5-Math-1.5B writes under
beam search: in the Nov 19 runs the median step was 44 tokens against 28, and
the 99th percentile 296 against 118. The tail is what makes ranks wait for one
another, so the mock must have the real one. Solution lengths stay from PRM800K.

The result, parabeam/backends/step_stats.json, is committed, so the mock needs
none of these files.
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
LEVEL_QUANTILES = 65  # per logged level; the last is the level's longest step
MIN_SOLUTIONS = 5  # problems with fewer say little about their own spread


def inverse_cdf(values: list[float], points: int = QUANTILES) -> list[int]:
    grid = np.linspace(0, 1, points)
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


def logged_levels(runs: list[Path], max_depth: int = 40) -> list[list[int]]:
    """Tokens of every step in the runs' rank logs, one list per level, all ranks merged.

    The last level is left out: it generates to EOS, several steps at once, as
    the mock's own last level does. Empty steps count as one token.
    """
    levels: dict[tuple, list[int]] = {}
    for run in runs:
        logs = sorted((run / "ranks").glob("rank-*.jsonl"))
        if not logs:
            raise SystemExit(f"no rank logs in {run / 'ranks'}")
        for log in logs:
            for line in log.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["level"] < max_depth:
                    key = (str(run), record.get("job"), record["id"], record["level"])
                    levels.setdefault(key, []).extend(max(1, s[0]) for s in record["steps"])
    return [levels[key] for key in sorted(levels, key=str)]


def refit_step_tokens(stats: dict, levels: list[list[int]], source: str) -> dict:
    """Return stats with tokens per step, pooled and per level, from logged levels."""
    pooled = [t for level in levels for t in level]
    return {**stats, "step_tokens": inverse_cdf(pooled),
            "level_steps": [inverse_cdf(level, LEVEL_QUANTILES) for level in levels],
            "steps": len(pooled), "step_source": source}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", type=Path, nargs="?")
    parser.add_argument("tokenizer", type=Path, nargs="?")
    parser.add_argument("--from-logs", type=Path, nargs="+", metavar="RUN_DIR")
    args = parser.parse_args()
    if args.from_logs:
        levels = logged_levels(args.from_logs)
        source = ", ".join(sorted({json.loads((run / "manifest.json").read_text())
                                   .get("slurm_job", run.name) for run in args.from_logs}))
        stats = refit_step_tokens(json.loads(OUTPUT.read_text()), levels,
                                  f"rank logs of jobs {source}")
        OUTPUT.write_text(json.dumps(stats) + "\n")
        print(f"{stats['steps']} logged steps in {len(levels)} levels -> {OUTPUT}")
        return
    if args.labels is None or args.tokenizer is None:
        parser.error("give PRM800K labels and a tokenizer, or --from-logs")
    from tokenizers import Tokenizer

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
