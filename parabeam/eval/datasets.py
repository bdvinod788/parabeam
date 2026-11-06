"""The benchmark datasets and the fixed subsets used for timing.

Datasets are pinned to exact revisions. A problem's id is its row in that
revision. Run `python -m parabeam.eval.datasets` to regenerate subsets.json,
which is committed and should not change once timing runs start.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SUBSETS_PATH = Path(__file__).with_name("subsets.json")

MATH500_REPO = "HuggingFaceH4/MATH-500"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
GSM8K_REPO = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"

# Changing this changes every subset.
SUBSET_SALT = "parabeam-subsets-v1"
MATH500_LEVELS = (1, 2, 3, 4, 5)
TIMING_SIZE = 100  # strong scaling, width and variance blocks
WEAK_SIZE = 50  # weak scaling and the serial baseline
PILOT_SIZE = 20  # pilot runs


@dataclass(frozen=True, slots=True)
class Example:
    id: int
    question: str
    answer: str
    level: int | None = None  # MATH500 difficulty


def gsm8k_gold(solution: str) -> str:
    """Return the final answer from a GSM8K solution, which follows ####."""
    _, separator, answer = solution.rpartition("####")
    if not separator:
        raise ValueError("GSM8K solution has no #### marker")
    return answer.strip().replace(",", "")


def load_math500() -> list[Example]:
    from datasets import load_dataset

    rows = load_dataset(MATH500_REPO, revision=MATH500_REVISION, split="test")
    return [
        Example(i, row["problem"], row["answer"], int(row["level"]))
        for i, row in enumerate(rows)
    ]


def load_gsm8k() -> list[Example]:
    from datasets import load_dataset

    rows = load_dataset(GSM8K_REPO, "main", revision=GSM8K_REVISION, split="test")
    return [Example(i, row["question"], gsm8k_gold(row["answer"])) for i, row in enumerate(rows)]


def fingerprint(examples: list[Example]) -> str:
    """Hash the questions and answers in order, to catch a changed dataset."""
    digest = hashlib.sha256()
    for example in examples:
        digest.update(json.dumps([example.question, example.answer]).encode("utf-8"))
    return digest.hexdigest()


def _shuffled(ids: list[int], salt: str) -> list[int]:
    return sorted(ids, key=lambda i: hashlib.sha256(f"{salt}:{i}".encode("ascii")).digest())


def build_subsets(math500: list[Example], gsm8k: list[Example]) -> dict:
    """Pick the subsets. Smaller subsets are prefixes of larger ones.

    MATH500 takes the same number of problems per level, so any prefix cut at a
    multiple of five stays balanced. Warm-up problems are outside every timing
    subset.
    """
    per_level = TIMING_SIZE // len(MATH500_LEVELS)
    columns = [
        _shuffled([e.id for e in math500 if e.level == level], f"{SUBSET_SALT}:math500:{level}")
        for level in MATH500_LEVELS
    ]
    if min(map(len, columns)) <= per_level:
        raise ValueError(f"every level needs more than {per_level} problems")
    math_order = [column[row] for row in range(per_level) for column in columns]
    middle = columns[len(columns) // 2]

    gsm_order = _shuffled([e.id for e in gsm8k], f"{SUBSET_SALT}:gsm8k")

    return {
        "math500": {
            "repo": MATH500_REPO,
            "revision": MATH500_REVISION,
            "fingerprint": fingerprint(math500),
            "subsets": {
                "pilot": math_order[:PILOT_SIZE],
                "weak": math_order[:WEAK_SIZE],
                "timing": math_order,
                "warmup": [middle[per_level]],
            },
        },
        "gsm8k": {
            "repo": GSM8K_REPO,
            "revision": GSM8K_REVISION,
            "fingerprint": fingerprint(gsm8k),
            "subsets": {
                "timing": gsm_order[:TIMING_SIZE],
                "warmup": [gsm_order[TIMING_SIZE]],
            },
        },
    }


def load_subset(dataset: str, subset: str) -> list[Example]:
    """Return one committed subset, or raise if the dataset no longer matches it."""
    entry = json.loads(SUBSETS_PATH.read_text())[dataset]
    examples = {"math500": load_math500, "gsm8k": load_gsm8k}[dataset]()
    if fingerprint(examples) != entry["fingerprint"]:
        raise ValueError(f"{dataset} does not match the data subsets.json was built from")
    return [examples[i] for i in entry["subsets"][subset]]


def main() -> None:
    subsets = build_subsets(load_math500(), load_gsm8k())
    SUBSETS_PATH.write_text(json.dumps(subsets, indent=2) + "\n")
    print(f"wrote {SUBSETS_PATH}")


if __name__ == "__main__":
    main()
