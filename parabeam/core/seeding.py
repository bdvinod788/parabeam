"""Path keys and path-derived seeds.

A beam's identity is its path through the search tree, so its seed cannot depend
on the rank, batch position or timing that produced it. Python's built-in hash()
is randomized per process, so seeds use SHA-256.
"""
from __future__ import annotations

import hashlib

Path = tuple[int, ...]


def path_key(problem: int, path: Path) -> str:
    """Return the readable key for a node, e.g. p17/L3/2-0-1."""
    return f"p{problem}/L{len(path)}/{'-'.join(map(str, path))}"


def child_seed(problem: int, parent_path: Path, child_index: int) -> int:
    """Return the seed for one child expansion, in [0, 2**63).

    The key encodes the problem, level, parent path and child index. The range
    fits the signed 64-bit seed vLLM accepts.
    """
    key = path_key(problem, (*parent_path, child_index))
    digest = hashlib.sha256(key.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") >> 1
