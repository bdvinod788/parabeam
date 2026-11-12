"""MpiComm and level-sync over real MPI processes.

Skipped unless mpi4py imports and an mpiexec is found: PB_MPIEXEC, then the one
beside this Python (the mpich wheel puts it there), then PATH. On CARC, load
the openmpi module first.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mpi4py")

ROOT = Path(__file__).resolve().parents[1]


def mpiexec() -> str | None:
    beside = Path(sys.executable).with_name("mpiexec")
    return os.environ.get("PB_MPIEXEC") or (str(beside) if beside.exists() else None) or (
        shutil.which("mpiexec")
    )


pytestmark = pytest.mark.skipif(mpiexec() is None, reason="no mpiexec")


def launch(ranks: int, *args: str, timeout: float = 300) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(
        [mpiexec(), "-n", str(ranks), sys.executable, *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout,
    )


@pytest.mark.parametrize("ranks", [2, 4])
def test_mpi_comm(ranks):
    done = launch(ranks, "tests/mpi_prog.py", "comm")
    assert done.returncode == 0, done.stdout + done.stderr
    assert sorted(done.stdout.split("\n"))[1:] == [f"rank {r} of {ranks}: OK" for r in range(ranks)]


def test_one_failing_rank_ends_the_job():
    done = launch(2, "tests/mpi_prog.py", "fail", timeout=60)
    assert done.returncode != 0
    assert "rank 1 fails on purpose" in done.stderr


@pytest.mark.parametrize(
    ("ranks", "extra"),
    [(1, []), (2, ["--placement", "round_robin"]), (4, ["--chaos", "3"])],
)
def test_level_sync_matches_serial_over_mpi(ranks, extra):
    done = launch(ranks, "-m", "parabeam.dist.levelsync", "--limit", "2", "--N", "16", *extra)
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"2 of 2 match the serial search, P={ranks}" in done.stdout
