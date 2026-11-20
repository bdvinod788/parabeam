"""Gantt data comes from the per-rank logs of a level-sync run."""
import sys

import pytest

from parabeam.eval import run
from parabeam.tools import gantt


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("ls")
    argv = ["run", "levelsync", "--backend", "mock", "--local", "2", "--subset", "pilot",
            "--limit", "1", "--N", "16", "--out", str(out)]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            run.main()
    return out


def problem(run_dir):
    return int((run_dir / run.RESULTS).read_text().split('"id": ')[1].split(",")[0])


def test_every_level_has_four_ordered_phases_per_rank(run_dir):
    lines = gantt.read_logs(run_dir, problem(run_dir))
    spans = gantt.spans(lines)
    assert len(spans) == 4 * len(lines)
    assert all(s.stop >= s.start for s in spans)


def test_the_running_count_starts_and_ends_at_zero(run_dir):
    lines = gantt.read_logs(run_dir, problem(run_dir))
    for rank in (0, 1):
        curve = gantt.running(lines, rank)
        counts = [n for _, n in curve]
        assert counts[0] == 0 and counts[-1] == 0 and min(counts) == 0
        assert max(counts) == max(line["requests"] for line in lines if line["rank"] == rank)


def test_level_tails_cover_every_request(run_dir):
    lines = gantt.read_logs(run_dir, problem(run_dir))
    rows = gantt.level_tails(lines)
    assert sum(row["requests"] for row in rows) == sum(line["requests"] for line in lines)
    assert all(row["longest_s"] >= row["median_s"] for row in rows)


def test_a_missing_problem_is_reported(run_dir):
    with pytest.raises(SystemExit, match="no log lines"):
        gantt.read_logs(run_dir, -1)


def test_it_draws_a_png(run_dir, tmp_path):
    pytest.importorskip("matplotlib")
    lines = gantt.read_logs(run_dir, problem(run_dir))
    gantt.draw(lines, "test", tmp_path / "g.png")
    assert (tmp_path / "g.png").stat().st_size > 10_000


def test_async_levels_have_three_phases(tmp_path):
    argv = ["run", "async", "--backend", "mock", "--local", "2", "--subset", "pilot",
            "--limit", "1", "--N", "16", "--cap", "4", "--out", str(tmp_path)]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            run.main()
    lines = gantt.read_logs(tmp_path, problem(tmp_path))
    spans = gantt.spans(lines)
    assert len(spans) == 3 * len(lines)
    assert {s.phase for s in spans} == {"move", "work", "wait"}
