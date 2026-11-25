"""The run driver, end to end on the mock backend."""
import json
import sys

import pytest

from parabeam.eval import run


def go(monkeypatch, out, *extra):
    argv = ["run", "serial", "--backend", "mock", "--subset", "pilot", "--limit", "3",
            "--N", "16", "--M", "4", "--out", str(out), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    run.main()
    return [json.loads(line) for line in (out / run.RESULTS).read_text().splitlines()]


def test_a_mock_run_writes_a_manifest_results_and_a_summary(monkeypatch, tmp_path):
    records = go(monkeypatch, tmp_path)
    assert len(records) == 3
    for record in records:
        assert record["seconds"] > 0 and record["levels"] >= 1
        assert record["requests"] == 16 * record["levels"]
        assert record["gen_tokens"] > 0 and len(record["per_level"]) == record["levels"]
        assert record["fingerprint"] and record["answer"] in {"0", "1", "2", "3"}
    manifest = json.loads((tmp_path / run.MANIFEST).read_text())
    assert manifest["config"]["N"] == 16 and manifest["config"]["backend"] == "mock"
    assert "commit" in manifest["code"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["problems"] == 3 and summary["seconds_mean"] > 0


def test_a_rerun_skips_finished_problems(monkeypatch, tmp_path):
    first = go(monkeypatch, tmp_path)
    again = go(monkeypatch, tmp_path)
    assert again == first  # nothing was appended


def test_a_rerun_is_listed_as_resumed(monkeypatch, tmp_path):
    go(monkeypatch, tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "222")
    go(monkeypatch, tmp_path, "--limit", "3")
    manifest = json.loads((tmp_path / run.MANIFEST).read_text())
    assert manifest["slurm_job"] != "222"
    assert [s["slurm_job"] for s in manifest["resumed"]] == ["222"]


def test_a_manifest_without_results_is_replaced(monkeypatch, tmp_path):
    stale = {"config": {"old": True}, "code": {"commit": "old"}, "slurm_job": "111"}
    (tmp_path / run.MANIFEST).write_text(json.dumps(stale))
    monkeypatch.setenv("SLURM_JOB_ID", "222")
    go(monkeypatch, tmp_path)
    manifest = json.loads((tmp_path / run.MANIFEST).read_text())
    assert manifest["slurm_job"] == "222" and manifest["code"] == run.code_version()
    assert "resumed" not in manifest


def test_results_from_other_code_are_not_resumed(monkeypatch, tmp_path):
    go(monkeypatch, tmp_path)
    monkeypatch.setattr(run, "code_version", lambda: {"commit": "other"})
    with pytest.raises(SystemExit, match="other code"):
        go(monkeypatch, tmp_path)


def test_a_rerun_with_other_settings_is_refused(monkeypatch, tmp_path):
    go(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        go(monkeypatch, tmp_path, "--N", "32")


def test_mock_runs_match_the_serial_search(monkeypatch, tmp_path):
    import asyncio

    from parabeam.backends.mock import MockBackend, mock_problem
    from parabeam.core.beam import SearchConfig
    from parabeam.core.serial import fingerprint, search

    records = go(monkeypatch, tmp_path)
    backend = MockBackend()
    cfg = SearchConfig(16, 4, 40, 2048)
    for record in records:
        result = asyncio.run(search(mock_problem(record["id"]), cfg, backend, backend.now))
        assert fingerprint(result) == record["fingerprint"]


def levelsync(monkeypatch, out, ranks=4, *extra):
    argv = ["run", "levelsync", "--backend", "mock", "--local", str(ranks), "--subset", "pilot",
            "--limit", "3", "--N", "16", "--M", "4", "--out", str(out), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit_:
        run.main()
    return exit_.value.code


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_level_sync_runs_match_serial_runs(monkeypatch, tmp_path):
    serial = go(monkeypatch, tmp_path / "serial")
    assert levelsync(monkeypatch, tmp_path / "ls") == 0
    records = lines(tmp_path / "ls" / run.RESULTS)
    assert [r["fingerprint"] for r in records] == [r["fingerprint"] for r in serial]
    assert all(r["requests"] == 16 * r["levels"] for r in records)
    manifest = json.loads((tmp_path / "ls" / run.MANIFEST).read_text())
    assert manifest["config"]["P"] == 4 and [r["rank"] for r in manifest["ranks"]] == [0, 1, 2, 3]


def test_each_rank_logs_its_levels_steps_and_phases(monkeypatch, tmp_path):
    assert levelsync(monkeypatch, tmp_path) == 0
    records = {r["id"]: r for r in lines(tmp_path / run.RESULTS)}
    logs = [lines(tmp_path / "ranks" / f"rank-{rank}.jsonl") for rank in range(4)]
    for rank, log in enumerate(logs):
        assert {line["rank"] for line in log} == {rank}
        for line in log:
            assert len(line["steps"]) == line["requests"]
            assert sum(tokens for tokens, _, _ in line["steps"]) == line["gen_tokens"]
            assert all(finish in {"stop", "length", "eos"} for _, finish, _ in line["steps"])
            assert line["marks"] == sorted(line["marks"]) and line["marks"][0] >= 0
    for problem_id, record in records.items():
        per_rank = [[line for line in log if line["id"] == problem_id] for log in logs]
        assert all(len(levels) == record["levels"] for levels in per_rank)
        assert sum(line["requests"] for levels in per_rank for line in levels) == record["requests"]
        # The ranks leave each level's allgather together, on the synced mock clocks.
        for level in range(record["levels"]):
            assert len({levels[level]["marks"][4] for levels in per_rank}) == 1


def test_a_level_sync_rerun_resumes_and_other_settings_are_refused(monkeypatch, tmp_path, capsys):
    assert levelsync(monkeypatch, tmp_path) == 0
    first = (tmp_path / run.RESULTS).read_text()
    assert levelsync(monkeypatch, tmp_path) == 0
    assert (tmp_path / run.RESULTS).read_text() == first
    capsys.readouterr()
    assert levelsync(monkeypatch, tmp_path, 2) == 1
    assert capsys.readouterr().err.count("other settings") == 1  # rank 0 reports it once


def test_local_ranks_need_the_mock(monkeypatch, tmp_path):
    argv = ["run", "levelsync", "--local", "2", "--gen-url", "u", "--prm-url", "u",
            "--out", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit_:
        run.main()
    assert exit_.value.code == 2


def run_async(monkeypatch, out, ranks=4, *extra):
    argv = ["run", "async", "--backend", "mock", "--local", str(ranks), "--subset", "pilot",
            "--limit", "3", "--N", "16", "--M", "4", "--cap", "4", "--out", str(out), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit_:
        run.main()
    return exit_.value.code


def test_async_runs_match_serial_runs_and_log_their_counters(monkeypatch, tmp_path):
    serial = go(monkeypatch, tmp_path / "serial")
    assert run_async(monkeypatch, tmp_path / "as") == 0
    records = lines(tmp_path / "as" / run.RESULTS)
    assert [r["fingerprint"] for r in records] == [r["fingerprint"] for r in serial]
    assert all({"stolen", "early_started", "early_kept"} <= set(r) for r in records)
    config = json.loads((tmp_path / "as" / run.MANIFEST).read_text())["config"]
    assert config["mode"] == "async" and config["cap"] == 4 and config["steal"] is True
    for rank in range(4):
        for line in lines(tmp_path / "as" / "ranks" / f"rank-{rank}.jsonl"):
            assert len(line["marks"]) == 4 and line["marks"] == sorted(line["marks"])
            assert len(line["steps"]) == line["requests"]


def test_async_and_level_sync_runs_do_not_share_a_directory(monkeypatch, tmp_path):
    assert levelsync(monkeypatch, tmp_path) == 0
    assert run_async(monkeypatch, tmp_path) == 1  # other settings: refused


def test_the_all_subset_is_the_whole_dataset_on_the_mock():
    items, _ = run.load_items("math500", "all", mock=True)
    assert [item.id for item in items] == list(range(500))
    assert len(run.load_items("gsm8k", "all", mock=True)[0]) == 1319
