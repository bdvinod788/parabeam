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
