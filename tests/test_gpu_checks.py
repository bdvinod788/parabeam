"""The parts of scripts/gpu_checks.py that need no GPU."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "scripts" / "gpu_checks.py"
spec = importlib.util.spec_from_file_location("gpu_checks", path)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def test_metric_sums_labels_and_accepts_counter_suffixes():
    text = "\n".join([
        "# HELP vllm:num_preemptions_total preemptions",
        'vllm:num_preemptions_total{engine="0",model_name="g"} 3.0',
        'vllm:num_preemptions_created{engine="0"} 1700000000.0',
        'vllm:kv_cache_usage_perc{engine="0"} 0.25',
        'vllm:kv_cache_usage_perc{engine="1"} 0.5',
        "vllm:prefix_cache_hits_total 7",
    ])
    assert checks.metric(text, "vllm:num_preemptions") == 3.0
    assert checks.metric(text, "vllm:kv_cache_usage_perc") == 0.75
    assert checks.metric(text, "vllm:prefix_cache_hits") == 7.0
    assert checks.metric(text, "vllm:prefix_cache_queries") is None
    assert checks.metric(text, "vllm:kv_cache_usage") is None


def test_split_steps_drops_empty_steps_and_caps_depth():
    assert checks.split_steps("a\n\nb\n\n\n\nc") == ["a", "b", "c"]
    assert checks.split_steps("a\n\n\nb") == ["a", "\nb"]
    assert checks.split_steps("\n\n".join("x" * 50), limit=3) == ["x", "x", "x"]
    assert checks.split_steps("  \n\n") == []


def test_level_prefixes_end_at_each_tag():
    assert checks.level_prefixes([1, 9, 2, 3, 9], 9) == [[1, 9], [1, 9, 2, 3, 9]]
    assert checks.level_prefixes([1, 2], 9) == []


def test_differences_and_tolerance():
    stats = checks.differences([0.5] * 99 + [0.6], [0.5] * 100)
    assert stats["count"] == 100 and stats["max"] == pytest.approx(0.1)
    assert stats["p95"] == 0 and stats["over_0.01"] == 1
    assert not checks.within_tolerance(stats)  # one diff over MAX_TOLERANCE
    assert checks.within_tolerance(checks.differences([0.5, 0.505], [0.5, 0.5]))
    with pytest.raises(ValueError):
        checks.differences([0.5], [])


def test_newest_reference_lines_up_levels_with_steps():
    items = [{"rewards": [0.1, 0.2, 0.3]}, {"rewards": [0.4]}]
    levels = [
        {"level": 1, "members": [0, 1], "rewards": [0.11, 0.41]},
        {"level": 2, "members": [0], "rewards": [0.21]},
    ]
    got, want = checks.newest_reference(items, levels)
    assert got == [0.11, 0.41, 0.21] and want == [0.1, 0.4, 0.2]


def test_the_card_example_is_the_one_whose_rewards_we_expect():
    import json

    card = json.loads(checks.CARD_EXAMPLE.read_text())
    assert len(card["response"]) == len(checks.CARD_REWARDS) == 4
    assert card["system"] == checks.SYSTEM
    assert card["response"][-1].endswith("The answer is (\\boxed{30}).")


def test_newest_reads_both_pooling_shapes():
    assert checks.newest([[0.9, 0.1], [0.3, 0.7]]) == 0.7  # token_classify rows
    assert checks.newest([0.3, 0.7]) == 0.7  # classify: the last token's row


def test_cached_share_counts_only_levels_after_the_first():
    levels = [
        {"level": 1, "cache_hits": 0, "cache_queries": 1000},
        {"level": 2, "cache_hits": 900, "cache_queries": 1000},
        {"level": 3, "cache_hits": 700, "cache_queries": 1000},
    ]
    assert checks.cached_share(levels) == (0.8, 2000)
    assert checks.cached_share(levels[:1]) == (0.0, 0)


def test_the_cached_prm_uses_the_plugin():
    assert checks.PRM_LAST_ARCH in " ".join(checks.CACHED_PRM)
    assert checks.CACHED_TASK == "classify"
