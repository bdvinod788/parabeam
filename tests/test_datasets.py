import json

import pytest

from parabeam.eval.datasets import (
    SUBSETS_PATH,
    Example,
    build_subsets,
    fingerprint,
    gsm8k_gold,
)


def fake_math500():
    return [Example(i, f"q{i}", str(i), level=i % 5 + 1) for i in range(150)]


def fake_gsm8k():
    return [Example(i, f"q{i}", str(i)) for i in range(150)]


@pytest.mark.parametrize(
    "solution, expected",
    [("work\n#### 18", "18"), ("#### 1,000", "1000"), ("a #### b\n#### -3", "-3")],
)
def test_gsm8k_gold(solution, expected):
    assert gsm8k_gold(solution) == expected


def test_gsm8k_gold_needs_the_marker():
    with pytest.raises(ValueError):
        gsm8k_gold("no marker")


def test_fingerprint_sees_a_changed_row():
    a = [Example(0, "q", "1")]
    assert fingerprint(a) != fingerprint([Example(0, "q", "2")])
    assert fingerprint(a) == fingerprint([Example(0, "q", "1", level=3)])


def test_subsets_are_reproducible():
    assert build_subsets(fake_math500(), fake_gsm8k()) == build_subsets(
        fake_math500(), fake_gsm8k()
    )


def test_subset_does_not_depend_on_row_order():
    rows = fake_math500()
    first = build_subsets(rows, fake_gsm8k())["math500"]["subsets"]
    second = build_subsets(rows[::-1], fake_gsm8k())["math500"]["subsets"]
    assert first == second


def test_too_few_problems_per_level_is_an_error():
    with pytest.raises(ValueError):
        build_subsets(fake_math500()[:50], fake_gsm8k())


def committed():
    return json.loads(SUBSETS_PATH.read_text())


def test_committed_subsets_have_the_planned_sizes():
    math500 = committed()["math500"]["subsets"]
    gsm8k = committed()["gsm8k"]["subsets"]
    assert [len(math500[k]) for k in ("pilot", "weak", "timing", "warmup")] == [20, 50, 100, 1]
    assert [len(gsm8k[k]) for k in ("timing", "warmup")] == [100, 1]


def test_committed_subsets_nest_and_have_no_repeats():
    math500 = committed()["math500"]["subsets"]
    assert math500["timing"][:50] == math500["weak"]
    assert math500["timing"][:20] == math500["pilot"]
    assert len(set(math500["timing"])) == 100


def test_warmup_problems_are_never_timed():
    for entry in committed().values():
        assert not set(entry["subsets"]["warmup"]) & set(entry["subsets"]["timing"])


def test_committed_ids_are_in_range():
    data = committed()
    assert all(0 <= i < 500 for i in data["math500"]["subsets"]["timing"])
    assert all(0 <= i < 1319 for i in data["gsm8k"]["subsets"]["timing"])
