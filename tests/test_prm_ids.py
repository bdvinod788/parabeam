"""PRM inputs, with a toy tokenizer standing in for Qwen's."""
import math

import pytest

from parabeam.backends.prm_ids import (
    SYSTEM,
    newest_reward,
    probabilities,
    prompt_text,
    solution_ids,
    step_ids,
    tag_rewards,
)

TAG = 99


def encode(text):
    return [ord(c) % 90 for c in text]


def test_the_prompt_matches_the_model_card_template():
    text = prompt_text("What is 1 + 1?")
    assert text == (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        "<|im_start|>user\nWhat is 1 + 1?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_a_step_drops_its_blank_line_and_ends_in_the_tag():
    assert step_ids(encode, "x = 2\n\n", TAG) == [*encode("x = 2"), TAG]
    assert step_ids(encode, "x = 2", TAG) == [*encode("x = 2"), TAG]
    assert step_ids(encode, "", TAG) == [TAG]


def test_each_level_extends_the_last():
    steps = ["a\n\n", "bb\n\n", "ccc"]
    levels = [solution_ids(encode, "q", steps[:k], TAG) for k in range(len(steps) + 1)]
    for shorter, longer in zip(levels, levels[1:]):
        assert longer[: len(shorter)] == shorter
    assert levels[-1].count(TAG) == 3


def test_tag_rewards_take_the_positive_column_at_tags():
    ids = [1, TAG, 2, 3, TAG]
    rows = [[0.5, 0.5], [0.2, 0.8], [0.9, 0.1], [0.4, 0.6], [0.7, 0.3]]
    assert tag_rewards(ids, rows, TAG) == [0.8, 0.3]


def test_tag_rewards_accept_one_row_per_tag():
    assert tag_rewards([1, TAG, 2, TAG], [[0.2, 0.8], [0.7, 0.3]], TAG) == [0.8, 0.3]
    assert tag_rewards([TAG, TAG], [[0.2, 0.8], [0.7, 0.3]], TAG) == [0.8, 0.3]


def test_tag_rewards_reject_other_row_counts():
    with pytest.raises(ValueError):
        tag_rewards([1, 2, TAG], [[0.5, 0.5], [0.5, 0.5]], TAG)


def test_the_newest_reward_is_the_last_row():
    assert newest_reward([[0.9, 0.1], [0.25, 0.75]]) == 0.75
    with pytest.raises(ValueError):
        newest_reward([])


def test_logits_become_probabilities():
    assert probabilities([0.3, 0.7]) == [0.3, 0.7]
    p = probabilities([0.0, math.log(3)])
    assert p == pytest.approx([0.25, 0.75])
    assert probabilities([2.0, -1.0])[0] == pytest.approx(1 / (1 + math.exp(-3)))
