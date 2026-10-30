from __future__ import annotations

import pytest

from router import backbones as bb

PROMPT = bb.build_mcq_prompt("Does this video contain violent activity?", ["Yes", "No"])
HEADER = "Additional evidence extracted from the video (treat as context, not ground truth):\n"


def _with_evidence(prompt: str, text: str) -> str:
    return prompt + "\n\n" + HEADER + text + "\n"


@pytest.fixture(autouse=True)
def _restore_default_position():
    yield
    bb.set_evidence_position("after")


def test_default_placement_leaves_the_prompt_unchanged():
    prompt = _with_evidence(PROMPT, "- [00:04] he threw a punch")
    assert bb.place_evidence(prompt) == prompt


def test_before_moves_the_evidence_block_ahead_of_the_question():
    bb.set_evidence_position("before")
    moved = bb.place_evidence(_with_evidence(PROMPT, "- [00:04] he threw a punch"))
    assert moved.startswith(HEADER)
    assert moved.endswith(PROMPT)
    assert moved.count("he threw a punch") == 1


def test_prompts_without_evidence_are_untouched_in_either_position():
    bb.set_evidence_position("before")
    assert bb.place_evidence(PROMPT) == PROMPT


def test_unknown_position_is_rejected():
    with pytest.raises(ValueError):
        bb.set_evidence_position("middle")


def test_question_position_puts_the_block_between_question_and_options():
    bb.set_evidence_position("question")
    moved = bb.place_evidence(_with_evidence(PROMPT, "- [00:04] he threw a punch"))
    lines = moved.splitlines()
    assert lines[0].startswith("Question:")
    assert lines[1] == HEADER.strip()
    assert lines[3] == "Options:"
    assert moved.count("he threw a punch") == 1 and moved.rstrip().endswith("nothing else.")


def test_instruction_position_puts_the_block_just_before_the_answer_instruction():
    bb.set_evidence_position("instruction")
    moved = bb.place_evidence(_with_evidence(PROMPT, "- [00:04] he threw a punch"))
    lines = moved.splitlines()
    assert lines[-1].startswith("Answer with the single letter")
    assert lines[-3] == HEADER.strip() and lines[-2].startswith("- [00:04]")
    assert moved.startswith(PROMPT.split("\nAnswer with")[0])


@pytest.mark.parametrize("position", ["before", "question", "instruction"])
def test_every_position_keeps_the_same_words(position):
    text = "- [00:04] he threw a punch"
    original = _with_evidence(PROMPT, text)
    bb.set_evidence_position(position)
    assert sorted(bb.place_evidence(original).split()) == sorted(original.split())


def test_audited_before_layout_is_unchanged():
    bb.set_evidence_position("before")
    text = "- [00:04] he threw a punch"
    assert bb.place_evidence(_with_evidence(PROMPT, text)) == HEADER + text + "\n\n" + PROMPT
