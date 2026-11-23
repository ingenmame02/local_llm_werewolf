from __future__ import annotations

from werewolf.ollama import normalize_harmony_response


def test_extracts_harmony_final_and_marks_thinking() -> None:
    raw = (
        "<|channel|> analysis<|message|>考えます。<|end|>"
        "<|start|> assistant<|channel|> final<|message|>最終回答です。<|return|>"
    )

    content, thinking_present = normalize_harmony_response(raw)

    assert content == "最終回答です。"
    assert thinking_present is True


def test_harmony_analysis_without_final_is_empty() -> None:
    raw = "<|channel|> analysis<|message|>まだ考え中です。<|end|>"

    content, thinking_present = normalize_harmony_response(raw)

    assert content == ""
    assert thinking_present is True


def test_plain_response_is_unchanged() -> None:
    content, thinking_present = normalize_harmony_response("通常の回答")

    assert content == "通常の回答"
    assert thinking_present is False
