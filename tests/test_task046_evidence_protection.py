"""TASK-046: 発言履歴予算の概算係数・下限と、直近発言の保護窓の回帰テスト（LLM不使用）。

game_22754212a062 では概算係数 0.8（実測 0.33 tok/字）が予算を 600字フロアまで潰し、
直近の他 player の発言が削除されていた。係数 0.5・下限 2000字・直近6件の保護が
維持されていることを固定する。
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.llm_actions import (
    CONTEXT_EVIDENCE_MIN_CHARS,
    CONTEXT_TOKENS_PER_CHAR,
    request_token_estimate,
)
from werewolf.models import Message, Role
from werewolf.prompts import RECENT_PROTECTED_SPEECHES, PromptBuilder
from werewolf.setup import create_game


def _day1_history(state, speech_count: int) -> tuple[Message, ...]:
    messages = [Message(speaker="system", content="Day 1 begins.")]
    names = [player.name for player in state.players]
    for index in range(speech_count):
        messages.append(
            Message(
                speaker=names[index % len(names)],
                content=(
                    f"議論 {index + 1}: Player {names[(index + 1) % len(names)]} の投票先と"
                    f"占い結果を確認したい。根拠を具体的に述べてください。"
                ),
            )
        )
    return tuple(messages)


def test_context_constants_are_calibrated() -> None:
    assert CONTEXT_TOKENS_PER_CHAR <= 0.5, "概算係数は実測(0.33)に対し安全側だが過大ではない"
    assert CONTEXT_EVIDENCE_MIN_CHARS >= 2000, "発言履歴予算の下限は直近議論を保持できる容量"


def test_token_estimate_stays_below_half_of_chars() -> None:
    messages = [{"content": "あ" * 1000}, {"content": "b" * 1000}]
    estimate = request_token_estimate(messages, 256)
    assert estimate <= 2000 * 0.5 + 256 + 600


def test_recent_speeches_survive_tiny_budget() -> None:
    state = create_game(seed=1001, config_path="config/players.json", response_language="日本語")
    actor = next(player for player in state.players if player.role is Role.VILLAGER)
    history = _day1_history(state, 24)
    context = PromptBuilder.build_discussion_prompt(
        actor,
        tuple(state.alive_players),
        history,
        day=1,
        state=state,
        include_claim=True,
        context_strategy="hybrid_retrieval",
        evidence_char_budget=600,
        question_hint_mode="none",
    )
    total = int(context.diagnostics["transcript_event_count"])
    kept = set(context.diagnostics["selected_event_indices"])
    protected = set(range(total - RECENT_PROTECTED_SPEECHES + 1, total + 1))
    assert protected <= kept, "直近発言は文字数予算の削除対象にならない"
    reasons = context.diagnostics["selected_reasons"]
    assert any("recent_protection" in value for value in reasons.values())
    assert int(context.diagnostics["evidence_char_count"]) > 600
