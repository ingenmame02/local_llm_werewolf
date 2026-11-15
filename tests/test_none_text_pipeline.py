"""None応答・None本文でも正規表現がTypeErrorにならない回帰テスト（LLM不使用）。"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.action_parser import ActionKind, ActionParser
from werewolf.discussion_format import (
    answers_previous_question,
    extract_discussion_body,
    find_repeated_topic,
    has_discussion_output_markers,
    parse_discussion_claim,
    parse_discussion_output,
    split_sentences,
)
from werewolf.engine import GameEngine, diagnose_discussion_body, extract_public_seer_result
from werewolf.llm_actions import OllamaActions
from werewolf.models import Message, Role
from werewolf.ollama import ChatResponse
from werewolf.secret_leak_guard import SecretLeakGuard
from werewolf.self_consistency import SelfConsistencyGuard
from werewolf.setup import create_game


class NoneContentClient:
    """Ollamaが本文を返さなかった場合を模擬する（content=None）。"""

    timeout = 60.0
    max_retries = 1
    timeout_retries = 1

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {"ollama_ps_context": None}

    def chat(self, model, messages, options=None):
        return None

    def chat_with_metadata(self, model, messages, options=None, tools=None):
        return ChatResponse(content=None, attempts=1)


def test_chat_response_normalizes_none_content() -> None:
    assert ChatResponse(content=None, attempts=1).content == ""


def test_text_helpers_tolerate_none() -> None:
    assert ActionParser.parse(None, ActionKind.VOTE) is None
    assert extract_discussion_body(None) is None
    assert parse_discussion_output(None) is None
    assert parse_discussion_claim(None) is None
    assert has_discussion_output_markers(None) is False
    assert split_sentences(None) == []
    assert find_repeated_topic(None) is False
    assert answers_previous_question(None, "質問") is False
    assert extract_public_seer_result(None, ("Player 1",)) is None


def test_guards_tolerate_none() -> None:
    state = create_game(seed=1001, config_path="config/players.json", response_language="日本語")
    actor = next(player for player in state.players if player.role is Role.SEER)
    assert SecretLeakGuard.inspect(None, actor, state) == ()
    assert SecretLeakGuard.allows_public_claim(None, actor) is False
    assert SelfConsistencyGuard.inspect(None, actor, state) == ()
    diagnosis = diagnose_discussion_body(None, actor, state)
    assert diagnosis.fatal and diagnosis.reasons == ("empty_body",)


def test_llm_actions_helpers_tolerate_none() -> None:
    assert OllamaActions._final_response(None) == ""
    assert OllamaActions._final_response("<think>x</think>本文") == "本文"
    assert OllamaActions._is_pass(None) is False
    assert OllamaActions._discussion_sentences(None) == ()


def test_discussion_with_none_response_falls_back_safely() -> None:
    state = create_game(seed=1001, config_path="config/players.json", response_language="日本語")
    state.conversation_log.append(Message(speaker="system", content="Day 1 begins"))
    actions = OllamaActions(
        state,
        NoneContentClient(),
        seed=1001,
        options={"question_hint_mode": "none"},
        discussion_claim_mode="legacy",
    )
    actor = state.players[0]
    alive = tuple(player.name for player in state.players if player.is_alive)
    body = actions.discuss(actor.name, 1, alive, tuple(state.conversation_log))
    assert isinstance(body, str) and body
    claim = actions.choose_discussion_claim(actor.name, 1, alive, tuple(state.conversation_log))
    assert isinstance(claim, str) and claim
