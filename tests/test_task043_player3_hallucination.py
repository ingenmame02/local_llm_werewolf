"""TASK-043の回帰テスト（LL不使用）: Player 3の幻覚、対象選択の別称解決、フォールバック基準。

実機ログ game_e2991441d6f0 では
- Player 3（mistral-small）が案内文のプレースホルダ（`プレイヤーX`）を模倣して
  実在しない対象の占い結果COを公開した
- 投票が別称・候補外（自分）表記で `invalid_target` → ランダムフォールバックになっていた
- 空応答（生成起因）の再試行失敗で議論がフォールバックになっていた
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.action_parser import ActionKind, ActionParser
from werewolf.config import PlayerConfig
from werewolf.engine import diagnose_discussion_body
from werewolf.llm_actions import (
    DISCUSSION_MAX_ATTEMPTS,
    TARGET_MAX_ATTEMPTS,
    TEMPLATE_PLACEHOLDER_ISSUE,
    OllamaActions,
)
from werewolf.models import Message
from werewolf.ollama import ChatResponse
from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game

CANDIDATES = tuple(f"Player {index}" for index in range(1, 9))


class ScriptedClient:
    """脚本化された応答を返すOllamaクライアントの代わり（リクエスト数を数える）。"""

    timeout = 60.0
    max_retries = 1
    timeout_retries = 1

    def __init__(self, responses: list[ChatResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.requests = 0

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {"ollama_ps_context": None}

    def _next(self) -> ChatResponse:
        self.requests += 1
        if self.responses:
            return self.responses.pop(0)
        return ChatResponse(content="", attempts=1)

    def chat(self, model, messages, options=None) -> str:
        return self._next().content

    def chat_with_metadata(self, model, messages, options=None, tools=None) -> ChatResponse:
        return self._next()


def _state(response_language: str = "日本語"):
    return create_game(
        seed=1001, config_path="config/players.json", response_language=response_language
    )


def _player(state, name: str):
    return next(player for player in state.players if player.name == name)


def _alive(state) -> tuple[str, ...]:
    return tuple(player.name for player in state.players if player.is_alive)


def test_action_parser_resolves_player_aliases() -> None:
    assert ActionParser.parse("Vote: プレイヤー3", ActionKind.VOTE, CANDIDATES) == "Player 3"
    assert ActionParser.parse("投票: P3", ActionKind.VOTE, CANDIDATES) == "Player 3"
    assert ActionParser.parse("Vote: Player 3さん", ActionKind.VOTE, CANDIDATES) == "Player 3"
    assert (
        ActionParser.parse("Seer target: プレイヤー 7", ActionKind.SEER, CANDIDATES)
        == "Player 7"
    )
    # 旧来の形式（候補名そのまま）は従来どおり
    assert ActionParser.parse("Vote: Player 1", ActionKind.VOTE, CANDIDATES) == "Player 1"
    # 対象行がない応答も、最初の行の単一候補名から対象を読む
    assert (
        ActionParser.parse(
            "投票はPlayer 2さんにしたい。根拠はCOの整合性です。", ActionKind.VOTE, CANDIDATES
        )
        == "Player 2"
    )
    # 複数候補に言及する応答は決め打ちしない（再要求へ回す）
    assert (
        ActionParser.parse(
            "Player 2さんとPlayer 4さんのCOを比較したい。", ActionKind.VOTE, CANDIDATES
        )
        is None
    )
    # 自分への投票は候補集合の検証で拒否される
    assert ActionParser.validate_target("Player 3", CANDIDATES, "Player 3") is False


def test_prompts_do_not_offer_copyable_placeholders() -> None:
    state = _state()
    actor = _player(state, "Player 3")
    alive = _alive(state)
    ctx = PromptBuilder.build_discussion_prompt(
        actor=actor,
        alive_players=alive,
        conversation_history=tuple(state.conversation_log),
        day=1,
        state=state,
        include_claim=True,
        question_hint_mode="none",
    )
    co_ctx = PromptBuilder.build_discussion_claim_prompt(
        actor=actor, alive_players=alive, day=1, state=state
    )
    prompt = ctx.system_prompt + ctx.user_prompt + co_ctx.user_prompt
    assert "Player X" not in prompt
    assert "プレイヤーX" not in prompt
    assert "[RESULT] Target=<name of a player you actually divined>" in ctx.user_prompt

def test_placeholder_echo_is_repairable_not_fatal() -> None:
    state = _state()
    state.conversation_log.append(Message(speaker="system", content="Day 1 begins."))
    actor = _player(state, "Player 3")
    response = (
        "[CO] Seer CO\n"
        "[RESULT] Target=Player X Result=Human\n"
        "[BODY] プレイヤーXが人間であることを示唆する新しい証拠です。"
        "Player 4さんのCOの整合性を確認したい。"
    )
    actions = OllamaActions(state, ScriptedClient(), seed=1001)
    assert actions._discussion_response_issue(actor, response) == TEMPLATE_PLACEHOLDER_ISSUE

    diagnosis = diagnose_discussion_body(response, actor, state)
    assert TEMPLATE_PLACEHOLDER_ISSUE in diagnosis.reasons
    assert diagnosis.fatal is False
    assert diagnosis.repairable is True

    repaired = actions._repair_discussion_response(
        actor, response, TEMPLATE_PLACEHOLDER_ISSUE, ()
    )
    assert repaired is not None
    assert "Player 4さんのCOの整合性" in repaired
    assert "Player X" not in repaired
    assert "プレイヤーX" not in repaired
    # 幻覚の対象を含む結果行は公開しない
    assert "【結果】" not in repaired
    assert actions._discussion_response_issue(actor, repaired) is None


def test_placeholder_response_is_repaired_without_fallback() -> None:
    state = _state()
    state.conversation_log.append(Message(speaker="system", content="Day 1 begins."))
    placeholder = (
        "[CO] Seer CO\n"
        "[RESULT] Target=Player X Result=Human\n"
        "[BODY] プレイヤーXが人間であることを示唆する。"
        "Player 4さんのCOの整合性を確認したい。"
    )
    client = ScriptedClient(
        [
            ChatResponse(content=placeholder, attempts=1),
            ChatResponse(content=placeholder, attempts=1),
        ]
    )
    actions = OllamaActions(
        state,
        client,
        seed=1001,
        options={"question_hint_mode": "none"},
        discussion_claim_mode="legacy",
    )
    body = actions.discuss("Player 3", 1, _alive(state), tuple(state.conversation_log))
    assert "Player X" not in body and "プレイヤーX" not in body
    assert "Player 4さんのCOの整合性" in body
    assert actions.traces[-1].fallback_used is False


def test_vote_recovers_on_third_attempt_without_random_fallback() -> None:
    state = _state()
    client = ScriptedClient(
        [
            ChatResponse(content="Vote: Player 3", attempts=1),  # 自分への投票
            ChatResponse(content="投票: プレイヤー3", attempts=1),  # 別称の自分への投票
            ChatResponse(content="Vote: Player 1", attempts=1),
        ]
    )
    actions = OllamaActions(state, client, seed=1001)
    target = actions.vote(
        "Player 3", tuple(name for name in CANDIDATES if name != "Player 3"), 1, False
    )
    assert target == "Player 1"
    assert TARGET_MAX_ATTEMPTS == 3
    assert client.requests == 3
    assert actions.traces[-1].fallback_used is False

def test_third_discussion_retry_only_for_generation_issues() -> None:
    state = _state()
    state.conversation_log.append(Message(speaker="system", content="Day 1 begins."))
    generation_client = ScriptedClient(
        [
            ChatResponse(content="", attempts=1),
            ChatResponse(content="<think>熟考</think>", attempts=1, thinking_present=True),
            ChatResponse(
                content="[CO] None\n[BODY] Player 2さんの投票理由とCOの整合性を確認したい。",
                attempts=1,
            ),
        ]
    )
    generation_actions = OllamaActions(
        state,
        generation_client,
        seed=1001,
        options={"question_hint_mode": "none"},
        discussion_claim_mode="legacy",
    )
    assert DISCUSSION_MAX_ATTEMPTS == 3
    body = generation_actions.discuss(
        "Player 3", 1, _alive(state), tuple(state.conversation_log)
    )
    assert generation_client.requests == 3
    assert generation_actions.traces[-1].fallback_used is False
    assert "Player 2さんの投票理由" in body

    # 意味検証の失敗（応答言語）は3回目の再要求を作らない（従来どおり2回）
    semantic_state = _state()
    semantic_state.conversation_log.append(
        Message(speaker="system", content="Day 1 begins.")
    )
    semantic_client = ScriptedClient(
        [
            ChatResponse(
                content="[CO] None\n[BODY] I want to check Player 2's voting intention.",
                attempts=1,
            ),
            ChatResponse(
                content="[CO] None\n[BODY] I still want to check Player 2's voting intention.",
                attempts=1,
            ),
            ChatResponse(content="[CO] None\n[BODY] この発言は使われないはず。", attempts=1),
        ]
    )
    semantic_actions = OllamaActions(
        semantic_state,
        semantic_client,
        seed=1001,
        options={"question_hint_mode": "none"},
        discussion_claim_mode="legacy",
    )
    semantic_actions.discuss(
        "Player 3", 1, _alive(semantic_state), tuple(semantic_state.conversation_log)
    )
    assert semantic_client.requests == 2


def test_player_sampling_options_apply_and_validate() -> None:
    state = _state()
    player3 = _player(state, "Player 3")
    player1 = _player(state, "Player 1")
    assert player3.temperature == 0.25
    assert player3.top_p == 0.9
    assert player3.top_k == 20
    assert player1.temperature is None

    actions = OllamaActions(
        state, ScriptedClient(), seed=1001, options={"temperature": 0.7}
    )
    options3, *_ = actions._options_for(player3, "discussion")
    options1, *_ = actions._options_for(player1, "discussion")
    # プレイヤー設定は全体設定（CLI）より優先される
    assert options3["temperature"] == 0.25
    assert options3["top_p"] == 0.9
    assert options3["top_k"] == 20
    assert options1["temperature"] == 0.7
    assert "top_p" not in options1

    for bad in ({"temperature": 2.5}, {"top_p": 1.5}, {"top_k": 401}, {"top_k": 1.5}):
        try:
            PlayerConfig(name="Player 9", model="m", **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
