"""Player 3のフォールバック抑制（TASK-041）と狂人の占い結果の共通認識の回帰テスト（LL不使用）。

実機ログ game_f6983c7cbcc2 では真占い師（Player 3 / 小モデル）が
`[CO] None` + `[RESULT] None` や `対象名=プレイヤー5 結果=Human` を書いたために
`seer_result_without_claim` → 再試行 → 英語の汎用フォールバックになっていました。
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.discussion_format import (
    normalize_result_field,
    parse_discussion_claim,
    parse_discussion_output,
)
from werewolf.engine import extract_public_seer_result
from werewolf.llm_actions import OllamaActions
from werewolf.models import Message, PublicSeerClaim, Role
from werewolf.ollama import ChatResponse
from werewolf.prompts import (
    ROLE_DESCRIPTIONS_EN,
    RULE_CARD,
    WEREWOLF_THEORY_CARD,
    PromptBuilder,
)
from werewolf.setup import create_game

PLAYER_NAMES = tuple(f"Player {index}" for index in range(1, 9))


class ScriptedClient:
    """脚本化された応答を返すOllamaクライアントの代わり（LL不使用）。"""

    timeout = 60.0
    max_retries = 1
    timeout_retries = 1

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {"ollama_ps_context": None}

    def _next(self) -> str:
        return self.responses.pop(0) if self.responses else ""

    def chat(self, model, messages, options=None):
        return self._next()

    def chat_with_metadata(self, model, messages, options=None, tools=None):
        return ChatResponse(content=self._next(), attempts=1)


def _seer_state(response_language: str = "日本語"):
    state = create_game(
        seed=1001, config_path="config/players.json", response_language=response_language
    )
    seer = next(player for player in state.players if player.role is Role.SEER)
    state.seer_results = {"Player 1": "human"}
    return state, seer


def test_unpublished_result_lines_normalize_to_no_result() -> None:
    assert normalize_result_field("None") == ""
    assert normalize_result_field("なし") == ""
    assert normalize_result_field("未公開") == ""
    assert normalize_result_field("") == ""
    assert normalize_result_field(None) == ""
    assert (
        normalize_result_field("Target=Player 1 Result=Human")
        == "Target=Player 1 Result=Human"
    )

    parsed = parse_discussion_output(
        "[CO] None\n[RESULT] None\n[BODY] Player 4さんのCOを踏まえて、他に質問したい。"
    )
    assert parsed is not None and parsed.result == ""
    parsed_empty = parse_discussion_output("[CO] None\n[RESULT]\n[BODY] 確認します。")
    assert parsed_empty is not None and parsed_empty.result == ""
    claim = parse_discussion_claim("[CO] None\n[RESULT] なし")
    assert claim is not None and claim.result == ""


def test_seer_result_without_claim_not_triggered_for_none_result() -> None:
    state, seer = _seer_state()
    actions = OllamaActions(state, ScriptedClient([""]), seed=1001)
    response = "[CO] None\n[RESULT] None\n[BODY] Player 4さんのCOを軸に、他に質問したい。"
    assert actions._discussion_response_issue(seer, response) is None


def test_japanese_player_aliases_are_extracted_as_seer_results() -> None:
    assert extract_public_seer_result(
        "対象名=プレイヤー5 結果=Human", PLAYER_NAMES
    ) == ("Player 5", "human")
    assert extract_public_seer_result(
        "占い対象: Player 5 / 判定: 人間", PLAYER_NAMES
    ) == ("Player 5", "human")
    assert extract_public_seer_result("P5は人間と判定された", PLAYER_NAMES) == (
        "Player 5",
        "human",
    )


def test_human_only_seer_can_hold_with_none_claim() -> None:
    state, seer = _seer_state()
    actions = OllamaActions(state, ScriptedClient([""]), seed=1001)
    assert (
        actions._holds_human_seer_result(
            "discussion", seer, "[CO] None\n[BODY] 他の候補を検討します。"
        )
        is True
    )
    assert (
        actions._holds_human_seer_result(
            "discussion", seer, "[CO] Hold\n[BODY] 保留します。"
        )
        is True
    )
    state.seer_results = {"Player 1": "wolf"}
    assert (
        actions._holds_human_seer_result(
            "discussion", seer, "[CO] None\n[BODY] 検討します。"
        )
        is False
    )

def test_seer_claim_prefix_repair_keeps_japanese_body() -> None:
    state, seer = _seer_state()
    actions = OllamaActions(state, ScriptedClient([""]), seed=1001)
    response = (
        "[CO] None\n[RESULT] 対象名=プレイヤー1 結果=人間\n"
        "[BODY] Player 1さんのCOと結果の矛盾を精査したい。"
    )
    repaired = actions._seer_claim_prefix_repair(seer, response)
    assert repaired is not None
    text, target, result = repaired
    assert target == "Player 1" and result == "human"
    assert text.startswith("[CO] Seer CO")
    assert "対象=Player 1" in text and "判定=Human" in text
    assert "Player 1さんのCOと結果の矛盾を精査したい。" in text
    assert "Suspicion toward" not in text
    assert actions._discussion_response_issue(seer, text) is None

    # CO行だけでRESULT行のない応答も実判定で決定補完する（捏造しない）。
    claim_only = actions._seer_claim_prefix_repair(seer, "[CO] Seer")
    assert claim_only is not None
    claim_text, claim_target, claim_result = claim_only
    assert claim_target == "Player 1" and claim_result == "human"
    assert "[BODY]" in claim_text

    # 台帳と一致しない対象は補完しない（再試行側の判断に委ねる）。
    hallucinated = actions._seer_claim_prefix_repair(
        seer, "[CO] None\n[RESULT] 対象名=プレイヤー5 結果=人間\n[BODY] 検討します。"
    )
    assert hallucinated is None


def test_seer_fallback_is_localized_and_preserves_body() -> None:
    state, seer = _seer_state()
    actions = OllamaActions(state, ScriptedClient([""]), seed=1001)
    fallback = actions._seer_info_fallback(seer, "Player 6さんの真偽を先に確認したい。")
    assert fallback is not None
    assert "[CO] Seer" in fallback
    assert "[RESULT] Target=Player 1 Result=Human" in fallback
    assert "Player 6さんの真偽を先に確認したい。" in fallback
    assert "Suspicion toward Player 1" not in fallback

    generic = actions._seer_info_fallback(seer, None)
    assert generic is not None and "占い結果" in generic

    english_state, english_seer = _seer_state("English")
    english_actions = OllamaActions(english_state, ScriptedClient([""]), seed=1001)
    english = english_actions._seer_info_fallback(english_seer, None)
    assert english is not None and "divination result" in english


def test_holdable_speech_avoids_fallback_end_to_end() -> None:
    state, seer = _seer_state()
    state.conversation_log.append(Message(speaker="system", content="Day 1 begins"))
    scripted = (
        "[CO] None\n[RESULT] None\n"
        "[BODY] Player 4さんのCOを踏まえて、投票方針を整理したい。"
    )
    actions = OllamaActions(
        state,
        ScriptedClient([scripted, scripted]),
        seed=1001,
        options={"question_hint_mode": "none"},
        discussion_claim_mode="legacy",
    )
    alive = tuple(player.name for player in state.players if player.is_alive)
    body = actions.discuss(seer.name, 1, alive, tuple(state.conversation_log))
    assert isinstance(body, str) and body
    assert actions.traces[-1].fallback_used is False


def test_published_verdict_without_claim_is_repaired_end_to_end() -> None:
    state, seer = _seer_state()
    state.conversation_log.append(Message(speaker="system", content="Day 1 begins"))
    scripted = (
        "[CO] None\n[RESULT] 対象名=プレイヤー1 結果=人間\n"
        "[BODY] Player 4さんのCOを踏まえて、投票方針を整理したい。"
    )
    actions = OllamaActions(
        state,
        ScriptedClient([scripted, scripted]),
        seed=1001,
        options={"question_hint_mode": "none"},
        discussion_claim_mode="legacy",
    )
    alive = tuple(player.name for player in state.players if player.is_alive)
    body = actions.discuss(seer.name, 1, alive, tuple(state.conversation_log))
    assert "[CO] Seer CO" in body
    assert "対象=Player 1" in body and "判定=Human" in body
    assert actions.traces[-1].fallback_used is False


def test_madman_human_verdict_is_common_knowledge() -> None:
    assert "a Madman is divined as Human" in RULE_CARD
    assert "狂人の占い結果は「人間」になる" in RULE_CARD
    assert "Villager OR a Madman" in WEREWOLF_THEORY_CARD
    assert "never reports a Madman as Werewolf" in WEREWOLF_THEORY_CARD
    for role in (Role.SEER, Role.MADMAN, Role.WOLF, Role.VILLAGER):
        assert "Madman" in ROLE_DESCRIPTIONS_EN[role]
    assert "divined as 'Human'" in ROLE_DESCRIPTIONS_EN[Role.SEER]
    assert "always 'Human'" in ROLE_DESCRIPTIONS_EN[Role.MADMAN]

    state, seer = _seer_state()
    state.seer_claims.append(
        PublicSeerClaim(
            day=1, claimant=seer.name, target="Player 1", result="human"
        )
    )
    ledger = PromptBuilder._discussion_claim_ledger(state)
    assert "Interpretation" in ledger
    assert "狂人の占い結果は「人間」になる" in ledger

    private_block = PromptBuilder._private_seer_block(
        actor=seer,
        current_seer_results=["Player 1: Human"],
        state=state,
        day=1,
        pending_wolf_results=[],
        human_results_can_be_held=True,
        include_claim=True,
    )
    assert "狂人は占い結果が「人間」になる" in private_block
    assert "Madman" in private_block

