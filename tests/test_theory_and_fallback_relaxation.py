"""TASK-042の回帰テスト（LL不使用）: 初日単独COの真置き、人狼相談の視点、フォールバック緩和。

実機ログ game_0055be638e37 では
- 初日の単独占い師COを「未確認」という理由だけで疑う動きがあった
- 人狼相談が村人の視点（「村の秩序」「村人としての動きが安定」）で書かれた
- 切り詰め（done_reason=length）だけで投票・議論がフォールバックになっていた
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.llm_actions import OllamaActions
from werewolf.models import Message, PublicRoleClaim, Role
from werewolf.ollama import ChatResponse
from werewolf.prompts import (
    ROLE_DESCRIPTIONS_EN,
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

    def __init__(self, responses: list[ChatResponse] | None = None) -> None:
        self.responses = list(responses or [])

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {"ollama_ps_context": None}

    def _next(self) -> ChatResponse:
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


def test_theory_card_states_day1_solo_co_genuine_placement() -> None:
    assert "Day 1" in WEREWOLF_THEORY_CARD
    assert "真置き" in WEREWOLF_THEORY_CARD
    assert "the longer that lone-CO state persists" in WEREWOLF_THEORY_CARD
    # 疑う理由として正当な不整合は残す（真確定扱いにはしない）
    assert "contradicting verdicts on the same target" in WEREWOLF_THEORY_CARD
    assert "definitive proof" in WEREWOLF_THEORY_CARD


def test_claimant_guidance_scales_with_solo_co_persistence() -> None:
    state = _state()
    alive = _alive(state)
    seer_claim = PublicRoleClaim(day=1, claimant="Player 4", claimed_role="seer")
    state.public_role_claims = [seer_claim]
    state.seer_claims = []

    # Day 2: 1日目COが対抗COなしで1日続いている → 真置きを明示
    state.discussion_turn = 1
    header_day2 = PromptBuilder._fact_header(
        _player(state, "Player 2"), alive, 2, "Day (Discussion)", state
    )
    assert "真置き" in header_day2
    assert "stood alone since Day 1" in header_day2

    # Day 1 巡2: 1巡終了・対抗なし → 真寄り
    state.discussion_turn = 2
    header_day1_round2 = PromptBuilder._fact_header(
        _player(state, "Player 2"), alive, 1, "Day (Discussion)", state
    )
    assert "first discussion round finished" in header_day1_round2

    # Day 1 巡1: まだ疑う根拠が薄い
    state.discussion_turn = 1
    header_day1_turn1 = PromptBuilder._fact_header(
        _player(state, "Player 2"), alive, 1, "Day (Discussion)", state
    )
    assert "little to doubt" in header_day1_turn1


def test_wolf_consult_prompt_uses_wolf_camp_perspective() -> None:
    state = _state()
    wolves = [player for player in state.players if player.role is Role.WOLF]
    wolf, partner = wolves[0], wolves[1]
    prompt_ctx = PromptBuilder.build_wolf_discuss_prompt(
        actor=wolf,
        partner_name=partner.name,
        alive_players=tuple(state.alive_players),
        wolf_chat_history=(),
        day=2,
        state=state,
    )
    assert "[Consult Perspective (internal to your camp)]" in prompt_ctx.user_prompt
    assert "村人陣営" in prompt_ctx.user_prompt
    assert "village order" in prompt_ctx.user_prompt
    assert "村の秩序" in prompt_ctx.user_prompt
    assert "Perspective Split" in ROLE_DESCRIPTIONS_EN[Role.WOLF]


def test_co_decision_prompt_encourages_lone_seer_co() -> None:
    state = _state()
    seer = next(player for player in state.players if player.role is Role.SEER)
    alive = _alive(state)
    state.seer_results = {"Player 1": "human"}
    prompt_ctx = PromptBuilder.build_discussion_claim_prompt(
        actor=seer, alive_players=alive, day=1, state=state
    )
    assert "CO now beats silence" in prompt_ctx.user_prompt
    assert "真置き" in prompt_ctx.user_prompt

    # 対抗COがある場合は旧来の「保留可」の guidance を使う
    rival = next(name for name in alive if name != seer.name)
    state.public_role_claims = [
        PublicRoleClaim(day=1, claimant=rival, claimed_role="seer")
    ]
    rival_prompt = PromptBuilder.build_discussion_claim_prompt(
        actor=seer, alive_players=alive, day=2, state=state
    )
    assert "CO now beats silence" not in rival_prompt.user_prompt


def test_target_selection_prompts_put_target_line_first() -> None:
    state = _state()
    actor = _player(state, "Player 2")
    alive = tuple(state.alive_players)
    vote_ctx = PromptBuilder.build_vote_prompt(
        actor=actor,
        alive_players=alive,
        valid_candidates=tuple(name for name in alive if name != actor.name),
        day=1,
        state=state,
    )
    assert "Required format (write this line first)" in vote_ctx.user_prompt
    assert "FIRST line" in vote_ctx.user_prompt

    seer = next(player for player in state.players if player.role is Role.SEER)
    seer_ctx = PromptBuilder.build_seer_prompt(
        actor=seer,
        alive_players=alive,
        seer_results={},
        day=1,
        state=state,
    )
    assert "Required format (write this line first)" in seer_ctx.user_prompt


def test_truncated_vote_with_parseable_target_is_accepted() -> None:
    state = _state()
    actions = OllamaActions(
        state,
        ScriptedClient(
            [
                ChatResponse(
                    content="Vote: Player 3\n（理由が途中で切れた）",
                    attempts=1,
                    done_reason="length",
                )
            ]
        ),
        seed=1001,
    )
    actor = _player(state, "Player 2")
    candidates = tuple(name for name in state.alive_players if name != actor.name)
    target = actions.vote(actor.name, candidates, 1, False)
    assert target == "Player 3"
    assert actions.traces[-1].fallback_used is False


def test_truncated_discussion_body_is_accepted_without_fallback() -> None:
    state = _state()
    state.conversation_log.append(Message(speaker="system", content="Day 1 begins."))
    scripted = ChatResponse(
        content="[CO] None\n[BODY] Player 3さんの投票先と根拠を確認したい。",
        attempts=1,
        done_reason="length",
    )
    actions = OllamaActions(
        state,
        ScriptedClient([scripted]),
        seed=1001,
        options={"question_hint_mode": "none"},
        discussion_claim_mode="legacy",
    )
    body = actions.discuss("Player 2", 1, _alive(state), tuple(state.conversation_log))
    assert actions.traces[-1].fallback_used is False
    assert "Player 3さんの投票先" in body


def test_truncated_output_usability_rules() -> None:
    actions = OllamaActions(_state(), ScriptedClient(), seed=1001)
    assert actions._truncated_output_usable("[CO] None\n[BODY] 確認します。", "discussion")
    assert not actions._truncated_output_usable("[CO] None\n[BODY]", "discussion")
    assert actions._truncated_output_usable("[CO] Seer CO", "discussion_claim")
    assert actions._truncated_output_usable(
        "今夜の攻撃対象を比較した。Player 3が候補だ。", "wolf_discussion"
    )
    assert not actions._truncated_output_usable(
        "今夜の攻撃対象を比較した。Player 3が", "wolf_discussion"
    )


def test_unknown_player_reference_is_repaired_not_regenerated() -> None:
    state = _state()
    actions = OllamaActions(state, ScriptedClient(), seed=1001)
    actor = _player(state, "Player 2")
    response = (
        "[CO] なし\n[BODY] Player 9さんの投票先を聞きたい。"
        "Player 3さんの占いの方向性を確認したい。"
    )
    assert actions._discussion_response_issue(actor, response) == "unknown_player_reference"
    repaired = actions._repair_discussion_response(
        actor, response, "unknown_player_reference", tuple(state.conversation_log)
    )
    assert repaired is not None
    assert "Player 9" not in repaired
    assert "Player 3さんの占いの方向性を確認したい。" in repaired
    assert actions._discussion_response_issue(actor, repaired) is None


def test_unverified_attributed_quote_is_repaired_not_regenerated() -> None:
    state = _state()
    state.conversation_log.append(
        Message(speaker="Player 3", content="今日はPlayer 2を確認したい。")
    )
    actions = OllamaActions(state, ScriptedClient(), seed=1001)
    actor = _player(state, "Player 2")
    response = (
        "[CO] なし\n[BODY] Player 3さんは「私は占い師です」と言っていた。"
        "Player 3さんの今日の発言の根拠を確認したい。"
    )
    assert actions._discussion_response_issue(actor, response) == "unverified_attributed_speech"
    repaired = actions._repair_discussion_response(
        actor, response, "unverified_attributed_speech", tuple(state.conversation_log)
    )
    assert repaired is not None
    assert "私は占い師です" not in repaired
    assert actions._discussion_response_issue(actor, repaired) is None


def test_warning_consistency_findings_do_not_force_retry() -> None:
    state = _state()
    state.conversation_log.append(Message(speaker="system", content="Day 1 begins."))
    state.conversation_log.append(
        Message(
            speaker="Player 3",
            content="Player 2さんのDay1の投票の理由を教えてください。",
        )
    )
    actor = _player(state, "Player 2")
    body = "[CO] None\n[BODY] Player 4さんのCOを踏まえて、他に質問したい。"

    # none/softではwarning（質問未回答）を再生成の材料にしない
    none_actions = OllamaActions(
        state, ScriptedClient(), seed=1001, options={"question_hint_mode": "none"}
    )
    assert none_actions._discussion_response_issue(actor, body) is None

    soft_actions = OllamaActions(
        state, ScriptedClient(), seed=1001, options={"question_hint_mode": "soft"}
    )
    assert soft_actions._discussion_response_issue(actor, body) is None

    # strictは形式として回答の先頭配置を要求する（従来動作）
    strict_actions = OllamaActions(
        state, ScriptedClient(), seed=1001, options={"question_hint_mode": "strict"}
    )
    assert (
        strict_actions._discussion_response_issue(actor, body)
        == "question_not_answered_first"
    )

    # repair級（記録にない引用）はモードに関係なく検出される
    quoted_body = (
        "[CO] None\n[BODY] Player 3さんは「私は占い師です」と言っていた。"
        "Player 3さんの今日の発言の根拠を確認したい。"
    )
    assert (
        none_actions._discussion_response_issue(actor, quoted_body)
        == "unverified_attributed_speech"
    )


def test_fallback_keeps_written_body_instead_of_pass() -> None:
    state = _state()
    actions = OllamaActions(state, ScriptedClient(), seed=1001)
    actor = _player(state, "Player 2")
    preserved = actions._preserved_fallback_response(
        "discussion",
        actor,
        "[CO] Villager CO\n[BODY] Player 3さんの投票意思と根拠を確認したい。",
        (),
    )
    assert preserved is not None
    assert "[CO]" in preserved
    assert "Player 3さんの投票意思" in preserved

    wolf = next(player for player in state.players if player.role is Role.WOLF)
    leaked = actions._preserved_fallback_response(
        "discussion",
        wolf,
        "[CO] なし\n[BODY] 俺の相方の人狼Player 6さんと今夜の襲撃を相談した。",
        (),
    )
    assert leaked is None


def test_empty_responses_fallback_to_content_bearing_sentence() -> None:
    state = _state()
    state.conversation_log.append(Message(speaker="system", content="Day 1 begins."))
    actions = OllamaActions(
        state,
        ScriptedClient(
            [ChatResponse(content="", attempts=1), ChatResponse(content="", attempts=1)]
        ),
        seed=1001,
        options={"question_hint_mode": "none"},
        discussion_claim_mode="legacy",
    )
    body = actions.discuss("Player 2", 1, _alive(state), tuple(state.conversation_log))
    assert actions.traces[-1].fallback_used is True
    assert body != "Pass"
    assert "確認したい" in body


def test_wolf_consult_fallback_is_localized() -> None:
    state = _state()
    actions = OllamaActions(state, ScriptedClient(), seed=1001)
    wolf = next(player for player in state.players if player.role is Role.WOLF)
    text = actions._consult_fallback_text(wolf)
    assert "Attack target:" in text
    assert "could not respond" not in text
    assert "強い根拠" in text

    # 相談本文が検証落ちしても、メタ発話のない日本語の相談文は保持する
    preserved = actions._preserved_fallback_response(
        "wolf_discussion",
        wolf,
        "今夜はPlayer 5を攻撃したい。Player 6はCO baitとして残す。",
    )
    assert preserved is not None
    assert "Player 5を攻撃したい" in preserved
