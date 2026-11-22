"""TASK-047の回帰テスト（LL不使用）: 確定事実台帳と未確定解釈台帳の分離。

最新ゲーム分析では Day 1 で3人が同じ Player 3 を占い、Day 2 で死亡した Player 4 へ
「本人に答えさせる」流れが複数回発生した（`past_action_question_unanswered` /
`dead_player_question`）。原因は発言の自由度ではなく、事実・主張・質問の状態がモデルへ
簡潔に見えていなかったこと。本テストは以下を固定する（実機LLMは使わない）。

1. 仮説（未確定解釈）は `HypothesisLedger` に `evidence_ids` 付きで分離され、確定事実
   （FactLedger）として再掲されない。
2. 死亡者への質問は本文で検出し、死亡者を対象とする仮説は historical（質問対象から除外）
   になり、死亡者宛の未回答質問は memo に残らない。
3. 未確定仮説を確定事実として断定した発言は `warning` のみ（修復・再生成しない）。
4. 同じ証拠の重複提示を削減しつつ、CO・投票・占い結果はIDで追跡できる。
5. 仮説欄は数値スコアを参考情報として示すだけで、結論・口調・論点を強制しない。
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.engine import UNVERIFIED_HYPOTHESIS_AS_FACT, diagnose_discussion_body
from werewolf.hypothesis_ledger import (
    REFUTATION_CONTESTED,
    REFUTATION_HISTORICAL,
    REFUTATION_SUPERSEDED,
    STANCE_SUSPECT,
    STANCE_TRUST,
    HypothesisLedger,
)
from werewolf.models import LedgerFact, Message, PublicRoleClaim, PublicSeerClaim, Role
from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game


def _state(response_language: str = "日本語"):
    return create_game(
        seed=1001, config_path="config/players.json", response_language=response_language
    )


def _player(state, name: str):
    return next(player for player in state.players if player.name == name)


def _alive(state) -> tuple[str, ...]:
    return tuple(player.name for player in state.players if player.is_alive)


def _day1(state) -> tuple[Message, ...]:
    state.conversation_log.append(Message(speaker="system", content="Day 1 begins."))
    return tuple(state.conversation_log)


def test_hypotheses_are_separated_from_confirmed_facts() -> None:
    state = _state()
    _day1(state)
    speech = "Player 3さんの投票先が疑わしい。人狼の可能性が高いと思います。"
    state.conversation_log.append(Message(speaker="Player 2", content=speech))
    created = state.record_hypotheses_from_speech("Player 2", speech, day=1, turn=1)
    assert created == 1
    record = state.hypothesis_ledger.records["H-Player 2-Player 3-suspect"]
    assert record.stance == STANCE_SUSPECT
    assert record.generated_by == "Player 2"
    assert record.subject == "Player 3"
    assert record.evidence_ids == ["[speech:1]"]
    assert record.day == 1 and record.turn == 1
    # 確定事実台帳には仮説を混ぜない（発言全文が FactLedger の確定事実として追加されない）
    assert not any(
        "Player 3さんの投票先" in fact.description
        for fact in state.fact_ledger.confirmed_facts
    )
    assert state.fact_ledger.confirmed_facts == []
    ledger_text = PromptBuilder._discussion_claim_ledger(state)
    assert "Player 3さんの投票先" not in ledger_text


def test_same_evidence_is_merged_not_recreated() -> None:
    ledger = HypothesisLedger()
    ledger.record(
        subject="Player 3",
        actor="Player 2",
        day=1,
        turn=1,
        stance=STANCE_SUSPECT,
        statement="Player 3の投票先が疑わしい",
        evidence_ids=["[speech:1]"],
    )
    ledger.record(
        subject="Player 3",
        actor="Player 2",
        day=1,
        turn=2,
        stance=STANCE_SUSPECT,
        statement="Player 3の投票先が疑わしい",
        evidence_ids=["[speech:4]"],
    )
    assert len(ledger.records) == 1
    assert ledger.duplicates_suppressed == 1
    assert ledger.records["H-Player 2-Player 3-suspect"].evidence_ids == [
        "[speech:1]",
        "[speech:4]",
    ]
    # 立場の変更は旧仮説を superseded にし、新仮説を別レコードとして立てる
    ledger.record(
        subject="Player 3",
        actor="Player 2",
        day=2,
        turn=1,
        stance=STANCE_TRUST,
        statement="Player 3は人間判定があるので信頼する",
        evidence_ids=["[seer_claim:1]"],
    )
    assert ledger.records["H-Player 2-Player 3-suspect"].refutation_status == (
        REFUTATION_SUPERSEDED
    )
    assert ledger.records["H-Player 2-Player 3-trust"].refutation_status != (
        REFUTATION_SUPERSEDED
    )
    assert len(ledger.records) == 2
    assert ledger.stance_changes == 1
    # superseded（立場変更で置き換えられた旧仮説）は本人にも提示しない
    block, diagnostics = ledger.render_for("Player 2")
    assert int(diagnostics["hypothesis_records_rendered"]) == 1
    assert "信頼する" in block
    assert "投票先が疑わしい" not in block


def test_dead_player_question_is_detected_and_hypothesis_is_historical() -> None:
    state = _state()
    _day1(state)
    question = "Player 4さん、なぜPlayer 3に投票したのですか？理由を教えてください。"
    state.conversation_log.append(Message(speaker="Player 2", content=question))
    state.record_hypotheses_from_speech("Player 2", question, day=1, turn=1)
    state.mark_dead("Player 4")
    state.resolve_hypothesis_ledger()
    record = state.hypothesis_ledger.records["H-Player 2-Player 4-suspect"]
    assert record.refutation_status == REFUTATION_HISTORICAL
    assert record.actionable is False
    assert state.hypothesis_ledger.metrics()["hypothesis_dead_references_suppressed"] >= 1
    # 死亡者宛の未回答質問は memo に残らない（回答不能な未解決項目を提示しない）
    state.refresh_reasoning_memos()
    assert state.reasoning_memos["Player 4"].unresolved_questions == []
    # 死亡者への言及を含む未回答質問は「回答不能」注記付きで渡す（質問誘導を誘導しない）
    state.conversation_log.append(
        Message(speaker="Player 2", content="Player 3さんに質問です。Player 4さんはなぜ投票したのですか？")
    )
    state.refresh_reasoning_memos()
    assert any(
        "dead players cannot answer" in entry
        for entry in state.reasoning_memos["Player 3"].unresolved_questions
    )
    # 生存者（Player 2）の本文が死亡者へ質問していれば検出する（修復対象）
    diagnosis = diagnose_discussion_body(
        "Player 4さん、なぜPlayer 3に投票したのですか？理由を聞きたい。",
        _player(state, "Player 2"),
        state,
        previous_message=None,
        round_transcript=(),
    )
    assert "dead_player_question" in diagnosis.reasons
    assert diagnosis.repairable is True
    # 死亡者対象の仮説は「過去記録」としてのみ提示され、質問誘導を作らない
    block, diagnostics = state.render_hypotheses_for("Player 2")
    assert "Past record of dead subjects" in block
    assert diagnostics["hypothesis_historical_dead_subjects"] >= 1
    assert "next_targets_to_check" not in block


def test_published_verdict_contests_opposing_hypothesis_without_repair() -> None:
    state = _state()
    _day1(state)
    state.public_role_claims.append(
        PublicRoleClaim(day=1, claimant="Player 1", claimed_role=Role.SEER.value)
    )
    state.seer_claims.append(
        PublicSeerClaim(day=1, claimant="Player 1", target="Player 3", result="human")
    )
    speech = "Player 3さんは人狼だと思います。投票先として疑っています。"
    state.conversation_log.append(Message(speaker="Player 2", content=speech))
    state.record_hypotheses_from_speech("Player 2", speech, day=1, turn=1)
    state.resolve_hypothesis_ledger()
    record = state.hypothesis_ledger.records["H-Player 2-Player 3-suspect"]
    assert record.refutation_status == REFUTATION_CONTESTED
    assert record.counter_evidence_ids == ["[seer_claim:1]"]
    block, diagnostics = state.render_hypotheses_for("Player 2")
    assert "Unresolved contradictions" in block
    assert diagnostics["hypothesis_contradictions"] >= 1
    # 矛盾は警告として渡すだけで、本文の自動修正・再生成は行わない
    diagnosis = diagnose_discussion_body(
        speech, _player(state, "Player 2"), state, previous_message=None, round_transcript=()
    )
    assert diagnosis.fatal is False
    assert diagnosis.repairable is False
    assert diagnosis.publishable is True


def test_unverified_hypothesis_asserted_as_fact_is_warning_only() -> None:
    state = _state()
    _day1(state)
    actor = _player(state, "Player 2")
    diagnosis = diagnose_discussion_body(
        "Player 3さんは確定人狼です。投票を集中させましょう。",
        actor,
        state,
        previous_message=None,
        round_transcript=(),
    )
    assert UNVERIFIED_HYPOTHESIS_AS_FACT in diagnosis.reasons
    assert diagnosis.warning is True
    assert diagnosis.repairable is False
    assert diagnosis.fatal is False
    assert diagnosis.publishable is True  # 本文はそのまま公開（自由な断定を修復しない）
    # 台帳の確定事実が裏付けになっている断定は指摘しない
    state.fact_ledger.confirmed_facts.append(
        LedgerFact(
            day=1,
            kind="execution",
            description="Player 3 was executed as a werewolf.",
            visibility="public",
        )
    )
    supported = diagnose_discussion_body(
        "Player 3さんは確定人狼です。",
        actor,
        state,
        previous_message=None,
        round_transcript=(),
    )
    assert UNVERIFIED_HYPOTHESIS_AS_FACT not in supported.reasons



def test_duplicate_evidence_suppressed_while_traceable() -> None:
    state = _state()
    _day1(state)
    # 確定事実（投票・死亡）と公開主張（CO・占い結果）を台帳へ登録する
    state.vote_history.append((1, False, {"Player 2": "Player 3", "Player 5": "Player 3"}))
    state.fact_ledger.confirmed_facts.append(
        LedgerFact(day=1, kind="attack_death", description="Player 4 was attacked.")
    )
    state.attack_deaths.append((1, "Player 4"))
    state.public_role_claims.append(
        PublicRoleClaim(day=1, claimant="Player 1", claimed_role=Role.SEER.value)
    )
    state.seer_claims.append(
        PublicSeerClaim(day=1, claimant="Player 1", target="Player 3", result="human")
    )
    # 同じ占い結果をCO行でも持つ主張（旧実装では2回重複提示されていた）
    state.public_role_claims.append(
        PublicRoleClaim(
            day=1,
            claimant="Player 1",
            claimed_role=Role.SEER.value,
            target="Player 3",
            result="human",
        )
    )
    actor = _player(state, "Player 2")
    long_statements = [
        LedgerFact(
            day=1,
            kind="unverified_model_statement",
            description="Player 3さんの投票先とCOの整合性を確認したい。" * 8,
            actor="Player 2",
        )
        for _ in range(6)
    ]
    state.fact_ledger.model_inferences.extend(long_statements)
    raw_inference_chars = sum(len(fact.description) for fact in long_statements)
    speech = "Player 3さんの投票先が疑わしい。"
    state.conversation_log.append(Message(speaker="Player 2", content=speech))
    state.record_hypotheses_from_speech("Player 2", speech, day=1, turn=1)
    state.refresh_reasoning_memos()
    context = PromptBuilder.build_discussion_prompt(
        actor,
        _alive(state),
        tuple(state.conversation_log),
        day=1,
        state=state,
        include_claim=True,
        context_strategy="hybrid_retrieval",
        question_hint_mode="soft",
    )
    diagnostics = context.diagnostics
    # 重複提示の削減：発言全文の再掲が仮説要約へ置き換わる
    assert int(diagnostics["inference_ledger_chars"]) < raw_inference_chars
    assert int(diagnostics["hypothesis_section_chars"]) < raw_inference_chars
    assert "[HYPOTHESIS_LEDGER]" in context.user_prompt
    # 追跡性：CO・投票・占い結果・死亡がすべて証拠IDで読める
    for marker in (
        "[claim:1]",
        "[seer_claim:1]",
        "[vote:1:initial:Player 2]",
        "[death:attack:1:Player 4]",
    ):
        assert marker in context.user_prompt
    # memo は本文を再掲せずIDだけ持つ（本文は CLAIM_LEDGER に同じIDで1回存在する）
    memo = state.reasoning_memos["Player 2"]
    assert all(entry.startswith("[") for entry in memo.confirmed_facts)
    assert all(entry.startswith("[") for entry in memo.public_claims)
    assert "[vote:1:initial:Player 2]" in memo.confirmed_facts
    # CO行の判定重複は統合される（同じ占い結果が本文に2回出ない）
    claim_lines = [
        line
        for line in context.user_prompt.splitlines()
        if line.startswith("- Public Claim [claim:")
    ]
    assert len(claim_lines) == 2
    assert sum("Player 3=Human" in line for line in claim_lines) == 0
    # TASK-055: 公開された占い結果行は全員一律「そのプレイヤー自身の申告」として表示される
    assert "Day 1: Player 1 states that Player 3 is Human" in context.user_prompt
    assert "(claimed by this player only)" in context.user_prompt


def test_hypothesis_block_does_not_constrain_conclusion_or_tone() -> None:
    state = _state()
    _day1(state)
    speech = "Player 3さんは人狼だと思います。"
    state.conversation_log.append(Message(speaker="Player 2", content=speech))
    state.record_hypotheses_from_speech("Player 2", speech, day=1, turn=1)
    confident = "Player 4さんは人狼だと確信しています。"
    state.conversation_log.append(Message(speaker="Player 2", content=confident))
    state.record_hypotheses_from_speech("Player 2", confident, day=1, turn=2)
    state.resolve_hypothesis_ledger()
    block, _ = state.render_hypotheses_for("Player 2")
    # 参考情報の明記と、結論・口調の強制文の不存在
    assert "reference hint only" in block
    for forbidden in ("必ず", "must", "禁止", "do not say", "回答しなければならない"):
        assert forbidden not in block
    # 数値スコアは参考値としてのみ表示する（断定度を押し付けない）
    assert "confidence=medium" in block
    assert "reference=0.50" in block
    assert "confidence=high" in block
    assert "reference=0.80" in block


def test_changed_since_last_turn_is_delta_only() -> None:
    state = _state()
    _day1(state)
    speech = "Player 3さんは人狼だと思います。"
    state.conversation_log.append(Message(speaker="Player 2", content=speech))
    state.record_hypotheses_from_speech("Player 2", speech, day=1, turn=1)
    state.resolve_hypothesis_ledger()
    _, first = state.render_hypotheses_for("Player 2")
    assert int(first["hypothesis_records_changed_since_last_turn"]) == 1
    state.snapshot_hypothesis_presentation("Player 2")
    _, second = state.render_hypotheses_for("Player 2")
    assert int(second["hypothesis_records_changed_since_last_turn"]) == 0
    # 新事実（占い結果の公開）で反証状態が変われば差分として提示される
    state.seer_claims.append(
        PublicSeerClaim(day=1, claimant="Player 1", target="Player 3", result="human")
    )
    state.resolve_hypothesis_ledger()
    block, third = state.render_hypotheses_for("Player 2")
    assert int(third["hypothesis_records_changed_since_last_turn"]) == 1
    assert "Changed since your previous turn" in block
    assert block.count("Player 3さん") >= 1


def test_private_memo_hypotheses_are_visible_only_to_their_owner() -> None:
    state = _state()
    _day1(state)
    memo = "Player 6さんを疑っています。占い結果をまだ公開しません。"
    state.record_hypotheses_from_reasoning_memo("Player 2", memo, day=1, turn=1)
    own_block, own_diag = state.render_hypotheses_for("Player 2")
    other_block, other_diag = state.render_hypotheses_for("Player 3")
    assert "Your own hypotheses" in own_block
    assert "Player 6" in own_block
    assert "Player 6" not in other_block
    assert int(own_diag["hypothesis_records_own"]) >= 1
    assert int(other_diag["hypothesis_records_own"]) == 0
    assert state.hypothesis_ledger.serialize_public()["records"] == []

