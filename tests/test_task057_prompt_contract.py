"""TASK-057 の回帰テスト（LL不使用）: プロンプト契約の簡素化と情報ブロック間の重複削減。

実ログ game_XXXX では Player 4 の初回議論で user prompt 18,109字 / 推定 15,630 tokens に達し、
FactLedger・ClaimLedger・推理メモ・仮説台帳・論点台帳・質問・自己行動履歴・出力形式が同時に並んで
同じ事実や制約を別の表現で繰り返していた。本テストは以下を固定する（実機 Ollama は使わない）。

1. セクション台帳は core / optional を分け、purpose・source・authority・expires_at を持つ。
2. 同じ発言本文は [CONVERSATION_EVIDENCE] の1箇所だけに出し、他の節は `[ref:sN]` 参照にできる
   （履歴から落ちた本文を指す空参照は作らない）。
3. optional 節は段階的にだけ落ち、core（確定事実・現在の質問・自分の行動記録・発言本文）は残る。
4. [BEHAVIOR_CONTRACT] が強い制約を1度だけ書き、旧来プロンプトで重複していた命令文を減らす。
   口調・文量・論点・結論の自由は自由行として明示し、推順は指示しない。
5. 契約行の静的検査で duplicate_rule / obligation_vs_freedom を検出でき、条件付き義務は矛盾に数えない。
6. `legacy` では旧挙動（重複した命令文と本文引用のまま）で、A/B比較の基準側になる。
"""
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.models import GameState, PlayerActionMemory
from werewolf.prompt_contract import (
    ALLOWED_DUPLICATE_RULES,
    AUTHORITY_TEMPORARY,
    CORE_NEVER_DROPPED,
    DIMENSION_EXPRESSION,
    DIMENSION_FORMAT,
    OPTIONAL_REMOVAL_MAX_STAGE,
    OPTIONAL_REMOVAL_STAGES,
    PROMPT_CONTRACT_POLICY_DEFAULT,
    PROMPT_CONTRACT_POLICY_LEGACY,
    PROMPT_CONTRACT_POLICY_SIMPLIFIED,
    PROMPT_SECTIONS,
    BehaviorContractLine,
    ContractDiagnostics,
    PromptSectionMeta,
    behavior_contract_lines,
    disallowed_duplicate_count,
    drop_sections,
    duplicated_constraints,
    evidence_entries_to_references,
    evidence_entry_reference,
    evidence_reference,
    expired_sections_present,
    expires_label,
    find_sections,
    guidance_expired,
    inspect_behavior_contract,
    measure_section_chars,
    normalize_prompt_contract_policy,
    optional_names_dropped_up_to,
    prompt_contract_policy_label,
    quote_or_reference,
    render_behavior_contract,
    scan_prompt_constraints,
    section_meta,
    tier_char_totals,
    tier_of,
)
from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game


def _state(seed: int = 1001, response_language: str = "日本語") -> GameState:
    return create_game(
        seed=seed,
        config_path="config/players.json",
        response_language=response_language,
    )


def _player(state: GameState, name: str):
    return next(player for player in state.players if player.name == name)


def _names(state: GameState) -> tuple[str, ...]:
    return tuple(player.name for player in state.players)


def _start_day1(state: GameState, turn: int = 2) -> None:
    from werewolf.models import Message, Phase

    state.public_timeline.record_day_start(state.day)
    state.conversation_log.append(Message("system", f"Day {state.day} begins."))
    state.phase = Phase.DAY_DISCUSSION
    state.discussion_turn = turn
    state.begin_discussion_turn()


def _say(state: GameState, name: str, content: str):
    from werewolf.models import Message

    return state.append_discussion_message(Message(name, content))


def _discussion_ctx(state: GameState, actor_name: str, **kwargs):
    actor = _player(state, actor_name)
    params = {
        "day": 1,
        "state": state,
        "include_claim": False,
        "context_strategy": "full_transcript",
        "question_hint_mode": "none",
    }
    params.update(kwargs)
    return PromptBuilder.build_discussion_prompt(
        actor,
        tuple(state.alive_players),
        tuple(state.conversation_log),
        **params,
    )


# ---------------------------------------------------------------------------
# 1. ポリシー解決（既定は簡素化、legacy はA/B比較用の旧挙動）
# ---------------------------------------------------------------------------
def test_policy_default_is_simplified_and_labels_are_normalized() -> None:
    assert PROMPT_CONTRACT_POLICY_DEFAULT == PROMPT_CONTRACT_POLICY_SIMPLIFIED
    assert normalize_prompt_contract_policy("legacy") == PROMPT_CONTRACT_POLICY_LEGACY
    assert normalize_prompt_contract_policy("OFF") == PROMPT_CONTRACT_POLICY_LEGACY
    assert normalize_prompt_contract_policy(False) == PROMPT_CONTRACT_POLICY_LEGACY
    assert normalize_prompt_contract_policy(None) == PROMPT_CONTRACT_POLICY_SIMPLIFIED
    assert normalize_prompt_contract_policy("unknown-value") == (
        PROMPT_CONTRACT_POLICY_SIMPLIFIED
    )
    assert prompt_contract_policy_label("off") == PROMPT_CONTRACT_POLICY_LEGACY




# ---------------------------------------------------------------------------
# 2. セクション台帳（purpose / source / authority / expires_at）と tier
# ---------------------------------------------------------------------------
def test_registry_has_metadata_and_unique_markers() -> None:
    names = [section.name for section in PROMPT_SECTIONS]
    assert len(names) == len(set(names))
    seen_markers: dict[str, str] = {}
    for section in PROMPT_SECTIONS:
        assert section.tier in {"core", "optional"}
        assert section.purpose and section.source and section.authority
        for marker in section.markers:
            # 同じマーカーを2つの節で使うと、位置計測と除去範囲がずれるため禁止
            assert marker not in seen_markers, f"{marker} shared by {names}"
            seen_markers[marker] = section.name
    # core に分類される必須ブロック（一貫性のanchor）
    assert CORE_NEVER_DROPPED
    for name in (
        "fact_ledger",
        "claim_ledger",
        "self_action_memory",
        "current_task",
        "conversation_evidence",
        "question_block",
    ):
        assert tier_of(name) == "core"


def test_temporary_guidance_carries_expires_at() -> None:
    meta = section_meta("day1_first_turn_guidance")
    assert meta.tier == "optional"
    assert meta.authority == AUTHORITY_TEMPORARY
    assert meta.expires_at == "D1T1"
    assert guidance_expired(meta.expires_at, 1, 1) is False
    assert guidance_expired(meta.expires_at, 1, 2) is True
    assert guidance_expired(meta.expires_at, 2, 1) is True
    assert guidance_expired(None, 9, 9) is False
    assert expires_label(2, 3) == "D2T3"
    assert expires_label(1, 0, prefix="N") == "N1"


# ---------------------------------------------------------------------------
# 3. optional 節の段階除去（core は絶対に削らない）
# ---------------------------------------------------------------------------
def _toy_prompt() -> str:
    return (
        "[SYSTEM_RULES]\nrules body\n"
        "[FACT_LEDGER]\nconfirmed: Player 1 alive\n"
        "[PREVIOUS_TOPICS]\ntopics digest\n"
        "[HYPOTHESIS_LEDGER]\nhypotheses\n"
        "[CONTINUITY_MEMO\nderived memo\n"
        "[INTEGRITY_ALERTS]\nalerts json\n"
        "[ACTOR_PRIVATE_MEMO\nprivate facts\n"
        "[CONVERSATION_EVIDENCE\n[D1 T2 s7 Player 2] \"speech body\"\n"
        "[CURRENT_TASK]\ndo it\n"
        "[Instructions]\nformat here\n"
    )


def test_drop_sections_removes_only_optional_and_keeps_core() -> None:
    text = _toy_prompt()
    pruned, removed, saved = drop_sections(text, ["previous_topics", "hypothesis_ledger"])
    assert removed == ("hypothesis_ledger", "previous_topics")
    assert saved > 0
    assert "[PREVIOUS_TOPICS]" not in pruned
    assert "[HYPOTHESIS_LEDGER]" not in pruned
    # core は1文字も減らない（ここが「一貫性を保つ」側の保証）
    for marker in (
        "[SYSTEM_RULES]",
        "[FACT_LEDGER]",
        "confirmed: Player 1 alive",
        "[ACTOR_PRIVATE_MEMO",
        "[CONVERSATION_EVIDENCE",
        "\"speech body\"",
        "[CURRENT_TASK]",
        "[Instructions]",
    ):
        assert marker in pruned
    # 入れ子見出しをまたがない（integrity alert は private memo を巻き込まない）
    assert "private facts" in pruned


def test_drop_sections_refuses_core_and_unknown_names() -> None:
    text = _toy_prompt()
    for name in CORE_NEVER_DROPPED[:3]:
        try:
            drop_sections(text, [name])
        except ValueError as error:
            assert "core sections are never dropped" in str(error)
        else:  # pragma: no cover - 設計上の禁止が破られたら失敗させる
            raise AssertionError(f"{name} must never be dropped")
    try:
        drop_sections(text, ["no_such_section"])
    except ValueError as error:
        assert "unknown prompt section names" in str(error)
    else:  # pragma: no cover
        raise AssertionError("unknown section name must raise")


def test_removal_stages_are_progressive_and_optional_only() -> None:
    assert optional_names_dropped_up_to(0) == ()
    stage1 = optional_names_dropped_up_to(1)
    assert "previous_topics" in stage1 and "reasoning_guidance" in stage1
    assert tier_of(stage1[0]) == "optional"
    cumulative: list[str] = []
    for stage in range(1, OPTIONAL_REMOVAL_MAX_STAGE + 1):
        names = optional_names_dropped_up_to(stage)
        assert len(names) > len(cumulative)
        for name in names:
            assert name not in CORE_NEVER_DROPPED
        cumulative = list(names)
    assert len(OPTIONAL_REMOVAL_STAGES) == OPTIONAL_REMOVAL_MAX_STAGE
    assert optional_names_dropped_up_to(99) == tuple(cumulative)




# ---------------------------------------------------------------------------
# 4. 証拠ID参照（同じ発言本文を2度送らない）
# ---------------------------------------------------------------------------
def test_quote_or_reference_needs_the_id_to_be_in_evidence() -> None:
    long_body = json.dumps(
        "これは実際の発言本文です。投票先を確認したいと考えていて、占い師の判定も見ておきたいと思っています。",
        ensure_ascii=False,
    )
    referenced, ok, saved = quote_or_reference(long_body, 12, frozenset({12}))
    assert ok is True
    assert referenced.startswith("[ref:s12]")
    assert "実際の発言本文" not in referenced
    assert saved == len(long_body) - len(referenced) > 0
    # IDが無い・履歴に載っていない場合は本文を残す（空参照を作らない）
    same, ok_missing, _ = quote_or_reference(long_body, None, frozenset({12}))
    assert (ok_missing, same) == (False, long_body)
    absent, ok_absent, _ = quote_or_reference(long_body, 99, frozenset({12}))
    assert (ok_absent, absent) == (False, long_body)
    assert evidence_reference(7) == "[ref:s7]"


def test_short_bodies_stay_as_plain_text() -> None:
    body = '"短"'
    kept, referenced, saved = quote_or_reference(body, 3, frozenset({3}))
    assert referenced is False
    assert saved == 0
    assert kept == body


# ---------------------------------------------------------------------------
# 5. behavior_contract（統合と静的検査）
# ---------------------------------------------------------------------------
def test_contract_lines_cover_each_strong_rule_once() -> None:
    lines = behavior_contract_lines(
        language="日本語", hint_mode="soft", include_claim=True
    )
    rule_ids = [line.rule_id for line in lines]
    assert len(rule_ids) == len(set(rule_ids))
    findings = inspect_behavior_contract(lines)
    kinds = {finding["kind"] for finding in findings}
    assert "duplicate_rule" not in kinds
    assert "obligation_vs_freedom" not in kinds
    # 強い制約（秘匿・形式）と自由行がそろう
    strengths = {line.strength for line in lines}
    assert "strong" in strengths and "freedom" in strengths
    # 口調・論点を決める文言は入らない（自由度を宣言する側）
    joined = "\n".join(line.text for line in lines)
    assert "- Free:" not in joined  # 日本語版なので英語ラベルは出ない
    assert "自由" in joined and "口調" in joined
    assert "3-5 sentences" not in joined


def test_contract_language_variants_and_optional_lines() -> None:
    japanese = behavior_contract_lines(language="日本語", hint_mode="strict")
    english = behavior_contract_lines(language="English", hint_mode="soft")
    assert any("強い契約" in line.text for line in japanese)
    assert any(line.text.startswith("- Strong:") for line in english)
    # strict では「直接問われたときだけ」と条件つきの義務が増える（無条件の自由とは衝突しない）
    strict_line = next(
        line for line in japanese if line.rule_id == "answer_when_asked"
    )
    assert strict_line.condition == "directly_addressed_and_strict_mode"
    findings = inspect_behavior_contract(japanese)
    assert not [f for f in findings if f["kind"] == "obligation_vs_freedom"]
    # CO依頼では「CO行と本文を混ぜない」が1行だけ足される
    claim_lines = behavior_contract_lines(
        language="English", hint_mode="soft", include_claim=True
    )
    body_free = behavior_contract_lines(
        language="English", hint_mode="soft", include_claim=False
    )
    assert "claim_separate_from_body" in {line.rule_id for line in claim_lines}
    assert "claim_separate_from_body" not in {line.rule_id for line in body_free}


def test_inspection_flags_duplicate_and_conflict() -> None:
    duplicated = [
        BehaviorContractLine("no_private_leak", "a", "strong", "secrecy"),
        BehaviorContractLine("no_private_leak", "b", "strong", "secrecy"),
        # 同じ次元に無条件の義務と無条件の自由を並べると矛盾として拾われる
        BehaviorContractLine("always_open_with_vote", "x", "strong", DIMENSION_EXPRESSION),
        BehaviorContractLine("expression_free", "free tone", "freedom", DIMENSION_EXPRESSION),
    ]
    findings = inspect_behavior_contract(duplicated)
    kinds = {finding["kind"] for finding in findings}
    assert "duplicate_rule" in kinds
    assert "obligation_vs_freedom" in kinds
    # 条件つき義務なら矛盾ではなく conditional_pair として扱う
    conditioned = [
        BehaviorContractLine(
            "answer_when_asked",
            "answer first when asked",
            "strong",
            DIMENSION_EXPRESSION,
            condition="directly_addressed_and_strict_mode",
        ),
        BehaviorContractLine("expression_free", "free tone", "freedom", DIMENSION_EXPRESSION),
    ]
    kinds2 = {finding["kind"] for finding in inspect_behavior_contract(conditioned)}
    assert "conditional_pair" in kinds2
    assert "obligation_vs_freedom" not in kinds2
    # 自由行のない制約だけのブロックは freedom_without_anchor として拾う
    only_strong = [
        BehaviorContractLine("body_not_empty", "x", "strong", DIMENSION_FORMAT),
    ]
    assert {f["kind"] for f in inspect_behavior_contract(only_strong)} == {
        "freedom_without_anchor"
    }




# ---------------------------------------------------------------------------
# 6. プロンプト組み立て（simplified / legacy の実測比較）
# ---------------------------------------------------------------------------
LONG_SPEECH = (
    "{name}の{index}番目の発言です。{rival}の占い師COと投票の動きを比べると、"
    "昨夜の襲撃先との整合が怪しく見えるので、今日の投票ではその点を確認したいと考えています。"
)


def _day1_with_speeches(seed: int = 1001) -> GameState:
    state = _state(seed=seed)
    _start_day1(state, turn=2)
    names = _names(state)
    for index, name in enumerate(names[:4]):
        rival = names[(index + 1) % len(names)]
        _say(
            state,
            name,
            LONG_SPEECH.format(name=name, index=index + 1, rival=rival),
        )
    return state


def test_simplified_prompt_keeps_each_speech_body_in_one_place() -> None:
    state = _day1_with_speeches()
    actor_name = _names(state)[4]
    # 直前の発言者がこの発言者へ質問を向ける（引用が複数の節に現れる状況を作る）
    asker = _names(state)[3]
    _say(
        state,
        asker,
        f"{actor_name}さんに聞きます。{asker}が先に述べた投票案と比べて、"
        f"{actor_name}はどう考えますか？根拠を具体的に教えてください。",
    )
    simplified = _discussion_ctx(
        state,
        actor_name,
        prompt_contract_policy="simplified",
        question_hint_mode="strict",
    )
    legacy = _discussion_ctx(
        state,
        actor_name,
        prompt_contract_policy="legacy",
        question_hint_mode="strict",
    )
    raw_body = state.conversation_log[-1].content
    assert simplified.user_prompt.count(raw_body) == 1  # [CONVERSATION_EVIDENCE] の1箇所だけ
    assert "[ref:s" in simplified.user_prompt
    assert legacy.user_prompt.count(raw_body) >= 3  # 履歴＋現在の質問＋直接回答欄に再掲
    assert "[BEHAVIOR_CONTRACT" in simplified.user_prompt
    assert "[BEHAVIOR_CONTRACT" not in legacy.user_prompt
    assert simplified.diagnostics is not None and legacy.diagnostics is not None
    assert int(simplified.diagnostics["dedup_reference_count"]) >= 2
    assert int(simplified.diagnostics["dedup_saved_chars"]) > 0
    assert len(simplified.user_prompt) < len(legacy.user_prompt)


def test_own_speeches_become_id_references_under_simplified() -> None:
    state = _day1_with_speeches()
    actor_name = _names(state)[1]  # すでに今日1発言している（本文は長文）
    simplified = _discussion_ctx(state, actor_name, prompt_contract_policy="simplified")
    legacy = _discussion_ctx(state, actor_name, prompt_contract_policy="legacy")
    own_body = next(
        message.content for message in state.conversation_log if message.speaker == actor_name
    )
    # 本人の発言は履歴に1回載っていれば足りる（simplifiedでは [Your Speeches Today] に再掲しない）
    assert simplified.user_prompt.count(own_body) == 1
    assert "full text in [CONVERSATION_EVIDENCE]" in simplified.user_prompt
    assert legacy.user_prompt.count(own_body) == 2  # 履歴と本人欄の2箇所に本文がある
    assert "[ref:s" in simplified.user_prompt
    assert simplified.diagnostics is not None
    assert int(simplified.diagnostics["dedup_saved_chars"]) > 0


def test_duplicate_constraints_fall_and_core_survives() -> None:
    state = _day1_with_speeches()
    actor_name = _names(state)[4]
    kwargs = {"include_claim": True, "question_hint_mode": "strict"}
    simplified = _discussion_ctx(
        state, actor_name, prompt_contract_policy="simplified", **kwargs
    )
    legacy = _discussion_ctx(state, actor_name, prompt_contract_policy="legacy", **kwargs)
    counts_simple = scan_prompt_constraints(simplified.user_prompt)
    counts_legacy = scan_prompt_constraints(legacy.user_prompt)
    assert disallowed_duplicate_count(counts_simple) < disallowed_duplicate_count(
        counts_legacy
    )
    # reasoning_not_in_body は契約節で1度だけ（旧来は案内文と出力形式で2回書かれていた）
    assert duplicated_constraints(counts_legacy).get("reasoning_not_in_body", 0) >= 2
    assert counts_simple.get("reasoning_not_in_body", 0) <= 1
    assert "Do NOT put reasoning in [BODY]" not in simplified.user_prompt
    # core の証拠・事実・自分の行動記録はどちらの設定でも同じように残る
    for marker in (
        "[FACT_LEDGER]",
        "[CLAIM_LEDGER]",
        "[SELF_ACTION_MEMORY",
        "[CONVERSATION_EVIDENCE",
    ):
        assert marker in simplified.user_prompt and marker in legacy.user_prompt
    # 議論を促す案内は残す（合理的な議論のため。口調や結論の強制はしない）
    assert "Form hypotheses" in simplified.user_prompt
    assert "- Even with limited info" in simplified.user_prompt


def test_derived_evidence_entries_become_id_references() -> None:
    entry = (
        "[speech:7] Player 2: "
        "Player 2の発言です。占い師COの理由と投票の動きが噛み合っていないので確認したいです。"
    )
    replaced, changed, saved = evidence_entry_reference(entry, frozenset({7}))
    assert changed is True
    assert replaced == "[speech:7] Player 2 [ref:s7]"
    assert saved > 0
    # IDが履歴に無いものは触らない（空参照を作らない）
    same, absent, _ = evidence_entry_reference(entry, frozenset({8}))
    assert (absent, same) == (False, entry)
    tiny, tiny_changed, _ = evidence_entry_reference("[speech:9] P: ok", frozenset({9}))
    assert tiny_changed is False
    other = "[speech:8] Player 3: 履歴から落ちた発言の本文なので参照にはできないはずです。"
    converted, diagnostics = evidence_entries_to_references([entry, other], frozenset({7}))
    assert converted[0] == replaced and converted[1] == other
    assert diagnostics == {"dedup_saved_chars": saved, "dedup_references": 1}


def test_every_speech_body_appears_only_once_under_simplified() -> None:
    state = _day1_with_speeches()
    actor_name = _names(state)[4]
    kwargs = {"include_claim": True, "question_hint_mode": "strict"}
    simplified = _discussion_ctx(
        state, actor_name, prompt_contract_policy="simplified", **kwargs
    )
    legacy = _discussion_ctx(state, actor_name, prompt_contract_policy="legacy", **kwargs)
    speeches = [
        message.content
        for message in state.conversation_log
        if message.speaker != "system" and len(message.content) > 30
    ]
    assert speeches
    for content in speeches:
        # 同じ本文は [CONVERSATION_EVIDENCE] の1箇所だけ（旧来は履歴＋推理メモ＋本人欄の2〜3箇所）
        assert simplified.user_prompt.count(content) == 1
    assert any(legacy.user_prompt.count(content) > 1 for content in speeches)
    # 推理メモは本文をやめ、IDと発話者名だけで「どの発言を見たか」を覚える（追跡性は維持）
    memo_line = next(
        line
        for line in simplified.user_prompt.splitlines()
        if "current_round_evidence" in line
    )
    assert "[ref:s" in memo_line
    assert "考えていま" not in memo_line


def test_own_action_history_uses_id_references_only_for_listed_bodies() -> None:
    state = _state()
    actor = _player(state, "Player 3")
    body = (
        "Player 3の発言です。投票の流れを確認したいので、先に占いの対象を明らかにしてほしいです。"
    )
    state.player_action_memories[actor.name] = [
        PlayerActionMemory(
            day=1,
            phase="day_discussion",
            action_type="vote",
            vote_target="Player 4",
            reasoning_summary="COの理由が噛み合っていない",
        ),
        PlayerActionMemory(
            day=1,
            phase="day_discussion",
            action_type="discussion",
            public_statement_summary=body,
        ),
    ]
    simplified_text = PromptBuilder._self_action_memory_text(
        actor,
        state,
        public_phase=True,
        body_reference_map={body: 12},
        evidence_speech_ids=frozenset({12}),
        use_id_references=True,
    )
    legacy_text = PromptBuilder._self_action_memory_text(actor, state, public_phase=True)
    # 証拠に残っている自分の発言だけ本文をやめ、ID参照にする（投票行はエンジン記録で不変）
    assert "[ref:s12]" in simplified_text
    assert body not in simplified_text
    assert body in legacy_text
    assert "- Day 1 Vote: for Player 4" in legacy_text
    assert "- Day 1 Vote: for Player 4" in simplified_text
    # 履歴から落ちた発言（IDが証拠に無い）は要約を残す＝空参照を作らない
    dangling = PromptBuilder._self_action_memory_text(
        actor,
        state,
        public_phase=True,
        body_reference_map={body: 99},
        evidence_speech_ids=frozenset({12}),
        use_id_references=True,
    )
    assert body in dangling
    tally: list[int] = []
    counted = PromptBuilder._self_action_memory_text(
        actor,
        state,
        public_phase=True,
        body_reference_map={body: 12},
        evidence_speech_ids=frozenset({12}),
        use_id_references=True,
        dedup_tally=tally,
    )
    assert len(tally) == 1 and tally[0] > 0 and "[ref:s12]" in counted


def test_contract_diagnostics_are_log_ready() -> None:
    state = _day1_with_speeches()
    actor_name = _names(state)[4]
    ctx = _discussion_ctx(state, actor_name, prompt_contract_policy="simplified")
    diagnostics = ctx.diagnostics
    assert diagnostics is not None
    json.dumps(diagnostics)  # JSONLへそのまま載せられること
    assert diagnostics["prompt_contract_policy"] == PROMPT_CONTRACT_POLICY_SIMPLIFIED
    assert diagnostics["prompt_user_char_count"] == len(ctx.user_prompt)
    assert diagnostics["prompt_core_chars"] > 0
    assert (
        diagnostics["prompt_core_chars"] + diagnostics["prompt_optional_chars"]
        <= len(ctx.user_prompt)
    )
    section_chars = diagnostics["prompt_section_chars"]
    assert isinstance(section_chars, dict) and section_chars
    assert section_chars.get("conversation_evidence", 0) > 0
    assert isinstance(diagnostics["prompt_duplicate_constraint_rules"], list)
    assert diagnostics["contract_removal_stage"] == 0
    assert diagnostics["dropped_optional_sections"] == []
    assert tier_char_totals(ctx.user_prompt)["core"] == diagnostics["prompt_core_chars"]
    measured = measure_section_chars(ctx.user_prompt)
    assert set(measured).issubset({section.name for section in PROMPT_SECTIONS})


def test_optional_sections_can_be_dropped_by_the_builder() -> None:
    state = _day1_with_speeches()
    actor_name = _names(state)[4]
    base = _discussion_ctx(state, actor_name, prompt_contract_policy="simplified")
    requested = optional_names_dropped_up_to(OPTIONAL_REMOVAL_MAX_STAGE)
    stage = _discussion_ctx(
        state,
        actor_name,
        prompt_contract_policy="simplified",
        optional_sections_dropped=requested,
    )
    diagnostics = stage.diagnostics
    assert diagnostics is not None
    dropped = list(diagnostics["dropped_optional_sections"])
    assert dropped  # 実際に存在した optional 節だけが落ちる
    assert set(dropped).issubset(set(requested))
    assert not [name for name in dropped if name in CORE_NEVER_DROPPED]
    assert int(diagnostics["optional_removed_chars"]) > 0
    # 落ちた名前のマーカーは本文から消えている（残っている名前は存在しなかっただけ）
    markers = {
        "previous_topics": "[PREVIOUS_TOPICS]",
        "hypothesis_ledger": "[HYPOTHESIS_LEDGER]",
        "reasoning_guidance": "[REASONING_MEMO_GUIDANCE]",
        "inference_line": "- Player Speech & Inferences (Unverified)",
        "integrity_alerts": "[INTEGRITY_ALERTS]",
        "continuity_memo": "[CONTINUITY_MEMO",
    }
    for name in dropped:
        marker = markers.get(name)
        if marker is not None:
            assert marker not in stage.user_prompt
    assert len(stage.user_prompt) < len(base.user_prompt)
    # core はどの段階でも無傷（事実・質問・時系列・出力形式・発言本文は残り続ける）
    for marker in (
        "[FACT_LEDGER]",
        "[CLAIM_LEDGER]",
        "[CURRENT_TASK]",
        "[Instructions]",
        "[DAY_TIMELINE]",
        "[CONVERSATION_EVIDENCE",
        "[BEHAVIOR_CONTRACT",
    ):
        assert marker in stage.user_prompt


def test_legacy_policy_reproduces_the_old_duplicate_shape() -> None:
    state = _day1_with_speeches()
    actor_name = _names(state)[4]
    legacy = _discussion_ctx(state, actor_name, prompt_contract_policy="legacy")
    simplified = _discussion_ctx(state, actor_name, prompt_contract_policy="simplified")
    # legacy では旧来の禁止文がそのまま残る（A/B比較の基準側）
    assert "- Do not describe yourself as a third person" in legacy.user_prompt
    assert "- Use only information presented in this game" in legacy.user_prompt
    assert "- Do not describe yourself as a third person" not in simplified.user_prompt
    assert (legacy.diagnostics or {})["prompt_contract_policy"] == "legacy"
    assert (simplified.diagnostics or {})["prompt_contract_policy"] == "simplified"


def test_contract_diagnostics_apply_to_adds_tier_measurements() -> None:
    text = _toy_prompt()
    payload = ContractDiagnostics(
        policy="legacy",
        stage=2,
        dropped=("previous_topics",),
        dedup_saved_chars=42,
        dedup_references=1,
    ).apply_to(text)
    assert payload["prompt_contract_policy"] == PROMPT_CONTRACT_POLICY_LEGACY
    assert payload["contract_removal_stage"] == 2
    assert payload["dropped_optional_sections"] == ["previous_topics"]
    assert payload["dedup_saved_chars"] == 42
    assert payload["dedup_reference_count"] == 1
    assert payload["prompt_user_char_count"] == len(text)
    assert payload["prompt_core_chars"] > 0 and payload["prompt_optional_chars"] > 0


class FakeContractClient:
    """プレイヤー設定のみを予算解決に使うモック（実Ollamaへは一切送らない）。"""

    def __init__(self) -> None:
        self.timeout = 120.0
        self.max_retries = 0
        self.timeout_retries = 0

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {}


def test_overflow_triggers_staged_optional_removal_without_touching_core() -> None:
    from werewolf.llm_actions import OllamaActions

    state = _day1_with_speeches()
    state.prompt_contract_policy = PROMPT_CONTRACT_POLICY_SIMPLIFIED
    actor = _player(state, _names(state)[4])
    actor.num_ctx = 3072  # 割り当てを意図的に小さくして overflow を作る
    actions = OllamaActions(state, FakeContractClient(), seed=1001)
    ctx = actions._build_discussion_prompt_with_budget(
        actor, tuple(state.alive_players), tuple(state.conversation_log), 1
    )
    diagnostics = ctx.diagnostics
    assert diagnostics is not None
    stage = int(diagnostics["contract_removal_stage"])
    assert stage >= 1
    dropped = list(diagnostics.get("dropped_optional_sections") or [])
    assert dropped and not [name for name in dropped if name in CORE_NEVER_DROPPED]
    # 落とされたのは参考情報のブロックだけ（事実・質問・自分の記録・発言本文は残る）
    for marker in (
        "[FACT_LEDGER]",
        "[CLAIM_LEDGER]",
        "[SELF_ACTION_MEMORY",
        "[CONVERSATION_EVIDENCE",
        "[BEHAVIOR_CONTRACT",
        "[Instructions]",
    ):
        assert marker in ctx.user_prompt
    # legacy では段階除去が走らない（比較基準側は旧来のまま）
    state.prompt_contract_policy = PROMPT_CONTRACT_POLICY_LEGACY
    legacy_ctx = actions._build_discussion_prompt_with_budget(
        actor, tuple(state.alive_players), tuple(state.conversation_log), 1
    )
    assert int((legacy_ctx.diagnostics or {})["contract_removal_stage"]) == 0
    assert int((legacy_ctx.diagnostics or {})["contract_stage_attempts"]) == 0


def test_no_body_is_printed_twice_for_any_actor() -> None:
    """simplifiedでは全生存者のプロンプトで同じ発言本文が2度以上載らない（TASK-057 完了条件）。"""
    state = _day1_with_speeches()
    asker, target = _names(state)[3], _names(state)[4]
    _say(
        state,
        asker,
        f"{target}さんに聞きます。{asker}が先に述べた投票案と比べて、"
        f"{target}はどう考えますか？根拠を挙げてください。",
    )
    bodies = {
        message.content
        for message in state.conversation_log
        if message.speaker != "system" and len(message.content) >= 24
    }
    seen_reference = False
    for actor_name in _names(state):
        ctx = _discussion_ctx(state, actor_name, prompt_contract_policy="simplified")
        for body in bodies:
            assert ctx.user_prompt.count(body) <= 1, f"{actor_name}: {body[:24]} が重複"
        if "[ref:s" in ctx.user_prompt:
            seen_reference = True
    assert seen_reference, "参照化された箇所が1つもない（節約が効いていない）"
    # legacy（A/B比較の基準側）では同じ本文が2箇所に現れる
    legacy_ctx = _discussion_ctx(
        state, _names(state)[2], prompt_contract_policy="legacy"
    )
    assert [body for body in bodies if legacy_ctx.user_prompt.count(body) > 1]


def test_inference_line_uses_id_reference_for_copied_speech() -> None:
    """台帳へ写し込まれた発言本文（推測行）も、証拠に在るあいだは ID参照に縮める。"""
    from werewolf.models import LedgerFact

    state = _state()
    _start_day1(state, turn=2)
    speaker, listener = _names(state)[0], _names(state)[1]
    _say(state, speaker, LONG_SPEECH.format(name=speaker, index=1, rival=listener))
    copied = next(m for m in state.conversation_log if m.speaker == speaker)
    speech_id = int(copied.speech_id)
    state.fact_ledger.model_inferences.append(
        LedgerFact(
            day=1,
            kind="unverified_model_statement",
            description=copied.content,
            actor=speaker,
        )
    )
    own_reasoning = "自分の推測: まだ誰のCOも今夜の襲撃先を説明できない"
    state.fact_ledger.model_inferences.append(
        LedgerFact(
            day=1,
            kind="unverified_model_statement",
            description=own_reasoning,
            actor=listener,
        )
    )
    text, dedup = PromptBuilder._inference_ledger_text(
        state,
        _player(state, listener),
        body_reference_map={copied.content.strip(): speech_id},
        evidence_speech_ids=frozenset({speech_id}),
    )
    assert f"[ref:s{speech_id}]" in text
    assert copied.content not in text
    assert own_reasoning in text  # 発言そのものを写していない行はそのまま残る
    assert dedup["dedup_references"] == 1 and dedup["dedup_saved_chars"] > 0
    legacy_text, legacy_dedup = PromptBuilder._inference_ledger_text(
        state, _player(state, listener)
    )
    assert copied.content in legacy_text
    assert "[ref:s" not in legacy_text
    assert int(legacy_dedup.get("dedup_references") or 0) == 0
    assert int(legacy_dedup.get("dedup_saved_chars") or 0) == 0


def test_stage_removal_does_not_rebuild_for_absent_sections() -> None:
    """段階除去は実在する optional 節ぶんだけしか組み直さない（ローカルの構築コスト抑制）。"""
    from werewolf.llm_actions import OllamaActions

    state = _day1_with_speeches()
    state.prompt_contract_policy = PROMPT_CONTRACT_POLICY_SIMPLIFIED
    actor = _player(state, _names(state)[4])
    actions = OllamaActions(state, FakeContractClient(), seed=1001)
    # 割り当てに余裕があるなら組み直し自体を行わない（無予算＋発言履歴予算の2回まで）
    actor.num_ctx = 32768
    roomy = actions._build_discussion_prompt_with_budget(
        actor, tuple(state.alive_players), tuple(state.conversation_log), 1
    )
    roomy_diag = roomy.diagnostics or {}
    assert int(roomy_diag["contract_stage_attempts"]) == 0
    assert int(roomy_diag["contract_removal_stage"]) == 0
    assert int(roomy_diag["prompt_build_calls"]) <= 2
    assert int(roomy_diag["optional_removed_chars"]) == 0
    # 窮屈な割り当てでも、組み直しは段階上限ぶんまで（存在しない節のために再構築しない）
    actor.num_ctx = 2560
    tight = actions._build_discussion_prompt_with_budget(
        actor, tuple(state.alive_players), tuple(state.conversation_log), 1
    )
    tight_diag = tight.diagnostics or {}
    assert int(tight_diag["contract_removal_stage"]) >= 1
    assert int(tight_diag["prompt_build_calls"]) <= 2 + OPTIONAL_REMOVAL_MAX_STAGE
    # core はどの段階でも無傷（自分の行動記録・発言本文・確定事実は落ちない）
    for marker in (
        "[FACT_LEDGER]",
        "[SELF_ACTION_MEMORY",
        "[CONVERSATION_EVIDENCE",
        "[BEHAVIOR_CONTRACT",
        "[Instructions]",
    ):
        assert marker in tight.user_prompt
