"""TASK-066の回帰テスト（実機LL不使用）: 再生成コンテキストの段階化と再生成節順の凍結。

実機ログ `game_87ff647746e6` では、形式修正・空応答・生成上限という「プロンプトが原因でない失敗」に
対しても、再生成1回目から発言履歴を core-ledger まで畳む短縮プロンプトへ切り替わっていた
（13,819字→2,745字）。モデルは直前までの議論・引用・反論・自分の立場を読めないまま再生成する。
本テストは決定的に次を検証する（モックのみ・実機Ollamaは使わない）。

- 方策は `tiered`（既定）／`legacy_early_compression`（旧挙動＝A/B基準）へ正規化する
- 圧縮は 同一証拠保持 → 派生節のみ → 連続失敗で履歴を畳む、の順にしか進まない
- `light_compressed` で落ちるのは低優先の派生節だけ（現在質問・確定事実台帳・役職主張・
  投票／死亡履歴・直近発言・反証・[CONTINUITY_CHECKPOINT] はどの段階でも残る）
- 再生成プロンプトの節順はハード制約を満たし、貪欲探索の最良順と一致する
- 診断ログには数値・時点・ID・節名だけを載せ、発言本文そのものは載せない
- 口調・文量・論点・結論・推理順を指定する新しい命令文を増やさない
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.retry_context_policy import (
    COMPRESSION_STAGES,
    FAILURE_CONTEXT_OVERFLOW,
    FAILURE_EMPTY_OR_THINKING_ONLY,
    FAILURE_GENERATION_CAP,
    FAILURE_OUTPUT_FORMAT,
    FAILURE_QUESTION_ANSWER,
    FAILURE_SEMANTIC_LIGHT,
    FAILURE_TIMEOUT,
    FAILURE_UNSPECIFIED,
    HISTORY_FOLDING_STAGES,
    RETRY_CONTEXT_POLICIES,
    RETRY_CONTEXT_POLICY_DEFAULT,
    RETRY_CONTEXT_POLICY_LEGACY_EARLY,
    RETRY_CONTEXT_POLICY_TIERED,
    RETRY_STAGES,
    RetryStageDecision,
    TRIGGER_FORMAT_REPAIR,
    TRIGGER_TIMEOUT_REPEAT,
    decide_retry_stage,
    dropped_evidence_id_safety,
    evidence_ids_in_text,
    normalize_retry_context_policy,
    retry_context_policy_label,
    retry_failure_class,
    retry_trigger_is_immediate_compression,
    retry_trigger_is_timeout,
    same_failure_run,
    stage_compresses,
    stage_folds_speech_history,
)

# 口調・文量・論点・結論・推理順を新しく縛る文言（新設してはいけないもの）
BANNED_FREEDOM_PHRASES = (
    "conclude that",
    "you must believe",
    "argue that",
    "in this reasoning order",
    "polite tone",
    "be concise and agreeable",
    "必ず次の順で",
    "こう結論せよ",
)


def test_policy_normalization_keeps_tiered_as_default() -> None:
    assert RETRY_CONTEXT_POLICY_DEFAULT == RETRY_CONTEXT_POLICY_TIERED
    assert normalize_retry_context_policy(None) == RETRY_CONTEXT_POLICY_TIERED
    assert normalize_retry_context_policy("TIERED") == RETRY_CONTEXT_POLICY_TIERED
    assert normalize_retry_context_policy("legacy") == RETRY_CONTEXT_POLICY_LEGACY_EARLY
    assert (
        normalize_retry_context_policy("legacy_early_compression")
        == RETRY_CONTEXT_POLICY_LEGACY_EARLY
    )
    # 未知値は「圧縮を進めない」側（既定）へ倒す
    assert normalize_retry_context_policy("unexpected") == RETRY_CONTEXT_POLICY_DEFAULT
    assert normalize_retry_context_policy(True) == RETRY_CONTEXT_POLICY_TIERED
    assert retry_context_policy_label("legacy") == RETRY_CONTEXT_POLICY_LEGACY_EARLY
    assert set(RETRY_CONTEXT_POLICIES) == {
        RETRY_CONTEXT_POLICY_TIERED,
        RETRY_CONTEXT_POLICY_LEGACY_EARLY,
    }


def test_failure_class_maps_issues_to_separate_sides() -> None:
    assert retry_failure_class("discussion_output_format") == FAILURE_OUTPUT_FORMAT
    assert retry_failure_class("template_placeholder_left") == FAILURE_OUTPUT_FORMAT
    assert retry_failure_class("question_not_answered_first") == FAILURE_QUESTION_ANSWER
    assert retry_failure_class("repeated_discussion_topic") == FAILURE_SEMANTIC_LIGHT
    assert (
        retry_failure_class("thinking_only_response") == FAILURE_EMPTY_OR_THINKING_ONLY
    )
    assert retry_failure_class("generation_truncated") == FAILURE_GENERATION_CAP
    assert retry_failure_class("timeout") == FAILURE_TIMEOUT
    assert retry_failure_class(None) == FAILURE_UNSPECIFIED
    # サーバーのコンテキスト超過は、他の名前より先に overflow へ寄せる
    assert (
        retry_failure_class("empty_content", context_overflow=True)
        == FAILURE_CONTEXT_OVERFLOW
    )


def test_stage_predicates_separate_compression_from_folding() -> None:
    assert stage_compresses("light_compressed") is True
    assert stage_compresses("last_resort_compressed") is True
    assert stage_compresses("full_preserved") is False
    assert stage_folds_speech_history("last_resort_compressed") is True
    assert stage_folds_speech_history("light_compressed") is False
    assert set(COMPRESSION_STAGES) == {"light_compressed", "last_resort_compressed"}
    assert set(HISTORY_FOLDING_STAGES) == {"last_resort_compressed"}
    assert RETRY_STAGES[0] == "normal_generation"


def test_same_failure_run_counts_only_the_trailing_streak() -> None:
    fmt = RetryStageDecision(
        stage="full_preserved",
        compression_trigger=TRIGGER_FORMAT_REPAIR,
        failure_class=FAILURE_OUTPUT_FORMAT,
        request_number=2,
    )
    timeout = RetryStageDecision(
        stage="last_resort_compressed",
        compression_trigger=TRIGGER_TIMEOUT_REPEAT,
        failure_class=FAILURE_TIMEOUT,
        request_number=3,
    )
    # 同じ条件が続いた回数だけを数える（今回の失敗を含めた個数で返す）
    assert same_failure_run((), FAILURE_OUTPUT_FORMAT) == 1
    assert same_failure_run((fmt,), FAILURE_OUTPUT_FORMAT) == 2
    assert same_failure_run((fmt, fmt), FAILURE_OUTPUT_FORMAT) == 3
    # 直前が別種の失敗なら連続失敗は途切れる（＝圧縮を進めない）
    assert same_failure_run((fmt, timeout), FAILURE_OUTPUT_FORMAT) == 1
    assert same_failure_run((fmt, timeout), FAILURE_TIMEOUT) == 2


from werewolf.retry_context_policy import (  # noqa: E402
    LAST_RESORT_AFTER_SAME_FAILURE_ATTEMPTS,
    LIGHT_STAGE_AFTER_SAME_FAILURE_ATTEMPTS,
    TRIGGER_BUDGET_ADJUSTMENT,
    TRIGGER_BUDGET_RETRY_REPEAT,
    TRIGGER_CONTEXT_OVERFLOW,
    TRIGGER_EFFECTIVE_CONTEXT_SHORTAGE,
    TRIGGER_POLICY_LEGACY_EARLY,
    TRIGGER_QUESTION_ANSWER_REPAIR,
    TRIGGER_SAME_FAILURE_SECOND,
    TRIGGER_SAME_FAILURE_THIRD,
)


def _light_retry(first):  # 同じ失敗の2回目（light_compressed を返すはず）
    return decide_retry_stage(
        failure_class=first.failure_class,
        retry_request_number=first.request_number + 1,
        decided=(first,),
    )


def test_first_retry_of_a_format_failure_keeps_every_evidence() -> None:
    decision = decide_retry_stage(failure_class=FAILURE_OUTPUT_FORMAT, retry_request_number=2)
    assert decision.stage == "full_preserved"
    assert decision.compression_trigger == TRIGGER_FORMAT_REPAIR
    assert decision.compresses is False
    assert decision.folds_speech_history is False
    # 形式だけ直すので、出力予算側（thinking・生成上限）もいじらない
    assert (decision.escalate_generation, decision.thinking_off, decision.retry_budget_fit) == (
        False,
        False,
        False,
    )
    # 質問への未回答も同じ（同一証拠でその問題だけを直す）
    answered = decide_retry_stage(failure_class=FAILURE_QUESTION_ANSWER, retry_request_number=2)
    assert answered.stage == "full_preserved"
    assert answered.compression_trigger == TRIGGER_QUESTION_ANSWER_REPAIR


def test_second_same_failure_drops_only_derived_sections() -> None:
    first = decide_retry_stage(failure_class=FAILURE_OUTPUT_FORMAT, retry_request_number=2)
    second = _light_retry(first)
    assert second.stage == "light_compressed"
    assert second.compression_trigger == TRIGGER_SAME_FAILURE_SECOND
    assert second.compresses is True
    # 派生節を落とすだけで発言履歴は畳まない（ここが一貫性の担保）
    assert second.folds_speech_history is False
    assert LIGHT_STAGE_AFTER_SAME_FAILURE_ATTEMPTS == 2


def test_third_strike_of_the_same_failure_is_the_only_ladder_that_folds_history() -> None:
    first = decide_retry_stage(failure_class=FAILURE_OUTPUT_FORMAT, retry_request_number=2)
    second = _light_retry(first)
    third = decide_retry_stage(
        failure_class=FAILURE_OUTPUT_FORMAT,
        retry_request_number=4,
        decided=(first, second),
    )
    assert third.stage == "last_resort_compressed"
    assert third.compression_trigger == TRIGGER_SAME_FAILURE_THIRD
    assert third.folds_speech_history is True
    assert LAST_RESORT_AFTER_SAME_FAILURE_ATTEMPTS == 3


def test_escalated_stage_never_walks_back_to_a_long_prompt() -> None:
    folded = decide_retry_stage(failure_class=FAILURE_CONTEXT_OVERFLOW, retry_request_number=2)
    # 次が別種の失敗でも、一度畳んだ枠から長いプロンプトへは戻さない
    follow_up = decide_retry_stage(
        failure_class=FAILURE_SEMANTIC_LIGHT,
        retry_request_number=3,
        decided=(folded,),
    )
    assert follow_up.stage == "last_resort_compressed"
    assert follow_up.compresses is True


def test_budget_failures_adjust_output_budget_before_any_shortening() -> None:
    thinking_only = decide_retry_stage(
        failure_class=FAILURE_EMPTY_OR_THINKING_ONLY, retry_request_number=2
    )
    assert thinking_only.stage == "full_preserved"
    assert thinking_only.compression_trigger == TRIGGER_BUDGET_ADJUSTMENT
    # thinking が本文を食い潰した失敗は、証拠を消さず thinking を止める
    assert thinking_only.thinking_off is True
    assert thinking_only.compresses is False
    assert thinking_only.folds_speech_history is False
    # 生成上限が足りない失敗は、まず上限を上げる（同一証拠のまま）
    capped = decide_retry_stage(failure_class=FAILURE_GENERATION_CAP, retry_request_number=2)
    assert capped.stage == "full_preserved"
    assert capped.escalate_generation is True
    assert capped.compresses is False
    # 同じ予算失敗が続いて初めて派生節だけを落とす（履歴はまだ畳まない）
    again = _light_retry(thinking_only)
    assert again.stage == "light_compressed"
    assert again.compression_trigger == TRIGGER_BUDGET_RETRY_REPEAT
    assert again.folds_speech_history is False


def test_timeout_and_real_overflows_shorten_immediately() -> None:
    timeout = decide_retry_stage(failure_class=FAILURE_TIMEOUT, retry_request_number=2)
    assert timeout.stage == "last_resort_compressed"
    assert timeout.compression_trigger == TRIGGER_TIMEOUT_REPEAT
    assert timeout.thinking_off is True and timeout.retry_budget_fit is True
    overflow = decide_retry_stage(
        failure_class=FAILURE_SEMANTIC_LIGHT,
        retry_request_number=2,
        context_overflow=True,
    )
    assert overflow.failure_class == FAILURE_CONTEXT_OVERFLOW
    assert overflow.compression_trigger == TRIGGER_CONTEXT_OVERFLOW
    assert overflow.folds_speech_history is True
    shortage = decide_retry_stage(
        failure_class=FAILURE_OUTPUT_FORMAT,
        retry_request_number=2,
        effective_context_shortage=True,
    )
    assert shortage.compression_trigger == TRIGGER_EFFECTIVE_CONTEXT_SHORTAGE
    # この行動で既に一度待たされているなら、長文を戻して再び待たせない
    slow_action = decide_retry_stage(
        failure_class=FAILURE_EMPTY_OR_THINKING_ONLY,
        retry_request_number=2,
        timeout_scope_applies=True,
    )
    assert slow_action.stage == "light_compressed"
    assert slow_action.compression_trigger == TRIGGER_TIMEOUT_REPEAT
    assert slow_action.retry_budget_fit is True


def test_legacy_policy_folds_from_the_first_retry_but_keeps_round_transcript_long() -> None:
    legacy_semantic = decide_retry_stage(
        failure_class=FAILURE_OUTPUT_FORMAT,
        policy=RETRY_CONTEXT_POLICY_LEGACY_EARLY,
        retry_request_number=2,
    )
    assert legacy_semantic.stage == "last_resort_compressed"
    assert legacy_semantic.compression_trigger == TRIGGER_POLICY_LEGACY_EARLY
    assert legacy_semantic.folds_speech_history is True
    # round_transcript_memory では旧挙動でも長いプロンプトを維持していた（再現させる）
    keeps = decide_retry_stage(
        failure_class=FAILURE_SEMANTIC_LIGHT,
        policy=RETRY_CONTEXT_POLICY_LEGACY_EARLY,
        retry_request_number=2,
        keeps_round_context=True,
    )
    assert keeps.stage == "full_preserved"
    assert keeps.compresses is False
    # budget起因（空・thinking-only・生成上限）は旧挙動どおり短縮＋予算調整から入る
    budget = decide_retry_stage(
        failure_class=FAILURE_EMPTY_OR_THINKING_ONLY,
        policy=RETRY_CONTEXT_POLICY_LEGACY_EARLY,
        retry_request_number=2,
        keeps_round_context=True,
    )
    assert budget.stage == "last_resort_compressed"
    assert budget.escalate_generation is True and budget.thinking_off is True


from types import SimpleNamespace  # noqa: E402

from werewolf.prompt_contract import SECTION_META_BY_NAME, TIER_OPTIONAL, tier_of  # noqa: E402,F401

from werewolf.retry_context_policy import (  # noqa: E402
    RETRY_BLOCK_CONFIRMED_FACTS,
    RETRY_BLOCK_CORRECTION_GUIDANCE,
    RETRY_BLOCK_INFORMATION_RULES,
    RETRY_BLOCK_OPENING_NOTE,
    RETRY_BLOCK_OUTPUT_FORMAT,
    RETRY_BLOCK_PRIVATE_SEER_RESULTS,
    RETRY_BLOCK_PUBLIC_CLAIMS,
    RETRY_BLOCK_RECENT_SPEECHES,
    RETRY_BLOCK_STATE_CHECKPOINT,
    RETRY_BLOCK_THEORY_LINE,
    RETRY_BLOCK_UNANSWERED_QUESTION,
    RETRY_LIGHT_DROP_CANDIDATES,
    RETRY_LIGHT_PROTECTED_SECTIONS,
    RETRY_PROMPT_BLOCKS,
    RETRY_PROMPT_ORDER_LEGACY,
    assemble_retry_prompt,
    drop_retry_light_sections,
    dropped_speech_stats,
    evidence_id_diff,
    evidence_ids_in_text,
    resolve_retry_prompt_order,
    retry_light_drop_names,
    score_retry_order,
    section_diff,
    section_names_in_prompt,
    validate_retry_order,
)


def _marker(name: str) -> str:
    """台帳に登録された実マーカー（engine が実際に書く見出し語）で合成プロンプトを作る。"""
    return SECTION_META_BY_NAME[name].markers[0]


# 派生節（落としてよいもの）と core 節（絶対に落とさないもの）の実マーカー例
DERIVED_OPTIONAL = "reasoning_guidance"
CORE_KEEP = "fact_ledger"


def test_light_drop_candidates_are_only_optional_derived_sections() -> None:
    names = retry_light_drop_names()
    assert names, "light_compressed で落とせる派生節が1つもない"
    for name in names:
        assert tier_of(name) == TIER_OPTIONAL, f"{name} は optional 登録ではない"
    # 保持すべき節（現在質問・台帳・発言本文・状態節）を除去候補に混ぜない
    overlap = set(RETRY_LIGHT_DROP_CANDIDATES) & set(RETRY_LIGHT_PROTECTED_SECTIONS)
    assert not overlap, sorted(overlap)


def test_drop_retry_light_sections_keeps_core_text_and_returns_removed_names() -> None:
    text = (
        f"{_marker(CORE_KEEP)}\n- Your Player Name: Player 1\n- Current Day: Day 2\n"
        f"{_marker(DERIVED_OPTIONAL)}\n- free-form guidance line\n"
    )
    pruned, removed, saved = drop_retry_light_sections(text)
    assert removed and set(removed) <= set(retry_light_drop_names())
    assert saved > 0
    assert _marker(CORE_KEEP) in pruned, "core節は1文字も触らない"
    assert "Player 1" in pruned and "Day 2" in pruned, "確定事実の本文はそのまま残る"
    assert _marker(DERIVED_OPTIONAL) not in pruned
    # 除去対象がなければ無傷のまま（段階を誤爆させない）
    untouched = f"{_marker(CORE_KEEP)}\n- Your Player Name: Player 1\n"
    same, none_removed, none_saved = drop_retry_light_sections(untouched)
    assert same == untouched and none_removed == () and none_saved == 0


def test_evidence_and_section_diffs_separate_kept_from_lost() -> None:
    initial = "seen [speech:3] and [claim:2] plus C7 reference; also [speech:9]"
    retry = "keeps [speech:3] and C7 only"
    ids = evidence_ids_in_text(initial)
    assert "[speech:3]" in ids and "[claim:2]" in ids and "C7" in ids
    kept, lost = evidence_id_diff(initial, retry)
    assert "[speech:3]" in kept and "C7" in kept
    assert "[claim:2]" in lost and "[speech:9]" in lost
    sections_initial = (
        f"{_marker(CORE_KEEP)}\nx\n{_marker(DERIVED_OPTIONAL)}\ny\n"
    )
    sections_retry = f"{_marker(CORE_KEEP)}\nx\n"
    present, gone = section_diff(sections_initial, sections_retry)
    assert CORE_KEEP in present and CORE_KEEP not in gone
    assert DERIVED_OPTIONAL in gone
    assert CORE_KEEP in section_names_in_prompt(sections_retry)


def test_dropped_speech_stats_records_count_chars_and_points() -> None:
    # 呼び出し側（llm_actions）は system 行を渡す前に除く。ここでは発言行だけを使う。
    messages = [
        SimpleNamespace(speaker="Player 2", content="最初の発言本文", day=1, turn=1),
        SimpleNamespace(speaker="Player 3", content="直前の発言本文", day=2, turn=2),
    ]
    prompt = "Recent Speech: Player 3: 直前の発言本文"
    stat = dropped_speech_stats(messages, prompt)
    # プロンプトに本文が載った Player 3 は数えない（落ちたのは古い1発言のみ）
    assert stat.dropped_count == 1
    assert stat.dropped_chars == len("最初の発言本文")
    assert stat.oldest_day_turn == "Day 1 Turn 1"
    assert stat.newest_day_turn == "Day 1 Turn 1"
    fields = stat.as_dict()
    assert fields["dropped_speech_count"] == 1
    assert fields["dropped_speech_oldest_point"] == "Day 1 Turn 1"
    # プロンプトが空なら全発言が落ちたと数える（それでも本文そのものは診断に出さない）
    empty_stat = dropped_speech_stats(messages, None)
    assert empty_stat.dropped_count == 2
    assert not any("発言本文" in str(value) for value in fields.values())


from werewolf.retry_context_policy import (  # noqa: E402
    RETRY_BLOCK_COUNTER_ARGUMENTS,
    RETRY_CHECKPOINT_MAX_INDEX,
    RETRY_FACTS_MAX_INDEX,
    RETRY_PROMPT_ORDER_LEGACY,
    RETRY_PROMPT_ORDER_V1,
    RETRY_REPRESENTATIVE_BLOCK_CHARS,
    RETRY_THEORY_MIN_INDEX,
    legacy_retry_order_score,
    retry_adopted_order_score,
    search_retry_orders,
)


def _positions(order):
    return {name: index for index, name in enumerate(order)}


def test_frozen_retry_order_meets_every_hard_constraint() -> None:
    assert validate_retry_order(RETRY_PROMPT_ORDER_V1) == []
    positions = _positions(RETRY_PROMPT_ORDER_V1)
    # 一貫性: 自分の身分・何日目を最初に読む／エンジンの論理状態も冒頭3ブロック以内
    assert positions[RETRY_BLOCK_CONFIRMED_FACTS] <= RETRY_FACTS_MAX_INDEX
    assert positions[RETRY_BLOCK_STATE_CHECKPOINT] <= RETRY_CHECKPOINT_MAX_INDEX
    # 誘導を縛らない: 未検証のセオリー行（読み枠を決める案内）を先頭2ブロックへ置かない
    assert positions[RETRY_BLOCK_THEORY_LINE] >= RETRY_THEORY_MIN_INDEX
    # 「いま使う」情報を終端側へ寄せる: 質問 → 修正指示 → 出力形式の順で隣接させる
    assert RETRY_PROMPT_ORDER_V1[-1] == RETRY_BLOCK_OUTPUT_FORMAT
    assert RETRY_PROMPT_ORDER_V1[-2] == RETRY_BLOCK_CORRECTION_GUIDANCE
    assert (
        positions[RETRY_BLOCK_UNANSWERED_QUESTION]
        < positions[RETRY_BLOCK_CORRECTION_GUIDANCE]
    )
    assert (
        positions[RETRY_BLOCK_RECENT_SPEECHES]
        < positions[RETRY_BLOCK_UNANSWERED_QUESTION]
    )
    # 判定語（占い師本人限定）を生成直前に置かず、チェックポイントIDは実主張行より前
    assert (
        positions[RETRY_BLOCK_PRIVATE_SEER_RESULTS]
        < positions[RETRY_BLOCK_CORRECTION_GUIDANCE]
    )
    assert (
        positions[RETRY_BLOCK_STATE_CHECKPOINT] < positions[RETRY_BLOCK_PUBLIC_CLAIMS]
    )


def test_each_hard_constraint_is_reported_by_name() -> None:
    def kinds(order) -> set[str]:
        return {str(item.get("kind")) for item in validate_retry_order(order)}

    assert kinds([]) == {"empty_order"}
    assert "duplicate_block" in kinds([*RETRY_PROMPT_ORDER_V1, RETRY_BLOCK_THEORY_LINE])
    assert "unknown_block" in kinds([*RETRY_PROMPT_ORDER_V1, "not_a_retry_block"])
    assert "facts_missing" in kinds(
        [name for name in RETRY_PROMPT_ORDER_V1 if name != RETRY_BLOCK_CONFIRMED_FACTS]
    )
    late_facts = list(RETRY_PROMPT_ORDER_V1)
    late_facts.remove(RETRY_BLOCK_CONFIRMED_FACTS)
    late_facts.insert(4, RETRY_BLOCK_CONFIRMED_FACTS)
    assert "facts_too_late" in kinds(late_facts)
    # 形式仕様を末尾に置かない配置は、出力形式の読み順が崩れることを示す
    not_last = list(RETRY_PROMPT_ORDER_V1)
    not_last.remove(RETRY_BLOCK_OUTPUT_FORMAT)
    not_last.insert(0, RETRY_BLOCK_OUTPUT_FORMAT)
    assert "output_format_not_last" in kinds(not_last)
    private_first = list(RETRY_PROMPT_ORDER_V1)
    private_first.remove(RETRY_BLOCK_PRIVATE_SEER_RESULTS)
    private_first.insert(0, RETRY_BLOCK_PRIVATE_SEER_RESULTS)
    reported = kinds(private_first)
    assert "private_results_first" in reported
    # 発言証拠を身分より前に読むと「自分が誰か」を読めないまま語り出す
    evidence_first = list(RETRY_PROMPT_ORDER_V1)
    evidence_first.remove(RETRY_BLOCK_RECENT_SPEECHES)
    evidence_first.insert(0, RETRY_BLOCK_RECENT_SPEECHES)
    assert "evidence_before_facts" in kinds(evidence_first)
    # 反論を修正指示より後（＝生成直後）に読む配置は、直前議論と接続しない原因になる
    after_correction = list(RETRY_PROMPT_ORDER_V1)
    after_correction.remove(RETRY_BLOCK_COUNTER_ARGUMENTS)
    after_correction.append(RETRY_BLOCK_COUNTER_ARGUMENTS)
    assert {"evidence_after_correction", "output_format_not_last"} <= kinds(after_correction)
    question_last = list(RETRY_PROMPT_ORDER_V1)
    question_last.remove(RETRY_BLOCK_UNANSWERED_QUESTION)
    question_last.insert(-1, RETRY_BLOCK_UNANSWERED_QUESTION)
    assert "question_after_correction" in kinds(question_last)


def test_legacy_order_is_legal_but_scores_below_the_frozen_order() -> None:
    assert validate_retry_order(RETRY_PROMPT_ORDER_LEGACY) == []
    chars = dict(RETRY_REPRESENTATIVE_BLOCK_CHARS)
    adopted = retry_adopted_order_score(chars)
    legacy = legacy_retry_order_score(chars)
    assert adopted.violations == () and legacy.violations == ()
    # 同じ内容（同じブロック字数）のまま配置だけが違う比較なので、順序の差がそのまま出る
    assert adopted.score > legacy.score
    assert adopted.attention_efficiency > legacy.attention_efficiency
    # 凍結順は隣接距離ペナルティゼロ（現在質問・修正指示・出力形式が離れていない）
    assert adopted.adjacency_penalty == 0.0


def test_frozen_order_is_the_greedy_search_best_and_deterministic() -> None:
    chars = dict(RETRY_REPRESENTATIVE_BLOCK_CHARS)
    best = search_retry_orders(chars, start=RETRY_PROMPT_ORDER_LEGACY, top_k=5)
    assert best, "貪欲探索が候補を1つも出さなかった"
    assert float(best[0]["retry_layout_score"]) == float(
        retry_adopted_order_score(chars).score
    )
    for entry in best:
        assert entry["retry_hard_violations"] == []
    # 決定的: 同じ出発順・同じ字数なら2度実行しても並ぶ順が変わらない（LL不使用）
    again = search_retry_orders(chars, start=RETRY_PROMPT_ORDER_LEGACY, top_k=5)
    assert [item["retry_layout_order"] for item in best] == [
        item["retry_layout_order"] for item in again
    ]


def test_resolve_and_assemble_follow_the_policy_without_losing_content() -> None:
    assert resolve_retry_prompt_order(RETRY_CONTEXT_POLICY_TIERED) == RETRY_PROMPT_ORDER_V1
    assert (
        resolve_retry_prompt_order(RETRY_CONTEXT_POLICY_LEGACY_EARLY)
        == RETRY_PROMPT_ORDER_LEGACY
    )
    blocks = {
        RETRY_BLOCK_OUTPUT_FORMAT: "output format spec",
        RETRY_BLOCK_CONFIRMED_FACTS: "confirmed facts",
        RETRY_BLOCK_STATE_CHECKPOINT: "",  # 空の節は載せない
        "an_unregistered_block": "must not vanish",
    }
    text, included = assemble_retry_prompt(blocks, resolve_retry_prompt_order("tiered"))
    assert included[0] == RETRY_BLOCK_CONFIRMED_FACTS
    assert included[-1] == RETRY_BLOCK_OUTPUT_FORMAT
    assert RETRY_BLOCK_STATE_CHECKPOINT not in included
    # 順番テーブルに無い節でも本文を失わない（宣言の末尾へ足す）
    assert "must not vanish" in text
    assert included[-2] == "an_unregistered_block"


import time  # noqa: E402

from werewolf.llm_actions import OllamaActions  # noqa: E402
from werewolf.models import Message  # noqa: E402
from werewolf.ollama import OllamaError  # noqa: E402
from werewolf.retry_context_policy import (  # noqa: E402
    RETRY_CONTEXT_POLICY_LEGACY_EARLY,
    RETRY_CONTEXT_POLICY_TIERED,
    RETRY_LIGHT_DROP_CANDIDATES,
    retry_stage_log_fields,
)
from werewolf.setup import create_game  # noqa: E402

VALID_BODY = (
    "[CO] None\n"
    "[BODY] まだ情報は揃っていませんが、まず各役職の主張を出し合わせるのが早いと思います。"
    "私は情報を未取得なので、結果を持っている方は先に共有してほしいです。"
)
# 最も古い発言（core-ledger短縮では「直近3発言」だけなので必ず落ちる位置に置く）
UNIQUE_OLD_SPEECH = "古い論点Aについて、私はあの人物に投票するつもりでいました。"
UNIQUE_NEW_SPEECH = "新しい論点Bです。投票先は誰にしますか？"


class _ScriptedClient:
    """送信を実際に発行しないモック（LL不使用）。プロンプト・オプションを記録する。"""

    def __init__(self, responses: list[object]) -> None:
        self.timeout = 120.0
        self.max_retries = 1
        self.timeout_retries = 0
        self._responses = list(responses)
        self.sent_prompts: list[list[dict[str, object]]] = []
        self.sent_options: list[dict[str, object]] = []

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {}

    def chat(
        self, model: str, messages: list[dict[str, object]], options: dict | None = None
    ) -> str:
        self.sent_prompts.append([dict(message) for message in messages])
        self.sent_options.append(dict(options or {}))
        entry = self._responses.pop(0) if self._responses else VALID_BODY
        if isinstance(entry, BaseException):
            raise entry
        assert isinstance(entry, str)
        return entry


def _user_prompt(client: _ScriptedClient, index: int) -> str:
    return str(client.sent_prompts[index][-1]["content"])


def _run(policy: str, client: _ScriptedClient, *, context_strategy: str = "hybrid_retrieval"):
    """同一seed・同一盤面で方策だけ差し替えて実行し、送信内容と段階診断を返す。"""
    state = create_game(
        seed=2066066, config_path="config/players.json", response_language="日本語"
    )
    state.retry_context_policy = policy
    actor = state.players[0]
    state.day = 2
    state.discussion_turn = 2
    others = [player.name for player in state.players if player.name != actor.name]
    # 古い発言を5つ積む（core-ledger短縮プロンプトは直近3発言しか持たないため、先頭は落ちる）
    for index, speaker in enumerate((others * 2)[:5]):
        state.conversation_log.append(
            Message(
                speaker=speaker,
                content=(UNIQUE_OLD_SPEECH if index == 0 else UNIQUE_NEW_SPEECH),
                day=2,
                turn=1,
                # エンジンが発行する発言IDを再現する（`[speech:N]` がどの節に載っていたかの
                # 分類と、IDが消えても本文が残るかの実測に必要なスタンプ）
                speech_id=index + 1,
                timeline_index=index + 1,
            )
        )
    events: list[dict[str, object]] = []
    actions = OllamaActions(
        state,
        client,
        seed=state.seed,
        event_callback=events.append,
        context_strategy=context_strategy,
    )
    prompt_ctx = actions._build_discussion_prompt_with_budget(
        actor,
        tuple(state.alive_players),
        tuple(state.conversation_log),
        2,
        include_claim=True,
        budget_key="discussion",
    )
    text = actions._free_text_with_context(
        "discussion",
        actor,
        prompt_ctx,
        "議論を続けます。",
        time.monotonic(),
        recent_messages=tuple(state.conversation_log),
        budget_key="discussion",
    )
    stages = [event for event in events if event.get("type") == "retry_context_stage"]
    return {
        "actions": actions,
        "client": client,
        "stages": stages,
        "events": events,
        "final_body": text,
        "initial_prompt": prompt_ctx.user_prompt,
    }


def test_tiered_first_retry_reuses_the_initial_evidence() -> None:
    client = _ScriptedClient(["", VALID_BODY])
    result = _run(RETRY_CONTEXT_POLICY_TIERED, client)
    stages = result["stages"]
    assert len(stages) == 1, [s.get("retry_stage") for s in stages]
    record = stages[0]
    assert record.get("retry_stage") == "full_preserved"
    assert record.get("retry_failure_class") == "empty_or_thinking_only"
    assert record.get("retry_context_policy") == RETRY_CONTEXT_POLICY_TIERED
    assert record.get("retry_compresses_prompt") is False
    assert record.get("retry_folds_speech_history") is False
    assert not record.get("dropped_sections"), "初回再生成で節を落とさない"
    retry_prompt = _user_prompt(client, 1)
    # 初回生成と同じ議論（最も古い発言まで）が読めたまま再生成している
    assert UNIQUE_OLD_SPEECH in retry_prompt
    assert UNIQUE_NEW_SPEECH in retry_prompt
    # core-ledger 短縮プロンプト（旧方策の証左）へは切り替えない
    assert "This is a short retry prompt" not in retry_prompt
    assert record.get("retry_prompt_style") == "full_preserved"


def test_tiered_second_retry_lights_out_only_derived_sections() -> None:
    client = _ScriptedClient(["", "", VALID_BODY])
    result = _run(RETRY_CONTEXT_POLICY_TIERED, client)
    stages = result["stages"]
    assert [s.get("retry_stage") for s in stages] == [
        "full_preserved",
        "light_compressed",
    ]
    light = stages[1]
    assert light.get("retry_compresses_prompt") is True
    assert light.get("retry_folds_speech_history") is False
    dropped = set(light.get("dropped_sections") or [])
    assert dropped <= set(RETRY_LIGHT_DROP_CANDIDATES), sorted(
        dropped - set(RETRY_LIGHT_DROP_CANDIDATES)
    )
    assert {"conversation_evidence", "question_block", "fact_ledger"} <= set(
        light.get("preserved_sections") or []
    )
    assert int(light.get("retry_prompt_saved_chars") or 0) > 0
    # 派生節だけを落としても発言本文（最も古い論点）は残り、直前議論と接続できる
    assert UNIQUE_OLD_SPEECH in _user_prompt(client, 2)


def test_legacy_policy_folds_speech_history_from_the_first_retry() -> None:
    """旧挙動（A/B基準）は再生成1回目から core-ledger 短縮へ落ち、古い発言が消えていた。"""
    client = _ScriptedClient(["", VALID_BODY])
    result = _run(RETRY_CONTEXT_POLICY_LEGACY_EARLY, client)
    stages = result["stages"]
    assert len(stages) == 1
    record = stages[0]
    assert record.get("retry_stage") == "last_resort_compressed"
    assert record.get("retry_folds_speech_history") is True
    assert record.get("compression_trigger") == "legacy_early_core_ledger"
    retry_prompt = _user_prompt(client, 1)
    assert "This is a short retry prompt" in retry_prompt
    assert UNIQUE_OLD_SPEECH not in retry_prompt, "旧方策では最も古い発言が消える"
    assert int(record.get("dropped_speech_count") or 0) > 0
    # 予算起因なので生成上限も引き上げられる（TASK-053からの挙動は保つ）
    first_limit = client.sent_options[0].get("num_predict")
    retry_limit = client.sent_options[1].get("num_predict")
    assert isinstance(first_limit, int) and isinstance(retry_limit, int)
    assert retry_limit > int(first_limit)


def test_timeout_repeat_never_returns_a_long_prompt() -> None:
    client = _ScriptedClient([OllamaError("timeout", "mock timeout", 1), VALID_BODY])
    result = _run(RETRY_CONTEXT_POLICY_TIERED, client)
    stages = result["stages"]
    assert len(stages) == 1, [s.get("retry_stage") for s in stages]
    record = stages[0]
    assert record.get("retry_stage") == "last_resort_compressed"
    assert record.get("compression_trigger") == "timeout_repeat_short_prompt"
    assert record.get("retry_budget_fit") is True
    assert len(_user_prompt(client, 1)) < len(str(result["initial_prompt"]))
    # タイムアウトの再送では枠も要求サイズへ絞る（TASK-035の方針を踏襲）
    first_ctx = client.sent_options[0].get("num_ctx")
    retry_ctx = client.sent_options[1].get("num_ctx")
    assert isinstance(first_ctx, int) and isinstance(retry_ctx, int)
    assert int(retry_ctx) < int(first_ctx)


def test_body_is_never_rewritten_and_success_leaves_no_stage_record() -> None:
    """圧縮段階があってもモデルが書いた本文は書き換えない。通常生成では段階診断を出さない。"""
    client = _ScriptedClient(["", VALID_BODY])
    result = _run(RETRY_CONTEXT_POLICY_TIERED, client)
    assert "結果を持っている方は先に共有してほしい" in str(result["final_body"])
    ok_client = _ScriptedClient([VALID_BODY])
    ok = _run(RETRY_CONTEXT_POLICY_TIERED, ok_client)
    assert not ok["stages"], "再生成が無い（初回成功）では段階診断を作らない"
    assert ok["actions"].last_body_source_stage == "normal_generation"


def test_retry_log_fields_carries_numbers_and_names_but_not_body_text() -> None:
    decision = decide_retry_stage(
        failure_class=FAILURE_OUTPUT_FORMAT, retry_request_number=2
    )
    body_sentence = "この発言本文そのものは診断ログへ載せてはいけない固有文です。"
    fields = retry_stage_log_fields(
        policy=RETRY_CONTEXT_POLICY_TIERED,
        decision=decision,
        initial_prompt_text=f"[FACT_LEDGER]\n- A\n{body_sentence}",
        retry_prompt_text="[FACT_LEDGER]\n- A\n",
        messages=[],
        previous_body_chars=len(body_sentence),
        new_body_chars=len(body_sentence) - 4,
        action="discussion",
    )
    assert fields["retry_stage"] == "full_preserved"
    assert fields["retry_context_policy"] == RETRY_CONTEXT_POLICY_TIERED
    assert (
        fields["initial_user_prompt_char_count"] > fields["retry_user_prompt_char_count"]
    )
    assert fields["retry_prompt_saved_chars"] > 0
    assert fields["retry_body_char_delta"] == -4
    assert fields["retry_body_shortened"] is True
    assert "fact_ledger" in fields["preserved_sections"]
    # 数値・時点・ID・節名だけで、発言本文を文字列値として残さない
    assert all(body_sentence not in str(value) for value in fields.values())


def test_new_retry_wording_does_not_steer_tone_topics_or_reasoning_order() -> None:
    """再生成で口調・文量・論点・結論・推理順を指定する新しい命令文を増やさない。"""
    actions = _run(RETRY_CONTEXT_POLICY_TIERED, _ScriptedClient(["", VALID_BODY]))["actions"]
    block = actions._retry_retained_public_sentences_block(  # type: ignore[attr-defined]
        "discussion", "私はこの順で考えました。続く部分を書きます。", True
    )
    assert "Already Written Draft" in block
    lowered = block.lower()
    for phrase in BANNED_FREEDOM_PHRASES:
        assert phrase.lower() not in lowered
    # 言い換え・結論・順序の強制ではなく「続きを書いてよい」自由の明記にとどめる
    assert "you choose the wording, length, points and conclusion" in lowered
    # 秘匿境界を越えた疑いがある本文は次プロンプトへ引き継がない
    assert actions._retry_retained_public_sentences_block(  # type: ignore[attr-defined]
        "discussion", "私はこの順で考えました。続く部分を書きます。", False
    ) == ""
    assert actions._retry_retained_public_sentences_block("vote", "投票します", True) == ""
    # 方策モジュール側の語彙（段階名・役割名）にも誘導系文言を混ぜない
    from werewolf.retry_context_policy import RETRY_BLOCK_ROLES

    joined = "\n".join(
        [*RETRY_CONTEXT_POLICIES, *RETRY_STAGES, *[str(text) for text in RETRY_BLOCK_ROLES]]
    ).lower()
    for phrase in BANNED_FREEDOM_PHRASES:
        assert phrase.lower() not in joined


def test_stage_records_body_length_baseline_without_body_text() -> None:
    """段階診断は「再生成前に出ていた本文の長さ」を持ち、本文そのものは載せない。

    1回目の再生成ではまだ本文が無いので基準長は 0、本文が得られた試行では
    `body_preserved` が立つ（短くなったかは何度本文が出たかで初めて計算する）。
    """
    client = _ScriptedClient(["", "", VALID_BODY])
    result = _run(RETRY_CONTEXT_POLICY_TIERED, client)
    stages = result["stages"]
    assert len(stages) == 2
    for record in stages:
        assert isinstance(record.get("retry_previous_body_chars"), int)
        assert int(record.get("retry_previous_body_chars") or 0) == 0
        assert record.get("retry_outcome") == "response"
        # 長短・個数・節名だけで、モデルが書いた文面を文字列値として残さない
        assert not any("[BODY]" in str(value) for value in record.values())
    # 本文が得られた試行でだけ body_preserved が立つ（空応答の試行は False のまま）
    assert stages[0].get("body_preserved") is False
    assert stages[-1].get("body_preserved") is True
    light = stages[1]
    dropped = list(light.get("retry_light_dropped_sections") or [])
    assert dropped and set(dropped) <= set(RETRY_LIGHT_DROP_CANDIDATES)


# ---------------------------------------------------------------------------
# TASK-066（レビュー対応2）: 削除した証拠IDの安全分類
# 「削除ID数」だけでなく、core節由来・現在質問由来・optional派生節由来に分ける。
# light_compressed でIDが1件消えても、それが保護対象で無いことを診断で確かめる。
# ---------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from werewolf.retry_context_policy import (  # noqa: E402
    TRIGGER_BUDGET_RETRY_REPEAT,
    TRIGGER_CONTEXT_OVERFLOW,
    TRIGGER_EFFECTIVE_CONTEXT_SHORTAGE,
    TRIGGER_SAME_FAILURE_THIRD,
)

_ID_INITIAL_PROMPT = (
    "Discord Bot Game prompt header\n"
    "[FACT_LEDGER]\n"
    "- Player 3 の確定事実 [ref:s4]\n"
    "[Current Question]\n"
    "- Player 2 への未回答質問 [speech:2]\n"
    "[HYPOTHESIS_LEDGER]\n"
    "- 派生情報（仮説行）[speech:1]、テンプレ案内は [claim:N] / [ref:sN]\n"
    "Trailing instruction footer\n"
)
# light_compressed 相当: optional の派生節だけ消え、core節と現在質問はそのまま残る
_ID_LIGHT_PROMPT = (
    "Discord Bot Game prompt header\n"
    "[FACT_LEDGER]\n"
    "- Player 3 の確定事実 [ref:s4]\n"
    "[Current Question]\n"
    "- Player 2 への未回答質問 [speech:2]\n"
    # ID行は消えたが、発言そのものの本文はプロンプトへ残り続ける
    "- Player 6: 先行きの議論を続けるための固有文です。\n"
    "Trailing instruction footer\n"
)
# last_resort_compressed 相当: 発言証拠ごと落ち、台帳の参照IDも失われる
_ID_FOLDED_PROMPT = (
    "Discord Bot Game prompt header\n"
    "[Current Question]\n"
    "- Player 2 への未回答質問（IDなし）\n"
    "Trailing instruction footer\n"
)
_ID_SPEECH_BODY = "先行きの議論を続けるための固有文です。"


def _id_message(speech_id: int, content: str) -> SimpleNamespace:
    # engine.Message の代わり（診断が必要とする属性だけ持たせる）
    return SimpleNamespace(
        speaker="Player 6",
        content=content,
        day=2,
        turn=1,
        speech_id=speech_id,
    )


def test_template_placeholder_ids_are_not_counted_as_evidence() -> None:
    # `[speech:N]` / `[claim:N]` / `[ref:sN]` は案内文のプレースホルダなので数えない
    assert evidence_ids_in_text("[HYPOTHESIS_LEDGER]\n- [speech:N] [claim:N] [ref:sN]\n") == ()
    ids = evidence_ids_in_text("[speech:12] and [ref:s7] and [claim:3]")
    assert "[speech:12]" in ids and "[ref:s7]" in ids and "[claim:3]" in ids


def test_dropped_ids_classified_by_carrier_sections_light_case() -> None:
    safety = dropped_evidence_id_safety(
        _ID_INITIAL_PROMPT,
        _ID_LIGHT_PROMPT,
        messages=(_id_message(1, _ID_SPEECH_BODY), _id_message(2, "別の発言です。")),
    )
    # 消えるのは optional 派生節（仮説行）に載っていたIDだけ
    assert safety.dropped_total == 1, safety.as_dict()
    assert safety.optional_only_drops == 1
    assert safety.protected_drops == ()
    assert safety.question_linked_drops == ()
    assert safety.clean is True
    data = safety.as_dict()
    assert data["dropped_ids_protected_count"] == 0
    assert data["dropped_ids_question_linked_count"] == 0
    assert data["dropped_ids_optional_only_count"] == 1
    assert data["dropped_ids_safety_clean"] is True
    # ID行は消えても発言本文はプロンプトに残る（議論の追跡は切れない）
    assert data.get("dropped_speech_bodies_still_present") == 1
    assert data.get("dropped_speech_bodies_lost") == 0


def test_dropped_ids_flag_core_and_question_carriers_when_history_folds() -> None:
    safety = dropped_evidence_id_safety(
        _ID_INITIAL_PROMPT,
        _ID_FOLDED_PROMPT,
        messages=(_id_message(2, "別の発言です。"),),
    )
    # 履歴を畳むと core節（台帳・現在質問）のIDも失われる＝想定内の畳みでも区別して記録
    assert safety.dropped_total == 3, safety.as_dict()
    assert any(token.endswith("@fact_ledger") for token in safety.protected_drops)
    assert any(token.endswith("@question_block") for token in safety.question_linked_drops)
    assert safety.clean is False
    data = safety.as_dict()
    assert int(data["dropped_ids_protected_count"]) >= 2
    assert int(data["dropped_ids_question_linked_count"]) >= 1
    assert data["dropped_ids_safety_clean"] is False
    assert data["dropped_ids_protected_detail"], "どのIDがどの節で消えたか詳細を残す"
    # 本文ごと消えた発言は bodies_lost に立つ
    assert data.get("dropped_speech_bodies_lost") == 1


def test_immediate_compression_flags_separate_timeout_from_the_ladder() -> None:
    # 意図的な例外（即時圧縮）は timeout・コンテキスト超過だけ。原則のとはしごは該当しない
    assert retry_trigger_is_immediate_compression(TRIGGER_TIMEOUT_REPEAT) is True
    assert retry_trigger_is_immediate_compression(TRIGGER_CONTEXT_OVERFLOW) is True
    assert retry_trigger_is_immediate_compression(TRIGGER_EFFECTIVE_CONTEXT_SHORTAGE) is True
    assert retry_trigger_is_immediate_compression(TRIGGER_FORMAT_REPAIR) is False
    assert retry_trigger_is_immediate_compression(TRIGGER_SAME_FAILURE_THIRD) is False
    assert retry_trigger_is_immediate_compression(TRIGGER_BUDGET_RETRY_REPEAT) is False
    # timeout起因の判定（A/Bで別建て評価するための分離）
    assert retry_trigger_is_timeout(TRIGGER_TIMEOUT_REPEAT) is True
    assert (
        retry_trigger_is_timeout(TRIGGER_BUDGET_RETRY_REPEAT, FAILURE_TIMEOUT) is True
    )
    assert (
        retry_trigger_is_timeout(TRIGGER_CONTEXT_OVERFLOW, FAILURE_CONTEXT_OVERFLOW) is False
    )
    assert retry_trigger_is_timeout(None, None) is False







