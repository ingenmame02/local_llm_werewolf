"""TASK-066: 再生成のコンテキスト段階化（証拠保持優先）と再生成プロンプトの節順最適化。

game_87ff647746e6 では、形式修正や空応答への再要求という「プロンプトが原因でない失敗」に
対しても、最初から発言履歴を core-ledger まで畳む短縮プロンプトへ切り替わっていた。その結果、
モデルは直前までの議論・引用・反論・自分の立場を読めないまま再生成し、再生成文だけが短く、
一般論的で、直前の議論と接続しない恐れがあった。

このモジュールは圧縮の「いつ・どこまで」を決める純粋なポリシーで、LLを呼ばず、決定論的である。
本文の言い換え・削除・中立化は一切行わない（圧縮は常にプロンプト上の表現だけを短くする）。

設計思想からの変換:

- コンテキスト節約: 段階を `full_preserved`（証拠そのまま・対象エラーだけ短い補足）→
  `light_compressed`（低優先の派生節だけ除去）→ `last_resort_compressed`（古い発言本文を
  ID行へ畳む）に分け、真に必要になるまで広い圧縮を行わない。不要なプロンプト組み直しも辞める。
- モデルの一貫性: 現在質問・確定事実・役職主張・投票／死亡台帳・直近発言・反証・
  [CONTINUITY_CHECKPOINT] はどの段階でも保持する。証拠ID（[speech:N] / [claim:N]）の
  保持・欠落を差分で記録し、再生成で追跡が切れないようにする。
- 合理的な議論: 失敗理由ごとに圧縮開始条件を分ける（形式は同一証拠で形式だけ直す、
  空/thinking-only は thinking と本文予算だけ触る、生成上限は num_predict を上げる）。
- 議論の自由: ここで決めるのは「どの情報を載せどこに置くか」だけで、口調・文量・論点・
  結論・推理順を決める文言は追加しない。誘導系（未検証のセオリー行）を先頭帯へ置かない
  ハード制約をプローブに入れる。

既存の `--core-compression`（TASK-058）や optional 段階除去（TASK-057）とは別系統で、
再生成専用の方策・診断として扱う（同じ名前の診断キーは使わない）。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .prompt_contract import TIER_CORE, TIER_OPTIONAL, drop_sections, find_sections, tier_of
from .prompt_layout import (
    ADJACENCY_PENALTY_PER_FRACTION,
    ATTENTION_PRIMACY_EDGE,
    ATTENTION_ZONES,
    ATTENTION_RECENCY_EDGE,
    HARD_VIOLATION_PENALTY,
    PRIMACY_CLUTTER_WEIGHT,
    POST_FORMAT_DILUTION_WEIGHT,
    SCORE_SCALE,
    WEAK_MIDDLE_MUST_USE_WEIGHT,
    attention_prior_span,
    attention_zone_for_center,
    middle_band_bounds,
    spans_from_order,
)

# ---------------------------------------------------------------------------
# ポリシー（既定 tiered / A/B比較基準として legacy_early_compression）
# ---------------------------------------------------------------------------

RETRY_CONTEXT_POLICY_TIERED = "tiered"
RETRY_CONTEXT_POLICY_LEGACY_EARLY = "legacy_early_compression"
RETRY_CONTEXT_POLICIES: tuple[str, ...] = (
    RETRY_CONTEXT_POLICY_TIERED,
    RETRY_CONTEXT_POLICY_LEGACY_EARLY,
)
RETRY_CONTEXT_POLICY_DEFAULT = RETRY_CONTEXT_POLICY_TIERED


def normalize_retry_context_policy(value: object) -> str:
    """設定値を `tiered` / `legacy_early_compression` へ正規化する（不明値は既定）。"""
    if isinstance(value, bool):
        # True: 段階化（既定）/ False: 再生成直後から早期圧縮していた旧挙動
        return (
            RETRY_CONTEXT_POLICY_TIERED if value else RETRY_CONTEXT_POLICY_LEGACY_EARLY
        )
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in RETRY_CONTEXT_POLICIES:
            return candidate
        if candidate in {"tier", "staged", "progressive", "task066", "on", "true"}:
            return RETRY_CONTEXT_POLICY_TIERED
        if candidate in {
            "legacy",
            "legacy_early",
            "early_compression",
            "core_ledger_first",
            "off",
            "false",
            "old",
            "pre_task066",
        }:
            return RETRY_CONTEXT_POLICY_LEGACY_EARLY
    return RETRY_CONTEXT_POLICY_DEFAULT


def retry_context_policy_label(policy: object) -> str:
    """CLI・ログ用の正規化ラベル。"""
    return normalize_retry_context_policy(policy)


# ---------------------------------------------------------------------------
# 段階（再生成のたびに1つだけ採用する。初回生成は normal_generation）
# ---------------------------------------------------------------------------

RETRY_STAGE_NORMAL = "normal_generation"
RETRY_STAGE_FULL_PRESERVED = "full_preserved"
RETRY_STAGE_LIGHT_COMPRESSED = "light_compressed"
RETRY_STAGE_LAST_RESORT_COMPRESSED = "last_resort_compressed"
RETRY_STAGES: tuple[str, ...] = (
    RETRY_STAGE_NORMAL,
    RETRY_STAGE_FULL_PRESERVED,
    RETRY_STAGE_LIGHT_COMPRESSED,
    RETRY_STAGE_LAST_RESORT_COMPRESSED,
)
# 圧縮が実際に起きた段階のみ（節約の実測や「早期圧縮を避けられたか」の判定に使う）
COMPRESSION_STAGES: frozenset[str] = frozenset(
    {RETRY_STAGE_LIGHT_COMPRESSED, RETRY_STAGE_LAST_RESORT_COMPRESSED}
)
# 発言履歴を広く畳む最後の手段（ここだけが speech history を削る）
HISTORY_FOLDING_STAGES: frozenset[str] = frozenset({RETRY_STAGE_LAST_RESORT_COMPRESSED})


def stage_compresses(stage: object) -> bool:
    """その段階がプロンプト上の表現を実際に短くしたか（公開本文は常に不変）。"""
    return str(stage or "") in COMPRESSION_STAGES


def stage_folds_speech_history(stage: object) -> bool:
    """その段階だけが古い発言本文をID行へ畳む（他段階では履歴は無傷）。"""
    return str(stage or "") in HISTORY_FOLDING_STAGES

# ---------------------------------------------------------------------------
# 失敗理由の分類（応答問題名・転送失敗を「なぜ直らなかったか」へ写す）
# ---------------------------------------------------------------------------

FAILURE_OUTPUT_FORMAT = "output_format"
FAILURE_QUESTION_ANSWER = "question_answer_format"
FAILURE_SEMANTIC_LIGHT = "semantic_light"
FAILURE_EMPTY_OR_THINKING_ONLY = "empty_or_thinking_only"
FAILURE_GENERATION_CAP = "generation_cap"
FAILURE_CONTEXT_OVERFLOW = "context_overflow"
FAILURE_TIMEOUT = "timeout_repeat"
FAILURE_SECRET_LEAK = "secret_leak_repair"
FAILURE_UNSPECIFIED = "unspecified_retry"

FAILURE_CLASSES: tuple[str, ...] = (
    FAILURE_OUTPUT_FORMAT,
    FAILURE_QUESTION_ANSWER,
    FAILURE_SEMANTIC_LIGHT,
    FAILURE_EMPTY_OR_THINKING_ONLY,
    FAILURE_GENERATION_CAP,
    FAILURE_CONTEXT_OVERFLOW,
    FAILURE_TIMEOUT,
    FAILURE_SECRET_LEAK,
    FAILURE_UNSPECIFIED,
)

# 本文は出ているが形だけが崩れている問題（TASK-063 の output_format 側と揃える）
FORMAT_ISSUE_NAMES: frozenset[str] = frozenset(
    {
        "discussion_output_format",
        "discussion_claim_format",
        "marker_missing",
        "template_placeholder_left",
        "wrong_response_language",
        "seer_result_without_claim",
        "wolf_result_not_disclosed",
    }
)
# 直前の質問・自分の過去行動への回答が足りていない問題（証拠が揃えば解ける）
QUESTION_ANSWER_ISSUE_NAMES: frozenset[str] = frozenset(
    {"question_not_answered_first", "past_action_question_unanswered"}
)
# 内容は出るが台帳・時点・名前との突合で弱い問題（軽微な意味的再要求）
LIGHT_SEMANTIC_ISSUE_NAMES: frozenset[str] = frozenset(
    {
        "unknown_player_reference",
        "unverified_game_event",
        "unknown_attacker_claim",
        "unverified_dead_role",
        "dead_player_action_target",
        "dead_player_question",
        "external_game_memory",
        "unverified_past_vote_reference",
        "out_of_phase_action_claim",
        "unverified_attributed_speech",
        "third_person_self_reference",
        "inconsistent_seer_result",
        "inconsistent_role_claim",
        "contradictory_past_vote",
        "recorded_action_denied",
        "vague_wait_and_see",
        "repeated_discussion_topic",
        "repeated_speech",
        "low_novelty",
        "duplicate_claim",
        "unsupported_reason",
    }
)
# 本文が無い失敗のうち、まず thinking と本文予算で直す方
EMPTY_OR_THINKING_ISSUE_NAMES: frozenset[str] = frozenset(
    {"empty_content", "thinking_only_response"}
)


def retry_failure_class(issue: object, *, context_overflow: bool = False) -> str:
    """応答問題名・転送失敗を再生成の失敗クラスへ写す（最初の不足原因を1つに寄せる）。"""
    name = str(issue or "")
    if context_overflow or name == "context_overflow_compacted":
        return FAILURE_CONTEXT_OVERFLOW
    if name in EMPTY_OR_THINKING_ISSUE_NAMES:
        return FAILURE_EMPTY_OR_THINKING_ONLY
    if name == "generation_truncated":
        return FAILURE_GENERATION_CAP
    if name in {"timeout", "transport", "connection_error", "http_error"}:
        return FAILURE_TIMEOUT
    if name in FORMAT_ISSUE_NAMES:
        return FAILURE_OUTPUT_FORMAT
    if name in QUESTION_ANSWER_ISSUE_NAMES:
        return FAILURE_QUESTION_ANSWER
    if not name:
        return FAILURE_UNSPECIFIED
    return FAILURE_SEMANTIC_LIGHT


# ---------------------------------------------------------------------------
# 段階決定（はしご）
# ---------------------------------------------------------------------------

# 同一失敗クラスが2回続いた時点で light_compressed（派生節だけを落とす）へ進む
LIGHT_STAGE_AFTER_SAME_FAILURE_ATTEMPTS = 2
# 短縮しない再要求が3回同じ失敗して初めて、古い発言本文を畳む
LAST_RESORT_AFTER_SAME_FAILURE_ATTEMPTS = 3

# 圧縮引き金の語彙（段階が進んだ理由だけをログへ載せる）
TRIGGER_FORMAT_REPAIR = "format_repair_only"
TRIGGER_QUESTION_ANSWER_REPAIR = "question_answer_repair_only"
TRIGGER_SEMANTIC_REPAIR = "semantic_repair_only"
TRIGGER_BUDGET_ADJUSTMENT = "budget_adjustment_same_evidence"
TRIGGER_BUDGET_RETRY_REPEAT = "budget_retry_repeat_light_compression"
TRIGGER_SAME_FAILURE_SECOND = "same_failure_second_attempt"
TRIGGER_SAME_FAILURE_THIRD = "same_failure_third_attempt"
TRIGGER_CONTEXT_OVERFLOW = "context_overflow_compacted"
TRIGGER_EFFECTIVE_CONTEXT_SHORTAGE = "effective_context_shortage"
TRIGGER_TIMEOUT_REPEAT = "timeout_repeat_short_prompt"
TRIGGER_SECRET_LEAK_REPAIR = "secret_leak_safety_retry"
TRIGGER_POLICY_LEGACY_EARLY = "legacy_early_core_ledger"

# 「圧縮は最終手段」という原則に対する**意図的な例外**（TASK-066レビューで明文化）。
# timeout とコンテキスト超過は、長文をそのまま再送しても待ち時間かエラーで再び失敗する
# ことが実測で確か（TASK-035 / TASK-046）。そのためこの2つに限っては、再生成1回目から
# 最後の手段（core-ledger 短縮＋num_ctx/num_predict の要求サイズ化）を使う。
# 実機A/Bでは、これらの引き金を引いた試行を別建てで評価する（節約と待ち時間のtrade-offは
# 「形式・質問回答で早期圧縮を避けられた率」と分けて見ないと誤判定になる）。
INTENTIONAL_IMMEDIATE_COMPRESSION_TRIGGERS: frozenset[str] = frozenset(
    {
        TRIGGER_TIMEOUT_REPEAT,
        TRIGGER_CONTEXT_OVERFLOW,
        TRIGGER_EFFECTIVE_CONTEXT_SHORTAGE,
    }
)


def retry_trigger_is_immediate_compression(trigger: object) -> bool:
    """即時圧縮を許した意図的な例外に該当するか（timeout・コンテキスト起因のみ）。"""
    return str(trigger or "") in INTENTIONAL_IMMEDIATE_COMPRESSION_TRIGGERS


def retry_trigger_is_timeout(trigger: object, failure_class: object = None) -> bool:
    """この再生成が timeout 起因か（実機A/Bで別集計するための判定）。"""
    name = str(trigger or "")
    if name == TRIGGER_TIMEOUT_REPEAT:
        return True
    if failure_class is not None and str(failure_class) == FAILURE_TIMEOUT:
        return True
    return False


def retry_compression_reason(decision: RetryStageDecision) -> str:
    """TASK-046 の `compression_reason` 語彙へ引き金を写す（圧縮が進んだ理由の分類）。

    timeout とコンテキスト超過は、長文をそのまま再送しても待ち時間・エラーで再び失敗する
    ため**意図的に即時圧縮**している（原則「圧縮は最終手段」の例外）。区別して評価できる
    ように語彙を分ける。
    """
    trigger = decision.compression_trigger
    if trigger == TRIGGER_TIMEOUT_REPEAT or decision.failure_class == FAILURE_TIMEOUT:
        return "timeout_retry"
    if trigger in {TRIGGER_CONTEXT_OVERFLOW, TRIGGER_EFFECTIVE_CONTEXT_SHORTAGE}:
        return "context_overflow_retry"
    if decision.failure_class in {FAILURE_EMPTY_OR_THINKING_ONLY, FAILURE_GENERATION_CAP}:
        # 圧縮そのものは予算失敗の連続で進んだが、壊れている本質は thinking・生成上限の
        # 割り当て。TASK-053の実測語彙（generation_retry）で比較できるままにし、
        # legacy起因の区別は compression_trigger 側（legacy_early_core_ledger）に残す。
        return "generation_retry"
    if trigger == TRIGGER_POLICY_LEGACY_EARLY:
        return "legacy_early_core_ledger"
    return "semantic_retry"


@dataclass(frozen=True)
class RetryStageDecision:
    """1回の再要求で採用する段階と、そこに至った理由・出力予算の調整フラグ。"""

    stage: str
    compression_trigger: str
    failure_class: str
    request_number: int
    # 再生成で適用する出力予算側の調整（プロンプト圧縮とは別物として分離して持つ）
    escalate_generation: bool = False
    thinking_off: bool = False
    retry_budget_fit: bool = False
    reasons: tuple[str, ...] = ()

    @property
    def compresses(self) -> bool:
        return stage_compresses(self.stage)

    @property
    def folds_speech_history(self) -> bool:
        return stage_folds_speech_history(self.stage)

    def as_dict(self) -> dict[str, object]:
        return {
            "retry_stage": self.stage,
            "compression_trigger": self.compression_trigger,
            "retry_failure_class": self.failure_class,
            "retry_request_number": int(self.request_number),
            "retry_escalate_generation": bool(self.escalate_generation),
            "retry_thinking_off": bool(self.thinking_off),
            "retry_budget_fit": bool(self.retry_budget_fit),
            "retry_stage_reasons": list(self.reasons),
            "retry_compresses_prompt": bool(self.compresses),
            "retry_folds_speech_history": bool(self.folds_speech_history),
        }


def same_failure_run(
    decided: Sequence[RetryStageDecision], failure_class: str
) -> int:
    """末尾から続く同一失敗クラスの回数（今回の失敗を足した個数で返す）。"""
    count = 1
    for item in reversed(list(decided)):
        if item.failure_class != failure_class:
            break
        count += 1
    return count


def _prior_stage(decided: Sequence[RetryStageDecision]) -> str | None:
    return decided[-1].stage if decided else None

def decide_retry_stage(
    *,
    failure_class: str,
    policy: object = RETRY_CONTEXT_POLICY_DEFAULT,
    retry_request_number: int = 1,
    decided: Sequence[RetryStageDecision] = (),
    same_failure_count: int | None = None,
    context_overflow: bool = False,
    effective_context_shortage: bool = False,
    timeout_scope_applies: bool = False,
    keeps_round_context: bool = False,
) -> RetryStageDecision:
    """失敗理由・連続失敗回数・コンテキスト実測から段階を決める（決定論的）。

    `retry_request_number` は 1 が最初の再要求（初回生成は対象にしない）。同じ条件
    （同一失敗クラス）での連続失敗回数だけを数え、通常の再要求で解けるうちは証拠を保持する。
    実効割り当て不足やコンテキスト超過は最初から最後の手段へ進む。
    """
    resolved_policy = normalize_retry_context_policy(policy)
    attempt = max(1, int(retry_request_number))
    klass = failure_class if failure_class in FAILURE_CLASSES else FAILURE_UNSPECIFIED
    if context_overflow:
        klass = FAILURE_CONTEXT_OVERFLOW
    repeated = (
        same_failure_run(decided, klass)
        if same_failure_count is None
        else max(1, int(same_failure_count))
    )
    prior = _prior_stage(decided)
    budget_bound = klass in {FAILURE_EMPTY_OR_THINKING_ONLY, FAILURE_GENERATION_CAP}

    if resolved_policy == RETRY_CONTEXT_POLICY_LEGACY_EARLY:
        # A/B基準: TASK-066以前は再生成の種別を問わず core-ledger 短縮プロンプトへ
        # 落ちていた（round_transcript_memory の意味的再要求だけ長文維持だった）。
        keep_long = keeps_round_context and not budget_bound and klass != FAILURE_TIMEOUT
        return RetryStageDecision(
            stage=(
                RETRY_STAGE_FULL_PRESERVED
                if keep_long
                else RETRY_STAGE_LAST_RESORT_COMPRESSED
            ),
            compression_trigger=(
                TRIGGER_FORMAT_REPAIR if keep_long else TRIGGER_POLICY_LEGACY_EARLY
            ),
            failure_class=klass,
            request_number=attempt,
            escalate_generation=budget_bound or klass == FAILURE_TIMEOUT,
            thinking_off=budget_bound or klass == FAILURE_TIMEOUT,
            retry_budget_fit=klass == FAILURE_TIMEOUT,
            reasons=("policy_legacy_early_compression",),
        )

    # 1. コンテキスト超過と実効割り当て不足は、最初から古い発言本文を畳む
    if klass == FAILURE_CONTEXT_OVERFLOW:
        return RetryStageDecision(
            stage=RETRY_STAGE_LAST_RESORT_COMPRESSED,
            compression_trigger=TRIGGER_CONTEXT_OVERFLOW,
            failure_class=klass,
            request_number=attempt,
            thinking_off=True,
            retry_budget_fit=True,
            reasons=("server_reported_context_overflow",),
        )
    if effective_context_shortage:
        return RetryStageDecision(
            stage=RETRY_STAGE_LAST_RESORT_COMPRESSED,
            compression_trigger=TRIGGER_EFFECTIVE_CONTEXT_SHORTAGE,
            failure_class=klass,
            request_number=attempt,
            thinking_off=True,
            retry_budget_fit=True,
            reasons=("prompt_does_not_fit_effective_num_ctx",),
        )
    # 2. タイムアウトの再送に長いプロンプトは戻さない（TASK-035の実測が主因）
    if klass == FAILURE_TIMEOUT:
        return RetryStageDecision(
            stage=RETRY_STAGE_LAST_RESORT_COMPRESSED,
            compression_trigger=TRIGGER_TIMEOUT_REPEAT,
            failure_class=klass,
            request_number=attempt,
            thinking_off=True,
            retry_budget_fit=True,
            reasons=("timeout_resend_costs_minutes",),
        )

    third_strike = (
        repeated >= LAST_RESORT_AFTER_SAME_FAILURE_ATTEMPTS
        or prior == RETRY_STAGE_LAST_RESORT_COMPRESSED
    )
    second_strike = prior == RETRY_STAGE_LIGHT_COMPRESSED or (
        repeated >= LIGHT_STAGE_AFTER_SAME_FAILURE_ATTEMPTS
        and prior != RETRY_STAGE_LAST_RESORT_COMPRESSED
    )

    if third_strike:
        # 同じ条件での連続失敗だけが、発言履歴を畳む理由になる
        return RetryStageDecision(
            stage=RETRY_STAGE_LAST_RESORT_COMPRESSED,
            compression_trigger=(
                TRIGGER_BUDGET_RETRY_REPEAT
                if budget_bound
                else TRIGGER_SAME_FAILURE_THIRD
            ),
            failure_class=klass,
            request_number=attempt,
            escalate_generation=budget_bound,
            thinking_off=True,
            retry_budget_fit=True,
            reasons=("free_reasks_without_shortening_did_not_fix_the_same_failure",),
        )

    if budget_bound:
        if second_strike:
            return RetryStageDecision(
                stage=RETRY_STAGE_LIGHT_COMPRESSED,
                compression_trigger=TRIGGER_BUDGET_RETRY_REPEAT,
                failure_class=klass,
                request_number=attempt,
                escalate_generation=klass == FAILURE_GENERATION_CAP,
                thinking_off=True,
                reasons=("same_budget_failure_twice_drop_only_derived_sections",),
            )
        if timeout_scope_applies:
            # この行動で既に一度待たされているなら、長文再送で再び待たせない
            return RetryStageDecision(
                stage=RETRY_STAGE_LIGHT_COMPRESSED,
                compression_trigger=TRIGGER_TIMEOUT_REPEAT,
                failure_class=klass,
                request_number=attempt,
                escalate_generation=klass == FAILURE_GENERATION_CAP,
                thinking_off=True,
                retry_budget_fit=True,
                reasons=("this_action_already_timed_out_once",),
            )
        # 初回の再要求は同一証拠のまま、thinking と本文予算だけ直す
        return RetryStageDecision(
            stage=RETRY_STAGE_FULL_PRESERVED,
            compression_trigger=TRIGGER_BUDGET_ADJUSTMENT,
            failure_class=klass,
            request_number=attempt,
            escalate_generation=klass == FAILURE_GENERATION_CAP,
            thinking_off=klass == FAILURE_EMPTY_OR_THINKING_ONLY,
            reasons=("first_retry_keeps_everything_and_fixes_output_budget_only",),
        )

    if second_strike:
        return RetryStageDecision(
            stage=RETRY_STAGE_LIGHT_COMPRESSED,
            compression_trigger=TRIGGER_SAME_FAILURE_SECOND,
            failure_class=klass,
            request_number=attempt,
            reasons=("same_failure_still_present_after_full_preserved_retry",),
        )

    # 形式・質問回答・軽微な意味的失敗: まず同一証拠でその問題だけを直す
    trigger = {
        FAILURE_OUTPUT_FORMAT: TRIGGER_FORMAT_REPAIR,
        FAILURE_QUESTION_ANSWER: TRIGGER_QUESTION_ANSWER_REPAIR,
        FAILURE_SECRET_LEAK: TRIGGER_SECRET_LEAK_REPAIR,
    }.get(klass, TRIGGER_SEMANTIC_REPAIR)
    return RetryStageDecision(
        stage=RETRY_STAGE_FULL_PRESERVED,
        compression_trigger=trigger,
        failure_class=klass,
        request_number=attempt,
        reasons=("first_retry_keeps_everything_and_fixes_the_error_only",),
    )


# ---------------------------------------------------------------------------
# light_compressed で落とす低優先の派生節（core は絶対に落とさない）
# ---------------------------------------------------------------------------

# 保持するもの（現在質問・確定事実台帳・役職主張・投票／死亡履歴・直近発言・反証・
# [CONTINUITY_CHECKPOINT]・自分の行動記録と推理メモ・秘匿境界・出力形式）はここに載せない。
# 落とすのは未検証の派生情報と期限付き案内だけ。
RETRY_LIGHT_DROP_CANDIDATES: tuple[str, ...] = (
    "previous_topics",
    "reasoning_guidance",
    "hypothesis_ledger",
    "inference_line",
    "werewolf_theory_card",
    "continuity_memo",
    "integrity_alerts",
    "direct_answer",
    "day1_first_turn_guidance",
    "first_night_guidance",
    "round_transcript",
)

# どの段階でも読み続ける節（ログで「保持された節」の代表として確認する）
RETRY_LIGHT_PROTECTED_SECTIONS: tuple[str, ...] = (
    "fact_ledger",
    "claim_ledger",
    "continuity_checkpoint",
    "role_action_boundary",
    "day_timeline",
    "conversation_evidence",
    "current_task",
    "question_block",
    "self_action_memory",
    "own_speeches",
    "reasoning_memo_own",
    "leak_boundary",
    "behavior_contract",
    "instructions",
    "output_checklist",
)


def retry_light_drop_names() -> tuple[str, ...]:
    """除去候補のうち、実際に optional 登録されている名前だけを返す。"""
    names: list[str] = []
    for name in RETRY_LIGHT_DROP_CANDIDATES:
        try:
            if tier_of(name) == TIER_OPTIONAL:
                names.append(name)
        except ValueError:
            continue
    return tuple(names)


def drop_retry_light_sections(text: str) -> tuple[str, tuple[str, ...], int]:
    """light_compressed の除去を適用する（coreには1文字も触れない）。

    戻り値は (新しいプロンプト, 実際に除いた節名, 減った字数)。未登録名や core 節は
    対象にならないため、現在質問・確定事実・発言本文・状態節は無傷のまま残る。
    """
    wanted = list(retry_light_drop_names())
    if not text or not wanted:
        return text, (), 0
    present = set(find_sections(text))
    targets = [name for name in wanted if name in present]
    if not targets:
        return text, (), 0
    return drop_sections(text, targets)

# ---------------------------------------------------------------------------
# 保持・削除の証拠をログへ出す（公開本文は一字一句変えない）
# ---------------------------------------------------------------------------

# プロンプト中に現れる証拠ID（エンジン発行の参照トークンだけを集める）
_EVIDENCE_ID_PATTERN = re.compile(r"\[(?:speech|claim|seer_claim|ref)[:\s][^\]]{0,24}\]|C\d+\b")
_MAX_LOGGED_IDS = 40


def evidence_ids_in_text(text: object) -> tuple[str, ...]:
    """プロンプト文面から証拠ID参照を順不同・重複なしで拾う（内容評価ではない）。

    テンプレート中の説明プレースホルダ（`[speech:N]` / `[claim:N]` / `[ref:sN]`）は
    実在の発言を指さず案内文に過ぎないため、数字を含むトークンだけを拾う。
    """
    if not isinstance(text, str) or not text:
        return ()
    found: list[str] = []
    for match in _EVIDENCE_ID_PATTERN.findall(text):
        token = re.sub(r"\s+", " ", match.strip())
        if not token or not re.search(r"\d", token):
            continue
        if token not in found:
            found.append(token)
    return tuple(found)


def evidence_id_diff(
    initial_prompt: object, retry_prompt: object
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """初回プロンプトと再生成プロンプトで、保持されたID／落ちたIDを分ける。"""
    initial_ids = set(evidence_ids_in_text(initial_prompt))
    retry_ids = set(evidence_ids_in_text(retry_prompt))
    dropped = sorted(initial_ids - retry_ids)
    preserved = sorted(initial_ids & retry_ids)
    return (
        tuple(preserved[:_MAX_LOGGED_IDS]),
        tuple(dropped[:_MAX_LOGGED_IDS]),
    )


def section_names_in_prompt(text: object) -> tuple[str, ...]:
    """実プロンプトに現れている登録節の名前を列挙する（find_sections の結果）。"""
    if not isinstance(text, str) or not text:
        return ()
    return tuple(find_sections(text).keys())


def section_diff(
    initial_prompt: object, retry_prompt: object
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """再生成で「残った節」と「落ちた節」を台帳名で分ける（本文は不変のままでの比較）。"""
    initial = set(section_names_in_prompt(initial_prompt))
    retry = set(section_names_in_prompt(retry_prompt))
    return tuple(sorted(retry & initial)), tuple(sorted(initial - retry))


# ---------------------------------------------------------------------------
# 削除した証拠IDの安全分類（TASK-066レビュー対応）
# ---------------------------------------------------------------------------
# 「削除ID数」だけでは、圧縮が安全だったかを判断できない。そこで初回プロンプトの
# **どの節に載っていたIDか**で分ける。契約上 core 節（現在質問・確定事実台帳・役職主張・
# 発言証拠・continuity checkpoint 等）に載っていたIDが消えているなら警告の余地、
# optional 派生節にしか無かったIDなら圧縮の想定内、現在質問が引用していたIDなら別扱いにする。
QUESTION_LINKED_SECTION_NAMES: frozenset[str] = frozenset({"question_block"})
_SPEECH_ID_NUMBER = re.compile(
    r"\[\s*(?:speech|ref)\s*[:\s]\s*s?(\d+)\s*\]", re.IGNORECASE
)
_MAX_LOGGED_ID_DETAILS = 12


@dataclass(frozen=True)
class DroppedIdSafety:
    """再生成で消えた証拠IDを「どこに載っていたか」で分類した実測。"""

    dropped_total: int = 0
    protected_drops: tuple[str, ...] = ()
    optional_only_drops: int = 0
    question_linked_drops: tuple[str, ...] = ()
    bodies_still_present: int | None = None
    bodies_lost: int | None = None

    @property
    def clean(self) -> bool:
        """core 節・現在質問からIDが消えていなければ安全（optional 由来のみ）。"""
        return not self.protected_drops and not self.question_linked_drops

    def as_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "dropped_evidence_id_total": int(self.dropped_total),
            "dropped_ids_protected_count": len(self.protected_drops),
            "dropped_ids_optional_only_count": int(self.optional_only_drops),
            "dropped_ids_question_linked_count": len(self.question_linked_drops),
            "dropped_ids_protected_detail": list(self.protected_drops)[:_MAX_LOGGED_ID_DETAILS],
            "dropped_ids_question_linked_detail": list(self.question_linked_drops)[
                :_MAX_LOGGED_ID_DETAILS
            ],
            "dropped_ids_safety_clean": bool(self.clean),
        }
        if self.bodies_still_present is not None:
            data["dropped_speech_bodies_still_present"] = int(self.bodies_still_present)
        if self.bodies_lost is not None:
            data["dropped_speech_bodies_lost"] = int(self.bodies_lost)
        return data


def dropped_evidence_id_safety(
    initial_prompt: object,
    retry_prompt: object,
    messages: Sequence[object] = (),
) -> DroppedIdSafety:
    """初回プロンプトから再生成プロンプトへ消えたIDを、載っていた節で分類する。

    - `protected`: 契約で落とさない core 節に載っていたIDが消えた（圧縮の想定外）
    - `optional_only`: optional 派生節にしか載っていなかったID（`light_compressed`の想定内）
    - `question_linked`: 現在質問（`[CURRENT_QUESTION]`等）が引用していたIDが消えた
    - 発言ID（`[speech:N]` / `[ref:sN]`）は、本文そのものが再生成プロンプトへ残っているかも
      数える（ID行だけ失れても議論が続く場合と、証拠ごと消える場合を区別するため）。
      判定は話者名＋本文冒頭の文字列照合だけで、公開ログの本文は参照するだけに変更しない。
    """
    if not isinstance(initial_prompt, str) or not isinstance(retry_prompt, str):
        return DroppedIdSafety()
    initial_ids = set(evidence_ids_in_text(initial_prompt))
    retry_ids = set(evidence_ids_in_text(retry_prompt))
    lost_tokens = sorted(initial_ids - retry_ids)
    # テンプレート中の説明プレースホルダ（`[speech:N]` / `[ref:sN]`）は証拠ではなく案内文なので、
    # 数字を含むトークンだけを「実在のID参照」として数える（旧プロンプトでも誤計上しない）。
    dropped = [token for token in lost_tokens if re.search(r"\d", token)]
    if not dropped:
        return DroppedIdSafety()

    spans = find_sections(initial_prompt)
    protected: list[str] = []
    question_linked: list[str] = []
    optional_only = 0
    for token in dropped:
        carriers: list[str] = []
        for name, (start, end) in spans.items():
            if token in initial_prompt[start:end]:
                carriers.append(name)
        core_carriers: list[str] = []
        question_carriers: list[str] = []
        for name in carriers:
            try:
                tier_is_core = tier_of(name) == TIER_CORE
            except ValueError:
                tier_is_core = False
            if tier_is_core:
                core_carriers.append(name)
            if name in QUESTION_LINKED_SECTION_NAMES:
                question_carriers.append(name)
        if question_carriers:
            question_linked.append(f"{token}@{question_carriers[0]}")
        if core_carriers:
            protected.append(f"{token}@{core_carriers[0]}")
        elif not carriers:
            # 登録節のスパン外（節マーカー-less 引用）は断定できないので optional 側へ寄せる
            optional_only += 1
        else:
            optional_only += 1

    still_present: int | None = None
    lost: int | None = None
    speech_ids = {
        int(match.group(1))
        for token in dropped
        for match in [_SPEECH_ID_NUMBER.search(token)]
        if match is not None
    }
    if speech_ids and messages:
        by_id: dict[int, object] = {}
        stamp_available = False
        for message in messages:
            value = getattr(message, "speech_id", None)
            if isinstance(value, int) and not isinstance(value, bool):
                by_id[int(value)] = message
                stamp_available = True
        if stamp_available:
            present = 0
            missing = 0
            for speech_id in sorted(speech_ids):
                message = by_id.get(speech_id)
                if message is None:
                    continue
                content = getattr(message, "content", "")
                if not isinstance(content, str):
                    continue
                head = re.sub(r"\s+", " ", content.strip())[:48]
                if head and (head in retry_prompt or head[:24] in retry_prompt):
                    present += 1
                else:
                    missing += 1
            still_present = present
            lost = missing

    return DroppedIdSafety(
        dropped_total=len(dropped),
        protected_drops=tuple(protected),
        optional_only_drops=optional_only,
        question_linked_drops=tuple(question_linked),
        bodies_still_present=still_present,
        bodies_lost=lost,
    )


@dataclass(frozen=True)
class DroppedSpeechStat:
    """発言履歴を畳んだときに、プロンプトから外れた発言の実測。"""

    dropped_count: int = 0
    dropped_chars: int = 0
    oldest_day_turn: str | None = None
    newest_day_turn: str | None = None
    oldest_speech_id: int | None = None
    newest_speech_id: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "dropped_speech_count": int(self.dropped_count),
            "dropped_speech_chars": int(self.dropped_chars),
            "dropped_speech_oldest_point": self.oldest_day_turn,
            "dropped_speech_newest_point": self.newest_day_turn,
            "dropped_speech_oldest_id": self.oldest_speech_id,
            "dropped_speech_newest_id": self.newest_speech_id,
        }


def _point_label(day: object, turn: object) -> str:
    try:
        return f"Day {int(day)} Turn {int(turn)}"
    except (TypeError, ValueError):
        return ""


def dropped_speech_stats(
    messages: Sequence[object], retry_prompt: object
) -> DroppedSpeechStat:
    """再生成プロンプトに本文が載らなかった発言を数え、時点と文字数を残す。

    判定は「話者名と本文の冒頭がプロンプトに現れるか」だけで、公開ログの本文は
    参照するだけ（変更しない）。IDが未取得の発言にも時点ラベルを付けられる。
    """
    text = retry_prompt if isinstance(retry_prompt, str) else ""
    dropped_count = 0
    dropped_chars = 0
    oldest: tuple[int, int, int | None] | None = None
    newest: tuple[int, int, int | None] | None = None
    for message in messages:
        speaker = str(getattr(message, "speaker", "") or "")
        content = getattr(message, "content", "")
        if not isinstance(content, str) or not speaker:
            continue
        head = re.sub(r"\s+", " ", content.strip())[:48]
        if not head:
            continue
        if speaker in text and (head in text or head[:24] in text):
            continue
        dropped_count += 1
        dropped_chars += len(content)
        try:
            day = int(getattr(message, "day", 0) or 0)
            turn = int(getattr(message, "turn", 0) or 0)
        except (TypeError, ValueError):
            day, turn = 0, 0
        speech_id_value = getattr(message, "speech_id", None)
        speech_id = (
            int(speech_id_value)
            if isinstance(speech_id_value, int) and not isinstance(speech_id_value, bool)
            else None
        )
        key = (day, turn, speech_id if speech_id is not None else 0)
        if oldest is None or key < oldest:
            oldest = key
        if newest is None or key > newest:
            newest = key
    return DroppedSpeechStat(
        dropped_count=dropped_count,
        dropped_chars=dropped_chars,
        oldest_day_turn=_point_label(*oldest[:2]) if oldest else None,
        newest_day_turn=_point_label(*newest[:2]) if newest else None,
        oldest_speech_id=(
            oldest[2] if oldest and oldest[2] else (None if not oldest else None)
        ),
        newest_speech_id=(
            newest[2] if newest and newest[2] else (None if not newest else None)
        ),
    )

# ---------------------------------------------------------------------------
# 再生成（短縮）プロンプトの節語彙と配置コストモデル（LL不使用・決定的）
# ---------------------------------------------------------------------------

# last_resort_compressed の短縮プロンプトは、この「再生成専用のブロック」で組み立てる。
# 通常生成の節順（PROMPT_LAYOUT_V5_ORDER）はそのまま維持し、再生成プロンプトだけ最適化する。
RETRY_BLOCK_OPENING_NOTE = "retry_opening_note"
RETRY_BLOCK_STATE_CHECKPOINT = "retry_state_checkpoint"
RETRY_BLOCK_CONFIRMED_FACTS = "retry_confirmed_facts"
RETRY_BLOCK_THEORY_LINE = "retry_theory_line"
RETRY_BLOCK_PUBLIC_CLAIMS = "retry_public_claims"
RETRY_BLOCK_RECENT_SPEECHES = "retry_recent_speeches"
RETRY_BLOCK_UNANSWERED_QUESTION = "retry_unanswered_question"
RETRY_BLOCK_COUNTER_ARGUMENTS = "retry_counter_arguments"
RETRY_BLOCK_INFORMATION_RULES = "retry_information_rules"
RETRY_BLOCK_PRIVATE_SEER_RESULTS = "retry_private_seer_results"
RETRY_BLOCK_CORRECTION_GUIDANCE = "retry_correction_guidance"
RETRY_BLOCK_OUTPUT_FORMAT = "retry_output_format"

RETRY_PROMPT_BLOCKS: tuple[str, ...] = (
    RETRY_BLOCK_OPENING_NOTE,
    RETRY_BLOCK_STATE_CHECKPOINT,
    RETRY_BLOCK_CONFIRMED_FACTS,
    RETRY_BLOCK_THEORY_LINE,
    RETRY_BLOCK_PUBLIC_CLAIMS,
    RETRY_BLOCK_RECENT_SPEECHES,
    RETRY_BLOCK_UNANSWERED_QUESTION,
    RETRY_BLOCK_COUNTER_ARGUMENTS,
    RETRY_BLOCK_INFORMATION_RULES,
    RETRY_BLOCK_PRIVATE_SEER_RESULTS,
    RETRY_BLOCK_CORRECTION_GUIDANCE,
    RETRY_BLOCK_OUTPUT_FORMAT,
)

# ブロックの役割（何を「いま」使うか）。重みが高いほど配置の恩恵が大きい。
RETRY_ROLE_FRAME_NOTE = "retry_frame_note"
RETRY_ROLE_ENGINE_STATE = "retry_engine_state"
RETRY_ROLE_ENGINE_FACT = "retry_engine_fact"
RETRY_ROLE_HEURISTIC = "retry_heuristic"
RETRY_ROLE_CLAIM_INDEX = "retry_claim_index"
RETRY_ROLE_SPEECH_EVIDENCE = "retry_speech_evidence"
RETRY_ROLE_CURRENT_QUESTION = "retry_current_question"
RETRY_ROLE_DISCIPLINE = "retry_discipline"
RETRY_ROLE_ENGINE_PRIVATE = "retry_engine_private"
RETRY_ROLE_CORRECTION = "retry_correction"
RETRY_ROLE_OUTPUT_FORMAT = "retry_output_format"

RETRY_BLOCK_ROLES: dict[str, str] = {
    RETRY_BLOCK_OPENING_NOTE: RETRY_ROLE_FRAME_NOTE,
    RETRY_BLOCK_STATE_CHECKPOINT: RETRY_ROLE_ENGINE_STATE,
    RETRY_BLOCK_CONFIRMED_FACTS: RETRY_ROLE_ENGINE_FACT,
    RETRY_BLOCK_THEORY_LINE: RETRY_ROLE_HEURISTIC,
    RETRY_BLOCK_PUBLIC_CLAIMS: RETRY_ROLE_CLAIM_INDEX,
    RETRY_BLOCK_RECENT_SPEECHES: RETRY_ROLE_SPEECH_EVIDENCE,
    RETRY_BLOCK_UNANSWERED_QUESTION: RETRY_ROLE_CURRENT_QUESTION,
    RETRY_BLOCK_COUNTER_ARGUMENTS: RETRY_ROLE_SPEECH_EVIDENCE,
    RETRY_BLOCK_INFORMATION_RULES: RETRY_ROLE_DISCIPLINE,
    RETRY_BLOCK_PRIVATE_SEER_RESULTS: RETRY_ROLE_ENGINE_PRIVATE,
    RETRY_BLOCK_CORRECTION_GUIDANCE: RETRY_ROLE_CORRECTION,
    RETRY_BLOCK_OUTPUT_FORMAT: RETRY_ROLE_OUTPUT_FORMAT,
}

# 重みは prompt_layout と同じ思想（状態・質問・修正・形式は高く、未検証の派生
# （セオリー行）は低く置く＝再生成でも議論の読み枠を先に決めない）。
RETRY_BLOCK_WEIGHTS: dict[str, float] = {
    RETRY_ROLE_FRAME_NOTE: 0.60,
    RETRY_ROLE_ENGINE_STATE: 1.25,
    RETRY_ROLE_ENGINE_FACT: 1.10,
    RETRY_ROLE_HEURISTIC: 0.55,
    RETRY_ROLE_CLAIM_INDEX: 1.00,
    RETRY_ROLE_SPEECH_EVIDENCE: 1.30,
    RETRY_ROLE_CURRENT_QUESTION: 1.45,
    RETRY_ROLE_DISCIPLINE: 1.15,
    RETRY_ROLE_ENGINE_PRIVATE: 1.05,
    RETRY_ROLE_CORRECTION: 1.50,
    RETRY_ROLE_OUTPUT_FORMAT: 1.40,
}

# 「いま使う」ブロックが中部帯に沈むことを減点する集合（記憶減衰への対策側）
RETRY_MUST_USE_ROLES: frozenset[str] = frozenset(
    {
        RETRY_ROLE_ENGINE_STATE,
        RETRY_ROLE_ENGINE_FACT,
        RETRY_ROLE_SPEECH_EVIDENCE,
        RETRY_ROLE_CURRENT_QUESTION,
        RETRY_ROLE_CORRECTION,
        RETRY_ROLE_OUTPUT_FORMAT,
        RETRY_ROLE_DISCIPLINE,
    }
)
# 先頭帯を占めると「読み枠を先に決める」側（中部帯へ寄せたいブロック）
RETRY_NON_ANCHOR_ROLES: frozenset[str] = frozenset(
    {RETRY_ROLE_HEURISTIC, RETRY_ROLE_FRAME_NOTE, RETRY_ROLE_CLAIM_INDEX}
)


@dataclass(frozen=True)
class RetryAdjacencyTarget:
    """隣接配置を測る2ブロックとペナルティ係数（距離/全体字数 × weight × 基本係数）。"""

    left: str
    right: str
    weight: float
    reason: str


RETRY_ADJACENCY_TARGETS: tuple[RetryAdjacencyTarget, ...] = (
    RetryAdjacencyTarget(
        RETRY_BLOCK_UNANSWERED_QUESTION,
        RETRY_BLOCK_CORRECTION_GUIDANCE,
        1.35,
        "現在質問の直後で修正指示を読むと、質問に答えたまま形式だけ直せる",
    ),
    RetryAdjacencyTarget(
        RETRY_BLOCK_RECENT_SPEECHES,
        RETRY_BLOCK_UNANSWERED_QUESTION,
        1.20,
        "直前の議論の直後に質問を載せると、引用の所在を辿って回答できる",
    ),
    RetryAdjacencyTarget(
        RETRY_BLOCK_STATE_CHECKPOINT,
        RETRY_BLOCK_PUBLIC_CLAIMS,
        1.00,
        "チェックポイントのID（[claim:N]）を実主張行の隣で解決させる",
    ),
    RetryAdjacencyTarget(
        RETRY_BLOCK_COUNTER_ARGUMENTS,
        RETRY_BLOCK_RECENT_SPEECHES,
        0.95,
        "反論を直近発言から離すと、再生成文が直前の議論と接続しにくい",
    ),
    RetryAdjacencyTarget(
        RETRY_BLOCK_CORRECTION_GUIDANCE,
        RETRY_BLOCK_OUTPUT_FORMAT,
        1.40,
        "何を書くか（修正点）とどう書くか（出力形式）をまとめて終端で読む",
    ),
)

# ハード制約：自分が誰で何日目かは即座に分ける（一貫性）
RETRY_FACTS_MAX_INDEX = 2
# ハード制約：エンジン正規化の論理状態（チェックポイント）は冒頭3ブロック以内
RETRY_CHECKPOINT_MAX_INDEX = 2
# ハード制約：未検証のセオリー行（誘導系）を先頭の2ブロックへ置かない
RETRY_THEORY_MIN_INDEX = 2

def validate_retry_order(order: Sequence[str]) -> list[dict[str, object]]:
    """宣言された再生成節順がハード制約を満たすかを静的検査する（字数不要）。"""
    names = [str(name) for name in order]
    problems: list[dict[str, object]] = []
    if not names:
        return [{"kind": "empty_order", "name": ""}]
    for name in sorted({n for n in names if names.count(n) > 1}):
        problems.append({"kind": "duplicate_block", "name": name})
    for name in [n for n in names if n not in RETRY_PROMPT_BLOCKS]:
        problems.append({"kind": "unknown_block", "name": name})
    if RETRY_BLOCK_CONFIRMED_FACTS not in names:
        problems.append({"kind": "facts_missing", "name": RETRY_BLOCK_CONFIRMED_FACTS})
    elif names.index(RETRY_BLOCK_CONFIRMED_FACTS) > RETRY_FACTS_MAX_INDEX:
        problems.append({"kind": "facts_too_late", "name": RETRY_BLOCK_CONFIRMED_FACTS})
    correction_index = (
        names.index(RETRY_BLOCK_CORRECTION_GUIDANCE)
        if RETRY_BLOCK_CORRECTION_GUIDANCE in names
        else None
    )
    format_index = (
        names.index(RETRY_BLOCK_OUTPUT_FORMAT)
        if RETRY_BLOCK_OUTPUT_FORMAT in names
        else None
    )
    if format_index is not None and names[-1] != RETRY_BLOCK_OUTPUT_FORMAT:
        problems.append({"kind": "output_format_not_last", "name": names[-1]})
    private_index = (
        names.index(RETRY_BLOCK_PRIVATE_SEER_RESULTS)
        if RETRY_BLOCK_PRIVATE_SEER_RESULTS in names
        else None
    )
    if private_index is not None:
        # 判定語を文脈なしで先頭に読むと、未公開結果をそのまま口にする余地が残る
        if private_index == 0:
            problems.append(
                {
                    "kind": "private_results_first",
                    "name": RETRY_BLOCK_PRIVATE_SEER_RESULTS,
                }
            )
        # 生成直前に判定語を置かない（TASK-058の終端回避方針を踏襲）
        if correction_index is not None and private_index > correction_index:
            problems.append(
                {
                    "kind": "private_results_after_correction",
                    "name": RETRY_BLOCK_PRIVATE_SEER_RESULTS,
                }
            )
    if correction_index is not None:
        for name in (RETRY_BLOCK_RECENT_SPEECHES, RETRY_BLOCK_COUNTER_ARGUMENTS):
            if name in names and names.index(name) > correction_index:
                problems.append({"kind": "evidence_after_correction", "name": name})
        if (
            RETRY_BLOCK_UNANSWERED_QUESTION in names
            and names.index(RETRY_BLOCK_UNANSWERED_QUESTION) > correction_index
        ):
            problems.append(
                {
                    "kind": "question_after_correction",
                    "name": RETRY_BLOCK_UNANSWERED_QUESTION,
                }
            )
    # 追加の順序制約（一貫性のため）: 自分の身分・確定事実より前に発言証拠を置かない。
    # 再生成文が「自分が誰か」を読めないまま他人の発言から語り出さないためで、口調・論点には
    # 触れない制約である。
    facts_index_for_order = (
        names.index(RETRY_BLOCK_CONFIRMED_FACTS)
        if RETRY_BLOCK_CONFIRMED_FACTS in names
        else None
    )
    if facts_index_for_order is not None:
        for name in (RETRY_BLOCK_RECENT_SPEECHES, RETRY_BLOCK_UNANSWERED_QUESTION):
            if name in names and names.index(name) < facts_index_for_order:
                problems.append({"kind": "evidence_before_facts", "name": name})
    # チェックポイントのID（[claim:N]/[speech:N]）は実主張行より前に置いて解決させる
    checkpoint_index = (
        names.index(RETRY_BLOCK_STATE_CHECKPOINT)
        if RETRY_BLOCK_STATE_CHECKPOINT in names
        else None
    )
    claims_index = (
        names.index(RETRY_BLOCK_PUBLIC_CLAIMS)
        if RETRY_BLOCK_PUBLIC_CLAIMS in names
        else None
    )
    if (
        checkpoint_index is not None
        and claims_index is not None
        and checkpoint_index > claims_index
    ):
        problems.append(
            {"kind": "checkpoint_after_claims", "name": RETRY_BLOCK_STATE_CHECKPOINT}
        )
    # チェックポイント（エンジン正規化の論理状態）は冒頭3ブロック以内に置く: 現在質問へ答える
    # 前に「今どこまで確定しているか」を読ませるためで、発言の内容・言い方は変えない制約。
    if checkpoint_index is not None and checkpoint_index > RETRY_CHECKPOINT_MAX_INDEX:
        problems.append(
            {"kind": "checkpoint_too_late", "name": RETRY_BLOCK_STATE_CHECKPOINT}
        )
    # 未検証のセオリー行を最初の2ブロックへ置かない: 再生成でも「読み枠」を誘導しないため
    # （TASK-060の方針と同じ趣向）。口調・論点・結論そのものには触れない。
    theory_index = (
        names.index(RETRY_BLOCK_THEORY_LINE)
        if RETRY_BLOCK_THEORY_LINE in names
        else None
    )
    if theory_index is not None and theory_index < RETRY_THEORY_MIN_INDEX:
        problems.append({"kind": "theory_in_leading_band", "name": RETRY_BLOCK_THEORY_LINE})
    return problems


def retry_role_of(name: object) -> str:
    """ブロックの役割（未登録名は heuristic として保守的に扱う＝先頭を任せない）。"""
    return RETRY_BLOCK_ROLES.get(str(name), RETRY_ROLE_HEURISTIC)


def retry_weight_of(name: object) -> float:
    return RETRY_BLOCK_WEIGHTS.get(retry_role_of(name), 1.0)


def _span_gap(spans: Mapping[str, tuple[int, int]], left: str, right: str) -> int:
    """2スパンの隙間（文字数）。重なり・逆順は 0 として扱う。"""
    left_span = spans.get(left)
    right_span = spans.get(right)
    if left_span is None or right_span is None:
        return 0
    first, second = sorted((left_span, right_span), key=lambda span: int(span[0]))
    return max(0, int(second[0]) - int(first[1]))


def _middle_overlap(start: int, end: int, total: int) -> int:
    lo, hi = middle_band_bounds(total)
    return max(0, int(min(end, hi) - max(start, lo)))

@dataclass
class RetryOrderScore:
    """再生成プロンプトの節順を測る構造的スコア（内容評価ではなく配置だけの代理指標）。"""

    order: tuple[str, ...]
    score: float
    attention_efficiency: float = 0.0
    adjacency_penalty: float = 0.0
    weak_middle_penalty: float = 0.0
    primacy_clutter_penalty: float = 0.0
    dilution_penalty: float = 0.0
    violations: tuple[dict[str, object], ...] = ()
    zone_chars: dict[str, int] = field(default_factory=dict)
    gaps: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "retry_layout_order": list(self.order),
            "retry_layout_score": round(float(self.score), 2),
            "retry_attention_efficiency": round(float(self.attention_efficiency), 4),
            "retry_adjacency_penalty": round(float(self.adjacency_penalty), 2),
            "retry_weak_middle_must_use_penalty": round(float(self.weak_middle_penalty), 2),
            "retry_primacy_clutter_penalty": round(float(self.primacy_clutter_penalty), 2),
            "retry_post_format_dilution_penalty": round(float(self.dilution_penalty), 2),
            "retry_hard_violations": [dict(item) for item in self.violations],
            "retry_simulated_zone_chars": dict(self.zone_chars),
            "retry_simulated_adjacency_gap_fraction": dict(self.gaps),
        }


def score_retry_order(
    order: Sequence[str],
    block_chars: Mapping[str, int],
) -> RetryOrderScore:
    """節順とブロック別字数から再生成プロンプトの配置を採点する（実LL不要の模擬）。

    式は prompt_layout のコストモデルと同じ形（attention_prior × role_weight の利得から、
    隣接距離・中部沈み・先頭帯の誘導・形式仕様より後ろの希釈を引く）。この値は実機LLMで
    校正した数値ではなく、同じ内容のまま配置だけを比べるための透明な指標として使う。
    """
    names = [str(name) for name in order]
    sizes = {name: max(0, int(block_chars.get(name, 0) or 0)) for name in names}
    violations = validate_retry_order(names)
    spans = spans_from_order(names, sizes)
    total = sum(end - start for start, end in spans.values())
    if total <= 0:
        return RetryOrderScore(order=tuple(names), score=0.0, violations=tuple(violations))

    gain = 0.0
    zone_chars: dict[str, int] = {zone: 0 for zone in ATTENTION_ZONES}
    weak_middle = 0
    primacy_clutter = 0
    for name, (start, end) in spans.items():
        center = (start + max(0, end - start) / 2.0) / float(total)
        role = retry_role_of(name)
        weight = RETRY_BLOCK_WEIGHTS.get(role, 1.0)
        gain += (end - start) * weight * attention_prior_span(start, end, total)
        zone = attention_zone_for_center(center)
        zone_chars[zone] += end - start
        if role in RETRY_MUST_USE_ROLES:
            weak_middle += _middle_overlap(start, end, total)
        elif role in RETRY_NON_ANCHOR_ROLES and zone == "primacy":
            primacy_clutter += end - start
    efficiency = gain / float(total)

    adjacency_penalty = 0.0
    gaps: dict[str, float] = {}
    for target in RETRY_ADJACENCY_TARGETS:
        fraction = _span_gap(spans, target.left, target.right) / float(total)
        gaps[f"{target.left}->{target.right}"] = round(fraction, 4)
        adjacency_penalty += fraction * ADJACENCY_PENALTY_PER_FRACTION * target.weight

    dilution_penalty = 0.0
    format_span = spans.get(RETRY_BLOCK_OUTPUT_FORMAT)
    if format_span is not None:
        after_chars = sum(
            end - start
            for name, (start, end) in spans.items()
            if start >= format_span[1]
            and retry_role_of(name) != RETRY_ROLE_OUTPUT_FORMAT
        )
        dilution_penalty = (after_chars / float(total)) * POST_FORMAT_DILUTION_WEIGHT

    weak_middle_penalty = (weak_middle / float(total)) * WEAK_MIDDLE_MUST_USE_WEIGHT
    clutter_penalty = (primacy_clutter / float(total)) * PRIMACY_CLUTTER_WEIGHT
    score = (
        efficiency * SCORE_SCALE
        - adjacency_penalty
        - dilution_penalty
        - weak_middle_penalty
        - clutter_penalty
        - HARD_VIOLATION_PENALTY * len(violations)
    )
    return RetryOrderScore(
        order=tuple(names),
        score=round(float(score), 2),
        attention_efficiency=round(float(efficiency), 4),
        adjacency_penalty=round(float(adjacency_penalty), 2),
        weak_middle_penalty=round(float(weak_middle_penalty), 2),
        primacy_clutter_penalty=round(float(clutter_penalty), 2),
        dilution_penalty=round(float(dilution_penalty), 2),
        violations=tuple(violations),
        zone_chars=dict(zone_chars),
        gaps=dict(gaps),
    )

# TASK-066以前の短縮プロンプトが実際に取っていた並び（A/B比較の基準側）
RETRY_PROMPT_ORDER_LEGACY: tuple[str, ...] = (
    RETRY_BLOCK_OPENING_NOTE,
    RETRY_BLOCK_STATE_CHECKPOINT,
    RETRY_BLOCK_CONFIRMED_FACTS,
    RETRY_BLOCK_THEORY_LINE,
    RETRY_BLOCK_PUBLIC_CLAIMS,
    RETRY_BLOCK_RECENT_SPEECHES,
    RETRY_BLOCK_UNANSWERED_QUESTION,
    RETRY_BLOCK_COUNTER_ARGUMENTS,
    RETRY_BLOCK_INFORMATION_RULES,
    RETRY_BLOCK_PRIVATE_SEER_RESULTS,
    RETRY_BLOCK_CORRECTION_GUIDANCE,
    RETRY_BLOCK_OUTPUT_FORMAT,
)

# 代表ブロック字数。`_analysis/task066_retry_order_probe.py` がモック状態（Day1＝発言のみ／
# Day2＝checkpointあり／Day3＝台帳・投票あり）で `_short_discussion_retry_prompt` の実本文から
# 測った3シナリオ平均（2026-11計測）。実際の台帳量が増えれば発言・質問ブロックはさらに伸びるが、
# 相対関係（状態と確定事実が太く、現在質問と直近発言が「いま」使う情報）は変わらない。
RETRY_REPRESENTATIVE_BLOCK_CHARS: dict[str, int] = {
    RETRY_BLOCK_OPENING_NOTE: 106,
    RETRY_BLOCK_STATE_CHECKPOINT: 127,
    RETRY_BLOCK_CONFIRMED_FACTS: 557,
    RETRY_BLOCK_THEORY_LINE: 306,
    RETRY_BLOCK_PUBLIC_CLAIMS: 72,
    RETRY_BLOCK_RECENT_SPEECHES: 215,
    RETRY_BLOCK_UNANSWERED_QUESTION: 84,
    RETRY_BLOCK_COUNTER_ARGUMENTS: 89,
    RETRY_BLOCK_INFORMATION_RULES: 312,
    RETRY_BLOCK_PRIVATE_SEER_RESULTS: 54,
    RETRY_BLOCK_CORRECTION_GUIDANCE: 217,
    RETRY_BLOCK_OUTPUT_FORMAT: 411,
}


def _order_index_key(order: Sequence[str]) -> tuple[int, ...]:
    return tuple(RETRY_PROMPT_BLOCKS.index(str(name)) for name in order)


def retry_order_variants(base: Sequence[str]) -> list[tuple[str, ...]]:
    """単一ブロック移動と2ブロック入替を1回適用した候補を決定的順で並べる。"""
    names = [str(name) for name in base]
    variants: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = {tuple(names)}
    for source in range(len(names)):
        for target in range(len(names)):
            if source == target:
                continue
            moved = list(names)
            item = moved.pop(source)
            moved.insert(target, item)
            candidate = tuple(moved)
            if candidate not in seen:
                seen.add(candidate)
                variants.append(candidate)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            swapped = list(names)
            swapped[left], swapped[right] = swapped[right], swapped[left]
            candidate = tuple(swapped)
            if candidate not in seen:
                seen.add(candidate)
                variants.append(candidate)
    return variants


def search_retry_orders(
    block_chars: Mapping[str, int] | None = None,
    *,
    start: Sequence[str] = RETRY_PROMPT_ORDER_LEGACY,
    top_k: int = 3,
    max_rounds: int = 24,
) -> list[dict[str, object]]:
    """代表字数で貪欲探索し、制約を満たす最良の再生成節順を上位 k 件返す（決定的）。

    反復は「単一ブロック移動」と「2ブロック入替」だけで、同じ出発順・同じ字数なら
    同じ結果になる（LLを呼ばない）。ハード制約を満たさない候補は採らない。
    """
    chars = dict(RETRY_REPRESENTATIVE_BLOCK_CHARS if block_chars is None else block_chars)
    present = [
        str(name)
        for name in (list(start) or list(RETRY_PROMPT_BLOCKS))
        if int(chars.get(str(name), 0) or 0) > 0
    ]
    missing = [
        name
        for name in RETRY_PROMPT_BLOCKS
        if name not in present and int(chars.get(name, 0) or 0) > 0
    ]
    current = tuple(present + missing)
    best = score_retry_order(current, chars)
    history: dict[tuple[str, ...], RetryOrderScore] = {current: best}
    for _ in range(max(1, int(max_rounds))):
        improved = False
        candidates: list[RetryOrderScore] = []
        for variant in retry_order_variants(best.order):
            if variant in history:
                continue
            scored = score_retry_order(variant, chars)
            history[variant] = scored
            if scored.violations:
                continue
            candidates.append(scored)
        if not candidates:
            break
        ranked = sorted(
            candidates,
            key=lambda item: (-float(item.score), _order_index_key(item.order)),
        )
        top = ranked[0]
        if float(top.score) > float(best.score) + 1e-9:
            best = top
            improved = True
        if not improved:
            break
    legal = [item for item in history.values() if not item.violations]
    ranked_all = sorted(
        legal,
        key=lambda item: (-float(item.score), _order_index_key(item.order)),
    )
    return [item.as_dict() for item in ranked_all[: max(1, int(top_k))]]


def retry_adopted_order_score(
    block_chars: Mapping[str, int] | None = None,
) -> RetryOrderScore:
    """現行既定（凍結順）を評価する。"""
    chars = dict(RETRY_REPRESENTATIVE_BLOCK_CHARS if block_chars is None else block_chars)
    present = tuple(name for name in RETRY_PROMPT_ORDER_V1 if int(chars.get(name, 0) or 0) > 0)
    return score_retry_order(present, chars)


def legacy_retry_order_score(
    block_chars: Mapping[str, int] | None = None,
) -> RetryOrderScore:
    """A/B基準（TASK-066以前の手順配置）を同じコストモデルで評価する。"""
    chars = dict(RETRY_REPRESENTATIVE_BLOCK_CHARS if block_chars is None else block_chars)
    present = tuple(name for name in RETRY_PROMPT_ORDER_LEGACY if int(chars.get(name, 0) or 0) > 0)
    return score_retry_order(present, chars)

# ---------------------------------------------------------------------------
# 凍結した再生成節順（TASK-066）と組み立て
# ---------------------------------------------------------------------------

# 実測ブロック字数での貪欲探索（単一ブロック移動＋2ブロック入替・決定的）が最良とした配置。
# スコアは legacy（TASK-066以前）で -20.0、この順で 47.56（同じ内容のまま配置だけ変更）。
# ・冒頭: 確定事実（自分が誰か・何日目か）→ 再生成である旨 → エンジンの論理状態
#   → 公開CO・判定行。誘導系（未検証のセオリー行）は先頭2ブロックに置かない。
# ・終端: 反論 → 直近発言 → 本人への未回答質問 → 修正指示 → 出力形式。
#   「いま使う」情報（現在質問・修正点・出力形式）を recency 側へ集約し、隣接距離0にする。
# ハード制約（現在質問・確定事実・役職主張・投票／死亡台帳・直近発言・反証・checkpoint を
# 落とさない／身分より前の証拠禁止／生成直前に判定語を置かない／形式仕様を末尾に置く）は
# `validate_retry_order` で静的検査し、プローブでも violation=0 を確認している。
RETRY_PROMPT_ORDER_V1: tuple[str, ...] = (
    RETRY_BLOCK_CONFIRMED_FACTS,
    RETRY_BLOCK_OPENING_NOTE,
    RETRY_BLOCK_STATE_CHECKPOINT,
    RETRY_BLOCK_PUBLIC_CLAIMS,
    RETRY_BLOCK_THEORY_LINE,
    RETRY_BLOCK_INFORMATION_RULES,
    RETRY_BLOCK_PRIVATE_SEER_RESULTS,
    RETRY_BLOCK_COUNTER_ARGUMENTS,
    RETRY_BLOCK_RECENT_SPEECHES,
    RETRY_BLOCK_UNANSWERED_QUESTION,
    RETRY_BLOCK_CORRECTION_GUIDANCE,
    RETRY_BLOCK_OUTPUT_FORMAT,
)


def resolve_retry_prompt_order(policy: object) -> tuple[str, ...]:
    """方策ラベルから実際に使う再生成節順を決める（legacy は従来配置でA/B基準）。"""
    resolved = normalize_retry_context_policy(policy)
    if resolved == RETRY_CONTEXT_POLICY_LEGACY_EARLY:
        return RETRY_PROMPT_ORDER_LEGACY
    return RETRY_PROMPT_ORDER_V1


def assemble_retry_prompt(
    blocks: Mapping[str, object],
    order: Sequence[str] | None = None,
) -> tuple[str, tuple[str, ...]]:
    """ブロック辞書を与えられた節順で1つのプロンプトへ組み立てる（決定論的）。

    空のブロックは載せない。宣言に無いキーは内容を失わないよう末尾側へ足すが、出力形式仕様
    （`retry_output_format`）が後ろに押されないように必ず最後に置く（ハード制約と同じ趣向で、
    「どう書くか」を生成直前に読ませるため）。返す値は (組み立てたテキスト, 実際に載った節順)。
    """
    declared = list(order if order is not None else RETRY_PROMPT_ORDER_V1)
    known = [name for name in declared if name in blocks]
    extra = [name for name in RETRY_PROMPT_BLOCKS if name in blocks and name not in known]
    unknown = sorted(str(name) for name in blocks if name not in RETRY_PROMPT_BLOCKS)
    candidates = known + extra + unknown
    # 出力形式仕様が末尾からずれないよう、存在する場合は常に最後へ寄せる
    names_in_order = [name for name in candidates if name != RETRY_BLOCK_OUTPUT_FORMAT]
    if RETRY_BLOCK_OUTPUT_FORMAT in candidates:
        names_in_order.append(RETRY_BLOCK_OUTPUT_FORMAT)
    parts: list[str] = []
    included: list[str] = []
    for name in names_in_order:
        raw = blocks.get(name)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        parts.append(text if text.endswith("\n") else f"{text}\n")
        included.append(str(name))
    return "".join(parts), tuple(included)


def retry_stage_log_fields(
    *,
    policy: object,
    decision: RetryStageDecision,
    initial_prompt_text: object = None,
    retry_prompt_text: object = None,
    initial_prompt_chars: int | None = None,
    retry_prompt_chars: int | None = None,
    messages: Sequence[object] = (),
    previous_body_chars: int | None = None,
    new_body_chars: int | None = None,
    action: str | None = None,
) -> dict[str, object]:
    """再生成1件ぶんの診断（保持・削除した証拠と節、本文長の変化）を組み立てる。

    本文そのものは載せない。公開ログの本文は一字一句不変であり、診断へは長さ・時点・IDだけを出す。
    """
    fields: dict[str, object] = {
        "retry_context_policy": retry_context_policy_label(policy),
        **decision.as_dict(),
    }
    # 「圧縮は最終手段」に対する意図的な例外（timeout・コンテキスト超過）かを明記する。
    # これが無いと、待ち時間優先の即時圧縮が原則違反に見える（TASK-066レビュー対応）。
    fields["retry_immediate_compression_exception"] = retry_trigger_is_immediate_compression(
        decision.compression_trigger
    )
    fields["retry_timeout_caused"] = retry_trigger_is_timeout(
        decision.compression_trigger, decision.failure_class
    )
    if action:
        fields["retry_action"] = str(action)
    initial_chars = (
        int(initial_prompt_chars)
        if initial_prompt_chars is not None
        else (len(initial_prompt_text) if isinstance(initial_prompt_text, str) else 0)
    )
    retry_chars = (
        int(retry_prompt_chars)
        if retry_prompt_chars is not None
        else (len(retry_prompt_text) if isinstance(retry_prompt_text, str) else 0)
    )
    fields["initial_user_prompt_char_count"] = int(initial_chars)
    fields["retry_user_prompt_char_count"] = int(retry_chars)
    if initial_chars > 0:
        fields["retry_prompt_saved_chars"] = int(max(0, initial_chars - retry_chars))
        fields["retry_prompt_ratio"] = round(retry_chars / float(initial_chars), 4)
    if isinstance(initial_prompt_text, str) and isinstance(retry_prompt_text, str):
        preserved_sections, dropped_sections = section_diff(
            initial_prompt_text, retry_prompt_text
        )
        preserved_ids, dropped_ids = evidence_id_diff(initial_prompt_text, retry_prompt_text)
        fields["preserved_sections"] = list(preserved_sections)
        fields["dropped_sections"] = list(dropped_sections)
        fields["preserved_evidence_ids"] = list(preserved_ids)
        fields["dropped_evidence_ids"] = list(dropped_ids)
        # 削除IDを「載っていた節」で分類する（TASK-066レビュー対応）。
        # core節由来の削除が 0 なら、落としたのは optional 派生節のIDだけということ。
        fields.update(
            dropped_evidence_id_safety(
                initial_prompt_text, retry_prompt_text, messages
            ).as_dict()
        )
        if decision.folds_speech_history:
            fields.update(dropped_speech_stats(messages, retry_prompt_text).as_dict())
    if previous_body_chars is not None:
        fields["retry_previous_body_chars"] = int(previous_body_chars)
        if new_body_chars is not None:
            delta = int(new_body_chars) - int(previous_body_chars)
            fields["body_preserved"] = bool(new_body_chars)
            fields["retry_body_char_delta"] = int(delta)
            fields["retry_body_shortened"] = bool(delta < 0)
    return fields

__all__ = [
    "COMPRESSION_STAGES",
    "DroppedIdSafety",
    "DroppedSpeechStat",
    "FAILURE_CLASSES",
    "FAILURE_CONTEXT_OVERFLOW",
    "FAILURE_EMPTY_OR_THINKING_ONLY",
    "FAILURE_GENERATION_CAP",
    "FAILURE_OUTPUT_FORMAT",
    "FAILURE_QUESTION_ANSWER",
    "FAILURE_SECRET_LEAK",
    "FAILURE_SEMANTIC_LIGHT",
    "FAILURE_TIMEOUT",
    "FAILURE_UNSPECIFIED",
    "HISTORY_FOLDING_STAGES",
    "INTENTIONAL_IMMEDIATE_COMPRESSION_TRIGGERS",
    "LAST_RESORT_AFTER_SAME_FAILURE_ATTEMPTS",
    "LIGHT_STAGE_AFTER_SAME_FAILURE_ATTEMPTS",
    "QUESTION_LINKED_SECTION_NAMES",
    "RETRY_ADJACENCY_TARGETS",
    "RETRY_BLOCK_ROLES",
    "RETRY_BLOCK_WEIGHTS",
    "RETRY_CONTEXT_POLICIES",
    "RETRY_CONTEXT_POLICY_DEFAULT",
    "RETRY_CONTEXT_POLICY_LEGACY_EARLY",
    "RETRY_CONTEXT_POLICY_TIERED",
    "RETRY_FACTS_MAX_INDEX",
    "RETRY_LIGHT_DROP_CANDIDATES",
    "RETRY_LIGHT_PROTECTED_SECTIONS",
    "RETRY_MUST_USE_ROLES",
    "RETRY_NON_ANCHOR_ROLES",
    "RETRY_PROMPT_BLOCKS",
    "RETRY_PROMPT_ORDER_LEGACY",
    "RETRY_PROMPT_ORDER_V1",
    "RETRY_REPRESENTATIVE_BLOCK_CHARS",
    "RETRY_STAGES",
    "RetryAdjacencyTarget",
    "RetryOrderScore",
    "RetryStageDecision",
    "TRIGGER_BUDGET_ADJUSTMENT",
    "TRIGGER_BUDGET_RETRY_REPEAT",
    "TRIGGER_CONTEXT_OVERFLOW",
    "TRIGGER_EFFECTIVE_CONTEXT_SHORTAGE",
    "TRIGGER_FORMAT_REPAIR",
    "TRIGGER_POLICY_LEGACY_EARLY",
    "TRIGGER_QUESTION_ANSWER_REPAIR",
    "TRIGGER_SAME_FAILURE_SECOND",
    "TRIGGER_SAME_FAILURE_THIRD",
    "TRIGGER_SECRET_LEAK_REPAIR",
    "TRIGGER_SEMANTIC_REPAIR",
    "TRIGGER_TIMEOUT_REPEAT",
    "assemble_retry_prompt",
    "decide_retry_stage",
    "drop_retry_light_sections",
    "dropped_evidence_id_safety",
    "dropped_speech_stats",
    "evidence_id_diff",
    "evidence_ids_in_text",
    "legacy_retry_order_score",
    "normalize_retry_context_policy",
    "resolve_retry_prompt_order",
    "retry_adopted_order_score",
    "retry_compression_reason",
    "retry_context_policy_label",
    "retry_failure_class",
    "retry_light_drop_names",
    "retry_order_variants",
    "retry_role_of",
    "retry_stage_log_fields",
    "retry_trigger_is_immediate_compression",
    "retry_trigger_is_timeout",
    "retry_weight_of",
    "same_failure_run",
    "score_retry_order",
    "search_retry_orders",
    "section_diff",
    "section_names_in_prompt",
    "stage_compresses",
    "stage_folds_speech_history",
    "validate_retry_order",
]










