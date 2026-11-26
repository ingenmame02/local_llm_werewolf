#!/usr/bin/env python3
"""TASK-006: CLI runner for Local LLM Werewolf."""

import argparse
import json
import os
import signal
import sys
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from werewolf import (
    ACTION_BUDGET_DEFAULT_LABELS,
    ACTION_BUDGET_KEYS,
    DEFAULT_ACTION_NUM_PREDICT,
    DEFAULT_THINKING_BY_ACTION,
    PERSONALITY_CHOICE_KEYS,
    THINKING_DEFAULT_LABELS,
    ActionPolicy,
    GameEngine,
    GameState,
    Message,
    MockActions,
    OllamaActions,
    OllamaClient,
    OllamaError,
    Phase,
    Player,
    Role,
    create_game,
    load_player_configs,
    normalize_action_budget_key,
    parse_action_num_predict_pairs,
    parse_thinking_by_action_pairs,
)
from werewolf.self_consistency import SelfConsistencyGuard
from werewolf.engine import UNVERIFIED_HYPOTHESIS_AS_FACT
from werewolf.discussion_format import (
    DEFAULT_QUESTION_HINT_MODE,
    normalize_question_hint_mode,
)
from werewolf.discussion_novelty import (
    DEFAULT_NOVELTY_POLICY,
    normalize_novelty_policy,
)
from werewolf.leak_gate_policy import (
    DEFAULT_LEAK_GUARD_POLICY,
    LeakGuardProfiles,
    normalize_leak_guard_policy,
)
# TASK-055: 役職主張と役職由来情報のプロヴェナンス分離
from werewolf.role_provenance import (
    ALLOW_BLUFF_ROLE_CLAIM_DEFAULT,
    RoleProvenanceLedger,
    bluff_role_claim_policy_label,
    normalize_allow_bluff_role_claim,
)
# TASK-056: Day内イベントの時系列明示と未来参照の観測（本文は書き換えない）
from werewolf.day_timeline import (
    DAY_TIMELINE_POLICY_DEFAULT,
    CITATION_STATUSES,
    FUTURE_REFERENCE_CATEGORIES,
    REFERENCE_TOKEN_CAUSES,
    day_timeline_policy_label,
    normalize_day_timeline_policy,
)
# TASK-057: プロンプト契約の簡素化（重複した制約・本文再掲を1箇所に集め、optional 節だけを
# 段階的に落とす）。ログ側ではセクション別字数・重複制約数・削除段階を比較できるようにする。
from werewolf.prompt_contract import (
    PROMPT_CONTRACT_POLICY_DEFAULT,
    prompt_contract_policy_label,
    normalize_prompt_contract_policy,
)
# TASK-058: core予算の段階圧縮（layered/off）。保持層（anchor/recent/historical）の字数と、
# 過去の低優先発言の本文を短い行へ畳んだ実績をログで比較できるようにする。発言の口調・
# 論点・推理順には干渉しない設定で、レイアウト（legacy_v1 / v2_attention）も区別して数える。
from werewolf.core_compression import (
    CORE_COMPRESSION_POLICY_DEFAULT,
    PROMPT_LAYOUT_LEGACY,
    PROMPT_LAYOUT_V2,
    core_compression_policy_label,
    normalize_core_compression_policy,
)
# TASK-060 / TASK-064: 議論プロンプトの節順（layout）。実測した帯字数・隣接距離・重複行の削減を
# ログで比べられるようにし、同じseedで v5 / v4 / v3 / v2 / auto を切り替えてA/B評価できるようにする。
from werewolf.prompt_layout import (
    PROMPT_LAYOUT_POLICY_DEFAULT,
    PROMPT_LAYOUT_V3,
    PROMPT_LAYOUT_V4,
    PROMPT_LAYOUT_V5,
    normalize_prompt_layout_policy,
    prompt_layout_policy_label,
)
# TASK-062: 日をまたぐ論理状態の正規化（ContinuityState）。エンジン発行の状態差分だけを
# 次の発言へ渡し、チェックポイントが正規化した日の平常発言本文をID行へ畳む。口調・論点・
# 結論には干渉しない設定で、off にすると旧挙動（本文の積み増しのまま）との比較ができる。
from werewolf.continuity_state import (
    CONTINUITY_POLICY_DEFAULT,
    continuity_policy_label,
    normalize_continuity_policy,
)
# TASK-065: 運用安定化の4テーマ（実効コンテキスト整合／thinking・timeout処置のスコープ、
# 未発言者に関する意図分離、プレイヤー名表記ゆれの内部解決）。いずれも観測・数値だけで、
# 議論本文の語彙・口調・結論には干渉しない。方策はA/B比較用に旧挙動へ戻せる。
from werewolf.context_integrity import (
    CONTEXT_INTEGRITY_POLICY_DEFAULT,
    assess_context_integrity,
    context_integrity_policy_label,
    normalize_context_integrity_policy,
)
from werewolf.context_budget import process_env_context
from werewolf.day_timeline import (
    TIMELINE_INTENT_POLICY_DEFAULT,
    UNSPOKEN_INTENTS,
    normalize_timeline_intent_policy,
    timeline_intent_policy_label,
)
from werewolf.name_resolution import (
    NAME_RESOLUTION_POLICY_DEFAULT,
    name_resolution_policy_label,
    normalize_name_resolution_policy,
)
from werewolf.output_stability import (
    OUTPUT_STABILIZATION_SCOPE_DEFAULT,
    output_stabilization_scope_label,
    normalize_output_stabilization_scope,
)

# フェーズ表示用ラベル
PHASE_NAMES_JA = {
    Phase.NIGHT_WOLF: "夜（人狼襲撃）",
    Phase.NIGHT_SEER: "夜（占い）",
    Phase.MORNING: "朝（襲撃結果発表）",
    Phase.DAY_DISCUSSION: "昼（議論）",
    Phase.VOTING: "投票（処刑）",
    Phase.ENDED: "ゲーム終了",
}

ROLE_NAMES_JA = {
    Role.WOLF: "人狼",
    Role.MADMAN: "狂人",
    Role.SEER: "占い師",
    Role.VILLAGER: "村人",
}

# 公開CO・占い結果の主張は台帳イベントとしてのみ発行され、構造化出力では本文メッセージに
# CO行が含まれない（二層形式は【本文】だけを公開会話へ残す）。そのためターミナルでは
# 台帳イベントを本文の直前に別行として表示する。本文に既に書かれている場合は重複表示しない。
PUBLIC_CLAIM_DISPLAY_TYPES: frozenset[str] = frozenset(
    {
        "claim_role_published",
        "claim_role_corrected",
        "seer_result_claim",
        "seer_result_corrected",
    }
)


def _role_label(value: object) -> str:
    claimed_role = str(value or "")
    try:
        return ROLE_NAMES_JA[Role(claimed_role)]
    except ValueError:
        return claimed_role


def public_claim_display_line(event: dict[str, object]) -> str:
    """台帳のCO・占い結果主張を表示用の1行にする（未確認の主張である旨を明記する）。"""
    claimant = str(event.get("claimant", "?"))
    event_type = str(event.get("type", ""))
    correction = "（訂正）" if event_type.endswith("corrected") else ""
    if event_type in {"claim_role_published", "claim_role_corrected"}:
        return (
            f"  [{claimant}] 公開CO: {_role_label(event.get('claimed_role'))}"
            f"を主張（未確認）{correction}"
        )
    result_label = "人狼" if str(event.get("claimed_result", "")) == "wolf" else "人間"
    return (
        f"  [{claimant}] 占い結果の主張: {event.get('target', '?')} = {result_label}"
        f"{correction}"
    )


def claim_covered_by_speech(event: dict[str, object], speech: str) -> bool:
    """その主張が発言本文に既に書かれているかを判定する（表示の重複を避ける）。"""
    event_type = str(event.get("type", ""))
    if event_type in {"claim_role_published", "claim_role_corrected"}:
        return _role_label(event.get("claimed_role")) in speech
    result_label = "人狼" if str(event.get("claimed_result", "")) == "wolf" else "人間"
    target = str(event.get("target", ""))
    if target and result_label in speech:
        return target in speech or "占い" in speech
    return False


class GameLogger:
    def __init__(
        self,
        game_id: str,
        log_dir: Path,
        metadata: dict[str, Any],
        private_enabled: bool = False,
    ) -> None:
        self.game_id = game_id
        self.log_dir = log_dir
        self.log_file = log_dir / f"game_{game_id}.jsonl"
        self.private_log_file = (
            log_dir / f"game_{game_id}.private.jsonl" if private_enabled else None
        )
        self.metadata = metadata
        self._written = False
        self._sequence = 0
        self._private_sequence = 0
        self._model_stats: dict[str, dict[str, Any]] = {}
        # TASK-048: 新規性計測（クラス別）と低新規性処理の別カウント
        self._novelty_class_counts: dict[str, int] = {}
        self._ledger_metrics: dict[str, int | float] = {
            "discussion_claim_calls": 0,
            "llm_requests_sent": 0,
            "retry_requests_sent": 0,
            "completed_actions": 0,
            "retries": 0,
            "fallbacks": 0,
            "discussion_bodies": 0,
            "discussion_bodies_preserved": 0,
            "question_not_answered": 0,
            "question_checks": 0,
            "question_directed": 0,
            "question_answered": 0,
            "question_answer_forced": 0,
            "ledger_repairs": 0,
            # TASK-048: 再生成（意味的再要求の試行）・低新規性処理・新規性計測
            "discussion_regenerations": 0,
            "novelty_measurements": 0,
            "novelty_present": 0,
            "novelty_restated_topics": 0,
            "novelty_legitimate_repetitions": 0,
            "low_novelty_kept": 0,
            "novelty_appends": 0,
            "discussion_elapsed_seconds": 0.0,
            "duplicate_claims_suppressed": 0,
            "public_claim_updates": 0,
            # TASK-055: 役職CO・占い結果申告の計上（中立。エンジン照合の真偽は数えない）
            "role_claims_recorded": 0,
            "verdict_lines_self_reported": 0,
            "role_claims_disallowed": 0,
            "secret_leaks_detected": 0,
            # TASK-054: ゲート分類の観測（soft/observe は公開したまま計上する）
            "secret_leak_hard": 0,
            "secret_leak_notes": 0,
            "secret_leak_memo_only_notes": 0,
            "secret_leak_false_positives_avoided": 0,
            "secret_leak_sentence_repairs": 0,
            "secret_leak_regenerations": 0,
            "secret_leak_redactions": 0,
            "secret_leak_retry_seconds": 0.0,
            "secret_leak_body_char_total": 0,
            "secret_leak_body_char_kept": 0,
            "secret_leak_repaired_char_kept": 0,
            "llm_elapsed_seconds": 0.0,
            "discussion_context_selections": 0,
            # TASK-056: 発言時点の時系列検査（future_event_reference と根拠追跡率）
            "timeline_reference_checks": 0,
            "future_event_references": 0,
            "timeline_attributed_references": 0,
            "timeline_unresolvable_references": 0,
            "day_timeline_chars_total": 0,
            "day_timeline_lines_total": 0,
            "day_timeline_sections": 0,
            "evidence_char_count_total": 0,
            # TASK-057: プロンプト総量・core/optional字数・参照化による節約・重複制約数・
            # 削除段階（A/B比較で「何を落として何が減ったか」を追跡する）
            "prompt_requests_contracted": 0,
            "prompt_user_char_total": 0,
            "prompt_user_char_max": 0,
            "prompt_core_char_total": 0,
            "prompt_optional_char_total": 0,
            "prompt_dedup_saved_chars_total": 0,
            "prompt_dedup_references": 0,
            "prompt_duplicate_constraint_total": 0,
            "prompt_optional_removed_chars_total": 0,
            # プロンプト構築回数と段階試行回数（段階除去で無駄な再構築をしていないかの確認用）
            "prompt_build_calls_total": 0,
            "contract_stage_attempts_total": 0,
            "contract_stage1_removals": 0,
            "contract_stage2_removals": 0,
            "contract_stage3_removals": 0,
            "contract_stage4_removals": 0,
            # TASK-058: core保持層の字数と、過去の発言を短い行へ畳んだ実績（削除とは別に、
            # 本文は公開ログに残ったままプロンプトだけ圧縮した量を追跡する）
            "core_anchor_char_total": 0,
            "core_recent_char_total": 0,
            "core_historical_char_total": 0,
            "evidence_withheld_body_total": 0,
            "evidence_withheld_saved_chars_total": 0,
            "layered_compression_requests": 0,
            # TASK-060: attention 帯（先頭／中部／終端）の字数、出力形式の後ろに載る字数、
            # 中部へ沈んだ「いま使う」字数、隣接距離（質問↔証拠など）の合計と件数、
            # セオリーカードの同字符数。配置変更が節約と一貫性にどう効いたかをA/Bで読む。
            "prompt_zone_primacy_char_total": 0,
            "prompt_zone_middle_char_total": 0,
            "prompt_zone_recency_char_total": 0,
            "prompt_instructions_to_end_char_total": 0,
            "prompt_weak_middle_must_use_char_total": 0,
            "prompt_theory_card_char_total": 0,
            "prompt_gap_question_to_evidence_sum": 0.0,
            "prompt_gap_timeline_to_evidence_sum": 0.0,
            "prompt_gap_own_speeches_to_question_sum": 0.0,
            # TASK-062: エンジン発行の論理状態（[CONTINUITY_CHECKPOINT]）の実測。差分として
            # 提示した件数、再掲せず個数だけ示した件数、チェックポイント側で本文を畳んだ
            # 節約字数。日をまたいでも状態が単調増加していない判断はこの合計と
            # continuity_metrics() の日別推移で見る（口調・論点の数値ではない）。
            "continuity_state_char_total": 0,
            "continuity_sections": 0,
            "continuity_items_changed_total": 0,
            "continuity_items_unchanged_total": 0,
            "continuity_open_questions_total": 0,
            "evidence_continuity_folded_total": 0,
            "continuity_fold_saved_chars_total": 0,
            "prompt_gap_state_to_evidence_sum": 0.0,
            "prompt_gap_state_to_question_sum": 0.0,
            # TASK-064: 役職能力の境界台帳の集計（公開ログには中立な合計値だけ載せる）。
            # 境界節の実測字数、engine_action/public_role_claim/unsupported_role_action/
            # inference の分類個数、エンジン結果との矛盾、CO補正の再要求で増えた能力表現、
            # 補正プロンプトの短縮効果（通常プロンプト長との比較用）。
            "role_boundary_char_total": 0,
            "role_boundary_sections": 0,
            "prompt_gap_rbound_to_question_sum": 0.0,
            "prompt_gap_rbound_to_own_speeches_sum": 0.0,
            "role_ability_assertions_observed": 0,
            "unsupported_role_action_count": 0,
            "claim_vs_engine_conflict_count": 0,
            "hedged_ability_mentions_observed": 0,
            "role_action_engine_actions": 0,
            "retry_added_ability_expressions": 0,
            "co_correction_reask_prompt_char_total": 0,
            "co_correction_normal_prompt_char_total": 0,
            "co_correction_events": 0,
            "claim_to_ability_delta_sum": 0.0,
            "claim_to_ability_delta_count": 0,
            # TASK-065: 意図分離（実質的未来参照／これから聞く質問行為／これからの行動推測）、
            # プレイヤー名表記ゆれの内部解決、実行コンテキスト整合、thinking・timeout処置の
            # スコープ別実績。すべて観測専用で、本文の書き換えは行っていない。
            "timeline_past_attributions_observed": 0,
            "timeline_future_question_acts": 0,
            "timeline_future_action_speculations": 0,
            "name_variant_mentions_absorbed": 0,
            "legacy_unknown_name_flags_avoided": 0,
            "unresolved_name_references": 0,
            "name_reference_scans": 0,
            "context_integrity_checks": 0,
            "context_integrity_mismatches": 0,
            "context_integrity_caps_applied": 0,
            "context_preflight_runs": 0,
            "context_resolutions_provisional": 0,
            "context_resolutions_refreshed_after_load": 0,
            "output_stabilization_scope_events": 0,
            "timeout_downgrade_events": 0,
            "timeout_downgrades_scoped_to_action": 0,
            "thinking_off_action_events": 0,
            "stabilization_relaxations": 0,
        }
        # TASK-056: 未来参照カテゴリ別の指摘数（公開本文は修正していない観測値）
        self._timeline_category_counts: dict[str, int] = {
            category: 0 for category in FUTURE_REFERENCE_CATEGORIES
        }
        # TASK-059: 引用された相手の時点別内訳（同一ターン先行／別ターン既発言／
        # 死亡後履歴／未発言）。違反の指摘ではなく「どの証拠を引こうとしたか」の分解。
        self._citation_status_counts: dict[str, int] = {
            status: 0 for status in CITATION_STATUSES
        }
        # TASK-059: 違反には数えない観察値（婉曲での仮定引用・未発言者名の一般言及）と、
        # 検査が飛んだ理由別の件数（空本文など。検査対象数の説明に使う）
        self._hedged_pending_citations = 0
        self._unattributed_name_mentions = 0
        self._timeline_checks_skipped_empty_body = 0
        # TASK-065: 未発言者に関する文の意図内訳（実質的未来参照／これから聞く質問行為／
        # これからの行動推測）。future_event_reference は past_attribution の断定だけなので、
        # 「質問したから未来参照」を数えない分離が効いているかをここで確認する（観測専用）。
        self._timeline_intent_counts: dict[str, int] = {
            intent: 0 for intent in UNSPOKEN_INTENTS
        }
        # TASK-065: モデル×行動別の処置スコープ観察（timeout縮小・thinking off・解除）と
        # コンテキスト整合チェックの実施モデル（設定値とサーバー観測値の不一致を観測する）
        self._stability_actions: dict[str, set[str]] = {}
        self._context_integrity_models: set[str] = set()
        self._context_mismatch_models: set[str] = set()
        # TASK-059: 証拠ID参照 `[speech:N]` / `[ref:sN]` の解決結果（原因別）。未解決参照の
        # 「ID欠落・旧ログ互換・実際の未来参照」を分解するために観測する（本文は触らない）。
        self._reference_cause_counts: dict[str, int] = {
            cause: 0 for cause in REFERENCE_TOKEN_CAUSES
        }
        # TASK-059: 公開情報の入力経路別の検査数（発言／CO行／投票／死亡）。検査対象外も
        # 理由付きで数えることで「なぜ53件だけだったのか」を説明できるようにする。
        self._timeline_public_path_counts: dict[str, int] = {}
        # TASK-058 / TASK-060 / TASK-062 / TASK-064: レイアウト別のプロンプト構築数
        # （legacy_v1 / v2_attention / v3_grounding / v4_continuity / v5_role_boundary）
        self._prompt_layout_counts: dict[str, int] = {
            PROMPT_LAYOUT_LEGACY: 0,
            PROMPT_LAYOUT_V2: 0,
            PROMPT_LAYOUT_V3: 0,
            PROMPT_LAYOUT_V4: 0,
            PROMPT_LAYOUT_V5: 0,
        }
        # v2の節順が崩れた回数（編集ミスの静的検査。0を維持する）
        self._prompt_layout_violations = 0

        self.log_dir.mkdir(parents=True, exist_ok=True)
        # 1行目にゲーム開始メタデータを書き込み
        init_record = {
            "type": "game_init",
            "game_id": self.game_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        public_init_record = {
            key: value for key, value in init_record.items() if key != "seed"
        }
        self.append_record(public_init_record, init_record)

    def _bump_public_path(self, path: str) -> None:
        """TASK-059: 公開情報の入力経路別の件数を1増やす（観測専用・本文は触らない）。"""
        self._timeline_public_path_counts[path] = (
            self._timeline_public_path_counts.get(path, 0) + 1
        )

    def append_record(
        self,
        record: dict[str, Any],
        private_record: dict[str, Any] | None = None,
    ) -> None:
        diagnostic_record = private_record if private_record is not None else record
        model = diagnostic_record.get("model", record.get("model"))
        event_type = diagnostic_record.get("type")
        action = diagnostic_record.get("action")
        if event_type == "llm_action_started" and action == "discussion_claim":
            self._ledger_metrics["discussion_claim_calls"] += 1
        if event_type == "llm_request_configuration":
            self._ledger_metrics["llm_requests_sent"] += 1
            request_number = diagnostic_record.get("request_number")
            if isinstance(request_number, int) and request_number > 1:
                self._ledger_metrics["retry_requests_sent"] += 1
        if event_type == "llm_action_completed":
            self._ledger_metrics["completed_actions"] += 1
            self._ledger_metrics["retries"] += int(
                diagnostic_record.get("total_retries", 0)
            )
            transport_retries = diagnostic_record.get("transport_retries", 0)
            if isinstance(transport_retries, int) and not isinstance(
                transport_retries, bool
            ):
                self._ledger_metrics["llm_requests_sent"] += transport_retries
                self._ledger_metrics["retry_requests_sent"] += transport_retries
            self._ledger_metrics["fallbacks"] += int(
                bool(diagnostic_record.get("fallback_used"))
            )
            if action in {"discussion", "discussion_body"}:
                self._ledger_metrics["discussion_bodies"] += 1
                self._ledger_metrics["discussion_bodies_preserved"] += int(
                    bool(diagnostic_record.get("content_preserved"))
                )
                # TASK-048: 議論1ターンの待ち時間（再生成・追記要求の遅延を測る基準）
                self._ledger_metrics["discussion_elapsed_seconds"] = round(
                    float(self._ledger_metrics["discussion_elapsed_seconds"])
                    + float(diagnostic_record.get("elapsed_seconds", 0.0)),
                    3,
                )
            self._ledger_metrics["llm_elapsed_seconds"] = round(
                float(self._ledger_metrics["llm_elapsed_seconds"])
                + float(diagnostic_record.get("elapsed_seconds", 0.0)),
                3,
            )
        if event_type == "llm_action_parse_failed":
            if diagnostic_record.get("reason") == "question_not_answered_first":
                self._ledger_metrics["question_not_answered"] += 1
        # TASK-038: 質問への回答観測（再生成しないモードでも回答率を測る）
        if event_type == "discussion_question_check":
            self._ledger_metrics["question_checks"] += 1
            if bool(diagnostic_record.get("question_directed")):
                self._ledger_metrics["question_directed"] += 1
                if bool(diagnostic_record.get("question_answered")):
                    self._ledger_metrics["question_answered"] += 1
            if bool(diagnostic_record.get("question_answer_forced")):
                self._ledger_metrics["question_answer_forced"] += 1
        if event_type == "ledger_repair":
            self._ledger_metrics["ledger_repairs"] += 1
        # --- TASK-065: プレイヤー名表記ゆれの内部解決（公開本文は変更しない観測のみ） ---
        if event_type == "name_reference_resolution":
            absorbed = int(diagnostic_record.get("name_variants_absorbed") or 0)
            unresolved = int(diagnostic_record.get("unresolved_name_references") or 0)
            avoided = int(diagnostic_record.get("legacy_unknown_flags_avoided") or 0)
            self._ledger_metrics["name_variant_mentions_absorbed"] += absorbed
            self._ledger_metrics["unresolved_name_references"] += unresolved
            self._ledger_metrics["legacy_unknown_name_flags_avoided"] += avoided
            self._ledger_metrics["name_reference_scans"] += 1
            if isinstance(model, str):
                stats = self._model_stats.setdefault(model, self._new_model_stats())
                stats["name_variant_mentions_absorbed"] += absorbed
                stats["unresolved_name_references"] += unresolved
                stats["legacy_unknown_name_flags_avoided"] += avoided
                labels = diagnostic_record.get("mention_labels")
                if isinstance(labels, dict):
                    bucket = stats.setdefault("name_mention_labels", {})
                    for label, count in labels.items():
                        bucket[str(label)] = int(bucket.get(str(label), 0)) + int(count or 0)
        # --- TASK-065: 実効コンテキスト整合（configured / server_observed / loaded / requested）
        if event_type == "effective_context_resolution":
            self._ledger_metrics["context_integrity_checks"] += 1
            if bool(diagnostic_record.get("context_integrity_mismatch")):
                self._ledger_metrics["context_integrity_mismatches"] += 1
            if bool(diagnostic_record.get("context_integrity_applied")):
                self._ledger_metrics["context_integrity_caps_applied"] += 1
            # TASK-065（レビュー対応）: 解決がロード値を持てたか（provisional）と、
            # 送信後に実ロード値へ取り直した回数を集計する（キャッシュ固定の検証用）
            if diagnostic_record.get("context_resolution_stage") == "post_load_refresh":
                self._ledger_metrics["context_resolutions_refreshed_after_load"] += 1
            elif bool(diagnostic_record.get("context_resolution_provisional")):
                self._ledger_metrics["context_resolutions_provisional"] += 1
            if isinstance(model, str):
                self._context_integrity_models.add(str(model))
                if bool(diagnostic_record.get("context_integrity_mismatch")):
                    self._context_mismatch_models.add(str(model))
                stats = self._model_stats.setdefault(model, self._new_model_stats())
                stats["context_integrity_checks"] += 1
                for key, metric in (
                    ("configured_context_limit", "context_configured_limit"),
                    ("server_observed_context_limit", "context_server_observed_limit"),
                    ("loaded_context_from_ps", "context_loaded_from_ps"),
                    ("modelfile_num_ctx", "context_modelfile_num_ctx"),
                    ("requested_num_ctx", "context_requested_num_ctx"),
                    ("adopted_num_ctx", "context_adopted_num_ctx"),
                ):
                    value = diagnostic_record.get(key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        stats[metric] = max(int(stats.get(metric, 0) or 0), value)
        # --- TASK-065: thinking・timeout処置のスコープ（モデル×行動へ限定し恒久化しない） ---
        if event_type == "model_timeout_downgrade":
            self._ledger_metrics["timeout_downgrade_events"] += 1
            self._ledger_metrics["output_stabilization_scope_events"] += 1
            if bool(diagnostic_record.get("scoped_to_action")):
                self._ledger_metrics["timeout_downgrades_scoped_to_action"] += 1
            action_key = str(diagnostic_record.get("action_budget_key") or "")
            scope = str(diagnostic_record.get("stabilization_scope") or "")
            if isinstance(model, str) and action_key:
                entry = f"timeout_fit/{action_key}"
                self._stability_actions.setdefault(str(model), set()).add(entry)
                stats = self._model_stats.setdefault(model, self._new_model_stats())
                stats["timeout_downgrade_events"] += 1
                scoped = stats.setdefault("stabilized_actions", [])
                marker = f"{scope}:{entry}"
                if marker not in scoped:
                    scoped.append(marker)
        if event_type == "thinking_stabilization":
            self._ledger_metrics["thinking_off_action_events"] += 1
            self._ledger_metrics["output_stabilization_scope_events"] += 1
            action_key = str(diagnostic_record.get("action_budget_key") or "")
            scope = str(diagnostic_record.get("stabilization_scope") or "")
            if isinstance(model, str) and action_key:
                self._stability_actions.setdefault(str(model), set()).add(
                    f"thinking_off/{action_key}"
                )
                stats = self._model_stats.setdefault(model, self._new_model_stats())
                stats["thinking_off_action_events"] += 1
                scoped = stats.setdefault("stabilized_actions", [])
                marker = f"{scope}:thinking_off/{action_key}"
                if marker not in scoped:
                    scoped.append(marker)
        if event_type == "output_stabilization_relaxed":
            self._ledger_metrics["stabilization_relaxations"] += 1
            action_key = str(diagnostic_record.get("action_budget_key") or "")
            cleared = diagnostic_record.get("cleared_downgrades")
            if isinstance(model, str):
                stats = self._model_stats.setdefault(model, self._new_model_stats())
                stats["stabilization_relaxations"] += 1
                if isinstance(cleared, list) and action_key:
                    bucket = self._stability_actions.setdefault(str(model), set())
                    for kind in cleared:
                        bucket.discard(f"{kind}/{action_key}")
        # TASK-048: 新規性の計測と低新規性処理（再生成回避回数）を別指標で積む
        if event_type == "discussion_novelty_measured":
            self._ledger_metrics["novelty_measurements"] += 1
            if bool(diagnostic_record.get("has_novelty")):
                self._ledger_metrics["novelty_present"] += 1
            self._ledger_metrics["novelty_restated_topics"] += int(
                diagnostic_record.get("novelty_restated_topics") or 0
            )
            self._ledger_metrics["novelty_legitimate_repetitions"] += int(
                diagnostic_record.get("novelty_legitimate_repetitions") or 0
            )
            primary = diagnostic_record.get("novelty_primary")
            if isinstance(primary, str):
                self._novelty_class_counts[primary] = (
                    self._novelty_class_counts.get(primary, 0) + 1
                )
                if isinstance(model, str):
                    stats = self._model_stats.setdefault(model, self._new_model_stats())
                    counts = stats.setdefault("novelty_primary_counts", {})
                    counts[primary] = counts.get(primary, 0) + 1
        if event_type == "llm_discussion_low_novelty_kept":
            self._ledger_metrics["low_novelty_kept"] += 1
        if event_type == "discussion_novelty_appended":
            self._ledger_metrics["novelty_appends"] += 1
        if (
            event_type == "llm_action_parse_failed"
            and diagnostic_record.get("will_retry") is True
            and action in {"discussion", "discussion_body"}
        ):
            self._ledger_metrics["discussion_regenerations"] += 1
        if event_type == "duplicate_claim_suppressed":
            self._ledger_metrics["duplicate_claims_suppressed"] += 1
        if event_type in {
            "claim_role_published",
            "claim_role_corrected",
            "seer_result_claim",
            "seer_result_corrected",
        }:
            self._ledger_metrics["public_claim_updates"] += 1
            # TASK-055: 公開されたCO行・結果行を中立に計上する（すべて本人申告）。
            self._ledger_metrics["role_claims_recorded"] += 1
            if diagnostic_record.get("target") and diagnostic_record.get("claimed_result"):
                self._ledger_metrics["verdict_lines_self_reported"] += 1
        if event_type == "role_claim_disallowed":
            # TASK-055: allow_bluff_role_claim=False の設定で受理しなかった行
            self._ledger_metrics["role_claims_disallowed"] += 1
        # TASK-059: 公開情報の入力経路別のカウント。発言本文は文中検査へかけ、CO行・投票・
        # 死亡の通知は台帳登録のみ（本文検査の外）。未検査も理由付きで数えることで、
        # 「議論54件のうち検査53件」のような内訳を後から説明できるようにする。
        if event_type == "discussion_message":
            if bool(diagnostic_record.get("timeline_observed")):
                self._bump_public_path("public_speech_body_inspected")
            elif diagnostic_record.get("timeline_observed") is False:
                self._bump_public_path("public_speech_uninspected_timeline_off")
        elif event_type in {
            "claim_role_published",
            "claim_role_corrected",
            "seer_result_claim",
            "seer_result_corrected",
            "seer_claim",
            "seer_results",
        }:
            self._bump_public_path("public_claim_or_verdict_line_ledger_only")
        elif event_type in {"voting_result", "vote_completed", "voting_phase_ended"}:
            self._bump_public_path("public_vote_outcome_ledger_only")
        elif event_type in {"morning_casualty", "first_night_no_attack"}:
            self._bump_public_path("public_death_notice_ledger_only")
        # TASK-056: 公開発言に添えた時系列検査を集計する（観測専用で本文は修正されない）
        if (
            event_type == "discussion_message"
            and bool(diagnostic_record.get("timeline_observed"))
        ):
            self._ledger_metrics["timeline_reference_checks"] += 1
            findings = int(diagnostic_record.get("future_event_reference_count") or 0)
            self._ledger_metrics["future_event_references"] += findings
            self._ledger_metrics["timeline_attributed_references"] += int(
                diagnostic_record.get("attributed_references_observed") or 0
            )
            self._ledger_metrics["timeline_unresolvable_references"] += int(
                diagnostic_record.get("unresolvable_references") or 0
            )
            categories = diagnostic_record.get("future_event_reference_categories")
            if isinstance(categories, dict):
                for category in FUTURE_REFERENCE_CATEGORIES:
                    count = int(categories.get(category) or 0)
                    if count:
                        self._timeline_category_counts[category] = (
                            self._timeline_category_counts.get(category, 0) + count
                        )
            # TASK-059: 引用時点の内訳を積む（どの時点の発言を根拠にしようとしたかの分解）。
            # future_event_reference（違反の指摘）とは別枠で、本文の書き換えは行わない。
            citation_counts = diagnostic_record.get("citation_status_counts")
            if isinstance(citation_counts, dict):
                for status in CITATION_STATUSES:
                    count = int(citation_counts.get(status) or 0)
                    if count:
                        self._citation_status_counts[status] = (
                            self._citation_status_counts.get(status, 0) + count
                        )
            self._hedged_pending_citations += int(
                diagnostic_record.get("hedged_pending_citations_observed") or 0
            )
            self._unattributed_name_mentions += int(
                diagnostic_record.get("mentions_without_attribution_observed") or 0
            )
            # TASK-065: 未発言者に関する文の意図内訳を積む。違反（future_event_reference）へ
            # 数えるのは past_attribution_to_unspoken の断定だけで、これから説明を求める質問
            # 行為とこれからの行動推測はここへ分ける（TASKS.mdの実装案3。本文は変更しない）。
            intent_counts = diagnostic_record.get("timeline_intent_counts")
            if isinstance(intent_counts, dict):
                for intent, count in intent_counts.items():
                    value = int(count or 0)
                    if value:
                        key = str(intent)
                        self._timeline_intent_counts[key] = (
                            self._timeline_intent_counts.get(key, 0) + value
                        )
            self._ledger_metrics["timeline_past_attributions_observed"] += int(
                diagnostic_record.get("past_attributions_observed") or 0
            )
            self._ledger_metrics["timeline_future_question_acts"] += int(
                diagnostic_record.get("future_question_acts_observed") or 0
            )
            self._ledger_metrics["timeline_future_action_speculations"] += int(
                diagnostic_record.get("future_action_speculations_observed") or 0
            )
            if diagnostic_record.get("timeline_checks_skipped_empty_body"):
                self._timeline_checks_skipped_empty_body += 1
                self._bump_public_path("public_speech_body_empty_skipped")
            # TASK-059: 本文の証拠ID参照を原因別に積む（解決済み／まだ存在しないID／
            # 旧ログ互換で検証不能／帰属なのにIDなし）。誤検出を増やさない観測値。
            reference_counts = diagnostic_record.get("evidence_reference_counts")
            if isinstance(reference_counts, dict):
                for cause in REFERENCE_TOKEN_CAUSES:
                    count = int(reference_counts.get(cause) or 0)
                    if count:
                        self._reference_cause_counts[cause] = (
                            self._reference_cause_counts.get(cause, 0) + count
                        )
            if isinstance(model, str):
                stats = self._model_stats.setdefault(model, self._new_model_stats())
                stats["timeline_reference_checks"] += 1
                stats["timeline_future_references"] += findings
                if isinstance(citation_counts, dict):
                    model_citations = stats["citation_status_counts"]
                    for status in CITATION_STATUSES:
                        count = int(citation_counts.get(status) or 0)
                        if count:
                            model_citations[status] = int(
                                model_citations.get(status, 0)
                            ) + count
                stats["hedged_pending_citations"] += int(
                    diagnostic_record.get("hedged_pending_citations_observed") or 0
                )
                stats["unattributed_name_mentions"] += int(
                    diagnostic_record.get("mentions_without_attribution_observed") or 0
                )
                if isinstance(reference_counts, dict):
                    model_references = stats["evidence_reference_cause_counts"]
                    for cause in REFERENCE_TOKEN_CAUSES:
                        count = int(reference_counts.get(cause) or 0)
                        if count:
                            model_references[cause] = int(
                                model_references.get(cause, 0)
                            ) + count
        if event_type == "discussion_context_selected":
            self._ledger_metrics["discussion_context_selections"] += 1
            # TASK-056: 時系列欄と発言履歴の文字数を積む。新しい節を増やしながら全体量を
            # 減らせているか（ヘッダ縮約の削減分）をA/Bで検証するための基準になる。
            timeline_chars = diagnostic_record.get("day_timeline_chars")
            if isinstance(timeline_chars, int) and not isinstance(timeline_chars, bool):
                self._ledger_metrics["day_timeline_chars_total"] += max(
                    0, int(timeline_chars)
                )
                self._ledger_metrics["day_timeline_sections"] += 1
                timeline_lines = diagnostic_record.get("day_timeline_lines")
                if (
                    isinstance(timeline_lines, int)
                    and not isinstance(timeline_lines, bool)
                ):
                    self._ledger_metrics["day_timeline_lines_total"] += max(
                        0, int(timeline_lines)
                    )
            evidence_chars = diagnostic_record.get("evidence_char_count")
            if isinstance(evidence_chars, int) and not isinstance(
                evidence_chars, bool
            ):
                self._ledger_metrics["evidence_char_count_total"] += max(
                    0, int(evidence_chars)
                )
            # TASK-057: プロンプト契約の診断を集計する（同じ制約・発言本文の再掲が減ったか、
            # core を保ったまま総量が減ったかをログ側で比較できるようにする）。
            user_chars = diagnostic_record.get("prompt_user_char_count")
            if isinstance(user_chars, int) and not isinstance(user_chars, bool):
                self._ledger_metrics["prompt_requests_contracted"] += 1
                self._ledger_metrics["prompt_user_char_total"] += max(
                    0, int(user_chars)
                )
                if int(user_chars) > int(self._ledger_metrics["prompt_user_char_max"]):
                    self._ledger_metrics["prompt_user_char_max"] = max(
                        0, int(user_chars)
                    )
            for key, metric in (
                ("prompt_core_chars", "prompt_core_char_total"),
                ("prompt_optional_chars", "prompt_optional_char_total"),
                ("dedup_saved_chars", "prompt_dedup_saved_chars_total"),
                (
                    "prompt_duplicate_constraint_count",
                    "prompt_duplicate_constraint_total",
                ),
                ("optional_removed_chars", "prompt_optional_removed_chars_total"),
                ("prompt_build_calls", "prompt_build_calls_total"),
                ("contract_stage_attempts", "contract_stage_attempts_total"),
                # TASK-058: core保持層の字数と、過去の発言を畳んだ実績（削除ではなく圧縮）
                ("core_anchor_chars", "core_anchor_char_total"),
                ("core_recent_chars", "core_recent_char_total"),
                ("core_historical_chars", "core_historical_char_total"),
                ("evidence_withheld_body_count", "evidence_withheld_body_total"),
                (
                    "evidence_withheld_saved_chars",
                    "evidence_withheld_saved_chars_total",
                ),
            ):
                value = diagnostic_record.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    self._ledger_metrics[metric] += max(0, int(value))
            # TASK-058: 実際に使われたレイアウトと、レイアウト崩れ（節順の逆転）を観測する。
            layout = diagnostic_record.get("prompt_layout")
            if isinstance(layout, str):
                self._prompt_layout_counts[layout] = (
                    self._prompt_layout_counts.get(layout, 0) + 1
                )
            # TASK-060: attention 帯の字数・出力形式以降の字数・中部へ沈んだ「いま使う」字数・
            # 隣接距離（割合）を集計する。同じseedで v3 / v2 を比べたときに、配置が実際に
            # 「質問が証拠の近くにあるか」「形式仕様が終端にあるか」を数値で示すため。
            for key, metric in (
                ("prompt_zone_primacy_chars", "prompt_zone_primacy_char_total"),
                ("prompt_zone_middle_chars", "prompt_zone_middle_char_total"),
                ("prompt_zone_recency_chars", "prompt_zone_recency_char_total"),
                ("prompt_instructions_to_end_chars", "prompt_instructions_to_end_char_total"),
                (
                    "prompt_weak_middle_must_use_chars",
                    "prompt_weak_middle_must_use_char_total",
                ),
                ("prompt_theory_card_chars", "prompt_theory_card_char_total"),
                # TASK-062: 論理状態の提示量（状態節の字数・差分件数・畳んだ本文の実績）
                ("continuity_state_chars", "continuity_state_char_total"),
                ("continuity_items_changed", "continuity_items_changed_total"),
                ("continuity_items_unchanged", "continuity_items_unchanged_total"),
                ("continuity_open_questions", "continuity_open_questions_total"),
                (
                    "evidence_continuity_folded_count",
                    "evidence_continuity_folded_total",
                ),
                (
                    "evidence_continuity_fold_saved_chars",
                    "continuity_fold_saved_chars_total",
                ),
            ):
                value = diagnostic_record.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    self._ledger_metrics[metric] += max(0, int(value))
            if int(diagnostic_record.get("continuity_state_chars") or 0) > 0:
                # 状態節を実際に載せたリクエスト数（off との比較で分母を揃える）
                self._ledger_metrics["continuity_sections"] += 1
            for key, metric in (
                (
                    "prompt_gap_question_to_evidence",
                    "prompt_gap_question_to_evidence_sum",
                ),
                (
                    "prompt_gap_timeline_to_evidence",
                    "prompt_gap_timeline_to_evidence_sum",
                ),
                (
                    "prompt_gap_own_speeches_to_question",
                    "prompt_gap_own_speeches_to_question_sum",
                ),
                # TASK-062: 論理状態節が「発言証拠の直後・現在質問の直前」に居るかの距離
                ("prompt_gap_state_to_evidence", "prompt_gap_state_to_evidence_sum"),
                ("prompt_gap_state_to_question", "prompt_gap_state_to_question_sum"),
            ):
                value = diagnostic_record.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._ledger_metrics[metric] += max(0.0, float(value))
            if bool(diagnostic_record.get("core_compression_applied")):
                self._ledger_metrics["layered_compression_requests"] += 1
            violations = diagnostic_record.get("prompt_layout_violations")
            if isinstance(violations, list) and violations:
                self._prompt_layout_violations += len(violations)
            dedup_refs = diagnostic_record.get("dedup_reference_count")
            if isinstance(dedup_refs, int) and not isinstance(dedup_refs, bool):
                self._ledger_metrics["prompt_dedup_references"] += max(0, int(dedup_refs))
            removal_stage = diagnostic_record.get("contract_removal_stage")
            if isinstance(removal_stage, int) and not isinstance(removal_stage, bool):
                limited = min(max(0, int(removal_stage)), 4)
                if limited > 0:
                    key = f"contract_stage{limited}_removals"
                    if key in self._ledger_metrics:
                        self._ledger_metrics[key] += 1
            if isinstance(model, str):
                stats = self._model_stats.setdefault(
                    model, self._new_model_stats()
                )
                strategy = diagnostic_record.get("context_strategy")
                if isinstance(strategy, str):
                    strategies = stats["context_strategy_counts"]
                    strategies[strategy] = strategies.get(strategy, 0) + 1
                header_style = diagnostic_record.get("evidence_header_style")
                if isinstance(header_style, str):
                    styles = stats["evidence_header_styles"]
                    styles[header_style] = styles.get(header_style, 0) + 1
                # TASK-057: モデル別の契約診断（送信プロンプト量と削減の実績）
                model_user_chars = diagnostic_record.get("prompt_user_char_count")
                if isinstance(model_user_chars, int) and not isinstance(
                    model_user_chars, bool
                ):
                    stats["prompt_contracted_requests"] += 1
                    stats["prompt_user_char_total"] += max(0, int(model_user_chars))
                    if int(model_user_chars) > int(stats["prompt_user_char_max"]):
                        stats["prompt_user_char_max"] = max(0, int(model_user_chars))
                for key, metric in (
                    ("prompt_core_chars", "prompt_core_char_total"),
                    ("prompt_optional_chars", "prompt_optional_char_total"),
                    ("dedup_saved_chars", "prompt_dedup_saved_chars_total"),
                    (
                        "prompt_duplicate_constraint_count",
                        "prompt_duplicate_constraint_total",
                    ),
                    # TASK-058: モデル別のcore保持層と圧縮実績（8k系でどこまで削れたかを比べる）
                    ("core_anchor_chars", "core_anchor_char_total"),
                    ("core_recent_chars", "core_recent_char_total"),
                    ("core_historical_chars", "core_historical_char_total"),
                    ("evidence_withheld_body_count", "evidence_withheld_body_total"),
                    (
                        "evidence_withheld_saved_chars",
                        "evidence_withheld_saved_chars_total",
                    ),
                ):
                    value = diagnostic_record.get(key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        stats[metric] += max(0, int(value))
                model_layout = diagnostic_record.get("prompt_layout")
                if isinstance(model_layout, str):
                    layouts = stats["prompt_layout_counts"]
                    layouts[model_layout] = int(layouts.get(model_layout, 0)) + 1
                # TASK-060: モデル別の配置品質（帯字数・出力形式以降の字数・隣接距離）。
                # 同じ盤面を v3 / v2 で走らせたときに、どのモデルが終端の質問と証拠の
                # 近さから利益を受けるかを比較する。
                for key, metric in (
                    ("prompt_zone_primacy_chars", "prompt_zone_primacy_char_total"),
                    ("prompt_zone_middle_chars", "prompt_zone_middle_char_total"),
                    ("prompt_zone_recency_chars", "prompt_zone_recency_char_total"),
                    (
                        "prompt_instructions_to_end_chars",
                        "prompt_instructions_to_end_char_total",
                    ),
                    (
                        "prompt_weak_middle_must_use_chars",
                        "prompt_weak_middle_must_use_char_total",
                    ),
                    ("prompt_theory_card_chars", "prompt_theory_card_char_total"),
                    # TASK-062: モデル別の論理状態提示量（差分件数・畳んだ本文の実績）
                    ("continuity_state_chars", "continuity_state_char_total"),
                    ("continuity_items_changed", "continuity_items_changed_total"),
                    ("continuity_items_unchanged", "continuity_items_unchanged_total"),
                    ("continuity_open_questions", "continuity_open_questions_total"),
                    (
                        "evidence_continuity_folded_count",
                        "evidence_continuity_folded_total",
                    ),
                    (
                        "evidence_continuity_fold_saved_chars",
                        "continuity_fold_saved_chars_total",
                    ),
                    # TASK-064: モデル別の能力境界節の提示量（字数・節を載せた回数）
                    ("role_boundary_section_chars", "role_boundary_char_total"),
                ):
                    value = diagnostic_record.get(key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        stats[metric] += max(0, int(value))
                if int(diagnostic_record.get("continuity_state_chars") or 0) > 0:
                    stats["continuity_sections"] += 1
                if int(diagnostic_record.get("role_boundary_section_chars") or 0) > 0:
                    stats["role_boundary_sections"] += 1
                for key, metric in (
                    (
                        "prompt_gap_question_to_evidence",
                        "prompt_gap_question_to_evidence_sum",
                    ),
                    (
                        "prompt_gap_timeline_to_evidence",
                        "prompt_gap_timeline_to_evidence_sum",
                    ),
                    (
                        "prompt_gap_own_speeches_to_question",
                        "prompt_gap_own_speeches_to_question_sum",
                    ),
                    # TASK-062: 論理状態節の隣接距離（証拠との距離／現在質問との距離）
                    ("prompt_gap_state_to_evidence", "prompt_gap_state_to_evidence_sum"),
                    ("prompt_gap_state_to_question", "prompt_gap_state_to_question_sum"),
                    # TASK-064: 能力境界節の隣接距離（現在質問・自分の発言との距離）。
                    # v5で凍結した位置から外れていないかをモデル別に見る。
                    (
                        "prompt_gap_role_boundary_to_question",
                        "prompt_gap_rbound_to_question_sum",
                    ),
                    (
                        "prompt_gap_role_boundary_to_own_speeches",
                        "prompt_gap_rbound_to_own_speeches_sum",
                    ),
                ):
                    value = diagnostic_record.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        stats[metric] += max(0.0, float(value))
                model_stage = diagnostic_record.get("contract_removal_stage")
                if isinstance(model_stage, int) and not isinstance(model_stage, bool):
                    limited = min(max(0, int(model_stage)), 4)
                    stage_counts = stats["contract_removal_stage_counts"]
                    stage_key = f"stage{limited}"
                    stage_counts[stage_key] = int(stage_counts.get(stage_key, 0)) + 1
        # TASK-064: 能力境界節の実測と隣接距離を集計する（公開指標は中立な数値だけ）。
        boundary_chars = diagnostic_record.get("role_boundary_section_chars")
        if isinstance(boundary_chars, int) and not isinstance(boundary_chars, bool):
            self._ledger_metrics["role_boundary_char_total"] += max(0, int(boundary_chars))
            if boundary_chars > 0:
                self._ledger_metrics["role_boundary_sections"] += 1
        for key, metric in (
            (
                "prompt_gap_role_boundary_to_question",
                "prompt_gap_rbound_to_question_sum",
            ),
            (
                "prompt_gap_role_boundary_to_own_speeches",
                "prompt_gap_rbound_to_own_speeches_sum",
            ),
        ):
            value = diagnostic_record.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._ledger_metrics[metric] += max(0.0, float(value))
        # TASK-064: 公開発言の能力行使表現の観測（エンジンが発行した管理者ビュー専用）。
        # カテゴリ・支援可否は保有者別の真偽を含むため、台帳の中立合計とイベントログ
        # だけに残し、プロンプトや公開行には絶対に流さない（TASK-055の線引きを踏襲）。
        observations = diagnostic_record.get("role_ability_observations")
        if isinstance(observations, list) and observations:
            obs_model = diagnostic_record.get("role_action_model")
            stats = None
            if isinstance(obs_model, str):
                stats = self._model_stats.setdefault(obs_model, self._new_model_stats())
            for observation in observations:
                if not isinstance(observation, dict):
                    continue
                self._ledger_metrics["role_ability_assertions_observed"] += 1
                if stats is not None:
                    stats["role_ability_assertions_observed"] = int(
                        stats.get("role_ability_assertions_observed", 0)
                    ) + 1
                category = observation.get("category")
                if category == "unsupported_role_action":
                    self._ledger_metrics["unsupported_role_action_count"] += 1
                    if stats is not None:
                        stats["unsupported_role_action_count"] = int(
                            stats.get("unsupported_role_action_count", 0)
                        ) + 1
                elif category == "engine_action":
                    self._ledger_metrics["role_action_engine_actions"] += 1
                if bool(observation.get("conflict")):
                    self._ledger_metrics["claim_vs_engine_conflict_count"] += 1
                    if stats is not None:
                        stats["claim_vs_engine_conflict_count"] = int(
                            stats.get("claim_vs_engine_conflict_count", 0)
                        ) + 1
            delta = diagnostic_record.get("claim_to_ability_turn_delta")
            if isinstance(delta, int) and not isinstance(delta, bool):
                self._ledger_metrics["claim_to_ability_delta_sum"] += float(
                    max(0, int(delta))
                )
                self._ledger_metrics["claim_to_ability_delta_count"] += 1
                if stats is not None:
                    stats["claim_to_ability_delta_sum"] = (
                        float(stats.get("claim_to_ability_delta_sum", 0.0))
                        + float(max(0, int(delta)))
                    )
                    stats["claim_to_ability_delta_count"] = int(
                        stats.get("claim_to_ability_delta_count", 0)
                    ) + 1
        # TASK-064: CO初出→能力表現（またはその逆）のターン距離イベントを集計する。
        # 数値のみで、どのCOが本物かを示す情報は含まれない。
        if event_type == "claim_to_ability_distance":
            distance_delta = diagnostic_record.get("turn_delta")
            if isinstance(distance_delta, int) and not isinstance(distance_delta, bool):
                self._ledger_metrics["claim_to_ability_delta_sum"] += float(
                    max(0, int(distance_delta))
                )
                self._ledger_metrics["claim_to_ability_delta_count"] += 1
                distance_model = diagnostic_record.get("model")
                if isinstance(distance_model, str):
                    stats = self._model_stats.setdefault(
                        distance_model, self._new_model_stats()
                    )
                    stats["claim_to_ability_delta_sum"] = (
                        float(stats.get("claim_to_ability_delta_sum", 0.0))
                        + float(max(0, int(distance_delta)))
                    )
                    stats["claim_to_ability_delta_count"] = int(
                        stats.get("claim_to_ability_delta_count", 0)
                    ) + 1
        # TASK-064: CO補正の再要求が短縮だったかと未検証の能力表現を増やさなかったかを
        # 行動別・モデル別に積む（観測専用。本文は変更されていない）。
        if event_type == "co_correction_boundary":
            reask_chars = diagnostic_record.get("reask_prompt_chars")
            normal_chars = diagnostic_record.get("normal_prompt_chars")
            added_expressions = diagnostic_record.get("ability_assertions_added")
            if isinstance(reask_chars, int) and not isinstance(reask_chars, bool):
                self._ledger_metrics["co_correction_events"] += 1
                self._ledger_metrics["co_correction_reask_prompt_char_total"] += max(
                    0, int(reask_chars)
                )
            if isinstance(normal_chars, int) and not isinstance(normal_chars, bool):
                self._ledger_metrics["co_correction_normal_prompt_char_total"] += max(
                    0, int(normal_chars)
                )
            if isinstance(added_expressions, int) and not isinstance(
                added_expressions, bool
            ):
                self._ledger_metrics["retry_added_ability_expressions"] += max(
                    0, int(added_expressions)
                )
            correction_model = diagnostic_record.get("model")
            if isinstance(correction_model, str):
                stats = self._model_stats.setdefault(
                    correction_model, self._new_model_stats()
                )
                stats["co_correction_events"] = int(
                    stats.get("co_correction_events", 0)
                ) + 1
                if isinstance(reask_chars, int) and not isinstance(reask_chars, bool):
                    stats["co_correction_reask_prompt_char_total"] = int(
                        stats.get("co_correction_reask_prompt_char_total", 0)
                    ) + max(0, int(reask_chars))
                if isinstance(normal_chars, int) and not isinstance(normal_chars, bool):
                    stats["co_correction_normal_prompt_char_total"] = int(
                        stats.get("co_correction_normal_prompt_char_total", 0)
                    ) + max(0, int(normal_chars))
                if isinstance(
                    added_expressions, int
                ) and not isinstance(added_expressions, bool):
                    stats["retry_added_ability_expressions"] = int(
                        stats.get("retry_added_ability_expressions", 0)
                    ) + max(0, int(added_expressions))
        if event_type == "secret_role_leak_detected":
            # TASK-054: hardゲート（本当に秘匿情報を出した場合）だけを数える。
            self._ledger_metrics["secret_leaks_detected"] += 1
            self._ledger_metrics["secret_leak_hard"] += 1
            if isinstance(model, str):
                stats = self._model_stats.setdefault(
                    model, self._new_model_stats()
                )
                stats["secret_leak_count"] += 1
                stats["leak_hard_count"] += 1
                leak_actions = stats["secret_leak_actions"]
                if isinstance(action, str):
                    leak_actions[action] = leak_actions.get(action, 0) + 1
                self._count_leak_rules(stats, diagnostic_record.get("detection_rules"))
                self._record_guard_strength(stats, diagnostic_record.get("guard_strength"))
                memo_rules = diagnostic_record.get("memo_region_rules")
                if isinstance(memo_rules, (list, tuple)) and memo_rules:
                    stats["leak_memo_only_count"] += 1
                    self._ledger_metrics["secret_leak_memo_only_notes"] += 1
        # TASK-054: soft / observe 指摘は本文をそのまま公開する（旧来はここで再生成していた）。
        # legacy_strict なら遮断されたはずの件数は誤検出自動回避として別指標に積む。
        if event_type == "secret_leak_note":
            self._ledger_metrics["secret_leak_notes"] += 1
            if bool(diagnostic_record.get("false_positive_avoided")):
                self._ledger_metrics["secret_leak_false_positives_avoided"] += 1
            body_chars = int(diagnostic_record.get("body_char_count") or 0)
            kept_chars = int(diagnostic_record.get("kept_char_count") or 0)
            self._ledger_metrics["secret_leak_body_char_total"] += body_chars
            self._ledger_metrics["secret_leak_body_char_kept"] += kept_chars
            if isinstance(model, str):
                stats = self._model_stats.setdefault(model, self._new_model_stats())
                stats["leak_soft_note_count"] += 1
                if bool(diagnostic_record.get("false_positive_avoided")):
                    stats["leak_false_positive_avoided"] += 1
                stats["leak_body_char_total"] += body_chars
                stats["leak_body_char_kept"] += kept_chars
                self._count_leak_rules(stats, diagnostic_record.get("note_rules"))
                self._record_guard_strength(stats, diagnostic_record.get("guard_strength"))
                memo_rules = diagnostic_record.get("memo_region_rules")
                if isinstance(memo_rules, (list, tuple)) and memo_rules:
                    stats["leak_memo_only_count"] += 1
                    self._ledger_metrics["secret_leak_memo_only_notes"] += 1
        # TASK-054実装案3: 漏洩文だけを選んだ補正（全文再生成を回避できた件数）。
        if event_type == "secret_leak_sentence_repair":
            self._ledger_metrics["secret_leak_sentence_repairs"] += 1
            body_chars = int(diagnostic_record.get("body_char_count") or 0)
            kept_chars = int(diagnostic_record.get("kept_char_count") or 0)
            self._ledger_metrics["secret_leak_body_char_total"] += body_chars
            self._ledger_metrics["secret_leak_body_char_kept"] += kept_chars
            self._ledger_metrics["secret_leak_repaired_char_kept"] += kept_chars
            if isinstance(model, str):
                stats = self._model_stats.setdefault(model, self._new_model_stats())
                stats["leak_sentence_repairs"] += 1
                stats["leak_body_char_total"] += body_chars
                stats["leak_body_char_kept"] += kept_chars
                rules: list[object] = []
                for key in ("replaced_rules", "dropped_rules"):
                    value = diagnostic_record.get(key)
                    if isinstance(value, (list, tuple)):
                        rules.extend(value)
                self._count_leak_rules(stats, rules)
                self._record_guard_strength(stats, diagnostic_record.get("guard_strength"))
        # TASK-054: 実際に再要求・削句へ回った負荷（モデル別の待ち時間比較に使う）。
        if event_type == "secret_leak_regeneration":
            self._ledger_metrics["secret_leak_regenerations"] += 1
            outcome = diagnostic_record.get("outcome")
            if isinstance(outcome, str) and "redacted" in outcome:
                self._ledger_metrics["secret_leak_redactions"] += 1
            retry_seconds = float(diagnostic_record.get("retry_seconds") or 0.0)
            self._ledger_metrics["secret_leak_retry_seconds"] = round(
                float(self._ledger_metrics["secret_leak_retry_seconds"]) + retry_seconds, 3
            )
            body_chars = int(diagnostic_record.get("body_char_count") or 0)
            kept_chars = int(diagnostic_record.get("kept_char_count") or 0)
            self._ledger_metrics["secret_leak_body_char_total"] += body_chars
            self._ledger_metrics["secret_leak_body_char_kept"] += kept_chars
            if isinstance(model, str):
                stats = self._model_stats.setdefault(model, self._new_model_stats())
                stats["leak_regenerations"] += 1
                if isinstance(outcome, str) and "redacted" in outcome:
                    stats["leak_redactions"] += 1
                stats["leak_retry_seconds"] = round(
                    float(stats["leak_retry_seconds"]) + retry_seconds, 3
                )
                stats["leak_body_char_total"] += body_chars
                stats["leak_body_char_kept"] += kept_chars
                self._count_leak_rules(stats, diagnostic_record.get("hard_rules"))
                self._record_guard_strength(stats, diagnostic_record.get("guard_strength"))
        if (
            isinstance(model, str)
            and diagnostic_record.get("type") == "llm_action_parse_failed"
        ):
            stats = self._model_stats.setdefault(model, self._new_model_stats())
            reason = diagnostic_record.get("reason")
            if isinstance(reason, str):
                reasons = stats["failure_reasons"]
                reasons[reason] = reasons.get(reason, 0) + 1
                stats["parse_failure_count"] += 1
                if reason in {"empty_content", "thinking_only_response"}:
                    stats["empty_response_count"] += 1
                if reason == "generation_truncated":
                    stats["truncated_count"] += 1
        # TASK-053: 1送信ごとの生成観測をモデル×アクション別に加算する（空応答率・
        # thinking消費率・待ち時間・本文長を行動単位で比較するための集計基盤）。
        if (
            isinstance(model, str)
            and diagnostic_record.get("type") == "llm_generation_attempt"
        ):
            stats = self._model_stats.setdefault(model, self._new_model_stats())
            action_key = diagnostic_record.get("action_budget_key") or action
            if isinstance(action_key, str):
                entry = self._action_profile_entry(stats, action_key)
                entry["requests"] += 1
                request_number = diagnostic_record.get("request_number")
                if (
                    isinstance(request_number, int)
                    and not isinstance(request_number, bool)
                    and request_number > 1
                ):
                    entry["retry_requests"] += 1
                if bool(diagnostic_record.get("body_empty")):
                    entry["body_empty_count"] += 1
                if bool(diagnostic_record.get("thinking_only_response")):
                    entry["thinking_only_count"] += 1
                if bool(diagnostic_record.get("thinking_present")):
                    entry["thinking_present_count"] += 1
                body_chars = diagnostic_record.get("body_char_count")
                if isinstance(body_chars, int) and not isinstance(body_chars, bool):
                    entry["body_char_total"] += max(0, body_chars)
                    if body_chars > 0:
                        entry["body_present_count"] += 1
                request_elapsed = diagnostic_record.get("request_elapsed_seconds")
                if isinstance(request_elapsed, (int, float)) and not isinstance(
                    request_elapsed, bool
                ):
                    entry["elapsed_seconds_total"] = round(
                        float(entry["elapsed_seconds_total"]) + float(request_elapsed), 3
                    )
                eval_count = diagnostic_record.get("eval_count")
                if isinstance(eval_count, int) and not isinstance(eval_count, bool):
                    entry["eval_count_total"] += eval_count
                if diagnostic_record.get("done_reason") == "length":
                    entry["truncated_count"] += 1
                issue = diagnostic_record.get("generation_issue")
                if isinstance(issue, str):
                    reasons = entry["failure_reasons"]
                    reasons[issue] = int(reasons.get(issue, 0)) + 1
                    if issue == "timeout":
                        entry["timeout_count"] += 1
                body_budget = diagnostic_record.get("body_budget")
                if isinstance(body_budget, dict):
                    entry["last_body_min_tokens"] = body_budget.get("body_min_tokens")
                    entry["last_thinking_allowance_tokens"] = body_budget.get(
                        "thinking_allowance_tokens"
                    )
                # TASK-063: 送信カテゴリ（thinking予算／timeout／生成上限／出力形式／通信／
                # プロンプト論理）と入力実字数をアクション別に加算する。失敗を「モデルの
                # 出力予算側」と「議論の論理側」へ分けて数えるための集計で、内容には干渉しない。
                trace_row = diagnostic_record.get("output_budget")
                if isinstance(trace_row, dict):
                    category = trace_row.get("category")
                    if isinstance(category, str) and category:
                        counts = entry["output_budget_categories"]
                        counts[category] = int(counts.get(category, 0)) + 1
                    if bool(trace_row.get("budget_bound")):
                        entry["budget_bound_failures"] += 1
                    if bool(trace_row.get("logic_bound")):
                        entry["logic_bound_failures"] += 1
                    if bool(trace_row.get("short_retry")):
                        entry["short_retry_requests"] += 1
                    prompt_chars = trace_row.get("prompt_char_count")
                    if isinstance(prompt_chars, int) and not isinstance(prompt_chars, bool):
                        entry["prompt_char_total"] += max(0, prompt_chars)
                        entry["prompt_char_count"] += 1
                        entry["max_prompt_char_count"] = max(
                            int(entry["max_prompt_char_count"]), max(0, prompt_chars)
                        )
        if record.get("type") == "llm_action_completed":
            if isinstance(model, str):
                stats = self._model_stats.setdefault(
                    model,
                    self._new_model_stats(),
                )
                stats["actions"] += 1
                stats["retries"] += int(record.get("total_retries", 0))
                stats["fallback_count"] += int(bool(record.get("fallback_used")))
                content_preserved = bool(record.get("content_preserved"))
                stats["content_preserved_count"] += int(content_preserved)
                stats["content_lost_fallback_count"] += int(
                    bool(record.get("fallback_used")) and not content_preserved
                )
                for failure_class in record.get("failure_classes", ()):
                    if isinstance(failure_class, str):
                        counts = stats["failure_classes"]
                        counts[failure_class] = counts.get(failure_class, 0) + 1
                stats["elapsed_seconds"] = round(
                    float(stats["elapsed_seconds"])
                    + float(record.get("elapsed_seconds", 0.0)),
                    3,
                )
                input_token_count = record.get("input_token_count")
                if isinstance(input_token_count, int) and not isinstance(
                    input_token_count, bool
                ):
                    stats["input_token_count_total"] += input_token_count
                response_metadata = record.get("response_metadata")
                if isinstance(response_metadata, dict):
                    eval_count = response_metadata.get("eval_count")
                    if isinstance(eval_count, int) and not isinstance(eval_count, bool):
                        stats["eval_count_total"] += eval_count
                        stats["output_token_count_total"] += eval_count
                    eval_duration_ms = response_metadata.get("eval_duration_ms")
                    if isinstance(eval_duration_ms, int) and not isinstance(eval_duration_ms, bool):
                        stats["eval_duration_ms_total"] += eval_duration_ms
                    done_reason = response_metadata.get("done_reason")
                    if isinstance(done_reason, str):
                        done_reasons = stats["done_reason_counts"]
                        done_reasons[done_reason] = done_reasons.get(done_reason, 0) + 1
                    stats["thinking_present_response_count"] += int(
                        bool(response_metadata.get("thinking_present"))
                    )
                budget_key = record.get("action_budget_key")
                num_predict = record.get("num_predict")
                if isinstance(budget_key, str) and isinstance(num_predict, int):
                    stats["action_budgets"][budget_key] = num_predict
                error = record.get("error")
                if record.get("failure_reason") == "timeout" or (
                    isinstance(error, dict) and error.get("kind") == "timeout"
                ):
                    stats["timeout_count"] += 1
        if isinstance(model, str) and record.get("type") == "llm_request_configuration":
            stats = self._model_stats.setdefault(model, self._new_model_stats())
            if record.get("thinking_enabled") is False:
                stats["thinking_disabled_requests"] += 1
            elif record.get("thinking_enabled") is True:
                stats["thinking_enabled_requests"] += 1
        if isinstance(model, str) and record.get("type") == "thinking_unavailable":
            stats = self._model_stats.setdefault(model, self._new_model_stats())
            stats["thinking_capability_fallbacks"] += 1
        if isinstance(model, str) and record.get("type") == "thinking_budget_exhausted":
            stats = self._model_stats.setdefault(model, self._new_model_stats())
            stats["thinking_budget_fallbacks"] += 1
            # TASK-053: 余白拡大で耐えたのか、最終的に thinking off へ切ったのかを
            # モデル×アクション別に分けて記録する（自由な思考をどこまで保てたかの検証用）。
            action_key = diagnostic_record.get("action_budget_key") or action
            if isinstance(action_key, str):
                entry = self._action_profile_entry(stats, action_key)
                entry["thinking_budget_exhaustions"] += 1
                if diagnostic_record.get("thinking_allowance_widened") is True:
                    entry["thinking_allowance_widenings"] += 1
                if diagnostic_record.get("thinking_off_switched") is True:
                    entry["thinking_off_switches"] += 1
                allowance_tokens = diagnostic_record.get("thinking_allowance_tokens")
                if isinstance(allowance_tokens, int) and not isinstance(allowance_tokens, bool):
                    entry["last_thinking_allowance_tokens"] = allowance_tokens
        self._append_to_file(record, self.log_file, private=False)
        if self.private_log_file is not None:
            self._append_to_file(
                private_record if private_record is not None else record,
                self.private_log_file,
                private=True,
            )

    @staticmethod
    def _new_model_stats() -> dict[str, Any]:
        return {
            "actions": 0,
            "retries": 0,
            "fallback_count": 0,
            "fallback_rate": 0.0,
            "content_preserved_count": 0,
            "content_lost_fallback_count": 0,
            "failure_classes": {},
            "input_token_count_total": 0,
            "output_token_count_total": 0,
            "timeout_count": 0,
            "parse_failure_count": 0,
            "empty_response_count": 0,
            "empty_response_rate": 0.0,
            "truncated_count": 0,
            "elapsed_seconds": 0.0,
            "failure_reasons": {},
            # eval_count / eval_duration_ms は Ollama の生成トーク数・生成時間メトリクス
            "eval_count_total": 0,
            "eval_duration_ms_total": 0,
            "action_budgets": {},
            # 思考生成（think=false）で送信したリクエスト数
            "thinking_disabled_requests": 0,
            "thinking_enabled_requests": 0,
            "thinking_present_response_count": 0,
            "done_reason_counts": {},
            "context_strategy_counts": {},
            "secret_leak_count": 0,
            "secret_leak_actions": {},
            # TASK-054: モデル別の漏洩ガード実績（強度を一律にしないための入力と、
            # 誤検出率・再要求時間・本文保持率の比較に使う）。
            "leak_hard_count": 0,
            "leak_soft_note_count": 0,
            "leak_memo_only_count": 0,
            "leak_false_positive_avoided": 0,
            "leak_sentence_repairs": 0,
            "leak_regenerations": 0,
            "leak_redactions": 0,
            "leak_retry_seconds": 0.0,
            "leak_body_char_total": 0,
            "leak_body_char_kept": 0,
            "leak_rule_counts": {},
            "leak_guard_strengths": {},
            "leak_rate": 0.0,
            "leak_false_positive_rate": 0.0,
            "leak_body_preservation_rate": 1.0,
            # TASK-048: モデル別の新規性計測（primaryクラス別）と低新規性処理
            "novelty_primary_counts": {},
            # TASK-056: 発言時点の時系列検査（未来参照の検出・ヘッダ様式の構成比）
            "timeline_reference_checks": 0,
            "timeline_future_references": 0,
            "evidence_header_styles": {},
            # TASK-057: モデル別のプロンプト契約実績（送信字数・core/optional・削減量・
            # 削除段階の内訳）。モデルごとに違う命令の受け取り方を比較できる。
            "prompt_user_char_total": 0,
            "prompt_user_char_max": 0,
            "prompt_contracted_requests": 0,
            "prompt_core_char_total": 0,
            "prompt_optional_char_total": 0,
            "prompt_dedup_saved_chars_total": 0,
            "prompt_duplicate_constraint_total": 0,
            "contract_removal_stage_counts": {},
            # TASK-058: モデル別のcore保持層（anchor/recent/historical）と、過去の発言を
            # 短い行へ畳んだ実績、使われたレイアウト（A/Bで入力量と挙動を比べる）
            "core_anchor_char_total": 0,
            "core_recent_char_total": 0,
            "core_historical_char_total": 0,
            "evidence_withheld_body_total": 0,
            "evidence_withheld_saved_chars_total": 0,
            "prompt_layout_counts": {},
            # TASK-060: モデル別のattention帯字数・出力形式以降の字数・中部に沈んだ「いま使う」
            # 字数・隣接距離（平均）。8k系で質問と証拠が離れていると回答率が下がるため、配置の
            # 効果をモデルごとに比べる。
            "prompt_zone_primacy_char_total": 0,
            "prompt_zone_middle_char_total": 0,
            "prompt_zone_recency_char_total": 0,
            "prompt_instructions_to_end_char_total": 0,
            "prompt_weak_middle_must_use_char_total": 0,
            "prompt_theory_card_char_total": 0,
            "prompt_gap_question_to_evidence_sum": 0.0,
            "prompt_gap_timeline_to_evidence_sum": 0.0,
            "prompt_gap_own_speeches_to_question_sum": 0.0,
            # TASK-062: モデル別の論理状態提示量（差分件数・畳んだ字数）。状態節が中部へ沈んで
            # いないか（距離）もここでモデルごとに比べる。
            "continuity_state_char_total": 0,
            "continuity_sections": 0,
            "continuity_items_changed_total": 0,
            "continuity_items_unchanged_total": 0,
            "continuity_open_questions_total": 0,
            "evidence_continuity_folded_total": 0,
            "continuity_fold_saved_chars_total": 0,
            "prompt_gap_state_to_evidence_sum": 0.0,
            "prompt_gap_state_to_question_sum": 0.0,
            # TASK-064: モデル別の役職能力の境界（境界節字数・隣接距離・観測カウンタ）。
            # unsupported_role_action_count / claim_vs_engine_conflict_count は観測専用で、
            # 本文の再生成には使われていない（ログ側で行動別と比較する）。CO補正の短縮効果は
            # reask/normal の字数合計で持ち、finalize で倍率を出す。
            "role_boundary_char_total": 0,
            "role_boundary_sections": 0,
            "prompt_gap_rbound_to_question_sum": 0.0,
            "prompt_gap_rbound_to_own_speeches_sum": 0.0,
            "claim_to_ability_delta_sum": 0.0,
            "claim_to_ability_delta_count": 0,
            "unsupported_role_action_count": 0,
            "claim_vs_engine_conflict_count": 0,
            "role_ability_assertions_observed": 0,
            "retry_added_ability_expressions": 0,
            "co_correction_events": 0,
            "co_correction_reask_prompt_char_total": 0,
            "co_correction_normal_prompt_char_total": 0,
            # TASK-065: モデル別の名前内部解決（吸収した表記ゆれ／未解決／旧判定なら未知名前と
            # 誤っていた数）と、実効コンテキストの証拠（設定・サーバー観測・ロード実測）。
            # timeout縮小・thinking off を掛けた行動名もモデル別に残す（恒久適用していない
            # ことをログ側で確かめられるようにする）。すべて観測専用で本文は変更しない。
            "name_variant_mentions_absorbed": 0,
            "unresolved_name_references": 0,
            "legacy_unknown_name_flags_avoided": 0,
            "name_mention_labels": {},
            "context_integrity_checks": 0,
            "context_configured_limit": 0,
            "context_server_observed_limit": 0,
            "context_loaded_from_ps": 0,
            "context_modelfile_num_ctx": 0,
            "context_requested_num_ctx": 0,
            "context_adopted_num_ctx": 0,
            "timeout_downgrade_events": 0,
            "thinking_off_action_events": 0,
            "stabilization_relaxations": 0,
            "stabilized_actions": [],
            # TASK-059: 引用時点の内訳と、違反には数えない観察値（婉曲・一般名指し）
            "citation_status_counts": {},
            "hedged_pending_citations": 0,
            "unattributed_name_mentions": 0,
            "evidence_reference_cause_counts": {},
            # 思考生成非対応と判明して think=off へ切り替えたモデルの数
            "thinking_capability_fallbacks": 0,
            # 思考が生成上限を消費して切り詰め、think=offへ自動切替した回数
            "thinking_budget_fallbacks": 0,
            # TASK-053: アクション別（議論・投票・占い…）の生成観測内訳。
            # モデル別の空応答率・thinking消費率・待ち時間を行動ごとに比較できる。
            "action_profiles": {},
        }

    @staticmethod
    def _new_action_profile_stats() -> dict[str, Any]:
        """TASK-053: モデル×アクション別の送信集計（1送信イベントから加算する）。"""
        return {
            "requests": 0,
            "retry_requests": 0,
            "body_present_count": 0,
            "body_empty_count": 0,
            "thinking_only_count": 0,
            "truncated_count": 0,
            "timeout_count": 0,
            "thinking_present_count": 0,
            "body_char_total": 0,
            "eval_count_total": 0,
            "elapsed_seconds_total": 0.0,
            "thinking_budget_exhaustions": 0,
            "thinking_allowance_widenings": 0,
            "thinking_off_switches": 0,
            "failure_reasons": {},
            # 直近の検査値（本文最低予算とモデル別thinking余白の内訳）
            "last_body_min_tokens": None,
            "last_thinking_allowance_tokens": None,
            # TASK-063: 送信の失敗カテゴリ内訳（thinking予算・timeout・生成上限・出力形式・
            # 通信・プロンプト論理）と、入力実字数・短縮再要求・本文喪失の有無
            "output_budget_categories": {},
            "prompt_char_total": 0,
            "prompt_char_count": 0,
            "max_prompt_char_count": 0,
            "short_retry_requests": 0,
            "budget_bound_failures": 0,
            "logic_bound_failures": 0,
            # finalize で算出する率・平均
            "empty_response_rate": 0.0,
            "thinking_only_rate": 0.0,
            "truncated_rate": 0.0,
            "thinking_share_rate": 0.0,
            "retry_request_rate": 0.0,
            "avg_elapsed_seconds": 0.0,
            "avg_body_chars": 0.0,
            "avg_eval_count": 0.0,
        }

    def _action_profile_entry(
        self, stats: dict[str, Any], action_key: str
    ) -> dict[str, Any]:
        """TASK-053: モデル別statsからアクション別エントリを取得（無ければ作成）。"""

        profiles = stats.setdefault("action_profiles", {})
        entry = profiles.get(action_key)
        if not isinstance(entry, dict):
            entry = self._new_action_profile_stats()
            profiles[action_key] = entry
        return entry

    @staticmethod
    def _count_leak_rules(stats: dict[str, Any], rules: object) -> None:
        """TASK-054: モデル別の検出ルール内訳を積む（分類精度の比較に使う）。"""
        if not isinstance(rules, (list, tuple)):
            return
        counts = stats.setdefault("leak_rule_counts", {})
        for rule in rules:
            if isinstance(rule, str) and rule:
                counts[rule] = counts.get(rule, 0) + 1

    @staticmethod
    def _record_guard_strength(stats: dict[str, Any], strength: object) -> None:
        """TASK-054: そのモデルで実際に使われたガード強度の使用回数を記録する。"""
        if isinstance(strength, str) and strength:
            strengths = stats.setdefault("leak_guard_strengths", {})
            strengths[strength] = strengths.get(strength, 0) + 1

    def _finalize_leak_stats(self, state: GameState) -> None:
        """TASK-054: モデル別の漏洩率・誤検出率・本文保持率を算出する。"""
        for stats in self._model_stats.values():
            actions = int(stats.get("actions", 0)) or 0
            hard = int(stats.get("leak_hard_count", 0))
            soft = int(stats.get("leak_soft_note_count", 0))
            avoided = int(stats.get("leak_false_positive_avoided", 0))
            total_char = int(stats.get("leak_body_char_total", 0))
            kept_char = int(stats.get("leak_body_char_kept", 0))
            stats["leak_rate"] = (
                round((hard + soft) / actions, 4) if actions else 0.0
            )
            stats["leak_false_positive_rate"] = (
                round(avoided / (hard + soft + avoided), 4)
                if (hard + soft + avoided)
                else 0.0
            )
            stats["leak_body_preservation_rate"] = (
                round(min(1.0, kept_char / total_char), 4) if total_char else 1.0
            )
            stats["leak_retry_seconds"] = round(float(stats.get("leak_retry_seconds", 0.0)), 3)

    def _append_to_file(
        self, record: dict[str, Any], path: Path, private: bool
    ) -> None:
        record = dict(record)
        record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        record["game_id"] = self.game_id
        sequence_attr = "_private_sequence" if private else "_sequence"
        record["sequence"] = getattr(self, sequence_attr)
        setattr(self, sequence_attr, getattr(self, sequence_attr) + 1)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finalize(self, state: GameState, traces: list[Any] | None = None, error: str | None = None) -> None:
        if self._written:
            return
        self._written = True
        for stats in self._model_stats.values():
            actions = stats["actions"]
            if actions:
                stats["fallback_rate"] = round(stats["fallback_count"] / actions, 4)
                stats["empty_response_rate"] = round(
                    stats["empty_response_count"] / actions, 4
                )
                stats["avg_eval_count"] = round(stats["eval_count_total"] / actions, 2)
                stats["avg_input_token_count"] = round(
                    stats["input_token_count_total"] / actions, 2
                )
                stats["avg_eval_duration_ms"] = round(
                    stats["eval_duration_ms_total"] / actions, 2
                )
            else:
                stats["avg_eval_count"] = 0.0
                stats["avg_input_token_count"] = 0.0
                stats["avg_eval_duration_ms"] = 0.0
            # TASK-053: アクション別の空応答率・thinking消費率・平均待ち時間などを算出する
            # （モデル×アクション別で比較し、本文予算制御の効果と悪化を抑止するための指標）。
            for entry in stats.get("action_profiles", {}).values():
                if not isinstance(entry, dict):
                    continue
                requests = int(entry.get("requests", 0))
                if not requests:
                    continue
                entry["empty_response_rate"] = round(
                    int(entry.get("body_empty_count", 0)) / requests, 4
                )
                entry["thinking_only_rate"] = round(
                    int(entry.get("thinking_only_count", 0)) / requests, 4
                )
                entry["truncated_rate"] = round(
                    int(entry.get("truncated_count", 0)) / requests, 4
                )
                entry["thinking_share_rate"] = round(
                    int(entry.get("thinking_present_count", 0)) / requests, 4
                )
                entry["retry_request_rate"] = round(
                    int(entry.get("retry_requests", 0)) / requests, 4
                )
                entry["avg_elapsed_seconds"] = round(
                    float(entry.get("elapsed_seconds_total", 0.0)) / requests, 3
                )
                present = int(entry.get("body_present_count", 0))
                entry["avg_body_chars"] = (
                    round(int(entry.get("body_char_total", 0)) / present, 1) if present else 0.0
                )
                entry["avg_eval_count"] = round(
                    int(entry.get("eval_count_total", 0)) / requests, 2
                )
                # TASK-063: 入力実字数の平均・本文生成率・短縮再要求率と、失敗を出力予算側と
                # 論理側へ分けた割合（強化モデルの事故を「論理の失敗」と取り違えないため）
                seen_chars = int(entry.get("prompt_char_count", 0))
                entry["avg_prompt_char_count"] = (
                    round(int(entry.get("prompt_char_total", 0)) / seen_chars, 1)
                    if seen_chars
                    else 0.0
                )
                entry["body_generation_rate"] = round(present / requests, 4)
                entry["short_retry_rate"] = round(
                    int(entry.get("short_retry_requests", 0)) / requests, 4
                )
                entry["budget_bound_failure_rate"] = round(
                    int(entry.get("budget_bound_failures", 0)) / requests, 4
                )
                entry["logic_bound_failure_rate"] = round(
                    int(entry.get("logic_bound_failures", 0)) / requests, 4
                )

        # TASK-054: モデル別の漏洩率・誤検出率・本文保持率を算出する
        self._finalize_leak_stats(state)
        # TASK-060: モデル別の配置品質（送信プロンプト量・attention帯・隣接距離の平均）。
        # 同じseedで --prompt-layout v3 / v2 を走らせたログをモデル単位で突き合わせられる。
        for stats in self._model_stats.values():
            prompts = int(stats.get("prompt_contracted_requests", 0))
            if not prompts:
                continue
            stats["avg_prompt_user_chars"] = round(
                int(stats["prompt_user_char_total"]) / prompts, 1
            )
            stats["avg_prompt_core_chars"] = round(
                int(stats["prompt_core_char_total"]) / prompts, 1
            )
            stats["avg_prompt_zone_primacy_chars"] = round(
                int(stats["prompt_zone_primacy_char_total"]) / prompts, 1
            )
            stats["avg_prompt_zone_middle_chars"] = round(
                int(stats["prompt_zone_middle_char_total"]) / prompts, 1
            )
            stats["avg_prompt_zone_recency_chars"] = round(
                int(stats["prompt_zone_recency_char_total"]) / prompts, 1
            )
            stats["avg_prompt_instructions_to_end_chars"] = round(
                int(stats["prompt_instructions_to_end_char_total"]) / prompts, 1
            )
            stats["avg_prompt_weak_middle_must_use_chars"] = round(
                int(stats["prompt_weak_middle_must_use_char_total"]) / prompts, 1
            )
            stats["avg_prompt_theory_card_chars"] = round(
                int(stats["prompt_theory_card_char_total"]) / prompts, 1
            )
            stats["avg_gap_question_to_evidence"] = round(
                float(stats["prompt_gap_question_to_evidence_sum"]) / prompts, 4
            )
            stats["avg_gap_timeline_to_evidence"] = round(
                float(stats["prompt_gap_timeline_to_evidence_sum"]) / prompts, 4
            )
            stats["avg_gap_own_speeches_to_question"] = round(
                float(stats["prompt_gap_own_speeches_to_question_sum"]) / prompts, 4
            )
            # TASK-062: 論理状態の実測平均。off（比較基準）では sections が 0 なので
            # avg_continuity_state_chars も 0 になる（同じ土俵で比較できる）。
            continuity_sections = int(stats.get("continuity_sections", 0))
            stats["avg_continuity_state_chars"] = (
                round(int(stats["continuity_state_char_total"]) / continuity_sections, 1)
                if continuity_sections
                else 0.0
            )
            stats["avg_continuity_items_changed"] = round(
                int(stats["continuity_items_changed_total"]) / max(1, continuity_sections), 2
            )
            stats["avg_continuity_items_unchanged"] = round(
                int(stats["continuity_items_unchanged_total"]) / max(1, continuity_sections), 2
            )
            stats["avg_gap_state_to_evidence"] = round(
                float(stats["prompt_gap_state_to_evidence_sum"]) / prompts, 4
            )
            stats["avg_gap_state_to_question"] = round(
                float(stats["prompt_gap_state_to_question_sum"]) / prompts, 4
            )
            # TASK-064: モデル別の能力境界節の実測（1リクエスト平均の字数・隣接距離）と、
            # 実行記録のない能力行使の観測数・エンジン結果との矛盾・CO補正の短縮効果。
            # v5で質問直後に置いた境界が実際にどこへ着地したかをA/B比較できる形にする。
            boundary_sections = int(stats.get("role_boundary_sections", 0))
            stats["avg_claim_boundary_chars"] = (
                round(int(stats.get("role_boundary_char_total", 0)) / boundary_sections, 1)
                if boundary_sections
                else 0.0
            )
            stats["avg_gap_role_boundary_to_question"] = round(
                float(stats.get("prompt_gap_rbound_to_question_sum", 0.0)) / prompts, 4
            )
            stats["avg_gap_role_boundary_to_own_speeches"] = round(
                float(stats.get("prompt_gap_rbound_to_own_speeches_sum", 0.0)) / prompts,
                4,
            )
            delta_count = int(stats.get("claim_to_ability_delta_count", 0))
            stats["avg_claim_to_ability_turn_delta"] = (
                round(
                    float(stats.get("claim_to_ability_delta_sum", 0.0)) / delta_count,
                    2,
                )
                if delta_count
                else 0.0
            )
            correction_events = int(stats.get("co_correction_events", 0))
            if correction_events:
                reask_chars = int(stats.get("co_correction_reask_prompt_char_total", 0))
                normal_chars = int(
                    stats.get("co_correction_normal_prompt_char_total", 0)
                )
                stats["avg_co_correction_reask_prompt_chars"] = round(
                    reask_chars / correction_events, 1
                )
                # 短縮倍率（1.0未満なら通常プロンプトより短い）。完了条件の
                # 「再要求が通常プロンプトより明確に短い」をモデル別数値で示す。
                stats["co_correction_shortening_ratio"] = (
                    round(reask_chars / normal_chars, 4) if normal_chars else None
                )

        public_players_data = [
            {
                "name": p.name,
                "model": p.model,
                "is_alive": p.is_alive,
            }
            for p in state.players
        ]
        conversation_data = [{"speaker": m.speaker, "content": m.content} for m in state.conversation_log]
        wolf_chat_data = [{"speaker": m.speaker, "content": m.content} for m in state.wolf_chat_log]

        private_traces_data = []
        if traces:
            for t in traces:
                private_traces_data.append(
                    {
                        "action": getattr(t, "action", str(t)),
                        "player": getattr(t, "player", ""),
                        "raw_responses": getattr(t, "raw_responses", ()),
                        "parsed_targets": getattr(t, "parsed_targets", ()),
                        "selected_target": getattr(t, "selected_target", None),
                        "fallback_used": getattr(t, "fallback_used", False),
                        "content_preserved": getattr(t, "content_preserved", False),
                        "error": getattr(t, "error", None),
                        "failure_reasons": getattr(t, "failure_reasons", ()),
                        "response_metadata": getattr(t, "response_metadata", ()),
                    }
                )

        public_summary = {
            "type": "game_result",
            "game_id": self.game_id,
            "day": state.day,
            "phase": state.phase.value,
            "winner": state.winner,
            "winner_reason": state.winner_reason,
            "winner_counts": dict(state.winner_counts),
            "error_occurred": error is not None,
            "discussion_claim_mode": self.metadata.get(
                "discussion_claim_mode", "legacy"
            ),
            "context_strategy": self.metadata.get(
                "context_strategy", "hybrid_retrieval"
            ),
            "players": public_players_data,
            "alive_players": state.alive_players,
            "seer_claims": [asdict(claim) for claim in state.seer_claims],
            "public_role_claims": [
                asdict(claim) for claim in state.public_role_claims
            ],
            "integrity_alerts": [
                asdict(alert) for alert in state.integrity_alerts
            ],
            "claim_ledger": {
                "public_claims": [
                    asdict(claim)
                    for claim in state.claim_ledger.public_claims.values()
                ],
                "public_seer_results": [
                    {
                        "claimant": claimant,
                        "target": target,
                        "result": result,
                    }
                    for (claimant, target), result
                    in state.claim_ledger.public_seer_results.items()
                ],
            },
            "fact_ledger": {
                "confirmed_facts": [
                    asdict(fact)
                    for fact in state.fact_ledger.confirmed_facts
                    if fact.visibility == "public"
                ],
                "public_claims": [
                    asdict(claim) for claim in state.fact_ledger.public_claims
                ],
                "model_inferences": [
                    asdict(fact) for fact in state.fact_ledger.model_inferences
                ],
            },
            # TASK-047: 未確定解釈台帳（公開仮説のみ）と矛盾・抑制カウントを評価ログへ残す
            "hypothesis_ledger": state.hypothesis_ledger.serialize_public(),
            # TASK-048: 既出論点台帳（要約・連呼・正当な繰り返し）を評価ログへ残す
            "topic_ledger": state.topic_ledger.serialize(),
            # TASK-062: エンジン発行の論理状態チェックポイント（公開情報のID付き短行のみ）。
            # 本人限定の未検証仮説はここへ載せず、日別字数と差分件数だけを残す。
            "continuity_state": state.continuity_state.serialize_public(),
            # TASK-055: 役職主張の出所台帳。公開面は全CO行を中立ラベルに揃え、エンジン照合の
            # 真偽は含まない（serialize_public() に真偽キーがないことはテストで検証する）。
            "role_provenance": state.role_provenance.serialize_public(),
            # TASK-064: 役職能力の境界台帳（engine_action / public_role_claim /
            # unsupported_role_action / inference）。公開面はカテゴリ合計と中立ラベルだけで、
            # どのCOが本物か・実行記録との一致可否は管理者ビュー専用に残す。
            "role_action_provenance": state.role_action_provenance.serialize_public(),
            "allow_bluff_role_claim": bool(
                getattr(state, "allow_bluff_role_claim", ALLOW_BLUFF_ROLE_CLAIM_DEFAULT)
            ),
            # TASK-056: 公開イベントの時系列（時点・ID・短い要約のみ。発言本文・占い結果・
            # 役職など非公開情報は含まれない）。`timeline_policy=off` でも台帳は回るが、
            # プロンプトへは時系列欄を出さない（A/B比較の基準側）。
            "public_timeline": state.public_timeline.serialize(),
            "day_timeline_policy": day_timeline_policy_label(
                getattr(state, "timeline_policy", DAY_TIMELINE_POLICY_DEFAULT)
            ),
            # TASK-057: プロンプト契約のポリシー（simplified/legacy）。同じ履歴でも重複した
            # 制約文と本文再掲を1箇所にまとめる設定が効いているかをログで判別できるようにする。
            "prompt_contract_policy": prompt_contract_policy_label(
                getattr(state, "prompt_contract_policy", PROMPT_CONTRACT_POLICY_DEFAULT)
            ),
            # TASK-058: core予算の段階圧縮（layered/off）。どの保持層を畳んだかを上の
            # ledger_metrics で追跡できるようにし、自由人格・契約・秘匿ガードの意味は変えない。
            "core_compression_policy": core_compression_policy_label(
                getattr(state, "core_compression_policy", CORE_COMPRESSION_POLICY_DEFAULT)
            ),
            # TASK-060: 議論プロンプトの節順ポリシー（v4/v3/v2/auto）。発言証拠・時点・現在
            # 質問・自分の連続性アンカーの位置と、同じ制約の言い換え行数にだけ効く設定。
            "prompt_layout_policy": prompt_layout_policy_label(
                getattr(state, "prompt_layout_policy", PROMPT_LAYOUT_POLICY_DEFAULT)
            ),
            # TASK-062: 日をまたぐ論理状態の正規化（normalized/off）。エンジン発行の状態差分
            # だけを次の発言へ渡す設定で、口調・論点・結論には干渉しない。
            "continuity_policy": continuity_policy_label(
                getattr(state, "continuity_policy", CONTINUITY_POLICY_DEFAULT)
            ),
            # TASK-065: 意図分離・名前正規化・コンテキスト整合・処置スコープの方策ラベル。
            # いずれも旧挙動へ戻せる（A/B比較用）で、発言の口調・論点・結論は変えない。
            "timeline_intent_policy": timeline_intent_policy_label(
                getattr(state, "timeline_intent_policy", TIMELINE_INTENT_POLICY_DEFAULT)
            ),
            "player_name_resolution_policy": name_resolution_policy_label(
                getattr(
                    state, "player_name_resolution_policy", NAME_RESOLUTION_POLICY_DEFAULT
                )
            ),
            "context_integrity_policy": context_integrity_policy_label(
                getattr(state, "context_integrity_policy", CONTEXT_INTEGRITY_POLICY_DEFAULT)
            ),
            "output_stabilization_scope": output_stabilization_scope_label(
                getattr(
                    state, "output_stabilization_scope", OUTPUT_STABILIZATION_SCOPE_DEFAULT
                )
            ),
            # 実効コンテキスト整合チェックで集めた証拠（数値のみ。公開プロンプトには載らない）
            "context_integrity_evidence": dict(
                getattr(state, "context_integrity_checks", {}) or {}
            ),
            "unverified_hypothesis_as_fact_alerts": sum(
                alert.category == UNVERIFIED_HYPOTHESIS_AS_FACT
                for alert in state.integrity_alerts
            ),
            "ledger_metrics": {
                **self._ledger_metrics,
                **state.hypothesis_ledger.metrics(),
                **state.topic_ledger.metrics(),
                # TASK-062: 論理状態の日別推移・差分件数・チェックポイント数（A/B比較の中心）
                **state.continuity_metrics(),
                "novelty_class_counts": dict(self._novelty_class_counts),
                # TASK-056: 発言時点の根拠追跡率と未来参照率。検出は観測専用で、公開本文の
                # 語彙・口調・結論は自動修正していない（body_modified=false で記録）。
                "future_event_reference_rate": round(
                    int(self._ledger_metrics["future_event_references"])
                    / max(1, int(self._ledger_metrics["timeline_reference_checks"])),
                    4,
                ),
                "evidence_grounding_rate_at_speech_time": round(
                    1.0
                    - int(self._ledger_metrics["timeline_unresolvable_references"])
                    / max(1, int(self._ledger_metrics["timeline_attributed_references"])),
                    4,
                ),
                "avg_day_timeline_chars": round(
                    int(self._ledger_metrics["day_timeline_chars_total"])
                    / max(1, int(self._ledger_metrics["day_timeline_sections"])),
                    1,
                ),
                "avg_evidence_char_count": round(
                    int(self._ledger_metrics["evidence_char_count_total"])
                    / max(1, int(self._ledger_metrics["discussion_context_selections"])),
                    1,
                ),
                "timeline_violation_counts": dict(self._timeline_category_counts),
                # TASK-065: 未発言者に関する文の意図内訳。違反（future_event_reference）へ数える
                # のは実質的な過去帰属の断定だけで、これから説明を求める質問行為とこれからの
                # 行動推測は別へ分ける（game_425755ce9a83 Day2 T1 Player 1 の誤検出を分離した効果）
                "timeline_unspoken_intent_counts": dict(self._timeline_intent_counts),
                "timeline_past_attributions_per_body": round(
                    int(self._ledger_metrics["timeline_past_attributions_observed"])
                    / max(1, int(self._ledger_metrics["timeline_reference_checks"])),
                    4,
                ),
                "timeline_question_acts_per_body": round(
                    int(self._ledger_metrics["timeline_future_question_acts"])
                    / max(1, int(self._ledger_metrics["timeline_reference_checks"])),
                    4,
                ),
                "timeline_action_speculations_per_body": round(
                    int(self._ledger_metrics["timeline_future_action_speculations"])
                    / max(1, int(self._ledger_metrics["timeline_reference_checks"])),
                    4,
                ),
                # TASK-065: プレイヤー名表記ゆれの内部解決。旧判定（文字列一致）なら未知名前と
                # 誤った件数を回避した数として分け、公開本文の表記は変えていない実測を残す
                "name_variants_absorbed_total": int(
                    self._ledger_metrics["name_variant_mentions_absorbed"]
                ),
                "legacy_unknown_name_flags_avoided_total": int(
                    self._ledger_metrics["legacy_unknown_name_flags_avoided"]
                ),
                "unresolved_name_references_total": int(
                    self._ledger_metrics["unresolved_name_references"]
                ),
                # TASK-065: thinking・timeout処置のスコープ（モデル×行動へ限定した実績）と
                # 実効コンテキスト整合チェックの実施・不一致・縮退（数値だけの観測）
                "timeout_downgrades_scoped_to_action_rate": round(
                    int(self._ledger_metrics["timeout_downgrades_scoped_to_action"])
                    / max(1, int(self._ledger_metrics["timeout_downgrade_events"])),
                    4,
                ),
                "stabilization_relaxations_total": int(
                    self._ledger_metrics["stabilization_relaxations"]
                ),
                "context_integrity_mismatch_models": sorted(self._context_mismatch_models),
                "context_integrity_checked_models": sorted(self._context_integrity_models),
                # TASK-065（レビュー対応）: ロード値を取れない解決数と、送信後に実ロード値へ
                # 取り直した回数。cap採用が実送信へ効いた件数も併せて出す（数値だけ）
                "context_resolutions_provisional": int(
                    self._ledger_metrics["context_resolutions_provisional"]
                ),
                "context_resolutions_refreshed_after_load": int(
                    self._ledger_metrics["context_resolutions_refreshed_after_load"]
                ),
                "context_integrity_caps_applied_events": int(
                    self._ledger_metrics["context_integrity_caps_applied"]
                ),
                # TASK-057: プロンプト契約の効果（core を保ったまま総量を減らせたか、同じ制約の
                # 再掲が減ったか、どの段階で何を落としたかをA/B比較できるようにする）
                "avg_prompt_user_chars": round(
                    int(self._ledger_metrics["prompt_user_char_total"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    1,
                ),
                "max_prompt_user_chars": int(self._ledger_metrics["prompt_user_char_max"]),
                "avg_prompt_core_chars": round(
                    int(self._ledger_metrics["prompt_core_char_total"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    1,
                ),
                "avg_prompt_optional_chars": round(
                    int(self._ledger_metrics["prompt_optional_char_total"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    1,
                ),
                "dedup_saved_chars_total": int(
                    self._ledger_metrics["prompt_dedup_saved_chars_total"]
                ),
                "dedup_references_total": int(self._ledger_metrics["prompt_dedup_references"]),
                "duplicate_constraint_total": int(
                    self._ledger_metrics["prompt_duplicate_constraint_total"]
                ),
                "optional_removed_chars_total": int(
                    self._ledger_metrics["prompt_optional_removed_chars_total"]
                ),
                # TASK-058: core保持層（anchor/recent/historical）の平均字数と、過去の発言を
                # 短い行へ畳んだ実績。削除ではなく「本文は公開ログに残したままプロンプトだけ
                # 省略」した量なので、future_event_reference や根拠追跡率と突き合わせて読む。
                "avg_core_anchor_chars": round(
                    int(self._ledger_metrics["core_anchor_char_total"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    1,
                ),
                "avg_core_recent_chars": round(
                    int(self._ledger_metrics["core_recent_char_total"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    1,
                ),
                "avg_core_historical_chars": round(
                    int(self._ledger_metrics["core_historical_char_total"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    1,
                ),
                "evidence_withheld_body_total": int(
                    self._ledger_metrics["evidence_withheld_body_total"]
                ),
                "evidence_withheld_saved_chars_total": int(
                    self._ledger_metrics["evidence_withheld_saved_chars_total"]
                ),
                "layered_compression_requests": int(
                    self._ledger_metrics["layered_compression_requests"]
                ),
                # 実際に使われた組み立て順（legacy_v1 / v2_attention / v3_grounding）と節順の崩れ
                "prompt_layout_counts": dict(self._prompt_layout_counts),
                "prompt_layout_violations": int(self._prompt_layout_violations),
                # TASK-060: 配置品質の実測平均（A/B比較用）。v3では dilution=0、
                # own_speeches↔質問 と day_timeline↔証拠 の距離が縮むはずで、それが
                # 回答率・立場一貫性・誤帰属の改善と対応するかを見る。
                "avg_prompt_zone_primacy_chars": round(
                    int(self._ledger_metrics["prompt_zone_primacy_char_total"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    1,
                ),
                "avg_prompt_zone_middle_chars": round(
                    int(self._ledger_metrics["prompt_zone_middle_char_total"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    1,
                ),
                "avg_prompt_zone_recency_chars": round(
                    int(self._ledger_metrics["prompt_zone_recency_char_total"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    1,
                ),
                "avg_prompt_instructions_to_end_chars": round(
                    int(self._ledger_metrics["prompt_instructions_to_end_char_total"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    1,
                ),
                "avg_prompt_weak_middle_must_use_chars": round(
                    int(self._ledger_metrics["prompt_weak_middle_must_use_char_total"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    1,
                ),
                "avg_prompt_theory_card_chars": round(
                    int(self._ledger_metrics["prompt_theory_card_char_total"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    1,
                ),
                "avg_gap_question_to_evidence": round(
                    float(self._ledger_metrics["prompt_gap_question_to_evidence_sum"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    4,
                ),
                "avg_gap_timeline_to_evidence": round(
                    float(self._ledger_metrics["prompt_gap_timeline_to_evidence_sum"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    4,
                ),
                "avg_gap_own_speeches_to_question": round(
                    float(self._ledger_metrics["prompt_gap_own_speeches_to_question_sum"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    4,
                ),
                # TASK-062: 論理状態チェックポイントの実測（A/B比較の要）。normalized では
                # avg_continuity_state_chars が出し、畳んだ本文ぶんだけ recent層が減る。
                # off（比較基準）では sections=0 / folded=0 / fold_saved=0 のままになる。
                "continuity_sections": int(self._ledger_metrics["continuity_sections"]),
                "avg_continuity_state_chars": round(
                    int(self._ledger_metrics["continuity_state_char_total"])
                    / max(1, int(self._ledger_metrics["continuity_sections"])),
                    1,
                ),
                "avg_continuity_items_changed": round(
                    int(self._ledger_metrics["continuity_items_changed_total"])
                    / max(1, int(self._ledger_metrics["continuity_sections"])),
                    2,
                ),
                "avg_continuity_items_unchanged": round(
                    int(self._ledger_metrics["continuity_items_unchanged_total"])
                    / max(1, int(self._ledger_metrics["continuity_sections"])),
                    2,
                ),
                "continuity_open_questions_total": int(
                    self._ledger_metrics["continuity_open_questions_total"]
                ),
                "evidence_continuity_folded_total": int(
                    self._ledger_metrics["evidence_continuity_folded_total"]
                ),
                "continuity_fold_saved_chars_total": int(
                    self._ledger_metrics["continuity_fold_saved_chars_total"]
                ),
                "avg_gap_state_to_evidence": round(
                    float(self._ledger_metrics["prompt_gap_state_to_evidence_sum"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    4,
                ),
                "avg_gap_state_to_question": round(
                    float(self._ledger_metrics["prompt_gap_state_to_question_sum"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    4,
                ),
                # TASK-064: 役職能力の境界節と観測の合計（公開指標は中立な数値だけ）。
                # 保有者別の分類内訳は管理者ビュー（serialize_admin）側だけで持つ。
                # unsupported_role_action は観測専用で本文の再生成には使われていないため、
                # ここでの増加は「発言が消えた」ことを意味しない。
                "role_boundary_sections": int(
                    self._ledger_metrics["role_boundary_sections"]
                ),
                "avg_claim_boundary_chars": round(
                    int(self._ledger_metrics["role_boundary_char_total"])
                    / max(1, int(self._ledger_metrics["role_boundary_sections"])),
                    1,
                ),
                "avg_gap_role_boundary_to_question": round(
                    float(self._ledger_metrics["prompt_gap_rbound_to_question_sum"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    4,
                ),
                "avg_gap_role_boundary_to_own_speeches": round(
                    float(self._ledger_metrics["prompt_gap_rbound_to_own_speeches_sum"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    4,
                ),
                "role_ability_assertions_observed": int(
                    self._ledger_metrics["role_ability_assertions_observed"]
                ),
                "unsupported_role_action_count": int(
                    self._ledger_metrics["unsupported_role_action_count"]
                ),
                "claim_vs_engine_conflict_count": int(
                    self._ledger_metrics["claim_vs_engine_conflict_count"]
                ),
                "retry_added_ability_expressions": int(
                    self._ledger_metrics["retry_added_ability_expressions"]
                ),
                # 婉曲（疑問・仮定・引用・否定）の能力言及は断定的な行使として数えない。
                # 台帳（RoleActionProvenance）が持った観測値をここへ写す（中立な件数のみ）。
                "hedged_ability_mentions_observed": int(
                    getattr(
                        getattr(state, "role_action_provenance", None),
                        "hedged_mentions_observed",
                        0,
                    )
                ),
                "co_correction_events": int(self._ledger_metrics["co_correction_events"]),
                "avg_co_correction_reask_prompt_chars": round(
                    int(self._ledger_metrics["co_correction_reask_prompt_char_total"])
                    / max(1, int(self._ledger_metrics["co_correction_events"])),
                    1,
                ),
                # 補正プロンプトの通常プロンプト比（<1 で短縮。完了条件の数値確認用）
                "co_correction_shortening_ratio": (
                    round(
                        int(self._ledger_metrics["co_correction_reask_prompt_char_total"])
                        / int(
                            self._ledger_metrics["co_correction_normal_prompt_char_total"]
                        ),
                        4,
                    )
                    if int(self._ledger_metrics["co_correction_events"]) > 0
                    and int(
                        self._ledger_metrics["co_correction_normal_prompt_char_total"]
                    )
                    > 0
                    else None
                ),
                "avg_claim_to_ability_turn_delta": (
                    round(
                        float(self._ledger_metrics["claim_to_ability_delta_sum"])
                        / int(self._ledger_metrics["claim_to_ability_delta_count"]),
                        2,
                    )
                    if int(self._ledger_metrics["claim_to_ability_delta_count"]) > 0
                    else 0.0
                ),
                # TASK-059: 引用した相手の時点別内訳（同一ターン先行／別ターン既発言／
                # 死亡後履歴／未発言）。違反指摘の future_event_reference_categories とは
                # 別枠で、婉曲な仮定引用・未発言者名の一般言及はここへ逃がして数える。
                "citation_status_counts": dict(self._citation_status_counts),
                "hedged_pending_citations_observed": int(self._hedged_pending_citations),
                "mentions_without_attribution_observed": int(
                    self._unattributed_name_mentions
                ),
                "timeline_checks_skipped_empty_body": int(
                    self._timeline_checks_skipped_empty_body
                ),
                # TASK-059: 証拠ID参照の解決内訳と、公開情報の入力経路別の検査数
                "evidence_reference_cause_counts": dict(self._reference_cause_counts),
                "timeline_public_info_paths": dict(self._timeline_public_path_counts),
                # ローカル側の構築コスト（段階除去の組み直し回数を確かめる指標）
                "avg_prompt_build_calls": round(
                    int(self._ledger_metrics["prompt_build_calls_total"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    2,
                ),
                "avg_contract_stage_attempts": round(
                    int(self._ledger_metrics["contract_stage_attempts_total"])
                    / max(1, int(self._ledger_metrics["prompt_requests_contracted"])),
                    2,
                ),
                # TASK-048: 新規性・修復・再生成・待ち時間の比較指標（old/new A/Bに使う）
                "novelty_rate": round(
                    int(self._ledger_metrics["novelty_present"])
                    / max(1, int(self._ledger_metrics["novelty_measurements"])),
                    4,
                ),
                "restated_topics_per_body": round(
                    int(self._ledger_metrics["novelty_restated_topics"])
                    / max(1, int(self._ledger_metrics["novelty_measurements"])),
                    4,
                ),
                "legitimate_repetitions_per_body": round(
                    int(self._ledger_metrics["novelty_legitimate_repetitions"])
                    / max(1, int(self._ledger_metrics["novelty_measurements"])),
                    4,
                ),
                "ledger_repair_rate": round(
                    int(self._ledger_metrics["ledger_repairs"])
                    / max(1, int(self._ledger_metrics["discussion_bodies"])),
                    4,
                ),
                "regeneration_rate": round(
                    int(self._ledger_metrics["discussion_regenerations"])
                    / max(1, int(self._ledger_metrics["discussion_bodies"])),
                    4,
                ),
                "avg_discussion_seconds": round(
                    float(self._ledger_metrics["discussion_elapsed_seconds"])
                    / max(1, int(self._ledger_metrics["discussion_bodies"])),
                    3,
                ),
                "retry_rate": round(
                    int(self._ledger_metrics["retries"])
                    / max(1, int(self._ledger_metrics["completed_actions"])),
                    4,
                ),
                "fallback_rate": round(
                    int(self._ledger_metrics["fallbacks"])
                    / max(1, int(self._ledger_metrics["completed_actions"])),
                    4,
                ),
                "discussion_body_preservation_rate": round(
                    int(self._ledger_metrics["discussion_bodies_preserved"])
                    / max(1, int(self._ledger_metrics["discussion_bodies"])),
                    4,
                ),
                "question_not_answered_rate": round(
                    int(self._ledger_metrics["question_not_answered"])
                    / max(1, int(self._ledger_metrics["discussion_bodies"])),
                    4,
                ),
                # TASK-038: 直前の発言が自分宛だった発言への回答率（観測ベース・再生成なし）
                "question_answer_rate": round(
                    int(self._ledger_metrics["question_answered"])
                    / max(1, int(self._ledger_metrics["question_directed"])),
                    4,
                ),
                "question_unanswered_rate": (
                    round(
                        (
                            int(self._ledger_metrics["question_directed"])
                            - int(self._ledger_metrics["question_answered"])
                        )
                        / max(1, int(self._ledger_metrics["question_directed"])),
                        4,
                    )
                    if int(self._ledger_metrics["question_directed"])
                    else 0.0
                ),
                "duplicate_claim_rate": round(
                    int(self._ledger_metrics["duplicate_claims_suppressed"])
                    / max(
                        1,
                        int(self._ledger_metrics["public_claim_updates"])
                        + int(self._ledger_metrics["duplicate_claims_suppressed"]),
                    ),
                    4,
                ),
                # TASK-054: ゲート分類の効率（誤検出を自動回避できた割合・本文保持率・
                # 再要求に掛かった合計秒）。legacy_strict とのA/B比較で使う。
                "leak_guard_policy": getattr(
                    state, "leak_guard_policy", DEFAULT_LEAK_GUARD_POLICY
                ),
                "leak_note_rate": round(
                    int(self._ledger_metrics["secret_leak_notes"])
                    / max(1, int(self._ledger_metrics["discussion_bodies"])),
                    4,
                ),
                "leak_false_positives_avoided": int(
                    self._ledger_metrics["secret_leak_false_positives_avoided"]
                ),
                "leak_false_positive_avoid_rate": round(
                    int(self._ledger_metrics["secret_leak_false_positives_avoided"])
                    / max(
                        1,
                        int(self._ledger_metrics["secret_leak_false_positives_avoided"])
                        + int(self._ledger_metrics["secret_leak_hard"]),
                    ),
                    4,
                ),
                "leak_sentence_repair_rate": round(
                    int(self._ledger_metrics["secret_leak_sentence_repairs"])
                    / max(
                        1,
                        int(self._ledger_metrics["secret_leak_sentence_repairs"])
                        + int(self._ledger_metrics["secret_leak_regenerations"]),
                    ),
                    4,
                ),
                "leak_body_preservation_rate": round(
                    min(
                        1.0,
                        int(self._ledger_metrics["secret_leak_body_char_kept"])
                        / max(1, int(self._ledger_metrics["secret_leak_body_char_total"])),
                    ),
                    4,
                ),
                "leak_retry_seconds": round(
                    float(self._ledger_metrics["secret_leak_retry_seconds"]), 3
                ),
                "leak_retry_seconds_per_body": round(
                    float(self._ledger_metrics["secret_leak_retry_seconds"])
                    / max(1, int(self._ledger_metrics["discussion_bodies"])),
                    3,
                ),
            },
            "conversation_log": conversation_data,
            "votes": state.votes,
            "action_memory_count": sum(
                len(memories)
                for memories in state.player_action_memories.values()
            ),
            "self_consistency_alert_count": sum(
                alert.category
                in {
                    "contradictory_past_vote",
                    "recorded_action_denied",
                    "dead_player_action_target",
                    "past_action_question_unanswered",
                }
                for alert in state.integrity_alerts
            ),
            "model_stats": self._model_stats,
            "model_capabilities": {
                model: dict(capabilities)
                for model, capabilities in state.model_capabilities.items()
            },
        }
        private_summary = {
            **public_summary,
            "error": error,
            "players": [
                {
                    "name": p.name,
                    "model": p.model,
                    "role": p.role.value,
                    "is_alive": p.is_alive,
                    "partner": p.partner,
                }
                for p in state.players
            ],
            "wolf_chat_log": wolf_chat_data,
            "seer_results": state.seer_results,
            "fact_ledger": {
                "confirmed_facts": [
                    asdict(fact) for fact in state.fact_ledger.confirmed_facts
                ],
                "public_claims": [
                    asdict(claim) for claim in state.fact_ledger.public_claims
                ],
                "model_inferences": [
                    asdict(fact) for fact in state.fact_ledger.model_inferences
                ],
            },
            # 私的仮説（推理メモ由来）も含めた完全な台帳
            "hypothesis_ledger": state.hypothesis_ledger.serialize(),
            "player_action_memories": {
                player: [
                    SelfConsistencyGuard.serialize(memory)
                    for memory in memories
                ]
                for player, memories in state.player_action_memories.items()
            },
            "action_candidate_sets": {
                key: asdict(candidate_set)
                for key, candidate_set in state.action_candidate_sets.items()
            },
            "ledger_snapshots": [
                asdict(snapshot) for snapshot in state.ledger_snapshots
            ],
            "reasoning_memos": {
                name: asdict(memo)
                for name, memo in state.reasoning_memos.items()
            },
            # TASK-054: モデル別の漏洩ガード実績（強度の判断材料・誤検出率・本文保持率）
            "leak_guard_policy": getattr(
                state, "leak_guard_policy", DEFAULT_LEAK_GUARD_POLICY
            ),
            "leak_guard_profiles": getattr(
                state, "leak_guard_profiles", LeakGuardProfiles()
            ).summary(),
            # TASK-055: 管理者用。エンジンが発行した占い結果（保有者名つき）と各CO行の照合結果を
            # ここにだけ残す。公開ログ側には真偽キーを含む形では出さない。
            "role_provenance": state.role_provenance.serialize_admin(),
            # TASK-064: 管理者用。エンジンが実行した能力行動（占い・襲撃）の実行記録と、
            # 公開発言で観測した能力行使表現のカテゴリ内訳・本人エンジン結果との矛盾を出す。
            # この情報を持つのは管理者ログだけで、公開台帳・checkpoint・プロンプトには混入しない。
            "role_action_provenance": state.role_action_provenance.serialize_admin(),
            "traces": private_traces_data,
        }
        self.append_record(public_summary, private_summary)


def run_context_integrity_preflight(
    client: object,
    state: object,
    logger: GameLogger,
) -> list[str]:
    """TASK-065: ゲーム開始前にモデル別の実効コンテキスト証拠を1度だけ集める。

    `configured`（プレイヤー/CLI設定）、`server_observed`（プロセスenv・Modelfile）、
    `loaded`（/api/psの実測）、`requested`（今回送りたい値）を1つの値へ潰さず分けて記録し、
    不一致は起動時の警告として出す。方策 cap のときだけ採用値を安全側へ縮め、実際の送信は
    OllamaActions側がstateを読んで反映する。議論本文・プロンプトには一切載らない（数値と
    運用警告のみ）。強化モデルで「timeout後に初めて不一致を知る」状態を避けるためのチェック。
    """
    policy = normalize_context_integrity_policy(
        getattr(state, "context_integrity_policy", CONTEXT_INTEGRITY_POLICY_DEFAULT)
    )
    if str(policy) == "off":
        return []
    env_value, env_key = process_env_context()
    warnings: list[str] = []
    seen: set[str] = set()
    for player in getattr(state, "players", ()) or ():
        model = str(getattr(player, "model", "") or "")
        if not model or model in seen:
            continue
        seen.add(model)
        diagnostics: dict[str, object] = {}
        getter = getattr(client, "context_diagnostics", None)
        if callable(getter):
            try:
                raw = getter(model)
            except Exception as error:  # noqa: BLE001 - 診断失敗でゲームを止めない
                raw = {"context_diagnostics_error": str(error)}
            if isinstance(raw, dict):
                diagnostics = raw
        configured = getattr(player, "num_ctx", None)
        source = "player_setting"
        if configured is None:
            configured = getattr(player, "recommended_num_ctx", None)
            source = "player_recommended_setting"
        if configured is None:
            configured = getattr(state, "cli_num_ctx", None)
            source = "cli_setting"
        check = assess_context_integrity(
            model,
            configured=configured,
            configured_source=source,
            ps_loaded_context=diagnostics.get("ollama_ps_context"),
            modelfile_num_ctx=diagnostics.get("modelfile_num_ctx"),
            process_env_num_ctx=env_value,
            max_num_ctx=getattr(player, "max_num_ctx", None),
            policy=policy,
        )
        checks = getattr(state, "context_integrity_checks", None)
        if isinstance(checks, dict):
            checks[model] = check.to_fields()
        record: dict[str, object] = {
            "type": "context_integrity_preflight",
            **check.to_fields(),
            "process_env_key": env_key,
            "context_diagnostics_error": diagnostics.get("context_diagnostics_error"),
            "note": "起動時に1度だけ確認した実効コンテキストの証拠（本文・役職は含まない）",
        }
        logger.append_record(record, record)
        for line in check.warning_lines():
            warnings.append(line)
            print(f"  [コンテキスト整合] {line}", flush=True)
    return warnings


def print_banner(text: str) -> None:
    print(f"\n{'=' * 60}\n{text}\n{'=' * 60}")


def print_round_header(day: int, phase: Phase, alive: list[str]) -> None:
    phase_str = PHASE_NAMES_JA.get(phase, phase.value)
    print(f"\n--- Day {day} : {phase_str} ---")
    print(f"生存者 ({len(alive)}名): {', '.join(alive)}")


def run_cli_game(args: argparse.Namespace) -> int:
    game_id = uuid.uuid4().hex[:12]
    log_dir = Path(args.log_dir)

    print_banner(f"人狼ゲーム CLI (Game ID: {game_id})")
    print(f"モード: {args.mode}")
    print(f"シード値: {args.seed}")
    print(f"プレイヤー設定: {args.models_file}")
    # TASK-044: --personality で全プレイヤーの個性プロファイルを上書きできる
    # （free は自由発言モード: 話し方の preset を適用しない）。
    personality_override = getattr(args, "personality", None)
    if personality_override is not None:
        free_mode = personality_override == "free"
        print(
            f"個性プロファイル上書き: {personality_override}"
            + ("（自由発言モード・preset なし）" if free_mode else "")
        )
    print(f"ログ保存先: {log_dir / f'game_{game_id}.jsonl'}")

    action_num_predict = parse_action_num_predict_pairs(args.num_predict_by_action)
    action_thinking = parse_thinking_by_action_pairs(args.thinking_by_action)

    options: dict[str, object] = {"temperature": args.temperature}
    if args.num_predict is not None:
        options["num_predict"] = args.num_predict
    if args.num_ctx is not None:
        options["num_ctx"] = args.num_ctx
    if getattr(args, "override_player_num_ctx", False) and args.num_ctx is None:
        raise ValueError("--override-player-num-ctx requires --num-ctx.")
    if getattr(args, "override_player_num_ctx", False):
        options["override_player_num_ctx"] = True
    options["dynamic_context"] = bool(getattr(args, "dynamic_context", True))
    # TASK-053: モデル×アクション別の本文予算予約（初回リクエストで thinking 余白と
    # 本文最低予算を分離）のon/off。既定on、比較ラン用に --no-body-reservation で切れる。
    options["body_reservation"] = bool(getattr(args, "body_reservation", True))
    question_hint_mode = normalize_question_hint_mode(
        getattr(args, "question_hint_mode", DEFAULT_QUESTION_HINT_MODE)
    )
    options["question_hint_mode"] = question_hint_mode
    # TASK-048: 低新規性時の処理（既定 publish=そのまま公開・再生成しない）
    novelty_policy = normalize_novelty_policy(
        getattr(args, "novelty_policy", DEFAULT_NOVELTY_POLICY)
    )
    options["novelty_policy"] = novelty_policy
    # TASK-054: 漏洩ガードのゲート分類（既定 tiered=非公開の真の情報だけ遮断）
    leak_guard_policy = normalize_leak_guard_policy(
        getattr(args, "leak_guard_policy", DEFAULT_LEAK_GUARD_POLICY)
    )
    options["leak_guard_policy"] = leak_guard_policy
    # TASK-055: ブラフ役職COの可否（既定は許可。議論の自由を保ち、出所だけを分離する）
    allow_bluff_role_claim = normalize_allow_bluff_role_claim(
        getattr(args, "bluff_role_claim", ALLOW_BLUFF_ROLE_CLAIM_DEFAULT)
    )
    options["allow_bluff_role_claim"] = allow_bluff_role_claim
    # TASK-056: 発言時点の時系列明示（既定 explicit。offはA/B比較用の旧挙動）
    day_timeline_policy = normalize_day_timeline_policy(
        getattr(args, "day_timeline", DAY_TIMELINE_POLICY_DEFAULT)
    )
    options["day_timeline_policy"] = day_timeline_policy
    # TASK-057: プロンプト契約の簡素化（既定 simplified。legacy は重複した命令文のままの旧挙動）
    prompt_contract_policy = normalize_prompt_contract_policy(
        getattr(args, "prompt_contract", PROMPT_CONTRACT_POLICY_DEFAULT)
    )
    options["prompt_contract_policy"] = prompt_contract_policy
    # TASK-058: core予算の段階圧縮（既定 layered。off はA/B比較用の旧挙動＝従来削除のみ）
    core_compression_policy = normalize_core_compression_policy(
        getattr(args, "core_compression", CORE_COMPRESSION_POLICY_DEFAULT)
    )
    options["core_compression_policy"] = core_compression_policy
    # TASK-060: 議論プロンプトの節順（既定 v4）。配置と重複行だけ変わり、発言内容の自由は不変。
    prompt_layout_policy = normalize_prompt_layout_policy(
        getattr(args, "prompt_layout", PROMPT_LAYOUT_POLICY_DEFAULT)
    )
    options["prompt_layout_policy"] = prompt_layout_policy
    # TASK-062: 日をまたぐ論理状態の正規化（既定 normalized）。エンジン発行の状態差分だけを
    # 次の発言へ渡し、正規化した日の平常発言本文はID行へ畳む。off は旧挙動（A/B比較用）。
    continuity_policy = normalize_continuity_policy(
        getattr(args, "continuity", CONTINUITY_POLICY_DEFAULT)
    )
    # TASK-065: 運用安定化の4方策を正規化する（既定は warn / separated / normalized / action。
    # いずれも旧挙動を明示設定で選べる＝同一seedでのA/B比較が可能。本文の自由は削らない）。
    context_integrity_policy = normalize_context_integrity_policy(
        getattr(args, "context_integrity", CONTEXT_INTEGRITY_POLICY_DEFAULT)
    )
    timeline_intent_policy = normalize_timeline_intent_policy(
        getattr(args, "timeline_intent", TIMELINE_INTENT_POLICY_DEFAULT)
    )
    player_name_resolution_policy = normalize_name_resolution_policy(
        getattr(args, "player_name_resolution", NAME_RESOLUTION_POLICY_DEFAULT)
    )
    output_stabilization_scope = normalize_output_stabilization_scope(
        getattr(args, "output_stability_scope", OUTPUT_STABILIZATION_SCOPE_DEFAULT)
    )
    options["continuity_policy"] = continuity_policy
    if args.thinking in {"on", "off"}:
        options["thinking"] = args.thinking == "on"
    for key, value in action_num_predict.items():
        options[f"num_predict_by_action:{key}"] = value
    for key, value in action_thinking.items():
        options[f"thinking_by_action:{key}"] = value

    # ゲーム初期化
    # TASK-038: --language引数が指定されている場合はresponse_languageを上書き
    # TASK-044: --personality引数が指定されている場合はpersonality_profileを上書き
    kwargs = {}
    if getattr(args, "language", None) is not None:
        kwargs["response_language"] = args.language
    if personality_override is not None:
        kwargs["personality_profile"] = personality_override
    state = create_game(seed=args.seed, config_path=Path(args.models_file), **kwargs)
    # TASK-038: engine側の本文診断も同じ強制度を使う（state経由で参照される）
    state.question_hint_mode = question_hint_mode
    # TASK-048: 台帳・新規性計測も同じポリシーを使う（state経由で参照される）
    state.novelty_policy = novelty_policy
    # TASK-054: engine/llm_actions の漏洩ガードも同じゲートポリシーを使う
    state.leak_guard_policy = leak_guard_policy
    state.leak_guard_profiles.policy = leak_guard_policy
    # TASK-055: engine/llm_actions の役職CO受理も同じルール設定を使う
    state.allow_bluff_role_claim = allow_bluff_role_claim
    # TASK-056: 発言時点の時系列明示をstateへ反映する（台帳のポリシーも揃う）
    state.timeline_policy = day_timeline_policy
    state.public_timeline.policy = day_timeline_policy
    # TASK-057: プロンプト契約のポリシーをstateへ反映する（llm_actions が state から読む）
    state.prompt_contract_policy = prompt_contract_policy
    # TASK-058: core圧縮のポリシーもstateへ反映する（プロンプト構築側が証拠欄の畳み方を選ぶ）
    state.core_compression_policy = core_compression_policy
    # TASK-060: 節順のポリシーもstateへ反映する（llm_actions が state から読んでbuilderへ渡す）
    state.prompt_layout_policy = prompt_layout_policy
    # TASK-062: 論理状態の正規化ポリシーをstateへ反映する（builder と engine が同じ値を使い、
    # 台帳側のポリシーとも揃える）。ここでの再正規化は行わず、台帳へも同じ値を書く。
    state.continuity_policy = continuity_policy
    state.continuity_state.policy = continuity_policy
    # TASK-065: 4つの運用ポリシーをstateへ反映する（builder・engine・llm_actions が同じ値を
    # 読む）。整合チェックは送信num_ctxの縮退にだけ使い、プロンプト本文へは書かない。
    state.context_integrity_policy = context_integrity_policy
    state.timeline_intent_policy = timeline_intent_policy
    state.player_name_resolution_policy = player_name_resolution_policy
    state.output_stabilization_scope = output_stabilization_scope

    meta = {
        "mode": args.mode,
        "seed": args.seed,
        "discussion_claim_mode": getattr(
            args, "discussion_claim_mode", "legacy"
        ),
        "context_strategy": getattr(
            args, "context_strategy", "hybrid_retrieval"
        ),
        "options": options,
        "timeout": args.timeout,
        "num_predict": args.num_predict,
        "num_predict_by_action": action_num_predict,
        "default_action_num_predict": dict(DEFAULT_ACTION_NUM_PREDICT),
        "thinking": args.thinking,
        "thinking_by_action": action_thinking,
        "default_thinking_by_action": dict(DEFAULT_THINKING_BY_ACTION),
        "dynamic_context": bool(getattr(args, "dynamic_context", True)),
        # TASK-053: 本文予算予約の設定値をログへ残す（A/B比較の判別用）
        "body_reservation": bool(getattr(args, "body_reservation", True)),
        "question_hint_mode": question_hint_mode,
        # TASK-048: 低新規性時の処理ポリシー（publish/append/legacy_repair）をログへ残す
        "novelty_policy": novelty_policy,
        # TASK-054: 漏洩ガードのゲートポリシー（tiered/legacy_strict）をログへ残す
        "leak_guard_policy": leak_guard_policy,
        # TASK-055: ブラフ役職COの可否（allow/deny）をログへ残す（A/B比較の判別用）
        "allow_bluff_role_claim": bluff_role_claim_policy_label(allow_bluff_role_claim),
        # TASK-056: 時系列明示のルール（explicit/off）をログへ残す（A/B比較の判別用）
        "day_timeline_policy": day_timeline_policy,
        # TASK-057: プロンプト契約の簡素化（simplified/legacy）をログへ残す（A/B比較の判別用）
        "prompt_contract_policy": prompt_contract_policy,
        # TASK-058: core圧縮の段階（layered/off）をログへ残す（A/B比較の判別用）
        "core_compression_policy": core_compression_policy,
        # TASK-060: 議論プロンプトの節順（v3/v2/auto）をログへ残す（A/B比較の判別用）。
        # 配置と重複行の変更のみで、発言の口調・論点・結論には干渉しない設定である。
        "prompt_layout_policy": prompt_layout_policy,
        # TASK-062: 論理状態の正規化ポリシー（normalized/off）をログへ残す
        "continuity_policy": continuity_policy,
        # TASK-065: 運用安定化の4方策をログへ残す（A/B比較の判別用）。整合チェックは
        # off/warn/cap、意図分離は separated/literal、名前正規化は normalized/literal、
        # 処置スコープは action/model。いずれも本文の自由には干渉しない設定のみ。
        "context_integrity_policy": context_integrity_policy,
        "timeline_intent_policy": timeline_intent_policy,
        "player_name_resolution_policy": player_name_resolution_policy,
        "output_stabilization_scope": output_stabilization_scope,
        "base_url": args.base_url,
        "show_private": args.show_private,
        "response_language": getattr(args, "language", None),
        # TASK-044: 个性の上書きと、実際に各プレイヤーへ適用されたプロファイルを残す
        # （free は自由発言モード。ログで個性とモデル・役職を区別できる）。
        "personality_override": personality_override,
        "personality_profiles": {
            player.name: player.personality_profile for player in state.players
        },
    }
    print(f"行動別生成上限（規定）: {' / '.join(ACTION_BUDGET_DEFAULT_LABELS.values())}")
    print(
        f"CLI全体num_predict: "
        f"{args.num_predict if args.num_predict is not None else '未設定（行動別既定を使う）'}"
    )
    print(
        "CLI行動別num_predict: "
        + (
            " / ".join(f"{key}={value}" for key, value in action_num_predict.items())
            if action_num_predict
            else "未設定"
        )
    )
    print(f"行動別思考生成（規定）: {' / '.join(THINKING_DEFAULT_LABELS.values())}")
    print(
        "CLI全体思考設定: "
        + (
            f"{'on' if args.thinking == 'on' else 'off'}"
            if args.thinking in {"on", "off"}
            else "未設定（行動別既定を使う）"
        )
    )
    print(
        "CLI行動別思考設定: "
        + (
            " / ".join(
                f"{key}={'on' if value else 'off'}"
                for key, value in action_thinking.items()
            )
            if action_thinking
            else "未設定"
        )
    )
    print(
        "動的コンテキスト: "
        + (
            "on（送信プロンプト実測サイズ＋生成上限へnum_ctxを調整・発言履歴を予算化）"
            if getattr(args, "dynamic_context", True)
            else "off（num_ctxは設定値をそのまま送信）"
        )
    )
    print(
        "本文予算予約（TASK-053）: "
        + (
            "on（初回リクエストでモデル別のthinking余白と本文最低予算を分離し、"
            "思考が上限を食い潰しても空応答にしない）"
            if bool(getattr(args, "body_reservation", True))
            else "off（num_predictは設定値のまま。比較ラン用）"
        )
    )
    question_hint_mode_labels = {
        "strict": "strict（1文目に回答対象・形式違反は1回の意味的再要求）",
        "soft": "soft（質問は観測として提示・回答の強制と再生成なし）",
        "none": "none（質問欄・回答指示なし・完全中立の証拠のみ）",
    }
    print(f"質問ヒントモード: {question_hint_mode_labels[question_hint_mode]}")
    novelty_policy_labels = {
        "publish": "publish（低新規性でも本文をそのまま公開・再生成も文の除去もしない）",
        "append": "append（本文を保持し、未論点への短い任意の追記要求を1回だけ送る）",
        "legacy_repair": "legacy_repair（旧挙動: 重複文の除去→1回の意味的再要求）",
    }
    print(f"低新規性ポリシー: {novelty_policy_labels[novelty_policy]}")
    leak_guard_policy_labels = {
        "tiered": (
            "tiered（未公開の真の情報だけ遮断。公開済みCO・判定の再言及や他者の役職推測は"
            "本文そのまま公開。漏洩文だけの中立化を優先し、全文再生成を避ける）"
        ),
        "legacy_strict": (
            "legacy_strict（旧挙動: 推理メモ含む全文照合で1件でも検出すれば全文再生成→削句）"
        ),
    }
    print(f"漏洩ガードポリシー: {leak_guard_policy_labels[leak_guard_policy]}")
    bluff_role_claim_labels = {
        True: (
            "allow（どの役職のCO・対抗COも自由。判定行は本人の申告として記録され、"
            "エンジンが発行した実際の占い結果とは台帳上で分離する。真偽は公開されない）"
        ),
        False: (
            "deny（実保有者だけがCO・判定を公開できる。非保有者のCO行と結果行は公開前に除去し、"
            "本文は一字一句そのまま残す。再生成は要求しない）"
        ),
    }
    print(f"ブラフ役職CO（TASK-055）: {bluff_role_claim_labels[allow_bluff_role_claim]}")
    print("player設定の行動別num_predict:")
    for player in state.players:
        per_action = (
            " / ".join(f"{key}={value}" for key, value in player.num_predict_by_action.items())
            if player.num_predict_by_action
            else "未設定"
        )
        per_thinking = (
            " / ".join(
                f"{key}={'on' if value else 'off'}"
                for key, value in player.thinking_by_action.items()
            )
            if player.thinking_by_action
            else "未設定"
        )
        print(
            f"  {player.name}: 全体="
            f"{player.num_predict if player.num_predict is not None else '未設定'} / 行動別={per_action}"
        )
        print(
            f"    思考生成: 全体="
            f"{'未設定' if player.thinking is None else ('on' if player.thinking else 'off')}"
            f" / 行動別={per_thinking}"
        )

    private_logging = args.show_private or args.private_log
    logger = GameLogger(game_id, log_dir, meta, private_enabled=private_logging)
    show_private = args.show_private
    if logger.private_log_file is not None:
        print(f"管理者用JSONLログ保存先: {logger.private_log_file}")

    def emit_action_event(event: dict[str, object]) -> None:
        public_event = {
            key: value
            for key, value in event.items()
            if key not in {
                "content",
                "selected_target",
                "parsed_target",
                "error",
                "response_metadata",
                "failure_reason",
                "failure_reasons",
            }
        }
        if event.get("error") is not None:
            public_event["error_occurred"] = True
        logger.append_record(public_event, event)
        event_type = event["type"]
        day = event["day"]
        phase = event["phase_label"]
        player = event["player"]
        action = event["action"]
        action_labels = {
            "wolf_discussion": "人狼相談",
            "attack": "襲撃選択",
            "seer": "占い",
            "discussion": "議論",
            "vote": "投票",
        }
        label = action_labels.get(str(action), str(action))
        prefix = f"[Day {day}][{phase}][{player}]"
        private = bool(event["private"])

        if event_type == "llm_action_started":
            print(f"{prefix} 応答待ち... ({label})", flush=True)
        elif event_type == "llm_action_parse_failed":
            if event.get("will_retry"):
                print(f"{prefix} 応答を解析できません。再要求します。", flush=True)
            else:
                print(f"{prefix} 再要求後も解析できません。", flush=True)
        elif event_type == "llm_request_configuration":
            request_options = event.get("request_options")
            request_options = (
                request_options if isinstance(request_options, dict) else {}
            )
            context = event.get("context_diagnostics")
            context = context if isinstance(context, dict) else {}
            requested_context = request_options.get("num_ctx", "未指定")
            loaded_context = context.get("ollama_ps_context", "不明")
            process_env_context = context.get(
                "process_ollama_context_length_env", "未設定"
            )
            modelfile_context = context.get("modelfile_num_ctx", "未設定")
            loaded_processor = context.get("ollama_ps_processor", "不明")
            loaded_runner = context.get("ollama_ps_runner", "不明")
            process_parallel = context.get(
                "process_ollama_num_parallel_env", "未設定"
            )
            request_number = event.get("request_number", 1)
            request_reason = event.get("request_reason", "初回")
            tool_based = event.get("tool_based", False)
            tool_label = "ツール付き" if tool_based else "文章のみ"
            thinking_enabled = event.get("thinking_enabled")
            thinking_label = (
                "未指定"
                if thinking_enabled is None
                else ("on" if thinking_enabled else "off")
            )
            retry_fit_note = ""
            if event.get("retry_budget_fit"):
                retry_fit_note = (
                    " (再要求でnum_predict・num_ctxを要求サイズへ縮小"
                    f": 元num_ctx={event.get('num_ctx_before_retry_fit', '未指定')})"
                )
            if event.get("dynamic_context_fit"):
                retry_fit_note += (
                    " (動的コンテキスト"
                    f": 元num_ctx={event.get('num_ctx_before_dynamic_fit', '未指定')}"
                    f"→必要量概算={event.get('context_needed_tokens', '不明')}"
                    f", 優先元={event.get('num_ctx_setting_source', '不明')})"
                )
            if event.get("effective_context_limit") is not None:
                # TASK-046: env・Modelfile・/api/ps・設定から解決した実効上限の単一値を表示する。
                retry_fit_note += (
                    f" (実効上限={event.get('effective_context_limit')}"
                    f", 解決元={event.get('context_budget_source', '不明')}"
                    + (", 設定値>サーバー観測値(不一致)" if event.get("context_limit_conflict") else "")
                    + ")"
                )
            print(
                f"{prefix} LLM要求 #{request_number} [{request_reason}] ({tool_label}): API要求num_ctx={requested_context}, "
                f"Modelfile PARAMETER num_ctx={modelfile_context}, "
                f"(優先元={event.get('num_ctx_source')}), "
                f"/api/ps CONTEXT={loaded_context}, PROCESSOR={loaded_processor}, "
                f"RUNNER={loaded_runner}, "
                f"PythonプロセスのOLLAMA_CONTEXT_LENGTH={process_env_context} "
                f"(サーバー側と異なる場合があります), OLLAMA_NUM_PARALLEL="
                f"{process_parallel} (サーバー環境とは異なる場合があります), "
                f"num_predict={request_options.get('num_predict', '未指定')} "
                f"(行動={event.get('action_budget_key')}, 優先元={event.get('num_predict_source')}), "
                f"think={thinking_label} (優先元={event.get('thinking_source')}), "
                f"max_retries={event.get('max_retries')}, "
                f"timeout再送上限={event.get('timeout_retries', event.get('max_retries'))}, "
                f"timeout={event.get('request_timeout_seconds')}秒{retry_fit_note}",
                flush=True,
            )
        elif event_type == "thinking_unavailable":
            print(
                f"{prefix} 思考生成（think）非対応と判明 → think=offで再送し、以後の行動もoffで送信します",
                flush=True,
            )
        elif event_type == "thinking_budget_exhausted":
            print(
                f"{prefix} 思考が生成上限を消費して本文が空 → 以後の議論・人狼相談はthink=offで送信します "
                f"(生成トーク数={event.get('eval_count')}, num_predict={event.get('num_predict')})",
                flush=True,
            )
        elif event_type == "model_timeout_downgrade":
            print(
                f"{prefix} タイムアウト観測 → 以後の議論・人狼相談はthink=off・"
                f"num_predict<={event.get('num_predict')}・発言予算{event.get('evidence_ratio')}で送信します",
                flush=True,
            )
        elif event_type == "effective_context_resolution":
            # TASK-046: env・Modelfile・/api/ps・設定から解決した単一値を表示する。
            # 設定由来・サーバー観測値・採用値を別々に示し、解釈の曖昧さを解消する。
            print(
                f"{prefix} 実効コンテキスト上限を解決: 採用={event.get('effective_context_limit', '未設定')} "
                f"(解決元={event.get('context_budget_source', '不明')}, "
                f"設定={event.get('configured_context_limit', '未設定')}, "
                f"サーバー観測={event.get('server_observed_context_limit', '未設定')})"
                + (
                    " — 設定値がサーバー観測値（/api/ps・env・Modelfile）を超える不一致あり"
                    if event.get("context_limit_conflict")
                    else ""
                ),
                flush=True,
            )
        elif event_type == "context_compression_stage":
            # TASK-046: 再要求のcore-ledger圧縮で保持した証拠を表示する（発言の自由度は維持）。
            print(
                f"{prefix} コンテキスト圧縮 [{event.get('compression_reason', 'retry')}]: "
                "公開CO・占い判定・投票・死亡・直近発言・本人への未回答質問・直近反証を保持",
                flush=True,
            )
        elif event_type == "ledger_repair":
            reason = event.get("reason")
            if reason == "meta_instruction_echo":
                print(
                    f"{prefix} メタ発話（システム・修復定型文）を本文から除去して公開します",
                    flush=True,
                )
            elif reason == "repeated_discussion_topic":
                print(
                    f"{prefix} 当日の議論で重複した文を本文から除去しました（メタ定型文は挿しません・legacy_repairのみ）",
                    flush=True,
                )
        elif event_type == "discussion_novelty_measured":
            # TASK-048: 出力後の本文を 新規証拠/反証/立場変更/質問への応答 などで計測する
            print(
                f"{prefix} 新規性計測 [{event.get('novelty_primary', 'unknown')}]: "
                f"score={event.get('novelty_score')}, 新論点={event.get('novelty_new_topics')}, "
                f"連呼={event.get('novelty_restated_topics')}, "
                f"正当な再確認={event.get('novelty_legitimate_repetitions')}",
                flush=True,
            )
        elif event_type == "llm_discussion_low_novelty_kept":
            print(
                f"{prefix} 低新規性 [{event.get('reason', 'repeated_discussion_topic')}]: "
                "本文をそのまま公開しました（再生成・文の除去なし）"
                + ("（追記要求は拒否/失敗）" if event.get("append_declined") else ""),
                flush=True,
            )
        elif event_type == "discussion_novelty_appended":
            print(
                f"{prefix} 低新規性 [{event.get('reason', 'repeated_discussion_topic')}]: "
                "既存本文を保持し、未論点への短い追記だけを追加しました（全文再生成なし）",
                flush=True,
            )
        elif event_type == "llm_action_completed":
            retries = int(event["total_retries"])
            fallback = "有" if event["fallback_used"] else "無"
            print(
                f"{prefix} 完了 ({event['elapsed_seconds']}s, "
                f"リトライ {retries}回, フォールバック {fallback})",
                flush=True,
            )
            if event.get("error"):
                print(f"  エラー: {event['error']}", flush=True)
            if event.get("selected_target") and (
                not private or show_private
            ):
                print(f"  {label}: {event['selected_target']}", flush=True)
            content = event.get("content")
            if (
                content
                and action not in {
                    "discussion",
                    "discussion_body",
                    "discussion_claim",
                    "discussion_safety_retry",
                }
                and (not private or show_private)
            ):
                print(f"  {player}: {content}", flush=True)

    # 台帳のCO・占い結果主張は本文メッセージに含まれない形式があるため、本文の直前に
    # 表示用の行を積んでおき、同一発言者の本文が出る時点で（本文に既に含まれていなければ）flushする。
    pending_claims: list[dict[str, object]] = []
    pending_claim_speaker: str | None = None

    def flush_pending_claims(speech: str | None = None) -> None:
        """未表示のCO・占い結果主張をターミナルへ出す。本文に既にあれば出さない。"""
        nonlocal pending_claims, pending_claim_speaker
        if not pending_claims:
            pending_claim_speaker = None
            return
        if speech is None or not all(
            claim_covered_by_speech(event, speech) for event in pending_claims
        ):
            for event in pending_claims:
                print(public_claim_display_line(event), flush=True)
        pending_claims = []
        pending_claim_speaker = None

    def on_engine_event(event: dict[str, object]) -> None:
        """Persist engine events and display public speeches as they arrive."""
        nonlocal pending_claims, pending_claim_speaker
        # TASK-064: 能力行使の分類観測（実行記録との一致可否・カテゴリ・矛盾）は保有者別の
        # 真偽を含むため、公開JSONLには出さない（TASK-055で決めた線引きと同じ）。
        # 管理者レコード（private側）だけに残し、公開側は中立な本文とID表示のままとする。
        public_event = {
            key: value
            for key, value in event.items()
            if key
            not in {
                "role_ability_observations",
                "claim_to_ability_turn_delta",
                "claim_to_ability_order",
            }
        }
        logger.append_record(public_event, event)
        event_type = event.get("type")
        if event_type in PUBLIC_CLAIM_DISPLAY_TYPES:
            claimant = str(event.get("claimant", "?"))
            if pending_claim_speaker not in (None, claimant):
                flush_pending_claims()
            pending_claim_speaker = claimant
            pending_claims.append(event)
            return
        if event_type == "discussion_message":
            speaker = str(event.get("speaker", event.get("player", "?")))
            content = str(event.get("content", ""))
            if pending_claim_speaker == speaker:
                # 本文にCO・占い結果が既に書かれている場合は重複表示しない
                flush_pending_claims(content)
            else:
                flush_pending_claims()
            print(
                f"  [{speaker}]: {content}",
                flush=True,
            )
            return
        flush_pending_claims()

    # クライアントとポリシーの設定
    client: OllamaClient | None = None
    actions: ActionPolicy
    active_call: dict[str, object] | None = None
    if args.mode == "ollama":
        client = OllamaClient(base_url=args.base_url, timeout=args.timeout, max_retries=2)
        if not client.ping():
            err_msg = f"Ollamaサーバー（{args.base_url}）に接続できません。Ollamaが起動しているか確認してください。"
            print_banner(f"[エラー] {err_msg}")
            logger.finalize(state, traces=None, error=err_msg)
            print(f"ログを保存しました: {logger.log_file}")
            return 1
        # TASK-065: ゲーム開始前に実効コンテキストの証拠を1度だけ集める。設定値（configured）と
        # サーバー観測値・ロード実測の不一致を、約360秒のtimeoutのあとで初めて知る状態を避ける。
        # 方策 cap のときだけ採用値を安全側へ縮め（warnは警告のみで送信値は不変）、議論本文へは
        # 何も書かない。診断が取れなくてもゲームはそのまま続行する。
        run_context_integrity_preflight(client, state, logger)
        def on_action_event(event: dict[str, object]) -> None:
            nonlocal active_call
            if event["type"] == "llm_action_started":
                active_call = {
                    "day": event["day"],
                    "phase": event["phase"],
                    "player": event["player"],
                    "action": event["action"],
                }
            emit_action_event(event)
            if event["type"] == "llm_action_completed":
                active_call = None

        actions = OllamaActions(
            state=state,
            client=client,
            seed=args.seed,
            options=options,
            event_callback=on_action_event,
            discussion_claim_mode=getattr(
                args, "discussion_claim_mode", "legacy"
            ),
            context_strategy=getattr(
                args, "context_strategy", "hybrid_retrieval"
            ),
        )
    else:
        actions = MockActions()

    engine = GameEngine(
        state,
        actions,
        seed=args.seed,
        event_callback=on_engine_event,
    )

    def cleanup_and_exit(signum=None, frame=None):
        print("\n\n[中断] シグナルを受信しました。ログを保存して終了します...")
        if active_call is not None:
            logger.append_record(
                {
                    "type": "llm_action_interrupted",
                    "active_action": active_call,
                }
            )
            print(f"  中断した処理: {active_call}", flush=True)
        flush_pending_claims()
        traces = getattr(actions, "traces", None)
        logger.finalize(state, traces=traces, error="interrupted")
        print(f"ログ保存完了: {logger.log_file}")
        sys.exit(130)

    # Ctrl+C ハンドラ
    try:
        signal.signal(signal.SIGINT, cleanup_and_exit)
    except (ValueError, AttributeError):
        pass  # Windowsや特定スレッドでsignal設定が制限される場合を考慮

    try:
        # 開始前の勝敗判定
        engine.check_winner()

        while state.winner is None:
            # 1. 夜フェーズ（人狼襲撃）
            print_round_header(state.day, Phase.NIGHT_WOLF, state.alive_players)
            engine.run_wolf_phase()
            if state.pending_attack:
                logger.append_record({
                    "type": "action_pending_attack",
                    "day": state.day,
                    "private": True,
                }, {
                    "type": "action_pending_attack",
                    "day": state.day,
                    "target": state.pending_attack,
                })
            elif state.day == 1:
                print("初夜のため人狼の相談・襲撃はありません。", flush=True)
                logger.append_record({
                    "type": "first_night_no_attack",
                    "day": state.day,
                    "reason": "first_night_rule",
                })
            if state.winner is not None:
                break

            # 2. 夜フェーズ（占い師）
            engine.run_seer_phase()
            logger.append_record({
                "type": "seer_results",
                "day": state.day,
                "private": True,
                "result_count": len(state.seer_results),
            }, {
                "type": "seer_results",
                "day": state.day,
                "results": dict(state.seer_results),
            })
            if show_private and state.seer_results:
                print(f"  占い結果（管理者表示）: {state.seer_results}", flush=True)

            # 3. 朝フェーズ（犠牲者発表）
            prev_alive = set(state.alive_players)
            engine.resolve_morning()
            dead_in_night = list(prev_alive - set(state.alive_players))
            print_round_header(state.day, Phase.MORNING, state.alive_players)
            if dead_in_night:
                print(f"昨夜の犠牲者: {', '.join(dead_in_night)}")
            else:
                print("昨夜の犠牲者はいませんでした。")
            logger.append_record({
                "type": "morning_casualty",
                "day": state.day,
                "dead": dead_in_night,
                "alive": state.alive_players,
            })
            if state.winner is not None:
                break

            # 4. 昼フェーズ（議論）
            print_round_header(state.day, Phase.DAY_DISCUSSION, state.alive_players)
            engine.run_discussion()

            # 5. 投票フェーズ（処刑）
            prev_alive = set(state.alive_players)
            vote_day = state.day
            engine.run_voting()
            dead_in_vote = list(prev_alive - set(state.alive_players))
            print(f"\n--- 投票結果 (Day {vote_day}) ---")
            for voter, target in state.votes.items():
                print(f"  {voter} -> {target}")
            if dead_in_vote:
                print(f"処刑されたプレイヤー: {', '.join(dead_in_vote)}")
            print(f"生存者 ({len(state.alive_players)}名): {', '.join(state.alive_players)}")
            logger.append_record({
                "type": "voting_result",
                "day": vote_day,
                "votes": state.votes,
                "executed": dead_in_vote,
                "alive": state.alive_players,
            })

    except KeyboardInterrupt:
        cleanup_and_exit()
    except OllamaError as err:
        print_banner(f"[エラー] Ollama通信エラーが発生しました: {err}")
        traces = getattr(actions, "traces", None)
        logger.finalize(state, traces=traces, error=str(err))
        print(f"ログを保存しました: {logger.log_file}")
        return 1
    except Exception as exc:
        print_banner(f"[エラー] 予期せぬエラーが発生しました: {exc}")
        # 予期せぬエラーはスタックトレースも出力する（メッセージだけでは原因箇所が不明なため）。
        traceback.print_exc()
        traces = getattr(actions, "traces", None)
        logger.finalize(state, traces=traces, error=str(exc))
        print(f"ログを保存しました: {logger.log_file}")
        return 1

    # ゲーム終了表示
    # 最終発言に付随するCO・占い結果主張が未表示ならここで出力する
    flush_pending_claims()
    winner_str = "村人陣営 (village)" if state.winner == "village" else "人狼陣営 (werewolf)"
    print_banner(f"ゲーム終了！ 勝者: {winner_str}")
    print(
        "勝敗判定人数: "
        f"人狼 {state.winner_counts.get('living_wolves', 0)}名 / "
        f"狂人 {state.winner_counts.get('living_madmen', 0)}名（勝敗判定から除外） / "
        f"村人陣営 {state.winner_counts.get('living_village', 0)}名"
    )
    winner_reason_labels = {
        "no_living_wolves": "生存している人狼が0名",
        "living_wolves_reached_village_parity": "生存人狼数が村人陣営人数以上",
    }
    if state.winner_reason is None:
        raise RuntimeError("The game ended without a winner reason.")
    reason_label = winner_reason_labels[state.winner_reason]
    print(f"勝敗理由: {reason_label} ({state.winner_reason})")

    print("\n【最終プレイヤー一覧】")
    for p in state.players:
        status_str = "生存" if p.is_alive else "死亡"
        role_str = ROLE_NAMES_JA.get(p.role, p.role.value)
        print(f"  - {p.name.ljust(10)}: {role_str.ljust(6)} ({p.model}) [{status_str}]")

    traces = getattr(actions, "traces", None)
    logger.finalize(state, traces=traces)
    print(f"\nゲームID: {game_id}")
    print(f"JSONLログ保存先: {logger.log_file}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Werewolf Game CLI")
    parser.add_argument(
        "--mode",
        choices=["mock", "ollama"],
        default="mock",
        help="Game execution mode ('mock' or 'ollama')",
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        default=REPO_ROOT / "config" / "players.json",
        help="Path to players.json configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable role assignment and tie-breaking",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout for Ollama (seconds)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for LLM generation",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help=(
            "Context window length sent to Ollama (num_ctx); when omitted, "
            "Ollama server defaults apply"
        ),
    )
    parser.add_argument(
        "--override-player-num-ctx",
        action="store_true",
        help=(
            "Make --num-ctx override per-player num_ctx settings; requires --num-ctx"
        ),
    )
    parser.add_argument(
        "--question-hint-mode",
        choices=["strict", "soft", "none"],
        default="soft",
        help=(
            "How strongly the prompt pushes answering the previous question. "
            "soft (default): shows the question as observation only and never "
            "regenerates for answer format. strict: legacy behavior (answer must "
            "appear in the first sentence, one semantic retry). none: no question "
            "section or answer instruction at all (fully neutral evidence)."
        ),
    )
    parser.add_argument(
        "--novelty-policy",
        choices=["publish", "append", "legacy_repair"],
        default="publish",
        help=(
            "What to do when a discussion body adds no new point (measured, not forced). "
            "publish (default): keep and publish the body as-is, no regeneration and no "
            "sentence removal. append: keep the body and send one short optional "
            "addition request only. legacy_repair: old behavior (remove repeated "
            "sentences, then one semantic retry) for A/B comparison."
        ),
    )
    parser.add_argument(
        "--leak-guard-policy",
        dest="leak_guard_policy",
        choices=["tiered", "legacy_strict"],
        default="tiered",
        help=(
            "How the secrecy guard classifies a discussion body before publishing. "
            "tiered (default): only unpublished true information is withheld; restating "
            "a published CO or verdict and speculating about other roles stays published, "
            "and only the leaking sentence is neutralized instead of regenerating the "
            "whole body. legacy_strict: pre-TASK-054 behavior (scan everything including "
            "the private reasoning memo, regenerate the whole body on any hit) for "
            "false-positive and latency comparison."
        ),
    )
    parser.add_argument(
        "--bluff-role-claim",
        dest="bluff_role_claim",
        choices=["allow", "deny"],
        default="allow",
        help=(
            "Whether a player may publish a role CO that they do not actually hold "
            "(bluff CO) and a divination verdict they do not really have. allow "
            "(default): any CO and counter-claim stays published; the line is recorded "
            "as the claimant's own statement and never as an engine-issued result, so "
            "no one in-game learns who is the real Seer. deny: only the true holder of "
            "a role may publish its CO or verdict; such lines are dropped before "
            "publishing while the rest of the body is kept word for word (no "
            "regeneration). Use this for A/B runs of bluff freedom."
        ),
    )
    # TASK-065: 運用安定化の4設定（実効コンテキスト整合／未発言者の意図分離／プレイヤー名
    # 表記ゆれの内部解決／thinking・timeout処置のスコープ）。いずれも観測と数値の制御だけで、
    # 議論本文の語彙・口調・結論には干渉せず、旧挙動へ戻してA/B比較できる。
    parser.add_argument(
        "--context-integrity",
        dest="context_integrity",
        choices=["off", "warn", "cap"],
        default="warn",
        help=(
            "How the effective context limit is checked before sending requests. "
            "warn (default): keep the configured value but log configured / "
            "server_observed / loaded separately and print a warning at startup, so "
            "a mismatch is known before a timeout instead of after it. cap: adopt "
            "the safe value confirmed from server evidence when the configuration "
            "exceeds it. off: skip the check (pre-TASK-065 behavior for A/B runs). "
            "This never adds constraint text to the discussion prompt."
        ),
    )
    parser.add_argument(
        "--timeline-intent",
        dest="timeline_intent",
        choices=["separated", "literal"],
        default="separated",
        help=(
            "How future-reference detection reads sentences about a player who has "
            "not spoken yet. separated (default): only claiming that such a player "
            "already said something counts as a future reference; asking them to "
            "explain from now on and speculating about their next action are logged "
            "as separate intents and never as violations. literal: pre-TASK-065 flat "
            "counting for A/B comparison of false-positive rates. Bodies stay "
            "published word for word either way; questioning and doubt stay free."
        ),
    )
    parser.add_argument(
        "--player-name-resolution",
        dest="player_name_resolution",
        choices=["normalized", "literal"],
        default="normalized",
        help=(
            "How player names written by a model are matched inside the ledger and "
            "diagnostics. normalized (default): resolve spacing variants, full-width "
            "text, lowercase spellings and alternative labels into the canonical name "
            "for internal lookups only, so a non-breaking space no longer triggers an "
            "unknown-player repair; numbers with no such player stay unresolved and "
            "are still observed. literal: old exact-string matching (A/B baseline). "
            "Published text is never rewritten in either mode."
        ),
    )
    parser.add_argument(
        "--output-stability-scope",
        dest="output_stability_scope",
        choices=["action", "model"],
        default="action",
        help=(
            "Where timeout narrowing and thinking-off switches apply after a failure. "
            "action (default): apply them to that model and action only, using the "
            "recent per-action history, and lift them once bodies keep arriving - a "
            "vote that ran out of budget no longer constrains discussion or divination "
            "for the same model. model: legacy whole-model downgrade (pre-TASK-065) "
            "for A/B comparison. Neither mode changes tone, topics or conclusions, and "
            "a generated body is always preserved."
        ),
    )
    # TASK-058: core予算の段階圧縮（anchor/recent/historical に分け、歴史情報の本文だけを短い行へ畳む）
    parser.add_argument(
        "--core-compression",
        dest="core_compression",
        choices=["layered", "off"],
        default="layered",
        help=(
            "How discussion prompts spend their core budget. layered (default): evidence keeps "
            "three retention layers - anchor (rules, secret boundaries, the current question, the "
            "action format, your own action record) always stays in full, recent holds this turn's "
            "prior speech and live claims, and historical collapses older public speech that is "
            "already normalized into the ledger into a short line that keeps the point, speaker "
            "and [speech:N] id instead of the body (the body never leaves the public log). Plain "
            "tail-truncation is disabled: the question target, counter-evidence, role COs and their "
            "reasons are withheld last. off: pre-TASK-058 behaviour (drop whole records only) for "
            "A/B comparison of prompt size, thinking-only output and truncations. Neither mode "
            "prescribes tone, topics or reasoning order."
        ),
    )
    # TASK-060 / TASK-062 / TASK-064: 議論プロンプトの節順（配置だけ変え、内容と発言の自由は変えない）
    parser.add_argument(
        "--prompt-layout",
        dest="prompt_layout",
        choices=["v5", "v4", "v3", "v2", "auto"],
        default=PROMPT_LAYOUT_POLICY_DEFAULT,
        help=(
            "Where each information block sits inside the discussion prompt. Only positions and "
            "duplicate wording change - tone, length, topics, conclusions and reasoning order stay "
            "the model's own. v5 (default): the v4 chain plus [ROLE_ACTION_BOUNDARY], the short "
            "engine-issued line pair that separates what the engine actually executed for you "
            "([DONE], divination/attack nights - results stay in your private section) from what "
            "you publicly claimed ([CLAIM], unverified), sitting directly after the current "
            "question and immediately before your own speeches, so bluffing, doubt and stance "
            "changes stay free while an ability sentence cannot be misread as an engine record; "
            "no bodies are restated. v4: the same chain without that boundary section (A/B base "
            "for prompt size and unsupported_role_action rates); it keeps [CONTINUITY_CHECKPOINT] "
            "after the speech evidence and before the current question. v3: no checkpoint either - "
            "your action record, board facts and engine-issued "
            "private info sit early; weak-middle band holds only droppable derived content (theory "
            "card, topic digest, hypothesis diff, temporary guidance, round header); the tail runs "
            "claim index -> timeline -> speech evidence -> current question -> your speeches and "
            "your reasoning memo -> neutral guidelines -> secrecy boundary -> behavior contract -> "
            "output format -> final checklist, so the format spec is no longer buried behind "
            "advice, and header lines that restate a rule written once elsewhere (first person, "
            "response language, off-record discipline, dead-player handling) are dropped. "
            "v2: TASK-058 ordering. auto: follow --prompt-contract (legacy keeps the old byte "
            "order). Compare one seed with prompt_zone_*_chars, prompt_gap_*, "
            "prompt_instructions_to_end_chars, prompt_theory_card_chars and "
            "duplicate_constraint_total, the continuity_* keys (avg_continuity_state_chars, "
            "evidence_continuity_folded_total, continuity_fold_saved_chars_total), and the "
            "TASK-064 keys (avg_claim_boundary_chars, avg_gap_role_boundary_to_question, "
            "unsupported_role_action_count, co_correction_shortening_ratio)."
        ),
    )
    # TASK-062: 日をまたぐ論理状態の正規化（エンジン発行の状態差分だけを使い、口調・論点には干渉しない）
    parser.add_argument(
        "--continuity",
        dest="continuity",
        choices=["normalized", "off"],
        default=CONTINUITY_POLICY_DEFAULT,
        help=(
            "How the player carries logical state across days and turns. normalized (default): "
            "the engine issues [CONTINUITY_CHECKPOINT] blocks at day boundaries and hands over "
            "only what changed since the player last saw it - confirmed facts (deaths, ballots, "
            "alive count), public claims and seer reports as unverified statements, the player's "
            "own standing readings, refuted hypotheses shown as history, and open questions - each "
            "as one short line with an observed stamp and evidence ids, with authority levels kept "
            "apart so an old inference is never read as a new fact. Unchanged items are not "
            "restated, and the bodies of ordinary speech from normalized days collapse into "
            "[speech:N] reference lines in the prompt while staying in the public log. Nothing "
            "here orders reasoning, fixes a conclusion or sets a tone; changing your mind and "
            "explaining why stays free. off: no checkpoint section (pre-TASK-062 behaviour) for "
            "A/B comparison of prompt size, answer rates and stance consistency."
        ),
    )
    # TASK-057: プロンプト契約の簡素化（同じ制約・発言本文の再掲を減らし、不足時は optional だけ落とす）
    parser.add_argument(
        "--prompt-contract",
        dest="prompt_contract",
        choices=["simplified", "legacy"],
        default="simplified",
        help=(
            "How discussion prompts state constraints and evidence. simplified (default): "
            "one [BEHAVIOR_CONTRACT] block carries each strong rule once, speech bodies live "
            "only in [CONVERSATION_EVIDENCE] (other sections refer to them by [ref:sN]), and "
            "when the allocation is still exceeded only optional blocks are dropped by stage "
            "(topic digest, hypothesis diff, temporary guidance, derived memo) - core facts, "
            "the current question, your own action record and your own reasoning memo are never "
            "dropped, and no tone, topic or conclusion is prescribed. legacy: pre-TASK-057 "
            "prompts for A/B comparison of prompt size, duplicate constraints and parse failures."
        ),
    )
    # TASK-056: Day内イベントの時系列明示（時点・IDだけを短く示す。offは旧挙動とのA/B比較用）
    parser.add_argument(
        "--day-timeline",
        dest="day_timeline",
        choices=["explicit", "off"],
        default="explicit",
        help=(
            "Whether day discussion prompts state when each public event happened. "
            "explicit (default): a short [DAY_TIMELINE] block lists the current point "
            "(Day / Turn / speech ID), what was already said before you, and which "
            "events have not happened yet (later speakers, today's vote, tonight's "
            "night actions) - it fixes no topic, tone or conclusion, and speech bodies "
            "are never rewritten. off: pre-TASK-056 prompts for A/B comparison of "
            "future-event references and context size."
        ),
    )
    parser.add_argument(
        "--no-dynamic-context",
        dest="dynamic_context",
        action="store_false",
        default=True,
        help=(
            "Disable dynamic context sizing. By default num_ctx is shrunk to the measured "
            "prompt size plus the per-action generation budget (and expanded up to "
            "max_num_ctx when the prompt needs it), and discussion history is budgeted to "
            "the allocated window. Use this flag for fixed-context comparison runs."
        ),
    )
    parser.add_argument(
        "--no-body-reservation",
        dest="body_reservation",
        action="store_false",
        default=True,
        help=(
            "Disable TASK-053 body-budget reservation. By default the first request per "
            "model x action reserves a separate thinking allowance plus a minimum body budget "
            "inside num_predict (raised only from measured thinking-only/empty evidence and "
            "capped per action), so thinking cannot eat the answer. Use this flag for A/B "
            "comparison runs."
        ),
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help=(
            "Maximum number of generated tokens (num_predict) as the global default. "
            "When omitted, per-action defaults from players.json or built-in defaults apply"
        ),
    )
    parser.add_argument(
        "--num-predict-by-action",
        action="append",
        default=[],
        metavar="ACTION=NUM_PREDICT",
        help=(
            "Per-action generation limit override (repeatable). Actions: "
            + ", ".join(ACTION_BUDGET_KEYS)
            + ". Example: --num-predict-by-action discussion=1024 "
            "--num-predict-by-action vote=256"
        ),
    )
    parser.add_argument(
        "--thinking",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Ollama thinking generation ('auto' keeps per-action defaults: "
            "discussion/wolf_discussion on, attack/seer/vote/co_correction off)"
        ),
    )
    parser.add_argument(
        "--discussion-claim-mode",
        choices=("legacy", "candidates", "all"),
        default="legacy",
        help=(
            "Discussion claim generation: legacy combined response, staged for "
            "seer/claimants, or staged for every player."
        ),
    )
    parser.add_argument(
        "--context-strategy",
        choices=(
            "full_transcript",
            "structured_ledger",
            "hybrid_retrieval",
            "round_transcript_memory",
        ),
        default="hybrid_retrieval",
        help=(
            "Discussion context: full public transcript, structured ledgers only, "
            "hybrid evidence retrieval, or current-round transcript plus memory."
        ),
    )
    parser.add_argument(
        "--thinking-by-action",
        action="append",
        default=[],
        metavar="ACTION=true|false",
        help=(
            "Per-action thinking generation override (repeatable). Actions: "
            + ", ".join(ACTION_BUDGET_KEYS)
            + ". Example: --thinking-by-action vote=false "
            "--thinking-by-action discussion=true"
        ),
    )
    parser.add_argument(
        "--language",
        choices=["English", "Japanese"],
        default=None,
        help="Language for player responses (English or Japanese)",
    ),
    parser.add_argument(
        "--personality",
        choices=list(PERSONALITY_CHOICE_KEYS),
        default=None,
        help=(
            "Override the configured personality profile for every player. "
            "Use 'free' for free-expression mode (no preset speaking style; "
            "tone, structure and focus are left to each model)."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=REPO_ROOT / "logs",
        help="Directory to save game JSONL logs",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API base URL",
    )
    parser.add_argument(
        "--show-private",
        action="store_true",
        help="Show private werewolf discussion and seer information in the terminal",
    )
    parser.add_argument(
        "--private-log",
        action="store_true",
        help="Write an administrator-only JSONL log without showing private content in the terminal",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    code = run_cli_game(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
