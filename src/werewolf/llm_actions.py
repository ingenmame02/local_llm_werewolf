import math
import random
import re
import time
import json
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Callable

from .action_parser import ActionKind, ActionParser
from .dead_player_resolution import (
    analyze_dead_player_references,
    classify_dead_player_sentence,
)
from .context_budget import (
    EffectiveContextResolution,
    process_env_context,
    resolve_effective_context_limit,
)
from .engine import (
    ActionPolicy,
    diagnose_discussion_body,
    extract_public_role_claim,
    extract_public_seer_result,
    is_public_seer_claim,
)
from .config import (
    ACTION_BUDGET_KEYS,
    DEFAULT_ACTION_NUM_PREDICT,
    DEFAULT_THINKING_BY_ACTION,
    GENERATION_SAMPLING_KEYS,
    MINIMAL_ACTION_MAX_NUM_PREDICT,
    MINIMAL_OUTPUT_ACTION_KEYS,
    THINKING_AFFORDABLE_NUM_PREDICT,
    THINKING_BUDGET_EXHAUSTED_REASON,
    THINKING_BUDGET_SOURCE,
    THINKING_CAPABILITY_SOURCE,
    THINKING_NUM_PREDICT_SOURCE,
)
from .thinking_profile import (
    BODY_MIN_TOKENS_BY_ACTION,
    BODY_RESERVATION_MAX_NUM_PREDICT,
    DEFAULT_THINKING_ALLOWANCE_TOKENS,
    EMPTY_CONTENT_REASON,
    GENERATION_TRUNCATED_REASON,
    MAX_THINKING_ALLOWANCE_TOKENS,
    RESERVATION_SKIPPED_TIMEOUT,
    THINKING_ONLY_REASON,
    THINKING_OFF_AFTER_EXHAUSTIONS,
    BodyReservation,
    ModelActionProfile,
    ThinkingProfiles,
    inspect_body_budget,
    reserve_body_budget,
    thinking_allowance,
)
from .discussion_format import (
    DEFAULT_QUESTION_HINT_MODE,
    QUESTION_HINT_NONE,
    QUESTION_HINT_SOFT,
    QUESTION_HINT_STRICT,
    answers_previous_question,
    extract_discussion_body,
    extract_reasoning_memo,
    has_discussion_output_markers,
    is_pass_body,
    normalize_question_hint_mode,
    normalize_result_field,
    parse_discussion_claim,
    parse_discussion_output,
)
from .discussion_novelty import (
    DEFAULT_NOVELTY_POLICY,
    NOVELTY_MEASURED_ISSUES,
    NOVELTY_POLICY_APPEND,
    NOVELTY_POLICY_LEGACY_REPAIR,
    NoveltyResult,
    classify_novelty,
    normalize_novelty_policy,
)
from .models import GameState, Message, Player, PublicRoleClaim, PublicSeerClaim, Role
from .ollama import CLAIM_ROLE_TOOL, ChatResponse, OllamaClient, OllamaError, ToolCall
from .prompts import (
    PromptBuilder,
    PromptContext,
    ROLE_NAMES_EN,
    is_japanese_language,
    language_directive,
)
# TASK-057: プロンプト契約のポリシー解決と、コンテキスト不足時の optional 節の段階除去を使う。
# core（確定事実・現在の質問・自分の行動記録・自分の推理メモ）は落とさない。
from .prompt_contract import (
    OPTIONAL_REMOVAL_MAX_STAGE,
    PROMPT_CONTRACT_POLICY_LEGACY,
    find_sections,
    normalize_prompt_contract_policy,
    optional_names_dropped_up_to,
)
from .leak_gate_policy import (
    DEFAULT_LEAK_GUARD_POLICY,
    RULE_INTERNAL_CONFIG,
    RULE_MODEL_IDENTIFIER,
    RULE_PARTNER_IDENTITY,
    RULE_PRIVATE_TEAM_REFERENCE,
    RULE_SELF_ROLE_META,
    RULE_SELF_ROLE_TRUE_CAMP,
    RULE_TEAM_ACTION_VIEW,
    RULE_THIRD_PERSON_SELF_META,
    RULE_UNPUBLISHED_SEER_RESULT,
    STRENGTH_BALANCED,
    STRENGTH_STRICT,
    LeakGuardProfiles,
    normalize_leak_guard_policy,
)
from .secret_leak_guard import LeakClassification, LeakFinding, SecretLeakGuard
from .self_consistency import SelfConsistencyGuard
# TASK-055: 役職主張の出所を分離する（CO自体は自由、実際の判定はエンジン発行情報のみ）
from .role_provenance import (
    PUBLIC_LINE_LABEL,
    SOURCE_PUBLIC_CLAIM,
)

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)

# 行動別生成上限の解決優先順位:
# player行動別 > CLI行動別 > 全体設定（対象選択・CO補正は行動別キャップへ降ろす）
# > player全体 > CLI全体 > 行動別既定値
ACTION_BUDGET_RESOLUTION_PRIORITY: tuple[str, ...] = (
    "player_by_action",
    "cli_by_action",
    "global_capped",
    "player_global",
    "cli_global",
    "default_by_action",
)

# Note shown when a light retry did not recover the output.
LIGHT_RETRY_EXHAUSTED_NOTE = (
    "If a light retry did not recover the output, adjust num_predict or the model quantization."
)

# 生成起因の応答失敗（再試行で生成上限を倍へ増やす理由）
GENERATION_ISSUE_REASONS: frozenset[str] = frozenset(
    {"empty_content", "thinking_only_response", "generation_truncated"}
)

# TASK-043: 対象選択（投票・占い・襲撃）の試行上限。再要求プロンプトは数行なので、
# 3回目の再要求はランダムフォールバックより player の意志を保つコストが低い。
TARGET_MAX_ATTEMPTS = 3
# TASK-043: 議論の試行上限。3回目は生成起因の失敗（空・思考のみ・切り詰め）だけを
# 再要求する（それ以外の失敗は2回目までの挙動をそのまま維持）。
DISCUSSION_MAX_ATTEMPTS = 3


# 再要求（タイムアウト・秘密漏洩セーフティ）では短いプロンプトで本文だけ生成するため、
# 生成上限とコンテキストを要求サイズへ絞る。思考offでも大上限・大コンテキストの再送は
# ローカル環境で数分かかる（TASK-035: Player 6のdeepseek-r1での長時間応答）。
# TASK-042: 768では投票・CO補正の再要求が切り詰め（generation_truncated）になりやすく、
# 再要求自体がフォールバックの原因になっていたため引き上げる。
RETRY_NUM_PREDICT_LIMIT = 1024
RETRY_NUM_PREDICT_SOURCE = "retry_generation_cap"
RETRY_NUM_CTX_SOURCE = "retry_context_fit"
# 日本語-heavyのプロンプト文字数からトークンを概算する係数と、ツール往復・余白の余裕。
# TASK-046: game_22754212a062 で実測 prompt_eval_count/送信文字数 = avg 0.33
# （min 0.25 / max 0.39, n=63）だった。0.8 は約2.4倍の過大概算で、動的発言履歴予算を
# 下限まで潰し、直近の発言が削除される直接原因になった。安全側の 0.5 を使う。
RETRY_NUM_CTX_TOKENS_PER_CHAR = 0.5
RETRY_NUM_CTX_MARGIN_TOKENS = 1024
RETRY_NUM_CTX_MIN_FIT = 2048

# TASK-036: 動的コンテキスト。送信プロンプトの実測サイズと行動別生成上限（思考込みの
# 残余を含む）から、実際に必要なコンテキストだけ確保する。ローカル環境では 65536 の
# 固定割り当ては KVキャッシュの確保・オフロードだけで数秒〜十数秒かかり、実際の入力が
# 5〜12kトークンでも応答時間を押し上げる。過大割り当てのみ縮小し、不足時はプレイヤーの
# max_num_ctx まで拡張する（設定の上限は超えない）。
CONTEXT_TOKENS_PER_CHAR = RETRY_NUM_CTX_TOKENS_PER_CHAR
CONTEXT_MARGIN_TOKENS = 512
CONTEXT_MIN_CTX = 2048
CONTEXT_SHRINK_RATIO = 2
# TASK-046: thinking on のリクエストでは、思考トークンが生成上限を消費し本文が
# 空・切り詰めになりやすい。実測（thinking_budget_exhausted・thinking_only）から、
# thinking分として別枠の余白を予算計算に含める。
# TASK-053: この値は「観測のない場合の既定」に降ろし、実測のあるモデル・行動では
# `ModelActionProfile` から算出したモデル別の余白（allowance）を優先して使う。
THINKING_MARGIN_TOKENS = DEFAULT_THINKING_ALLOWANCE_TOKENS
# 発言履歴（CONVERSATION_EVIDENCE相当）の文字数予算の下限・上限。
# TASK-046: 600字は直近の議論数件しか残さなかった（game_22754212a062 では42回のうち
# 40回が下限に潰れ、Day1 6発言→2件だけ保持）。議論の断裂を防ぐため下限を引き上げる。
CONTEXT_EVIDENCE_MIN_CHARS = 2000
CONTEXT_EVIDENCE_MAX_CHARS = 12000
DYNAMIC_CONTEXT_SOURCE = "dynamic_context_fit"
# タイムアウト observed モデルの議論・人狼相談では、思考off・生成上限・発言予算を
# 段階的に絞って再送する（Player 6のdeepseek-r1での長時間応答対策）。
TIMEOUT_SAFE_NUM_PREDICT = 1536
TIMEOUT_SAFE_NUM_PREDICT_SOURCE = "model_timeout_fit"
TIMEOUT_SAFE_THINKING_SOURCE = "model_timeout_fit"
TIMEOUT_EVIDENCE_RATIO = 0.6


def context_budget_breakdown(
    messages: list[dict[str, object]],
    num_predict: object,
    *,
    thinking_enabled: bool = False,
    effective_limit: object = None,
    thinking_margin: int | None = None,
) -> dict[str, object]:
    """リクエスト予算を 入力/出力(生成上限)/thinking/安全余白 に分けた構成で返す（TASK-046）。

    日本語混在プロンプトは文字数よりトーク数が大きくなりやすいため、概算係数は
    安全側を取る。thinking on では思考トークンが生成上限を消費して本文を削るため、
    thinking 分を別枠の余白として確保する。`effective_limit`（実効コンテキスト上限）
    を渡すと、収まる余地（headroom）と超過量を別々に測定できる。
    TASK-053: `thinking_margin` を渡すとモデル別の実測余白を反映する（観測のない
    モデルでは既定値を使い、過大な余白で応答時間を増やさない）。
    """
    prompt_chars = sum(len(str(message.get("content", ""))) for message in messages)
    generation = (
        num_predict
        if isinstance(num_predict, int) and not isinstance(num_predict, bool)
        else 512
    )
    input_tokens = math.ceil(prompt_chars * CONTEXT_TOKENS_PER_CHAR)
    if thinking_enabled:
        thinking_tokens = (
            int(thinking_margin)
            if isinstance(thinking_margin, int) and not isinstance(thinking_margin, bool)
            else THINKING_MARGIN_TOKENS
        )
    else:
        thinking_tokens = 0
    total = input_tokens + int(generation) + thinking_tokens + CONTEXT_MARGIN_TOKENS
    breakdown: dict[str, object] = {
        "input_tokens_estimate": input_tokens,
        "generation_limit_tokens": int(generation),
        "thinking_margin_tokens": thinking_tokens,
        "safety_margin_tokens": CONTEXT_MARGIN_TOKENS,
        "total_needed_tokens": total,
    }
    if isinstance(effective_limit, int) and not isinstance(effective_limit, bool):
        breakdown["effective_context_limit"] = effective_limit
        breakdown["headroom_tokens"] = int(effective_limit) - total
        breakdown["overflow_tokens"] = max(0, total - int(effective_limit))
    return breakdown


def request_token_estimate(
    messages: list[dict[str, object]],
    num_predict: object,
    *,
    thinking: bool = False,
    thinking_margin: int | None = None,
) -> int:
    """送信プロンプトの実測文字数と生成上限から、必要なコンテキストトーク数を概算する。

    日本語混在のプロンプトは文字数よりトーク数が大きくなりやすいため、概算係数は
    安全側を取る。生成上限は思考onの残余を含む値を受け取る前提。
    thinking on の場合は思考消費分の別枠余白（既定は `THINKING_MARGIN_TOKENS`、
    TASK-053ではモデル別実測値を `thinking_margin` で渡せる）も加える。
    """
    breakdown = context_budget_breakdown(
        messages, num_predict, thinking_enabled=thinking, thinking_margin=thinking_margin
    )
    return int(breakdown["total_needed_tokens"])


def fit_dynamic_context(
    options: dict[str, object],
    messages: list[dict[str, object]],
    *,
    ceiling: int | None = None,
    shrink_ratio: int = CONTEXT_SHRINK_RATIO,
    min_ctx: int = CONTEXT_MIN_CTX,
    thinking: bool = False,
    thinking_margin: int | None = None,
) -> tuple[dict[str, object], int | None, bool]:
    """送信オプションの `num_ctx` を実測必要量へ合わせる（ゲームと評価ハーネスで共用）。

    必要量は `request_token_estimate`（プロンプト実測文字数＋生成上限＋余白＋thinking分）。
    過大割り当て（設定値が必要量の `shrink_ratio` 倍以上）のみ縮小し、必要量不足は `ceiling`
    （プレイヤーの `max_num_ctx` 等）まで拡張する。`ceiling` 未指定時は拡張しない。
    TASK-053: `thinking_margin` でモデル別のthinking余白を渡すと、思考を実際に使う
    モデルだけコンテキストを増やし、使わないモデルの割り当ては節約したままにする。
    """
    requested_ctx = options.get("num_ctx")
    if not isinstance(requested_ctx, int) or isinstance(requested_ctx, bool):
        return options, None, False
    needed = request_token_estimate(
        messages,
        options.get("num_predict"),
        thinking=thinking,
        thinking_margin=thinking_margin,
    )
    # TASK-046: `ceiling`（実効コンテキスト上限・max_num_ctx）を超えるnum_ctxは、
    # 過大割り当ての判定にかかわらず上限へ縮小する（上限を超えるリクエストを
    # 生成しない完了条件の担保）。入力は発言履歴予算の圧縮で上限内に収める。
    if (
        isinstance(ceiling, int)
        and not isinstance(ceiling, bool)
        and ceiling > 0
        and requested_ctx > ceiling
    ):
        fitted_options = dict(options)
        fitted_options["num_ctx"] = max(min_ctx, min(needed, ceiling))
        return fitted_options, needed, True
    limit = (
        ceiling
        if isinstance(ceiling, int) and not isinstance(ceiling, bool)
        else requested_ctx
    )
    if requested_ctx >= needed * shrink_ratio:
        fitted = max(min_ctx, min(needed, limit))
        if fitted >= requested_ctx:
            return options, needed, False
        fitted_options = dict(options)
        fitted_options["num_ctx"] = fitted
        return fitted_options, needed, True
    if requested_ctx < needed and limit > requested_ctx:
        expanded = min(needed, limit)
        if expanded > requested_ctx:
            fitted_options = dict(options)
            fitted_options["num_ctx"] = expanded
            return fitted_options, needed, True
    return options, needed, False


# 修復用定型文・システムメタ語の本文混入（メタ発話の連呼）を検出する。
META_ECHO_ISSUE = "meta_instruction_echo"
META_ECHO_PATTERNS: tuple[str, ...] = (
    r"論点の繰り返しを避け",
    r"繰り返しを避け.{0,24}(新しい|別の)",
    r"新しい根拠や.{0,12}確認事項",
    r"確認事項を述べます",
    r"(生成上限|num_predict|num_ctx|num_keep|top_k|temperature)\s*(=|は|を|の|:)",
    r"(リクエスト|プロンプト|システムプロンプト|思考生成|ツール呼び出し)",
    r"(台帳の?修復|台帳補正|再要求|安全再要求|フォールバック)",
    r"(修復|補正)されました|差し止め",
    r"avoid repeating|repeating the same|repeat the same point",
    r"\b(previous response|retry|safety retry|fallback|ledger repair|corrected|suppressed|held back)\b",
    r"\b(num_predict|num_ctx|generation limit|system prompt|thinking block|tool call)\b",
    r"\bpoints to check\b|state new (?:basis|evidence|points)|wait-and-see statements",
    r"simply saying .no information.|whom you want to check what|drive discussion",
)

# プロンプトの案内文・プレースホルダをそのまま出力する幻覚（TASK-043）。
# 実機ログ game_e2991441d6f0 の Player 3（mistral-small）は
# `[RESULT] ターゲット=プレイヤーX 結果=人間` を出力し、実在しない対象の占い結果COが
# 公開された（その後「まだ占っていない」と矛盾する発言になった）。致命的扱いではなく
# 決定的修復（該当行の除去）で本文を保持し、再試行→フォールバックを作らない。
TEMPLATE_PLACEHOLDER_ISSUE = "template_placeholder_echo"
_PLACEHOLDER_PATTERNS: tuple[str, ...] = (
    r"\bplayer\s+[xX]\b",
    r"プレイヤー\s*[xXｘ]",
    r"(?:対象名|対象|ターゲット|target)\s*[:=＝]\s*[xXｘ]\b",
    r"werewolf\s*/\s*human",
    r"人狼\s*/\s*人間",
    r"\bplayer\s+name\b",
    r"\bplaceholder\b|プレースホルダ",
)

SOFT_DISCUSSION_ISSUES: frozenset[str] = frozenset(
    {
        "discussion_output_format",
        "question_not_answered_first",
        "past_action_question_unanswered",
        "repeated_discussion_topic",
    }
)

# TASK-048: 論点の再掲・同一発言の反復は `NOVELTY_MEASURED_ISSUES`（discussion_novelty）で
# 「低新規性」として計測する（誘導しない）。publish / append では本文をそのまま公開し、
# 再生成も文の除去も行わない。legacy_repair のみ旧挙動（重複文の除去→意味的再要求）を使う。

LEDGER_REPAIR_ISSUES: frozenset[str] = frozenset(
    {
        "external_game_memory",
        "unverified_past_vote_reference",
        "unverified_dead_role",
        "dead_player_question",
        "dead_player_action_target",
        # TASK-048: 旧挙動（legacy_repair）でのみ台帳修復対象。既定ポリシーでは
        # 台帳修復・再生成せず、新規性計測だけ行う。
        "repeated_discussion_topic",
        "contradictory_past_vote",
        "contradictory_past_action",
        "recorded_action_denied",
        "public_fact_denied",
        "third_person_self_reference",
        # TASK-042: 存在しないプレイヤー名・記録にない引用は、文単位で決定的に除去できるため
        # 再生成（→フォールバック）ではなく台帳修復で扱う（game_0055be638e37 Player 7/8）。
        "unknown_player_reference",
        "unverified_attributed_speech",
        # TASK-043: プレースホルダ・案内文の模倣は該当行/文を決定的に除去できるため、
        # 再生成（→フォールバック）ではなく台帳修復で扱う。
        TEMPLATE_PLACEHOLDER_ISSUE,
        META_ECHO_ISSUE,
    }
)


def classify_failure_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    if reason in {"communication_failure", "request_timeout", "http_error"}:
        return "communication"
    if reason in {"empty_content", "thinking_only_response"}:
        return "empty_or_thinking_only"
    if reason in {"generation_truncated", "thinking_budget_exhausted"}:
        return "generation_limit"
    if reason in {"invalid_target", "invalid_action_response"}:
        return "invalid_target"
    if reason in SOFT_DISCUSSION_ISSUES:
        return "soft_discussion_quality"
    if reason in {
        "discussion_output_format",
        "seer_result_without_claim",
        "wolf_result_not_disclosed",
        "inconsistent_seer_result",
        "inconsistent_role_claim",
        TEMPLATE_PLACEHOLDER_ISSUE,
    }:
        return "required_format_or_claim"
    if reason.startswith(("http_", "connection_", "timeout")):
        return "communication"
    return "semantic_integrity"


# 行動キーの別名（設定ファイルで使いやすい名前を正規キーへ寄せる）
ACTION_BUDGET_KEY_ALIASES: dict[str, str] = {
    "wolf_consult": "wolf_discussion",
    "target": "vote",
    "discussion_seer_co": "co_correction",
}


def normalize_action_budget_key(budget_key: str, *, fallback: str = "discussion") -> str:
    """行動別生成上限のキーを正規化し、未知のキーは fallback に寄せる。"""
    normalized = budget_key.strip().lower()
    if normalized in ACTION_BUDGET_KEYS:
        return normalized
    if normalized in ACTION_BUDGET_KEY_ALIASES:
        return ACTION_BUDGET_KEY_ALIASES[normalized]
    return fallback


def _global_num_predict(
    player: Player | None, cli_options: dict[str, object] | None
) -> tuple[int | None, str | None]:
    """player全体 > CLI全体 の順で全体生成上限を返す。"""
    if player is not None and player.num_predict is not None:
        return player.num_predict, "player_global"
    if cli_options is not None:
        cli_global = cli_options.get("num_predict")
        if isinstance(cli_global, int) and not isinstance(cli_global, bool):
            return cli_global, "cli_global"
    return None, None


def resolve_action_num_predict(
    player: Player | None,
    cli_options: dict[str, object] | None,
    budget_key: str,
) -> tuple[int | None, str]:
    """行動別生成上限を優先順位で解決する。

    player行動別 > CLI行動別 > 全体設定 > 行動別既定値。
    対象選択・CO補正は全体設定が大きい場合、行動別キャップへ降ろす。
    議論・人狼相談は根拠を説明できる文量を確保するため全体設定をそのまま使う。
    """
    key = normalize_action_budget_key(budget_key)
    if player is not None:
        by_action = (
            player.num_predict_by_action.get(key)
            if player.num_predict_by_action is not None
            else None
        )
        if isinstance(by_action, int) and not isinstance(by_action, bool):
            return by_action, "player_by_action"
    if cli_options is not None:
        cli_by_action = cli_options.get(f"num_predict_by_action:{key}")
        if isinstance(cli_by_action, int) and not isinstance(cli_by_action, bool):
            return cli_by_action, "cli_by_action"

    global_value, global_source = _global_num_predict(player, cli_options)
    default = DEFAULT_ACTION_NUM_PREDICT.get(key)

    if key in MINIMAL_OUTPUT_ACTION_KEYS:
        cap = MINIMAL_ACTION_MAX_NUM_PREDICT.get(key)
        if global_value is not None:
            if cap is not None and global_value > cap:
                return cap, "global_capped"
            return global_value, global_source or "default_by_action"
        if default is not None:
            return default, "default_by_action"
        return None, "ollama_service_default"

    if global_value is not None:
        return global_value, global_source or "default_by_action"
    if default is not None:
        return default, "default_by_action"
    return None, "ollama_service_default"
def resolve_thinking(
    player: Player | None,
    cli_options: dict[str, object] | None,
    budget_key: str,
) -> tuple[bool | None, str]:
    """行動別の思考生成（Ollamaの `think`）を優先順位で解決する。

    優先順位: player行動別 > CLI行動別 > player全体 > CLI全体 > 行動別既定値
    解決できない場合は (None, "ollama_service_default") でOllama側既定に委ねる。
    """
    key = normalize_action_budget_key(budget_key)
    if player is not None and player.thinking_by_action is not None:
        by_action = player.thinking_by_action.get(key)
        if isinstance(by_action, bool):
            return by_action, "player_by_action"
    if cli_options is not None:
        cli_by_action = cli_options.get(f"thinking_by_action:{key}")
        if isinstance(cli_by_action, bool):
            return cli_by_action, "cli_by_action"
    if player is not None and isinstance(player.thinking, bool):
        return player.thinking, "player_global"
    if cli_options is not None and isinstance(cli_options.get("thinking"), bool):
        return cli_options["thinking"], "cli_global"
    default = DEFAULT_THINKING_BY_ACTION.get(key)
    if default is not None:
        return default, "default_by_action"
    return None, "ollama_service_default"


def apply_thinking_floor(
    num_predict: int | None,
    num_predict_source: str,
    thinking_enabled: bool | None,
    budget_key: str,
) -> tuple[int | None, str]:
    """思考onの行動で、設定未指定（行動別既定値）時のみ思考残余用の下限を適用する。

    思考onでは思考トークンがnum_predictを消費するため、既定値の議論・人狼相談では
    最終回答（5〜8文）が最後まで生成できる残余を下限として確保する（TASK-024）。
    player・CLIの行動別/全体設定は優先順位上より高いため、下限で上書きしない。
    """
    if thinking_enabled is not True or num_predict_source != "default_by_action":
        return num_predict, num_predict_source
    floor = THINKING_AFFORDABLE_NUM_PREDICT.get(normalize_action_budget_key(budget_key))
    if (
        floor is None
        or not isinstance(num_predict, int)
        or isinstance(num_predict, bool)
        or num_predict >= floor
    ):
        return num_predict, num_predict_source
    return floor, THINKING_NUM_PREDICT_SOURCE


@dataclass(frozen=True)
class ActionTrace:
    action: ActionKind | str
    player: str
    raw_responses: tuple[str, ...]
    parsed_targets: tuple[str | None, ...]
    selected_target: str | None
    fallback_used: bool
    error: dict[str, str | int | None] | None = None
    failure_reasons: tuple[str, ...] = ()
    response_metadata: tuple[dict[str, str | int | bool | None], ...] = ()
    content_preserved: bool = False
    # TASK-034: フォールバックとは別に、本文保持・修復・安全フォールバックの内訳を記録する。
    recovered_content: bool = False
    repair_applied: bool = False
    fatal_safety_fallback: bool = False
    validation_severity: str = "ok"


@dataclass
class ModelDiscussionMetrics:
    """モデル別の議論本文品質メトリクス。形式弱モデルの判定と予算調整に使う。"""

    model: str
    calls: int = 0
    extraction_success: int = 0
    format_compliant: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)
    input_token_counts: list[int] = field(default_factory=list)
    output_token_counts: list[int] = field(default_factory=list)
    post_retry_success: int = 0
    fallbacks: int = 0
    fatal_safety_fallbacks: int = 0

    @property
    def extraction_success_rate(self) -> float:
        return self.extraction_success / max(1, self.calls)

    @property
    def format_compliance_rate(self) -> float:
        return self.format_compliant / max(1, self.calls)

    @property
    def fallback_rate(self) -> float:
        return self.fallbacks / max(1, self.calls)


class OllamaActions(ActionPolicy):
    def __init__(
        self,
        state: GameState,
        client: OllamaClient,
        seed: int | None = None,
        options: dict[str, object] | None = None,
        event_callback: Callable[[dict[str, object]], None] | None = None,
        discussion_claim_mode: str = "legacy",
        context_strategy: str = "hybrid_retrieval",
        novelty_policy: str | None = None,
    ) -> None:
        if discussion_claim_mode not in {"legacy", "candidates", "all"}:
            raise ValueError(
                "discussion_claim_mode must be 'legacy', 'candidates', or 'all'."
            )
        if context_strategy not in {
            "full_transcript",
            "structured_ledger",
            "hybrid_retrieval",
            "round_transcript_memory",
        }:
            raise ValueError(
                "context_strategy must be 'full_transcript', "
                "'structured_ledger', 'hybrid_retrieval', or "
                "'round_transcript_memory'."
            )
        self.state = state
        self.client = client
        self.options = options
        self.event_callback = event_callback
        self.discussion_claim_mode = discussion_claim_mode
        self.context_strategy = context_strategy
        # TASK-038: 質問への回答を強制する度合い（strict/soft/none）。既定soft。
        self.question_hint_mode = normalize_question_hint_mode(
            options.get("question_hint_mode")
            if options is not None
            else getattr(state, "question_hint_mode", DEFAULT_QUESTION_HINT_MODE)
        )
        # TASK-048: 低新規性時の処理ポリシー。既定 publish（そのまま公開・再生成しない）。
        # append は短い任意の追記要求のみ（既存本文は保持）、legacy_repair は旧挙動。
        self.novelty_policy = normalize_novelty_policy(
            novelty_policy
            if novelty_policy is not None
            else (
                options.get("novelty_policy")
                if options is not None
                else getattr(state, "novelty_policy", DEFAULT_NOVELTY_POLICY)
            )
        )
        self._rng = random.Random(state.seed if seed is None else seed)
        self.traces: list[ActionTrace] = []
        self._seer_info_used_day: dict[str, int] = {}
        # TASK-048: 直近の議論本文の新規性計測結果（呼び出し側が同じ本文を再計測しないため）
        self._last_novelty: NoveltyResult | None = None
        # TASK-034: モデル別の議論本文メトリクス・形式弱モデル判定。
        self._discussion_metrics: dict[str, ModelDiscussionMetrics] = {}
        self._discussion_format_weak_models: set[str] = set()
        # TASK-046: モデル別の実効コンテキスト上限（env・Modelfile・/api/ps・設定の単一値解決）。
        # 初回リクエスト時に1回だけ解決してキャッシュし、解決イベントを記録する。
        self._context_budgets: dict[str, EffectiveContextResolution] = {}
        # TASK-053: モデル×アクション別の生成観測プロファイル（thinking消費・空応答率・
        # 待ち時間）。初回リクエストの本文予約とモデル別thinking余白の算出元に使う。
        self._thinking_profiles = ThinkingProfiles()

    def thinking_profiles_snapshot(self) -> dict[str, dict[str, dict[str, object]]]:
        """TASK-053: モデル×アクション別のプロファイル（評価ログ・CLI表示用）。"""
        return self._thinking_profiles.snapshot()

    def _profile_for(self, actor: Player, budget_key: str) -> ModelActionProfile:
        key = normalize_action_budget_key(budget_key)
        return self._thinking_profiles.get_or_create(actor.model, key)

    def _body_reservation_enabled(self) -> bool:
        """TASK-053: 本文予算予約を適用するか（既定on、CLIで `--no-body-reservation`）。"""
        if self.options is None:
            return True
        return self.options.get("body_reservation") is not False

    def _thinking_margin_for(
        self, actor: Player, budget_key: str, thinking_enabled: bool | None
    ) -> int:
        """モデル別の実測thinking余白（TASK-046の全局余白ではなく観測値を使う）。"""

        return thinking_allowance(
            self._thinking_profiles.get(actor.model, normalize_action_budget_key(budget_key)),
            thinking_enabled=thinking_enabled,
            default_margin=THINKING_MARGIN_TOKENS,
        )

    def _body_reservation(
        self,
        actor: Player,
        budget_key: str,
        num_predict: object,
        num_predict_source: str,
        thinking_enabled: bool | None,
        *,
        retry: bool = False,
    ) -> BodyReservation:
        """初回リクエスト向けに「本文最低予算 + モデル別thinking余白」を分離する。

        タイムアウト観測済みのモデルでは応答時間を優先して適用しない。
        再要求は本文専用に上限を絞るため対象外（`retry=True`）。
        """

        key = normalize_action_budget_key(budget_key)
        profile = self._thinking_profiles.get(actor.model, key)
        cap = BODY_RESERVATION_MAX_NUM_PREDICT.get(key)
        timeout_fit = self._slow_model(actor) and key in {"discussion", "wolf_discussion"}
        reservation = reserve_body_budget(
            num_predict,
            num_predict_source,
            profile,
            key,
            thinking_enabled=thinking_enabled,
            cap=cap,
            enabled=self._body_reservation_enabled() and not timeout_fit,
            retry_request=retry,
        )
        if timeout_fit and not reservation.applied:
            return replace(reservation, skip_reason=RESERVATION_SKIPPED_TIMEOUT)
        return reservation

    def _resolve_context_budget(
        self,
        actor: Player,
        action: str | None = None,
        started_at: float | None = None,
    ) -> EffectiveContextResolution:
        """プロセスenv・Modelfile・Ollama API実効値・プレイヤー/CLI設定から、
        実効コンテキスト予算の採用値を1つの値として解決する（TASK-046、モデル別キャッシュ）。

        由来別名称で解釈を明確化する（レビュー指摘）: `configured`（設定由来）・
        `server_observed`（サーバー観測値）・送信値（`effective_requested_num_ctx`）。
        旧名称 `effective_context_limit` は採用値 `limit` の互換表記として記録する。
        採用値がサーバー側観測値を超える場合は conflict を記録し、設定値と実効値の
        境界の曖昧さをログで検証可能にする。発言内容・口調には干渉しない。
        """
        model = actor.model
        cached = self._context_budgets.get(model)
        if cached is not None:
            return cached
        options, _num_predict_source, _thinking, _thinking_source = self._options_for(
            actor, "discussion"
        )
        env_value, env_key = process_env_context()
        modelfile_value: object = None
        ps_value: object = None
        if callable(getattr(self.client, "context_diagnostics", None)):
            diagnostics = self.client.context_diagnostics(model)
            if isinstance(diagnostics, dict):
                modelfile_value = diagnostics.get("modelfile_num_ctx")
                ps_value = diagnostics.get("ollama_ps_context")
        resolved = resolve_effective_context_limit(
            {
                "player_setting": options.get("num_ctx") if options else None,
                "cli_setting": (
                    self.options.get("num_ctx") if self.options else None
                ),
                "api_ps_loaded": ps_value,
                "process_env": env_value,
                "modelfile": modelfile_value,
            }
        )
        self._context_budgets[model] = resolved
        # 初回解決時に必ず記録する（呼び出し元がactionを持たない経路でも、
        # genericな action名で解決イベントを残す）。
        self._emit(
            "effective_context_resolution",
            action if action is not None else "context_resolution",
            actor,
            started_at if started_at is not None else time.perf_counter(),
                effective_context_limit=resolved.limit,
                context_budget_source=resolved.source,
                context_budget_candidates=resolved.candidates,
                context_limit_conflict=resolved.conflict,
                # 由来別の名称でログの解釈を明確化する（TASK-046レビュー指摘）:
                # configured=設定由来 / server_observed=サーバー観測値 / limit=採用単一値。
                configured_context_limit=resolved.configured,
                configured_context_source=resolved.configured_source,
                server_observed_context_limit=resolved.server_observed,
                server_observed_context_source=resolved.server_observed_source,
                process_env_key=env_key,
                note=(
                    "設定値がサーバー観測値（/api/ps・env・Modelfile）を超えるため、"
                    "実機でnum_ctxが上書きされるか再確認が必要"
                    if resolved.conflict
                    else ""
                ),
            )
        return resolved

    def _record_discussion_body_metrics(
        self,
        model: str,
        extracted: str | None,
        diagnosis: DiscussionBodyDiagnosis,
        repair_applied: bool,
        fatal_safety_fallback: bool,
        final_body: str,
    ) -> None:
        """Record TASK034 body-quality metrics without affecting game flow."""
        metrics = self._discussion_metrics.setdefault(
            model, ModelDiscussionMetrics(model=model)
        )
        metrics.calls += 1
        if extracted and extracted.strip():
            metrics.extraction_success += 1
        if diagnosis.severity == "ok":
            metrics.format_compliant += 1
        if repair_applied and extracted and extracted.strip():
            metrics.post_retry_success += 1
        if is_pass_body(final_body):
            metrics.fallbacks += 1
        if fatal_safety_fallback:
            metrics.fatal_safety_fallbacks += 1
        for reason in diagnosis.reasons:
            metrics.failure_reasons[reason] = (
                metrics.failure_reasons.get(reason, 0) + 1
            )

    def wolf_discuss(
        self, wolf: str, partner: str, history: tuple[Message, ...]
    ) -> str:
        started_at = self._start_action("wolf_discussion", self._player(wolf))
        player = self._player(wolf)
        prompt_ctx = PromptBuilder.build_wolf_discuss_prompt(
            actor=player,
            partner_name=partner,
            alive_players=tuple(self.state.alive_players),
            wolf_chat_history=history,
            day=self.state.day,
            state=self.state,
        )
        return self._free_text_with_context(
            "wolf_discussion",
            player,
            prompt_ctx,
            f"{wolf} could not respond to the consultation.",
            started_at,
            recent_messages=history,
            budget_key="wolf_discussion",
        )

    def choose_attack(self, wolves: tuple[str, ...], candidates: tuple[str, ...]) -> str:
        started_at = self._start_action("attack", self._player(wolves[0]))
        actor = self._player(wolves[0])
        valid_candidates = self._valid_candidates(
            actor, candidates, forbidden=wolves
        )
        prompt_ctx = PromptBuilder.build_attack_prompt(
            actor=actor,
            wolves=wolves,
            valid_candidates=valid_candidates,
            day=self.state.day,
            state=self.state,
        )
        return self._select_target(
            ActionKind.ATTACK,
            actor,
            valid_candidates,
            prompt_ctx,
            started_at,
            forbidden=wolves,
            budget_key="attack",
        )

    def choose_seer_target(
        self, seer: str, candidates: tuple[str, ...], results: tuple[tuple[str, str], ...]
    ) -> str:
        started_at = self._start_action("seer", self._player(seer))
        actor = self._player(seer)
        living_candidates = self._valid_candidates(
            actor, candidates, forbid_self=True
        )
        unscanned_candidates = tuple(
            candidate
            for candidate in living_candidates
            if candidate not in self.state.seer_results
        )
        eligible_candidates = unscanned_candidates
        if not eligible_candidates:
            raise ValueError(f"No valid unscanned seer targets for {seer}.")
        prompt_ctx = PromptBuilder.build_seer_prompt(
            actor=actor,
            alive_players=tuple(self.state.alive_players),
            seer_results=results,
            day=self.state.day,
            state=self.state,
        )
        return self._select_target(
            ActionKind.SEER,
            actor,
            eligible_candidates,
            prompt_ctx,
            started_at,
            budget_key="seer",
        )

    def _build_discussion_prompt_with_budget(
        self,
        actor: Player,
        alive: tuple[str, ...],
        history: tuple[Message, ...],
        day: int,
        include_claim: bool = True,
        budget_key: str = "discussion",
        free_form_priority: bool = False,
    ) -> PromptContext:
        """動的発言履歴予算を適用した議論プロンプトを作る（TASK-036 / TASK-057）。

        まず予算なしで作り、発言履歴が割り当てコンテキストに収まらない場合だけ
        予算付きで再構築する。確定扱いの発言（直近・現在の質問・公開CO台帳）は削らない。

        TASK-057: `prompt_contract_policy="simplified"` では、発言履歴を削ってもなお
        割り当てを超えるときだけ optional 節（論点要約・仮説差分・暫定案内・派生メモ）を
        段階的に落とす。core（ルール・生存者・確定事実・現在の質問・自分の行動記録・
        自分の推理メモ・発言本文）はどの段階でも落とさないため、立場の一貫性は保たれる。
        口調・論点・結論への指示は追加しない（落とすのは参考情報のブロックだけ）。
        """
        context_strategy = actor.context_strategy or self.context_strategy
        contract_policy = normalize_prompt_contract_policy(
            getattr(self.state, "prompt_contract_policy", None)
        )
        stage_limit = (
            OPTIONAL_REMOVAL_MAX_STAGE
            if contract_policy != PROMPT_CONTRACT_POLICY_LEGACY
            and self._dynamic_context_enabled()
            else 0
        )

        def build(dropped: tuple[str, ...], budget: int | None) -> PromptContext:
            build_calls[0] += 1
            return PromptBuilder.build_discussion_prompt(
                actor=actor,
                alive_players=alive,
                conversation_history=history,
                day=day,
                state=self.state,
                include_claim=include_claim,
                context_strategy=context_strategy,
                evidence_char_budget=budget,
                free_form_priority=free_form_priority,
                question_hint_mode=self.question_hint_mode,
                prompt_contract_policy=contract_policy,
                optional_sections_dropped=dropped,
            )

        # 構築回数は観測する（段階除去の組み直しで無駄な再構築をしていないかの判断材料）。
        build_calls = [0]
        dropped: tuple[str, ...] = ()
        prompt_ctx = build((), None)
        budget = self._dynamic_evidence_budget(actor, budget_key, prompt_ctx)
        if budget is not None:
            prompt_ctx = build(dropped, budget)
        stage = 0
        applied_stage = 0
        overflow_tokens = self._prompt_overflow_tokens(actor, budget_key, prompt_ctx)
        # 実際にプロンプトへ現れている optional 節だけを削る。存在しない節のために組み直すと、
        # その分だけローカルでの構築コストが増えるため、見出しの存在を先に1度だけ数える。
        present_sections = set(find_sections(prompt_ctx.user_prompt))
        while stage < stage_limit and overflow_tokens > 0:
            stage += 1
            newly_present = [
                name
                for name in optional_names_dropped_up_to(stage)
                if name not in dropped and name in present_sections
            ]
            if not newly_present:
                continue
            dropped = optional_names_dropped_up_to(stage)
            prompt_ctx = build(dropped, budget)
            applied_stage = stage
            overflow_tokens = self._prompt_overflow_tokens(actor, budget_key, prompt_ctx)
        # TASK-046: 実効上限・圧縮段階を診断に記録し、「プロンプト総量」と「必須証拠の
        # 欠落」を別々に検証できるようにする（発言内容・口調の制約は行わない）。
        if prompt_ctx.diagnostics is not None:
            resolved = self._resolve_context_budget(actor)
            prompt_ctx.diagnostics["effective_context_limit"] = resolved.limit
            prompt_ctx.diagnostics["context_budget_source"] = resolved.source
            prompt_ctx.diagnostics["context_limit_conflict"] = resolved.conflict
            prompt_ctx.diagnostics["compression_stage"] = (
                "evidence_budget_applied" if budget is not None else "full"
            )
            # TASK-057: 削除段階・試行段階・構築回数と overflow を記録する（実機比較の判断材料）。
            prompt_ctx.diagnostics["contract_removal_stage"] = int(applied_stage)
            prompt_ctx.diagnostics["contract_stage_attempts"] = int(stage)
            prompt_ctx.diagnostics["prompt_build_calls"] = int(build_calls[0])
            if dropped:
                prompt_ctx.diagnostics["dropped_optional_sections"] = list(dropped)
            prompt_ctx.diagnostics["prompt_overflow_tokens"] = int(overflow_tokens)
        return prompt_ctx

    def _prompt_overflow_tokens(
        self,
        actor: Player,
        budget_key: str,
        prompt_ctx: PromptContext,
    ) -> int:
        """送信見積もりの必要トーク数が割り当てを何度超えるか（不明なら 0）。

        `num_ctx` 未指定（サーバー既定に委ねる）では解決済み実効上限を使い、それも無ければ
        判定しない（0 を返して段階除去を進めない）。thinking 分はモデル別実測余白を引く。
        """
        options, _source, thinking_enabled, _thinking_source = self._options_for(
            actor, budget_key
        )
        allocatable = options.get("num_ctx") if options else None
        if not isinstance(allocatable, int) or isinstance(allocatable, bool):
            allocatable = self._resolve_context_budget(actor).limit
        if not isinstance(allocatable, int) or isinstance(allocatable, bool):
            return 0
        generation = options.get("num_predict") if options else None
        if not isinstance(generation, int) or isinstance(generation, bool):
            generation = DEFAULT_ACTION_NUM_PREDICT.get(
                normalize_action_budget_key(budget_key), 512
            )
        prompt_chars = len(prompt_ctx.system_prompt) + len(prompt_ctx.user_prompt)
        needed = (
            math.ceil(prompt_chars * CONTEXT_TOKENS_PER_CHAR)
            + int(generation)
            + CONTEXT_MARGIN_TOKENS
            + self._thinking_margin_for(actor, budget_key, bool(thinking_enabled))
        )
        return max(0, needed - int(allocatable))


    def _question_answer_check(
        self,
        action: str,
        actor: Player,
        body: str,
        history: tuple[Message, ...],
        started_at: float,
    ) -> dict[str, object]:
        """TASK-038: 直前の発言への回答状況を観測して記録する（再生成はしない）。

        soft/noneでは形式合わせの再生成を行わないため、本文が直前の発言へ答えているかを
        内容ベースで判定し、メトリクスだけ残す。
        """
        messages = tuple(message for message in history if message.speaker != "system")
        current_day_history = PromptBuilder.current_day_messages(messages)
        previous_message = current_day_history[-1] if current_day_history else None
        if self.question_hint_mode == QUESTION_HINT_NONE or previous_message is None:
            # noneは観測も記録しない（完全中立・メトリクスも質問回答欄を使わない）
            return {}
        directed = PromptBuilder._is_directed_to_player(
            actor, previous_message, current_day_history
        )
        extracted = extract_discussion_body(body) or body
        answered = bool(
            previous_message is not None
            and previous_message.speaker != actor.name
            and extracted.strip()
            and not is_pass_body(extracted)
            and answers_previous_question(extracted, previous_message.content)
        )
        details: dict[str, object] = {
            "question_hint_mode": self.question_hint_mode,
            "question_directed": bool(directed),
            "question_answered": answered,
            "question_answer_forced": bool(
                directed and self.question_hint_mode == QUESTION_HINT_STRICT
            ),
            "question_retry_allowed": self.question_hint_mode == QUESTION_HINT_STRICT,
        }
        self._emit("discussion_question_check", action, actor, started_at, **details)
        return details

    def _novelty_result(
        self, actor: Player, body: str, history: tuple[Message, ...] | None
    ) -> NoveltyResult:
        """出力後の本文を新規性で計測する（観測のみ・再生成や書き換えの指示ではない）。

        既出論点台帳（`state.topic_ledger`）と本人のこれまでの立場
        （`state.own_stances_for`）を照合し、新規証拠・反証・立場変更・質問への応答・
        新論点と、新シグナルのない連呼を区別する。
        """
        extracted = extract_discussion_body(body) or body
        messages = (
            tuple(message for message in history if message.speaker != "system")
            if history is not None
            else tuple(self.state.conversation_log)
        )
        current_day_history = PromptBuilder.current_day_messages(messages)
        previous_message = current_day_history[-1] if current_day_history else None
        question_answered = bool(
            previous_message is not None
            and previous_message.speaker != actor.name
            and extracted.strip()
            and not is_pass_body(extracted)
            and PromptBuilder._is_directed_to_player(
                actor, previous_message, current_day_history
            )
            and answers_previous_question(extracted, previous_message.content)
        )
        return classify_novelty(
            extracted,
            actor_name=actor.name,
            player_names=tuple(player.name for player in self.state.players),
            prior_topics=self.state.topic_ledger.topics_for_day(self.state.day),
            question_answered=question_answered,
            own_prior_stances=self.state.own_stances_for(actor.name),
        )

    def _measure_discussion_novelty(
        self,
        action: str,
        actor: Player,
        body: str,
        history: tuple[Message, ...] | None,
        started_at: float,
    ) -> dict[str, object]:
        """公開される本文の新規性を計測して記録する（誘導・再生成は行わない）。"""
        novelty = self._novelty_result(actor, body, history)
        self._last_novelty = novelty
        details: dict[str, object] = {
            "novelty_primary": novelty.primary,
            "novelty_classes": list(novelty.classes),
            "novelty_score": novelty.novelty_score,
            "novelty_restated_topics": len(novelty.restated_topic_ids),
            "novelty_legitimate_repetitions": len(novelty.legitimate_repetition_ids),
            "novelty_new_topics": len(novelty.new_topic_ids),
            "novelty_topic_sentences": novelty.topic_sentence_count,
            "novelty_policy": self.novelty_policy,
            "has_novelty": novelty.has_novelty,
        }
        self._emit("discussion_novelty_measured", action, actor, started_at, **details)
        return details

    def _novelty_append_prompt(
        self, actor: Player, existing_body: str, recent_messages: tuple[Message, ...] | None
    ) -> str:
        """既存本文をそのまま公開した上で、任意の短い追記だけ求める短いプロンプト。"""
        summary, _ = self.state.topic_summary_for(actor.name, self.state.day)
        history = (
            recent_messages
            if recent_messages is not None
            else tuple(self.state.conversation_log)
        )
        recent_speech = [
            f"{message.speaker}: {message.content[:160]}"
            for message in PromptBuilder.current_day_messages(history)[-2:]
        ]
        return (
            "[Optional Addition Request]\n"
            "Your statement below is published as-is; do not rewrite or shorten it.\n"
            f"Topics already raised today:\n{summary or '(none recorded yet)'}\n"
            "If you want to add anything, write 1-3 sentences in your own words about a point "
            "not yet raised (a new reason, a counter-argument, a question, or a changed position). "
            "If you have nothing to add, output exactly 'Pass'. Do not repeat the statement below "
            "and do not include meta-talk about the system.\n\n"
            "[Your published statement (unchanged)]\n"
            f"{existing_body}\n\n"
            "[Recent speech]\n"
            + ("\n".join(recent_speech) or "(none)")
            + "\n"
            + language_directive(actor.response_language)
        )

    def _novelty_append_request(
        self,
        actor: Player,
        existing_body: str,
        recent_messages: tuple[Message, ...] | None,
        started_at: float,
    ) -> str | None:
        """低新規性の本文に短い追記だけを求め、追加が得られなければ None（元本文を公開）。"""
        messages = [
            {"role": "system", "content": PromptBuilder.build_system_prompt(actor)},
            {
                "role": "user",
                "content": self._novelty_append_prompt(
                    actor, existing_body, recent_messages
                ),
            },
        ]
        try:
            response = self._chat(
                actor,
                messages,
                action="discussion_novelty_append",
                started_at=started_at,
                request_number=1,
                request_reason="novelty_append",
                budget_key="discussion",
                retry_budget_fit=True,
            )
        except OllamaError as error:
            self._emit(
                "llm_action_parse_failed",
                "discussion_novelty_append",
                actor,
                started_at,
                reason=self._transport_failure_reason(error),
                will_retry=False,
            )
            return None
        addition = self._strip_meta_echo(
            extract_discussion_body(response.content) or response.content
        ).strip()
        if not addition or is_pass_body(addition):
            return None
        existing_normalized = re.sub(r"[\W_]+", "", existing_body, flags=re.UNICODE)
        addition_normalized = re.sub(r"[\W_]+", "", addition, flags=re.UNICODE)
        if (
            len(addition_normalized) >= 20
            and SequenceMatcher(None, addition_normalized, existing_normalized).ratio()
            >= 0.82
        ):
            # 追記が既存本文と実質同じなら重複を増やさないため元本文をそのまま公開する
            return None
        structured = parse_discussion_output(existing_body)
        if structured is None:
            return f"{existing_body}\n{addition}"
        blocks: list[str] = [f"[CO] {structured.claim}"]
        if structured.result:
            blocks.append(f"[RESULT] {structured.result}")
        blocks.append(f"[BODY] {structured.body} {addition}".strip())
        if structured.reasoning:
            blocks.append(f"[REASONING] {structured.reasoning}")
        return "\n".join(blocks)

    def discuss(
        self, player: str, day: int, alive: tuple[str, ...], history: tuple[Message, ...]
    ) -> str:
        started_at = self._start_action("discussion", self._player(player))
        actor = self._player(player)
        if (
            self.state.reasoning_memory_enabled
            and actor.name not in self.state.reasoning_memos
        ):
            self.state.refresh_reasoning_memos()
        prompt_ctx = self._build_discussion_prompt_with_budget(
            actor=actor,
            alive=alive,
            history=history,
            day=day,
        )
        body = self._free_text_with_context(
            "discussion",
            actor,
            prompt_ctx,
            f"{player} could not speak this turn.",
            started_at,
            recent_messages=history,
            budget_key="discussion",
            context_diagnostics=prompt_ctx.diagnostics,
        )
        self._question_answer_check("discussion", actor, body, history, started_at)
        # TASK-048: 公開本文の新規性を計測する（誘導せず、観測だけ残す）
        self._measure_discussion_novelty("discussion", actor, body, history, started_at)
        return body

    def should_stage_discussion_claim(self, player: str) -> bool:
        if self.discussion_claim_mode == "legacy":
            return False
        if self.discussion_claim_mode == "all":
            checked_key = (self.state.day, player)
            has_claim = player in self.state.claim_ledger.public_claims or any(
                claim.claimant == player for claim in self.state.public_role_claims
            )
            has_unpublished_result = self._has_unpublished_seer_result(player)
            if checked_key not in self.state.discussion_claim_checks:
                self.state.discussion_claim_checks.add(checked_key)
                return not has_claim or has_unpublished_result
            if has_unpublished_result:
                return True
            return any(
                message.speaker != player
                and re.search(
                    r"(?:CO|主張).*(?:訂正|更新|見直)",
                    message.content if isinstance(message.content, str) else "",
                )
                for message in self.state.conversation_log[-4:]
            )
        actor = self._player(player)
        return actor.role is Role.SEER or any(
            claim.claimant == player for claim in self.state.public_role_claims
        )

    def _has_unpublished_seer_result(self, player: str) -> bool:
        return any(
            (player, target) not in self.state.claim_ledger.public_seer_results
            and not self._has_published_seer_result(player, target, result)
            for target, result in self.state.seer_results.items()
        )

    def choose_discussion_claim(
        self,
        player: str,
        day: int,
        alive: tuple[str, ...],
        history: tuple[Message, ...],
    ) -> str:
        actor = self._player(player)
        started_at = self._start_action("discussion_claim", actor)
        prompt_ctx = PromptBuilder.build_discussion_claim_prompt(
            actor=actor,
            alive_players=alive,
            day=day,
            state=self.state,
        )
        response = self._free_text_with_context(
            "discussion_claim",
            actor,
            prompt_ctx,
            "[CO] None",
            started_at,
            recent_messages=history,
            budget_key="co_correction",
        )
        return response

    def deterministic_discussion_claim(self, player: str) -> str:
        return self._deterministic_discussion_claim(self._player(player)) or "[CO] None"

    def discuss_body(
        self,
        player: str,
        day: int,
        alive: tuple[str, ...],
        history: tuple[Message, ...],
    ) -> str:
        actor = self._player(player)
        started_at = self._start_action("discussion_body", actor)
        round_transcript = tuple(self.state.current_round_transcript)
        prompt_ctx = self._build_discussion_prompt_with_budget(
            actor=actor,
            alive=alive,
            history=history,
            day=day,
            include_claim=False,
            free_form_priority=actor.model in self._discussion_format_weak_models,
        )
        # ステージ1: 通常要求。_free_text_with_context 内で1回の意味的再試行まで行う。
        body = self._free_text_with_context(
            "discussion_body",
            actor,
            prompt_ctx,
            "Pass",
            started_at,
            recent_messages=history,
            budget_key="discussion",
            context_diagnostics=prompt_ctx.diagnostics,
        )
        # TASK-034: 本文抽出＋重大度診断＋段階的処理。
        # 本文が存在する場合はパスへ置換せず情報を保持。空または致命的違反のみ安全フォールバック。
        extracted = extract_discussion_body(body)
        # モデル推理メモの抽出・保存
        reasoning_memo = extract_reasoning_memo(body)
        if reasoning_memo:
            self.state.store_model_reasoning_memo(player, reasoning_memo)
        previous_message = round_transcript[-1] if round_transcript else None
        diagnosis = diagnose_discussion_body(
            extracted,
            actor,
            self.state,
            previous_message=previous_message,
            round_transcript=round_transcript,
            question_hint_mode=self.question_hint_mode,
        )
        repair_applied = self._last_free_text_retries >= 1
        fatal_safety_fallback = False
        recovered_content = False

        if diagnosis.fatal or extracted is None:
            fatal_safety_fallback = diagnosis.fatal
            final_body = "Pass"
        elif diagnosis.repairable and extracted and extracted.strip():
            # 修復可能だが有用な本文あり：本文を保持し、すでに再試行で回収済みとみなす。
            recovered_content = True
            final_body = body
        else:
            # 警告または問題なし：本文をそのまま採用。
            recovered_content = bool(extracted and extracted.strip())
            final_body = body

        self._record_discussion_body_metrics(
            actor.model,
            extracted,
            diagnosis,
            repair_applied,
            fatal_safety_fallback,
            final_body,
        )
        self._question_answer_check("discussion_body", actor, final_body, history, started_at)
        # TASK-048: 公開本文の新規性を計測する（誘導せず、観測だけ残す）
        self._measure_discussion_novelty(
            "discussion_body", actor, final_body, history, started_at
        )
        return final_body

    def retry_after_secret_leak(
        self,
        player: str,
        day: int,
        alive: tuple[str, ...],
        history: tuple[Message, ...],
        leak_categories: tuple[str, ...],
        *,
        rule_ids: tuple[str, ...] = (),
        retained_body: str | None = None,
    ) -> str | None:
        actor = self._player(player)
        started_at = self._start_action("discussion_safety_retry", actor)
        recent_messages = tuple(
            message for message in history if message.speaker != "system"
        )[-5:]
        # 全文プロンプトの再送は思考offでも数分かかるため、公開事実・直近発言だけの
        # 短縮プロンプトで本文を作り直し、生成上限・コンテキストも要求サイズへ絞る（TASK-035）。
        # 漏洩分類ごとの修復案内を出す（TASK-039）。安全再生成では役職・結果の
        # 公開形式を案内しないため、真占い師が正当なCOを出せず三人称の自己参照だけ
        # を語る発言になっていた（game_20da533bc4a6 Player 1）。
        leak_repair_guidance: dict[str, str] = {
            "hidden_result_leak": (
                "You mentioned an unpublished divination result. Either publish it with a Seer CO "
                "at the start (Seer CO: Target=<the player you actually divined>, Result=Werewolf or "
                "Human, Reason=reason, Vote=proposal), or remove the target name, verdict, and divination words."
            ),
            "secret_role_leak": (
                "You stated your role as meta-talk or in the third person. State your role only "
                "as an in-game CO claim in the first person (I/me/my)."
            ),
            "strategy_leak": (
                "You revealed internal role details (bluffing, configuration, true role). Mention "
                "your role only as the CO you publish."
            ),
            "secret_team_leak": (
                "You used the wolf camp's internal perspective or mentioned your partner. Speak in "
                "a villager's first person about public info: doubts, questions, and vote intention."
            ),
            "prompt_or_config_leak": (
                "You referenced the system or model configuration. Write only in-game dialogue."
            ),
        }
        # TASK-054: 検出ルール単位で必要な案内だけを1行出す（カテゴリ横断の長文案内より
        # 短く、論点・口調・結論は縛らない）。ルールが不明なときだけ旧カテゴリ案内へ落とす。
        rule_repair_guidance: dict[str, str] = {
            RULE_SELF_ROLE_TRUE_CAMP: (
                "You confirmed your hidden camp as a fact. State your role only through the "
                "in-game claims you choose to publish."
            ),
            RULE_PARTNER_IDENTITY: (
                "You bound a named player as your ally. Remove the alliance wording and argue "
                "from publicly recorded speech, votes, and deaths."
            ),
            RULE_PRIVATE_TEAM_REFERENCE: (
                "You referenced a private consult or partner. Speak only about what the public "
                "record shows."
            ),
            RULE_TEAM_ACTION_VIEW: (
                "You spoke from the wolf camp's planning side. Judge and propose votes using "
                "public information."
            ),
            RULE_UNPUBLISHED_SEER_RESULT: (
                "You wrote an unpublished verdict outside a CO. Either publish it inside a Seer CO "
                "line or remove the target name, verdict, and divination words."
            ),
            RULE_SELF_ROLE_META: (
                "You explained role meta (bluffing, configuration, true role). Write only the CO "
                "you choose to publish and in-game reasoning."
            ),
            RULE_THIRD_PERSON_SELF_META: (
                "You discussed your own role as an outside description. Talk about yourself in "
                "the first person without meta-explanations."
            ),
            RULE_INTERNAL_CONFIG: (
                "You referenced the system or generation setup. Write only in-game dialogue."
            ),
            RULE_MODEL_IDENTIFIER: (
                "You named your own model. Write only in-game dialogue."
            ),
        }
        leak_lines = [
            rule_repair_guidance[rule_id]
            for rule_id in rule_ids
            if rule_id in rule_repair_guidance
        ]
        if not leak_lines:
            leak_lines = [
                leak_repair_guidance[category]
                for category in leak_categories
                if category in leak_repair_guidance
            ]
        user_prompt = (
            self._short_discussion_retry_prompt(
                actor,
                None,
                recent_messages,
                body_only=False,
            )
            + "\n[Self Identity]\n"
            + f"You are {actor.name}. Speak about yourself in the first person (I/me/my); "
            + f"do not discuss {actor.name} in the third person or treat yourself as another player.\n"
            + (
                "You are the Seer. A Seer CO and revealing divination results is a legitimate "
                "village claim, not a leak. If you reveal a result, put the CO format in your first "
                "sentence. Holding your result stays fully free; if holding, do not write the "
                "target name, verdict, or divination words in the body.\n"
                if actor.role is Role.SEER
                else ""
            )
            + "\n[Safety Retry]\n"
            + "Replace only the sentences that exposed hidden information. Your topics, order, "
            "reasoning, and tone stay yours; this note does not prescribe a conclusion.\n"
            + "Never expose an unpublished verdict, your real camp, or an ally's name. Repeating a "
            "claim or verdict already published in the ledger is allowed and is not a leak.\n"
            + "Speak in singular first person (I/me/my) about publicly recorded speech, votes, and deaths.\n"
            + "Detected: "
            + (", ".join(rule_ids) if rule_ids else ", ".join(leak_categories))
            + "\n"
            + "".join(f"- {line}\n" for line in leak_lines)
            + (
                "\n[Keep These Public Sentences]\n"
                f"{retained_body.strip()[:1200]}\n"
                "You may reuse these already-public sentences where they fit. They are evidence of "
                "what you said before, not a template for your answer.\n"
                if retained_body and retained_body.strip()
                else ""
            )
            + self._attempt_generation_guidance(
                actor,
                "discussion",
                thinking_off=True,
                retry_cap=True,
            )
        )
        messages = [
            {
                "role": "system",
                "content": PromptBuilder.build_system_prompt(actor),
            },
            {"role": "user", "content": user_prompt},
        ]
        # TASK-046: 漏洩セーフティ再要求も core-ledger圧縮（CO・判定・投票・死亡・
        # 直近発言・未回答質問・反証を保持）であり、圧縮段階を記録する。
        self._emit(
            "context_compression_stage",
            "discussion_safety_retry",
            actor,
            started_at,
            compression_stage="core_ledger",
            compression_reason="secret_leak_safety_retry",
            protected_evidence=[
                "public_claims",
                "vote_ledger",
                "confirmed_deaths",
                "private_seer_results",
                "recent_speeches",
                "unanswered_question_to_actor",
                "recent_counter_arguments",
            ],
            dropped_scope="speech_history_beyond_recent_and_ledger",
            request_number=1,
        )
        try:
            response = self._chat(
                actor,
                messages,
                action="discussion_safety_retry",
                started_at=started_at,
                request_number=1,
                request_reason="secret_leak_safety_retry",
                budget_key="discussion",
                disable_thinking=True,
                retry_budget_fit=True,
            )
        except OllamaError as error:
            self._complete_action(
                "discussion_safety_retry",
                actor,
                started_at,
                semantic_retries=0,
                transport_attempts=error.attempts,
                fallback_used=True,
                error=error,
                failure_reason="secret_leak_retry_failed",
                failure_reasons=("secret_leak_retry_failed",),
            )
            return None
        metadata = response.to_metadata()
        self._complete_action(
            "discussion_safety_retry",
            actor,
            started_at,
            semantic_retries=0,
            transport_attempts=response.attempts,
            fallback_used=False,
            content=response.content,
            response_metadata=metadata,
            content_preserved=bool(response.content.strip()),
        )
        # TASK-036: セーフティ再要求の応答にもメタ発話混入を除去する（公開会話への残留防止）。
        return self._strip_meta_echo(response.content)

    def vote(
        self, voter: str, candidates: tuple[str, ...], day: int, runoff: bool
    ) -> str:
        started_at = self._start_action("vote", self._player(voter))
        actor = self._player(voter)
        valid_candidates = self._valid_candidates(
            actor, candidates, forbid_self=True
        )
        recent_speech = [
            f"{message.speaker}: {message.content}"
            for message in self.state.conversation_log
            if message.speaker != "system"
        ][-8:]
        prompt_ctx = PromptBuilder.build_vote_prompt(
            actor=actor,
            alive_players=tuple(self.state.alive_players),
            valid_candidates=valid_candidates,
            today_summary="\n".join(recent_speech),
            runoff=runoff,
            day=day,
            state=self.state,
        )
        return self._select_target(
            ActionKind.VOTE,
            actor,
            candidates,
            prompt_ctx,
            started_at,
            forbid_self=True,
            budget_key="vote",
        )

    def _select_target(
        self,
        action: ActionKind,
        actor: Player,
        candidates: tuple[str, ...],
        prompt_ctx: PromptContext,
        started_at: float,
        forbid_self: bool = False,
        forbidden: tuple[str, ...] = (),
        budget_key: str = "vote",
    ) -> str:
        valid_candidates = self._valid_candidates(
            actor, candidates, forbid_self=forbid_self, forbidden=forbidden
        )
        if not valid_candidates:
            raise ValueError(f"No valid candidates for {action.value} by {actor.name}.")

        def _with_budget(
            prompt: str,
            *,
            escalated: bool = False,
            thinking_off: bool = False,
        ) -> str:
            return prompt + self._attempt_generation_guidance(
                actor,
                budget_key,
                escalated=escalated,
                thinking_off=thinking_off,
            )

        user_prompts = [_with_budget(prompt_ctx.user_prompt)]
        raw_responses: list[str] = []
        parsed_targets: list[str | None] = []
        last_error: OllamaError | None = None
        transport_attempts = 0
        semantic_retries = 0
        last_response_issue: str | None = None
        failure_reasons: list[str] = []
        response_metadata_history: list[dict[str, str | int | bool | None]] = []

        for request_number in range(TARGET_MAX_ATTEMPTS):
            # TASK-043: 対象選択は再要求プロンプトが数行なので、3回目まで許す。
            # ランダムフォールバックより、player が選んだ対象を保持する方が意志が残る。
            can_retry = request_number < TARGET_MAX_ATTEMPTS - 1
            messages = [
                {"role": "system", "content": prompt_ctx.system_prompt},
                {"role": "user", "content": user_prompts[-1]},
            ]
            try:
                chat_response = self._chat(
                    actor,
                    messages,
                    increase_generation=(
                        request_number >= 1
                        and last_response_issue in GENERATION_ISSUE_REASONS
                    ),
                    action=action,
                    started_at=started_at,
                    request_number=request_number + 1,
                    request_reason=(
                        "initial"
                        if request_number == 0
                        else "semantic retry"
                    ),
                    tool_based=False,
                    budget_key=budget_key,
                    # 再要求は1行形式だけを求めるため、思考をoffにして
                    # 思考トークンが num_predict を消費しないようにする。
                    disable_thinking=request_number >= 1,
                )
            except OllamaError as error:
                last_error = error
                context_overflow = (
                    error.kind != "timeout" and self._is_context_overflow(error)
                )
                failure_reason = (
                    "context_overflow_compacted"
                    if context_overflow
                    else self._transport_failure_reason(error)
                )
                failure_reasons.append(failure_reason)
                transport_attempts += error.attempts
                if can_retry and context_overflow:
                    self._emit(
                        "llm_action_parse_failed",
                        action,
                        actor,
                        started_at,
                        reason="context_overflow_compacted",
                        will_retry=True,
                    )
                    user_prompts.append(
                        _with_budget(
                            self._target_retry_prompt(
                                action,
                                actor,
                                valid_candidates,
                                "context_overflow_compacted",
                            ),
                            thinking_off=True,
                        )
                    )
                    semantic_retries += 1
                    continue
                self._emit(
                    "llm_action_parse_failed",
                    action,
                    actor,
                    started_at,
                    reason=failure_reason,
                    will_retry=False,
                )
                break

            transport_attempts += chat_response.attempts
            response_metadata = chat_response.to_metadata()
            response_metadata_history.append(response_metadata)
            raw_responses.append(chat_response.content)
            final_response = self._final_response(chat_response.content)
            truncated = response_metadata.get("done_reason") == "length"
            target = (
                ActionParser.parse(final_response, action, valid_candidates)
                if final_response
                else None
            )
            response_issue: str | None = None
            if not final_response:
                response_issue = (
                    "thinking_only_response"
                    if response_metadata.get("thinking_present")
                    or self._has_thinking(chat_response.content)
                    else "empty_content"
                )
            elif truncated and target is None:
                # TASK-042: 切り詰めは「対象行が未生成」のときだけ失敗扱いにする。
                # 対象行が読み取れるなら再試行・ランダムフォールバックを作らない。
                response_issue = "generation_truncated"
            last_response_issue = response_issue
            parsed_targets.append(target)
            if ActionParser.validate_target(
                target, valid_candidates, actor.name if forbid_self else None
            ):
                if truncated:
                    self._emit(
                        "llm_truncated_output_kept",
                        action,
                        actor,
                        started_at,
                        reason="generation_truncated_target_recovered",
                        failure_class="generation_limit",
                        selected_target=target,
                    )
                trace = ActionTrace(
                    action, actor.name, tuple(raw_responses),
                    tuple(parsed_targets), target, False,
                    failure_reasons=tuple(failure_reasons),
                    response_metadata=tuple(response_metadata_history),
                )
                self.traces.append(trace)
                self._complete_action(
                    action,
                    actor,
                    started_at,
                    semantic_retries,
                    transport_attempts,
                    False,
                    selected_target=target,
                    failure_reasons=tuple(failure_reasons),
                    response_metadata=response_metadata,
                )
                return target

            if response_issue is not None:
                failure_reason = response_issue
                self._emit(
                    "llm_action_parse_failed",
                    action,
                    actor,
                    started_at,
                    transport_retries=max(0, transport_attempts - len(raw_responses)),
                    parsed_target=None,
                    reason=failure_reason,
                    response_metadata=response_metadata,
                    will_retry=can_retry,
                )
                failure_reasons.append(failure_reason)
                if can_retry:
                    semantic_retries += 1
                    user_prompts.append(
                        _with_budget(
                            self._target_retry_prompt(
                                action,
                                actor,
                                valid_candidates,
                                failure_reason,
                            ),
                            escalated=failure_reason in GENERATION_ISSUE_REASONS,
                            thinking_off=True,
                        )
                    )
                continue

            self._emit(
                "llm_action_parse_failed",
                action,
                actor,
                started_at,
                transport_retries=max(0, transport_attempts - len(raw_responses)),
                parsed_target=target,
                reason=(
                    "invalid_target"
                    if target is not None
                    else "invalid_action_response"
                ),
                response_metadata=response_metadata,
                will_retry=can_retry,
            )
            failure_reasons.append(
                "invalid_target" if target is not None else "invalid_action_response"
            )
            if can_retry:
                semantic_retries += 1
                user_prompts.append(
                    _with_budget(
                        self._target_retry_prompt(
                            action,
                            actor,
                            valid_candidates,
                            "invalid_target" if target is not None else "invalid_action_response",
                        ),
                        thinking_off=True,
                    )
                )

        fallback = self._rng.choice(valid_candidates)
        self.traces.append(
            ActionTrace(
                action,
                actor.name,
                tuple(raw_responses),
                tuple(parsed_targets),
                fallback,
                True,
                last_error.to_dict() if last_error is not None else None,
                tuple(failure_reasons),
                tuple(response_metadata_history),
            )
        )
        self._complete_action(
            action,
            actor,
            started_at,
            semantic_retries,
            transport_attempts,
            True,
            selected_target=fallback,
            error=last_error,
            failure_reasons=tuple(failure_reasons),
            failure_reason=(
                failure_reasons[-1] if failure_reasons else "invalid_action_response"
            ),
            response_metadata=(
                response_metadata_history[-1]
                if response_metadata_history
                else None
            ),
        )
        return fallback

    def _valid_candidates(
        self,
        actor: Player,
        candidates: tuple[str, ...],
        forbid_self: bool = False,
        forbidden: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        alive = set(self.state.alive_players)
        return tuple(
            candidate
            for candidate in candidates
            if candidate in alive
            and candidate not in forbidden
            and (not forbid_self or candidate != actor.name)
        )

    def _target_retry_prompt(
        self,
        action: ActionKind,
        actor: Player,
        valid_candidates: tuple[str, ...],
        failure_reason: str,
    ) -> str:
        corrections = {
            # TASK-053: 失敗理由は thinking_profile の定数で揃える（ログ側と同じ語）。
            # いずれも形式だけを直し、結論・口調・論点は固定しない。
            EMPTY_CONTENT_REASON: "Previous response was empty.",
            THINKING_ONLY_REASON: "Previous response was only thinking. Output only the answer.",
            GENERATION_TRUNCATED_REASON: "Previous response was truncated. Output only the format without explanations.",
            "invalid_target": "Previous target was outside the valid candidates. Choose only from eligible candidates.",
            "invalid_action_response": "Could not parse target from previous response.",
            "context_overflow_compacted": "Previous input was too long. Pick concisely from the candidates.",
        }
        label = self._label(action)
        return (
            f"[Retry] {corrections.get(failure_reason, 'Please correct your previous response.')}\n"
            f"- Action: {label}\n"
            f"- You: {actor.name}\n"
            f"- Day: {self.state.day}\n"
            + self._retry_public_fact_lines()
            + f"- Eligible candidates: {', '.join(valid_candidates)}\n"
            f"Choose exactly 1 candidate and reply with 1 line only: '{label}: Name'.\n"
            # TASK-043: 自分・候補外の名前・案内文のプレースホルダの出力を明確に禁止する
            # （実機ログでは自分への投票とプレースホルダ模倣が再試行原因になった）。
            "- Never write your own name, a player outside the list, or the format template's "
            "example text; use exactly one living candidate name from the list above.\n"
            f"- 候補リスト以外の名前（自分「{actor.name}」・存在しない名前・案内文の例）は書かない。\n"
            # TASK-053: 公開事実だけを添え、結論・口調・論点は固定しない。長いプロンプトの
            # 再送をやめて待ち時間を削りつつ、自分の判断で対象を選ばせる（一貫性の担保は
            # 事実の提示だけで行う）。
            "- Decide by your own reading of the public facts above. This retry fixes only the "
            "output format; it does not prescribe your conclusion, reasoning, or tone.\n"
            f"出力は厳密に{actor.response_language}で行ってください。"
        )

    def _retry_public_fact_lines(self) -> str:
        """再要求プロンプトへ添える最小限の公開事実（TASK-053・一貫性の根拠）。"""

        dead_players = tuple(player.name for player in self.state.players if not player.is_alive)
        latest_vote = (
            ", ".join(
                f"{voter}->{target}"
                for voter, target in self.state.vote_history[-1][2].items()
            )
            if self.state.vote_history
            else "None"
        )
        death_events = sorted(
            [(day, name) for day, name in self.state.attack_deaths]
            + [(day, name) for day, name in self.state.executions]
        )[-3:]
        death_text = ", ".join(f"Day {day} {name}" for day, name in death_events) or "None"
        return (
            f"- Alive: {', '.join(self.state.alive_players)}\n"
            f"- Known dead: {', '.join(dead_players) or 'None'}\n"
            f"- Confirmed deaths: {death_text}\n"
            f"- Latest vote: {latest_vote}\n"
        )

    def _public_response_leak_findings(
        self, action: str, actor: Player, content: str | None
    ) -> tuple[LeakFinding, ...]:
        if action not in {"discussion", "discussion_body", "discussion_claim"}:
            return ()
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        structured = parse_discussion_output(content)
        claim_text = ""
        result_text = ""
        if structured is not None:
            claim_text = structured.claim
            result_text = structured.result
        elif action == "discussion_claim":
            decision = parse_discussion_claim(content)
            if decision is not None:
                claim_text = decision.claim
                result_text = decision.result
        else:
            claim_text = content
            result_text = content
        allowed_claim = extract_public_role_claim(claim_text, actor.name) is not None
        allowed_results: frozenset[tuple[str, str]] = frozenset()
        if (
            actor.role is Role.SEER
            and extract_public_role_claim(claim_text, actor.name) == Role.SEER.value
        ):
            result = extract_public_seer_result(
                f"{claim_text}\n{result_text}",
                tuple(player.name for player in self.state.players),
            )
            if (
                result is not None
                and self.state.seer_results.get(result[0]) == result[1]
            ):
                allowed_results = frozenset({result})
        # TASK-054: 公開面のみを、モデル別強度つきで分類する（soft/observe は再生成しない）。
        policy = normalize_leak_guard_policy(
            getattr(self.state, "leak_guard_policy", DEFAULT_LEAK_GUARD_POLICY)
        )
        profiles = getattr(self.state, "leak_guard_profiles", None)
        if not isinstance(profiles, LeakGuardProfiles):
            profiles = LeakGuardProfiles(policy=policy)
            self.state.leak_guard_profiles = profiles
        profiles.policy = policy
        return SecretLeakGuard.classify(
            content,
            actor,
            self.state,
            allowed_public_claim=(
                allowed_claim
                or SecretLeakGuard.allows_public_claim(content, actor)
            ),
            allowed_public_results=allowed_results,
            guard_policy=policy,
            guard_strength=profiles.strength_for(actor.model),
        ).hard

    def _free_text_with_context(
        self,
        action: str,
        actor: Player,
        prompt_ctx: PromptContext,
        fallback: str,
        started_at: float,
        recent_messages: tuple[Message, ...] | None = None,
        budget_key: str = "discussion",
        context_diagnostics: dict[str, object] | None = None,
    ) -> str:
        if context_diagnostics is not None:
            self._emit(
                "discussion_context_selected",
                action,
                actor,
                started_at,
                **context_diagnostics,
            )
        raw_responses: list[str] = []
        last_error: OllamaError | None = None
        transport_attempts = 0
        last_response_issue: str | None = None
        failure_reasons: list[str] = []
        # TASK-034: 直近の_free_text_with_context で意味的再試行が行われたかを呼び出し側で参照する。
        self._last_free_text_retries = 0
        response_metadata_history: list[dict[str, str | int | bool | None]] = []
        last_spoken_result: tuple[str, str] | None = None
        last_soft_discussion: tuple[str, str, dict[str, str | int | bool | None]] | None = None
        for attempt in range(DISCUSSION_MAX_ATTEMPTS):
            # TASK-043: 3回目の再要求は生成起因の失敗（空・思考のみ・切り詰め）だけ。
            # 意味検証の失敗は従来の2回までの挙動を維持し、再試行コストを増やさない。
            if attempt >= 2 and last_response_issue not in GENERATION_ISSUE_REASONS:
                break
            # TASK-034: 再要求が行われたかをdiscussion_bodyの後処理で参照する。
            self._last_free_text_retries = attempt
            is_co_correction = (
                attempt
                and action == "discussion"
                and last_response_issue == "seer_result_without_claim"
            )
            is_light_retry = bool(attempt) and not is_co_correction
            keeps_round_context = (
                self.context_strategy == "round_transcript_memory"
                or actor.context_strategy == "round_transcript_memory"
            )
            # TASK-053: thinking-only・空・切り詰めでの再要求は、戦略が
            # round_transcript_memory でも同一の長いプロンプトを再送しない。公開台帳と
            # 直近発言だけを保持した短縮プロンプトへ差し替え、口調・結論・論点は固定しない
            # （長いままの再送が待ち時間と再要求ループの主因だった）。
            uses_core_ledger_retry = (
                not keeps_round_context
                or last_response_issue in GENERATION_ISSUE_REASONS
            )
            user_prompt = (
                self._seer_co_correction_prompt(actor, last_spoken_result)
                if is_co_correction
                else (
                    prompt_ctx.user_prompt
                    + self._discussion_retry_instruction(
                        action, actor, last_response_issue
                    )
                    + "\nKeep the full current-round transcript and continuity memo; fix only the problematic part."
                    if (keeps_round_context and not uses_core_ledger_retry)
                    else self._short_discussion_retry_prompt(
                    actor,
                    last_response_issue,
                    recent_messages,
                    body_only=action == "discussion_body",
                    )
                )
                if is_light_retry and action in {"discussion", "discussion_body"}
                else self._wolf_consult_light_retry_prompt(
                    actor,
                    recent_messages,
                )
                if is_light_retry and action == "wolf_discussion"
                else prompt_ctx.user_prompt
                + (
                    self._discussion_retry_instruction(
                        action, actor, last_response_issue
                    )
                    if attempt
                    else ""
                )
            )
            attempt_budget_key = "co_correction" if is_co_correction else budget_key
            generation_escalation = (
                attempt >= 1 and last_response_issue in GENERATION_ISSUE_REASONS
            )
            # タイムアウト再要求は短縮プロンプト・思考offに加えて、生成上限と
            # コンテキストを要求サイズへ絞って再送の応答時間を削る（TASK-035）。
            timeout_retry = attempt >= 1 and last_response_issue == "timeout"
            thinking_off_retry = attempt >= 1 and (
                last_response_issue in GENERATION_ISSUE_REASONS or timeout_retry
            )
            user_prompt += self._attempt_generation_guidance(
                actor,
                attempt_budget_key,
                escalated=generation_escalation,
                thinking_off=thinking_off_retry,
                retry_cap=timeout_retry,
            )
            # TASK-046: 軽再試行の短縮プロンプト（core-ledger圧縮）では、保持した証拠
            # （公開CO・占い判定・投票・死亡・直近発言・本人への未回答質問・直近反証）と
            # 圧縮段階を記録し、プロンプト総量と証拠欠落を別々に検証できるようにする。
            if (
                is_light_retry
                and action in {"discussion", "discussion_body"}
                and uses_core_ledger_retry
            ):
                self._emit(
                    "context_compression_stage",
                    action,
                    actor,
                    started_at,
                    compression_stage="core_ledger",
                    compression_reason=(
                        "timeout_retry"
                        if timeout_retry
                        else (
                            "generation_retry"
                            if thinking_off_retry
                            else "semantic_retry"
                        )
                    ),
                    protected_evidence=[
                        "public_claims",
                        "vote_ledger",
                        "confirmed_deaths",
                        "private_seer_results",
                        "recent_speeches",
                        "unanswered_question_to_actor",
                        "recent_counter_arguments",
                    ],
                    # TASK-053: 短縮プロンプトでどこまでコンテキストを節約できたかを、初回
                    # 送信のプロンプト長と直接比較できるようにする（長いプロンプト再送の排除）。
                    retry_prompt_style="core_ledger_short",
                    initial_user_prompt_char_count=len(prompt_ctx.user_prompt),
                    retry_user_prompt_char_count=len(user_prompt),
                    dropped_scope="speech_history_beyond_recent_and_ledger",
                    request_number=attempt + 1,
                )
            messages = [
                {"role": "system", "content": prompt_ctx.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            try:
                chat_response = self._chat_with_claim_tools(
                    actor,
                    messages,
                    action,
                    increase_generation=generation_escalation,
                    started_at=started_at,
                    request_number=attempt + 1,
                    request_reason=(
                        "initial"
                        if attempt == 0
                        else (
                            "CO correction"
                            if is_co_correction
                            else "light retry"
                            if is_light_retry and action == "discussion"
                            else "semantic retry"
                        )
                    ),
                    tool_based=(action == "discussion"),
                    budget_key=attempt_budget_key,
                    # TASK-034: 思考オフは生成上限（generation limit等）に限定し、
                    # 意味検証失敗では思考を維持する。
                    disable_thinking=thinking_off_retry,
                    retry_budget_fit=timeout_retry,
                )
            except OllamaError as error:
                last_error = error
                context_overflow = (
                    error.kind != "timeout" and self._is_context_overflow(error)
                )
                last_response_issue = (
                    "context_overflow_compacted"
                    if context_overflow
                    else self._transport_failure_reason(error)
                )
                failure_reasons.append(last_response_issue)
                transport_attempts += error.attempts
                if (
                    attempt < DISCUSSION_MAX_ATTEMPTS - 1
                    and context_overflow
                ):
                    self._emit(
                        "llm_action_parse_failed",
                        action,
                        actor,
                        started_at,
                        reason="context_overflow_compacted",
                        will_retry=True,
                    )
                    continue
                if (
                    attempt < DISCUSSION_MAX_ATTEMPTS - 1
                    and error.kind == "timeout"
                    and action
                    in {
                        "discussion",
                        "discussion_body",
                        "discussion_claim",
                        "wolf_discussion",
                    }
                ):
                    self._emit(
                        "llm_action_parse_failed",
                        action,
                        actor,
                        started_at,
                        failure_class=classify_failure_reason(last_response_issue),
                        reason=last_response_issue,
                        will_retry=True,
                        retry_strategy="short_prompt_thinking_off",
                    )
                    continue
                self._emit(
                    "llm_action_parse_failed",
                    action,
                    actor,
                    started_at,
                    failure_class=classify_failure_reason(last_response_issue),
                    reason=last_response_issue,
                    will_retry=False,
                )
                break

            transport_attempts += chat_response.attempts
            metadata = chat_response.to_metadata()
            response_metadata_history.append(metadata)
            raw_responses.append(chat_response.content)
            response_text = self._final_response(chat_response.content).strip()
            is_pass = self._is_pass(response_text)
            leak_findings = self._public_response_leak_findings(
                action, actor, response_text
            )
            # 真占い師の役職主張・未公開結果への言及は、安全再生成ではなくCO形式の
            # 再要求と決定的補完で直す漏洩として扱う（TASK-039）。再生成だと本文が
            # 失われ、三人称の自己参照だけの発言が公開されていた（game_20da533bc4a6）。
            seer_claim_repairable = (
                actor.role is Role.SEER
                and action in {"discussion", "discussion_body", "discussion_claim"}
                and bool(leak_findings)
                and all(
                    finding.category in {"secret_role_leak", "hidden_result_leak"}
                    for finding in leak_findings
                )
                and self._has_unpublished_seer_result(actor.name)
            )
            leak_findings = tuple(
                finding
                for finding in leak_findings
                if finding.category != "hidden_result_leak"
            )
            if leak_findings and not seer_claim_repairable:
                self.traces.append(
                    ActionTrace(
                        action,
                        actor.name,
                        tuple(raw_responses),
                        (),
                        None,
                        False,
                        failure_reasons=tuple(failure_reasons),
                        response_metadata=tuple(response_metadata_history),
                        content_preserved=True,
                    )
                )
                self._complete_action(
                    action,
                    actor,
                    started_at,
                    attempt,
                    transport_attempts,
                    False,
                    content=response_text,
                    failure_reasons=tuple(failure_reasons),
                    content_preserved=True,
                    recovered_content=False,
                    repair_applied=self._last_free_text_retries >= 1,
                    fatal_safety_fallback=False,
                    validation_severity="ok",
                )
                return response_text
            must_use_seer_info = self._must_use_seer_info(action, actor)
            uses_seer_info = self._references_seer_result(actor, response_text)
            holds_human_seer_result = self._holds_human_seer_result(
                action, actor, response_text
            )
            response_issue = (
                self._discussion_claim_response_issue(actor, response_text)
                if action == "discussion_claim" and response_text
                else self._discussion_response_issue(
                    actor, response_text, recent_messages
                )
                if action in {"discussion", "discussion_body"}
                and response_text
                and (
                    not is_pass or has_discussion_output_markers(response_text)
                )
                else None
            )
            claim_decision = (
                parse_discussion_claim(response_text)
                if action == "discussion_claim" and response_text
                else None
            )
            if claim_decision is not None and claim_decision.ignored_trailing_text:
                self._emit(
                    "discussion_claim_extra_text_discarded",
                    action,
                    actor,
                    started_at,
                    reason="structured_claim_recovered",
                )
            if metadata.get("done_reason") == "length":
                # TASK-042: 切り詰めそのものを失敗にしない。内容が検証を通り、文末が
                # 文として閉じているならそのまま受理し、再試行・フォールバックを作らない。
                if not response_text:
                    response_issue = (
                        "thinking_only_response"
                        if metadata.get("thinking_present")
                        or self._has_thinking(chat_response.content)
                        else "empty_content"
                    )
                elif response_issue is None and self._truncated_output_usable(
                    response_text, action
                ):
                    self._emit(
                        "llm_truncated_output_kept",
                        action,
                        actor,
                        started_at,
                        reason="generation_truncated_content_recovered",
                        failure_class="generation_limit",
                    )
                else:
                    response_issue = "generation_truncated"
            elif not response_text and response_issue is None:
                response_issue = (
                    "thinking_only_response"
                    if metadata.get("thinking_present")
                    or self._has_thinking(chat_response.content)
                    else "empty_content"
                )
            # TASK-053: 思考が生成上限を食い潰して本文が空になった観測は、`_chat` の
            # `_record_generation_response` で中心記録する（投票・襲撃・占いも同じ経路で
            # 拾え、まずモデル別thinking余白の拡大で解消するため行動別の記述は不要）。
            # TASK-048: 論点の再掲・同一発言の反復は「低新規性」として計測し、再生成しない。
            # publish（既定）はそのまま公開、append は既存本文を保持した短い任意の追記だけ、
            # legacy_repair のみ旧挙動（重複文の除去→意味的再要求）を通す。
            if (
                action in {"discussion", "discussion_body"}
                and response_text
                and response_issue in NOVELTY_MEASURED_ISSUES
                and self.novelty_policy != NOVELTY_POLICY_LEGACY_REPAIR
            ):
                low_novelty_reason = response_issue
                novelty = self._novelty_result(actor, response_text, recent_messages)
                if not novelty.has_novelty and self.novelty_policy == NOVELTY_POLICY_APPEND:
                    appended = self._novelty_append_request(
                        actor, response_text, recent_messages, started_at
                    )
                    if appended is not None:
                        merged_novelty = self._novelty_result(
                            actor, appended, recent_messages
                        )
                        self._last_novelty = merged_novelty
                        response_text = appended
                        self._emit(
                            "discussion_novelty_appended",
                            action,
                            actor,
                            started_at,
                            reason=low_novelty_reason,
                            novelty_primary=merged_novelty.primary,
                            novelty_score=merged_novelty.novelty_score,
                            body_preserved=True,
                            regeneration_avoided=True,
                        )
                    else:
                        self._emit(
                            "llm_discussion_low_novelty_kept",
                            action,
                            actor,
                            started_at,
                            reason=low_novelty_reason,
                            novelty_primary=novelty.primary,
                            novelty_score=novelty.novelty_score,
                            body_preserved=True,
                            regeneration_avoided=True,
                            append_declined=True,
                        )
                else:
                    self._emit(
                        "llm_discussion_low_novelty_kept",
                        action,
                        actor,
                        started_at,
                        reason=low_novelty_reason,
                        novelty_primary=novelty.primary,
                        novelty_classes=list(novelty.classes),
                        novelty_score=novelty.novelty_score,
                        novelty_restated_topics=len(novelty.restated_topic_ids),
                        novelty_legitimate_repetitions=len(
                            novelty.legitimate_repetition_ids
                        ),
                        body_preserved=True,
                        regeneration_avoided=True,
                    )
                # 低新規性は失敗として扱わない（本文保持・再要求を作らない）
                response_issue = None
                uses_seer_info = self._references_seer_result(actor, response_text)
                holds_human_seer_result = self._holds_human_seer_result(
                    action, actor, response_text
                )
            if (
                attempt == 0
                and action in {"discussion", "discussion_body"}
                and response_issue in LEDGER_REPAIR_ISSUES
            ):
                original_issue = response_issue
                original_text_before_repair = response_text
                repaired = self._repair_discussion_response(
                    actor, response_text, original_issue, recent_messages
                )
                if repaired is not None:
                    repaired_issue = self._discussion_response_issue(
                        actor, repaired, recent_messages
                    )
                    if repaired_issue is None:
                        response_text = repaired
                        response_issue = None
                        uses_seer_info = self._references_seer_result(
                            actor, response_text
                        )
                        holds_human_seer_result = self._holds_human_seer_result(
                            action, actor, response_text
                        )
                        dead_details = {}
                        if original_issue in {
                            "dead_player_question",
                            "dead_player_action_target",
                        }:
                            dead_names = frozenset(
                                p.name
                                for p in self.state.players
                                if not p.is_alive
                            )
                            living_names = frozenset(
                                p.name
                                for p in self.state.players
                                if p.is_alive
                            )
                            classifications = analyze_dead_player_references(
                                original_text_before_repair,
                                dead_names,
                                living_names,
                            )
                            dead_details = {
                                "historical_references": [
                                    c.sentence
                                    for c in classifications
                                    if c.is_historical_reference
                                ],
                                "actionable_targets": [
                                    c.sentence
                                    for c in classifications
                                    if c.is_actionable_target
                                ],
                                "dead_player_questions": [
                                    c.sentence
                                    for c in classifications
                                    if c.is_dead_player_question
                                ],
                                "original_content": original_text_before_repair,
                                "repaired_content": response_text,
                            }
                        self._emit(
                            "ledger_repair",
                            action,
                            actor,
                            started_at,
                            reason=original_issue,
                            body_preserved=bool(response_text),
                            regeneration_avoided=True,
                            **dead_details,
                        )
            if (
                attempt >= 1
                and action in {"discussion", "discussion_body"}
                and response_issue in LEDGER_REPAIR_ISSUES
            ):
                original_issue = response_issue
                repaired = self._repair_discussion_response(
                    actor, response_text, original_issue, recent_messages
                )
                if repaired is not None:
                    repaired_issue = self._discussion_response_issue(
                        actor, repaired, recent_messages
                    )
                    if repaired_issue is None:
                        response_text = repaired
                        response_issue = None
                        uses_seer_info = self._references_seer_result(
                            actor, response_text
                        )
                        holds_human_seer_result = self._holds_human_seer_result(
                            action, actor, response_text
                        )
                        self._emit(
                            "discussion_integrity_repaired",
                            action,
                            actor,
                            started_at,
                            reason=original_issue,
                        )
                        self._emit(
                            "ledger_repair",
                            action,
                            actor,
                            started_at,
                            reason=original_issue,
                            body_preserved=response_text == repaired,
                            regeneration_avoided=False,
                        )
            # TASK-041: 真占い師のCO不足・判定言及は、再試行→フォールバックではなくCO行の
            # 決定補完で受理する（game_f6983c7cbcc2 Player 3のフォールバックの主因）。
            if (
                action in {"discussion", "discussion_body"}
                and response_text
                and response_issue in {"seer_result_without_claim", "discussion_output_format"}
            ):
                repaired_claim = self._seer_claim_prefix_repair(actor, response_text)
                if repaired_claim is not None:
                    repaired_text, repaired_target, repaired_result = repaired_claim
                    repaired_issue = self._discussion_response_issue(
                        actor, repaired_text, recent_messages
                    )
                    if repaired_issue is None:
                        original_issue = response_issue
                        response_text = repaired_text
                        response_issue = None
                        uses_seer_info = True
                        holds_human_seer_result = self._holds_human_seer_result(
                            action, actor, response_text
                        )
                        self._seer_info_used_day[actor.name] = self.state.day
                        self._emit(
                            "discussion_claim_deterministically_repaired",
                            action,
                            actor,
                            started_at,
                            reason=original_issue,
                            target=repaired_target,
                            result=repaired_result,
                        )
                        self._emit(
                            "ledger_repair",
                            action,
                            actor,
                            started_at,
                            reason=original_issue,
                            body_preserved=True,
                            regeneration_avoided=True,
                        )
            if (
                action in {"discussion", "discussion_body"}
                and response_text
                and response_issue in SOFT_DISCUSSION_ISSUES
            ):
                last_soft_discussion = (
                    response_text,
                    response_issue,
                    metadata,
                )
            last_response_issue = response_issue
            if response_issue == "seer_result_without_claim":
                last_spoken_result = extract_public_seer_result(
                    response_text,
                    tuple(player.name for player in self.state.players),
                )
                self._emit(
                    "seer_result_claim_missing",
                    action,
                    actor,
                    started_at,
                    reason=response_issue,
                    will_retry=attempt == 0,
                )
            if response_text and response_issue is None and (
                not must_use_seer_info
                or (
                    not is_pass
                    and (uses_seer_info or holds_human_seer_result)
                )
            ):
                if is_pass:
                    response_text = "Pass"
                elif must_use_seer_info and (
                    uses_seer_info or holds_human_seer_result
                ):
                    self._seer_info_used_day[actor.name] = self.state.day
                elif action == "discussion_claim" and claim_decision is not None:
                    if claim_decision.claim in {"none", "hold"}:
                        response_text = (
                            "[CO] None" if claim_decision.claim == "none" else "[CO] Hold"
                        )
                    else:
                        claimed_role = extract_public_role_claim(
                            claim_decision.claim
                        )
                        role_labels = {
                            Role.SEER.value: "Seer CO",
                            Role.WOLF.value: "Werewolf CO",
                            Role.MADMAN.value: "Madman CO",
                            Role.VILLAGER.value: "Villager CO",
                        }
                        role_label = role_labels.get(claimed_role or "")
                        if role_label is not None:
                            response_text = f"[CO] {role_label}"
                            if claim_decision.result:
                                result = extract_public_seer_result(
                                    claim_decision.result,
                                    tuple(
                                        player.name for player in self.state.players
                                    ),
                                )
                                if result is not None:
                                    target, result_kind = result
                                    result_label = (
                                        "Werewolf" if result_kind == "wolf" else "Human"
                                    )
                                    response_text += (
                                        f"\n[RESULT] Target={target} Result={result_label}"
                                    )
                self.traces.append(
                    ActionTrace(
                        action,
                        actor.name,
                        tuple(raw_responses),
                        (),
                        None,
                        False,
                        failure_reasons=tuple(failure_reasons),
                        response_metadata=tuple(response_metadata_history),
                        content_preserved=True,
                    )
                )
                self._complete_action(
                    action,
                    actor,
                    started_at,
                    attempt,
                    transport_attempts,
                    False,
                    content=response_text,
                    failure_reasons=tuple(failure_reasons),
                    content_preserved=True,
                    response_metadata=metadata,
                )
                return response_text

            reason = (
                response_issue
                if response_issue is not None
                else "seer_result_required"
                if must_use_seer_info
                else "invalid_discussion_response"
            )
            failure_reasons.append(reason)
            self._emit(
                "llm_action_parse_failed",
                action,
                actor,
                started_at,
                transport_retries=max(0, transport_attempts - len(raw_responses)),
                failure_class=classify_failure_reason(reason),
                reason=reason,
                response_metadata=metadata,
                # TASK-043: 3回目の再要求は生成起因の失敗だけなので、will_retryもそれに合わせる。
                will_retry=(
                    attempt < DISCUSSION_MAX_ATTEMPTS - 1
                    and (attempt == 0 or reason in GENERATION_ISSUE_REASONS)
                ),
            )

        if (
            action == "discussion"
            and "seer_result_without_claim" in failure_reasons
            and last_spoken_result is not None
        ):
            target, result = last_spoken_result
            if (
                actor.role is Role.SEER
                and self.state.seer_results.get(target) == result
                and not self._has_published_seer_result(actor.name, target, result)
            ):
                raw_content = raw_responses[-1] if raw_responses else ""
                structured = parse_discussion_output(raw_content)
                body = structured.body if structured is not None else re.sub(
                    r"(?m)^\s*【(?:CO|結果|本文)】\s*", "", raw_content
                ).strip()
                if is_japanese_language(actor.response_language):
                    fallback_body = (
                        f"占い結果に基づき、{target}さんへの投票を提案する。"
                        if result == "wolf"
                        else f"{target}さんへの疑いは薄く、他の可能性を慎重に検討したい。"
                    )
                    claim_line = (
                        f"[CO] Seer CO: 対象={target}, "
                        f"判定={'人狼' if result == 'wolf' else '人間'}, "
                        "理由=実際の占い結果, 投票="
                        + (
                            f"{target}さんへの投票を提案する"
                            if result == "wolf"
                            else "他の可能性を慎重に検討する"
                        )
                    )
                else:
                    fallback_body = (
                        f"I propose voting for {target}."
                        if result == "wolf"
                        else f"Suspicion toward {target} is weak; I want to consider other possibilities carefully."
                    )
                    claim_line = (
                        "[CO] Seer CO: "
                        f"Target={target}, Result={'Werewolf' if result == 'wolf' else 'Human'}, "
                        "Reason=my actual divination result, Vote="
                        + (
                            f"Vote for {target}"
                            if result == "wolf"
                            else "I want to consider other possibilities carefully"
                        )
                    )
                repaired = f"{claim_line}\n[BODY] {body or fallback_body}"
                self._emit(
                    "seer_result_claim_fallback",
                    action,
                    actor,
                    started_at,
                    reason="seer_result_without_claim",
                    target=target,
                    result=result,
                )
                self._emit(
                    "discussion_claim_deterministically_repaired",
                    action,
                    actor,
                    started_at,
                    reason="seer_result_without_claim",
                    target=target,
                    result=result,
                )
                self._seer_info_used_day[actor.name] = self.state.day
                self.traces.append(
                    ActionTrace(
                        action,
                        actor.name,
                        tuple(raw_responses),
                        (),
                        None,
                        True,
                        failure_reasons=tuple(failure_reasons),
                        response_metadata=tuple(response_metadata_history),
                        content_preserved=bool(body),
                    )
                )
                self._complete_action(
                    action,
                    actor,
                    started_at,
                    max(0, len(raw_responses) - 1),
                    transport_attempts,
                    True,
                    content=repaired,
                    failure_reasons=tuple(failure_reasons),
                    content_preserved=bool(body),
                    response_metadata=(
                        response_metadata_history[-1]
                        if response_metadata_history
                        else None
                    ),
                )
                return repaired

        if last_soft_discussion is not None:
            response_text, soft_issue, metadata = last_soft_discussion
            if soft_issue == "discussion_output_format":
                body_match = re.search(
                    r"【本文】\s*(.*)", response_text, re.DOTALL
                )
                if response_text.count("【本文】") > 1:
                    body_match = None
                response_text = (
                    body_match.group(1).strip()
                    if body_match is not None
                    else re.sub(
                        r"(?m)^\s*【(?:CO|結果|本文)】\s*",
                        "",
                        response_text,
                    ).strip()
                )
            elif soft_issue == "repeated_discussion_topic":
                repaired = self._repair_discussion_response(
                    actor,
                    response_text,
                    soft_issue,
                    recent_messages,
                )
                if repaired is not None:
                    response_text = repaired
            if (
                soft_issue == "discussion_output_format"
                and body_match is None
                and has_discussion_output_markers(
                    last_soft_discussion[0]
                )
            ):
                response_text = ""
            if response_text and self._discussion_response_issue(
                actor, response_text, recent_messages
            ) not in {
                "external_game_memory",
                "unverified_past_vote_reference",
                "unverified_dead_role",
                "dead_player_question",
                "dead_player_action_target",
                "out_of_phase_action_claim",
                "unknown_attacker_claim",
                "unverified_game_event",
                "unverified_attributed_speech",
            }:
                self._emit(
                    "llm_discussion_soft_issue_kept",
                    action,
                    actor,
                    started_at,
                    reason=soft_issue,
                    failure_class=classify_failure_reason(soft_issue),
                    body_preserved=True,
                )
                self.traces.append(
                    ActionTrace(
                        action,
                        actor.name,
                        tuple(raw_responses),
                        (),
                        None,
                        False,
                        last_error.to_dict() if last_error is not None else None,
                        tuple(failure_reasons),
                        tuple(response_metadata_history),
                    )
                )
                self._complete_action(
                    action,
                    actor,
                    started_at,
                    max(0, len(raw_responses) - 1),
                    transport_attempts,
                    False,
                    content=response_text,
                    error=last_error,
                    failure_reasons=tuple(failure_reasons),
                    content_preserved=True,
                    response_metadata=metadata,
                )
                return response_text

        # TASK-041/042: フォールバックは直前の応答本文を保持し、応答言語で書く（英語の汎用文や
        # 「Pass」を公開すると他プレイヤーが「中身のない発言」「沈黙」と誤認した）。
        raw_last = raw_responses[-1] if raw_responses else ""
        preserved_body = extract_discussion_body(raw_last) if raw_last else None
        forced_seer_message = (
            self._seer_info_fallback(actor, preserved_body)
            if self._must_use_seer_info(action, actor)
            or (
                action == "discussion"
                and "seer_result_without_claim" in failure_reasons
            )
            else None
        )
        if forced_seer_message is not None:
            self._seer_info_used_day[actor.name] = self.state.day
            if "seer_result_without_claim" in failure_reasons:
                self._emit(
                    "seer_result_claim_fallback",
                    action,
                    actor,
                    started_at,
                    reason="seer_result_without_claim",
                )
            fallback_response = forced_seer_message
        elif action == "discussion_claim":
            fallback_response = (
                self._deterministic_discussion_claim(actor) or "[CO] None"
            )
        else:
            preserved_fallback = self._preserved_fallback_response(
                action, actor, raw_last, recent_messages
            )
            if preserved_fallback is not None:
                self._emit(
                    "fallback_body_preserved",
                    action,
                    actor,
                    started_at,
                    reason=(
                        failure_reasons[-1] if failure_reasons else "unavailable_response"
                    ),
                    body_preserved=True,
                )
                fallback_response = preserved_fallback
            elif action == "wolf_discussion":
                fallback_response = self._consult_fallback_text(actor)
            else:
                # 本文が本当に使い物にならない場合も、無内容な「Pass」ではなく
                # 生存者への確認点を述べる内容のある文を返す。
                fallback_response = self._fallback_discussion_turn(
                    actor, recent_messages
                )
        self.traces.append(
            ActionTrace(
                action,
                actor.name,
                tuple(raw_responses),
                (),
                None,
                True,
                last_error.to_dict() if last_error is not None else None,
                tuple(failure_reasons),
                tuple(response_metadata_history),
            )
        )
        self._complete_action(
            action,
            actor,
            started_at,
            max(0, len(raw_responses) - 1),
            transport_attempts,
            True,
            content=fallback_response,
            error=last_error,
            failure_reasons=tuple(failure_reasons),
            failure_reason=(
                failure_reasons[-1] if failure_reasons else "unavailable_response"
            ),
            response_metadata=(
                response_metadata_history[-1]
                if response_metadata_history
                else None
            ),
        )
        return fallback_response

    @staticmethod
    def _final_response(response: str | None) -> str:
        """応答本文から思考ブロック除去後の最終文を取り出す（Noneは空文字扱い）。"""
        if response is None or not isinstance(response, str):
            return "" if response is None else str(response)
        final = _THINK_BLOCK.sub("", response)
        final = _UNCLOSED_THINK_BLOCK.sub("", final)
        return final.replace("</think>", "").strip()

    @staticmethod
    def _is_pass(response: str | None) -> bool:
        if response is None or not isinstance(response, str):
            return False
        normalized = response.strip().strip("。.!！?？「」『』\"' ")
        return normalized.casefold() in {"パス", "pass"}

    def _must_use_seer_info(self, action: str, actor: Player) -> bool:
        return (
            action == "discussion"
            and actor.role.value == "seer"
            and bool(self.state.seer_results)
            and self._seer_info_used_day.get(actor.name) != self.state.day
        )

    def _holds_human_seer_result(
        self, action: str, actor: Player, response: str
    ) -> bool:
        structured = parse_discussion_output(response)
        # TASK-041: 人間判定のみの占い師は `[CO] None` も保留と同等に扱う。弱いモデルは
        # 指示の `[CO] Hold` を `[CO] None` で返すため、旧来は再試行→フォールバックになっていた。
        return (
            action == "discussion"
            and actor.role is Role.SEER
            and structured is not None
            and structured.claim in {"hold", "none"}
            and not structured.result
            and bool(self.state.seer_results)
            and all(result == "human" for result in self.state.seer_results.values())
        )

    def _references_seer_result(self, actor: Player, response: str) -> bool:
        if (
            actor.role.value != "seer"
            or not response
            or not self.state.seer_results
        ):
            return False
        latest_target, _ = self._latest_seer_result()
        if any(
            claim.claimant == actor.name
            and claim.claimed_role == Role.SEER.value
            and claim.target == latest_target
            for claim in self.state.public_role_claims
        ):
            return True
        mentions_result_target = latest_target in response
        usage_terms = (
            "占い",
            "結果",
            "情報",
            "疑い",
            "警戒",
            "信頼",
            "擁護",
            "推理",
            "seer",
            "result",
            "information",
            "suspect",
        )
        return mentions_result_target and any(
            term.casefold() in response.casefold() for term in usage_terms
        )

    def _seer_info_fallback(
        self, actor: Player, last_body: str | None = None
    ) -> str | None:
        if actor.role.value != "seer" or not self.state.seer_results:
            return None
        target, result = self._latest_seer_result()
        label = "Werewolf" if result == "wolf" else "Human"
        # TASK-041: 応答言語で書き、モデルが書いた本文を可能な限り保持する。旧来は英語の
        # 汎用文へ完全に置換されていたため、実機では中身のない発言（他プレイヤーが
        # 「沈黙」と疑う発言）が公開されていた（game_f6983c7cbcc2 Player 3）。
        preserved = (last_body or "").strip()
        if preserved and not self._meta_echo_sentences(preserved):
            body = preserved
        elif is_japanese_language(actor.response_language):
            body = (
                f"占い結果により{target}さんは人狼と判定した。本日{target}さんへの投票を提案する。"
                if result == "wolf"
                else f"占い結果により{target}さんは人間と判定した。{target}さんへの疑いは薄く、他の可能性を慎重に検討したい。"
            )
        elif result == "wolf":
            body = f"My divination result. I propose voting for {target}."
        else:
            body = (
                f"My divination result weakens suspicion toward {target}; "
                "I want to consider other possibilities carefully."
            )
        return (
            f"[CO] Seer\n[RESULT] Target={target} Result={label}\n[BODY] {body}"
        )

    def _seer_claim_prefix_repair(
        self, actor: Player, response: str
    ) -> tuple[str, str, str] | None:
        """真占い師の応答へCO行を決定的に補完する（TASK-041、再試行・フォールバック回避）。

        対象・判定は台帳の実際の占い結果と一致する場合だけ公開し、CO行に隔離する。
        本文はそのまま保持するため、議論の内容を失わず、英語のフォールバック文も出ない。
        """
        if actor.role is not Role.SEER or not self.state.seer_results:
            return None
        japanese = is_japanese_language(actor.response_language)
        structured = parse_discussion_output(response)
        if structured is None:
            # マーカー行のみ（`[CO] Seer` 単独等）の応答は、CO行を本文へ複製しないよう
            # 行頭マーカーを剥がして本文として扱う。
            body = re.sub(
                r"(?m)^\s*(?:【(?:CO|結果|本文)】|\[(?:CO|RESULT|BODY)\])\s*",
                "",
                response,
            ).strip()
        else:
            body = structured.body
        claim_text = structured.claim if structured is not None else response
        spoken = extract_public_seer_result(
            response, tuple(player.name for player in self.state.players)
        )
        if spoken is not None:
            target, result = spoken
            if self.state.seer_results.get(target) != result:
                return None
            if self._has_published_seer_result(actor.name, target, result):
                return None
        elif is_public_seer_claim(claim_text, actor.name):
            # COだけでRESULT行のない応答は、最新の実判定を決定補完する（捏造しない）。
            target, result = self._latest_seer_result()
            if self._has_published_seer_result(actor.name, target, result):
                return None
        else:
            return None
        label = "Werewolf" if result == "wolf" else "Human"
        if japanese:
            claim_line = (
                f"[CO] Seer CO: 対象={target}, 判定={label}, 理由=実際の占い結果, 投票="
                + (
                    f"{target}さんへの投票を提案する"
                    if result == "wolf"
                    else "他の可能性を慎重に検討する"
                )
            )
            fallback_body = (
                f"占い結果により{target}さんは人狼と判定した。本日{target}さんへの投票を提案する。"
                if result == "wolf"
                else f"占い結果により{target}さんは人間と判定した。{target}さんへの疑いは薄く、他の可能性を慎重に検討したい。"
            )
        else:
            claim_line = (
                f"[CO] Seer CO: Target={target}, Result={label}, "
                + "Reason=my actual divination result, Vote="
                + (
                    f"Vote for {target}"
                    if result == "wolf"
                    else "I want to consider other possibilities carefully"
                )
            )
            fallback_body = (
                f"My divination result shows {target} as Werewolf; I propose voting for {target}."
                if result == "wolf"
                else (
                    f"My divination result shows {target} as Human; suspicion toward {target} is "
                    "weak, so I want to consider other possibilities carefully."
                )
            )
        repaired_body = (
            body.strip() if body.strip() and not self._meta_echo_sentences(body) else fallback_body
        )
        return f"{claim_line}\n[BODY] {repaired_body}", target, result

    def _discussion_retry_instruction(
        self, action: str, actor: Player, response_issue: str | None = None
    ) -> str:
        retry_guidance = {
            "empty_content": (
                "Previous response had an empty body. Write your final answer into [BODY] "
                "(3-5 sentences). The point you raise, your tone, and your conclusion stay "
                "yours to choose; only the output format is fixed."
            ),
            "thinking_only_response": (
                "Previous response had only thinking and no final response. Move your own "
                "judgement into [BODY] now (3-5 sentences) and finish it inside the limit; "
                "do not restate the thinking text, and keep your points, tone and conclusion free."
            ),
            "generation_truncated": (
                "Previous response was cut off by generation limit. Using the structured format, output 3-5 sentences in [BODY] "
                "completely to the end without truncation. Exclude thinking, greetings, or meta-explanations; include "
                "answers to questions, new evidence/counter-arguments, and current suspicion/vote intentions."
            ),
            "discussion_output_format": "Previous format was invalid. Put '[CO] None' (or role claim) on line 1, and '[BODY] text' on line 2. For hold, use '[CO] Hold'. Do not leave [BODY] empty.",
            "discussion_claim_format": (
                "Re-answer only the CO. Line 1 must be one of '[CO] None', '[CO] Hold', "
                "'[CO] Seer CO', '[CO] Werewolf CO', '[CO] Madman CO', or '[CO] Villager CO'. "
                "Only when claiming Seer AND revealing results, add line 2: "
                "'[RESULT] Target=<the player you actually divined> Result=Werewolf or Human'. "
                "Never copy the placeholder text itself as the target. "
                "Do not output other lines, reasons, or body."
            ),
            "wrong_response_language": (
                f"Output strictly in {actor.response_language}. Do not use any other language."
            ),
            "third_person_self_reference": "Do not describe yourself in the third person by name; speak in the first person (I/me/my).",
            "unknown_player_reference": "Do not mention non-existent player names; refer only to current players.",
            # TASK-043: 案内文の模倣（プレースホルダ）は実在しない対象の幻覚になるため、
            # 実在する名前・実際の結果だけへ書き直させる。
            TEMPLATE_PLACEHOLDER_ISSUE: (
                "You copied the prompt's template text (e.g. 'Player X', 'プレイヤーX', "
                "'Werewolf/Human' as a choice). Name only players that actually exist in this game "
                "and state only a result you actually received; if you have no result, remove that "
                "line and write '[CO] None' or '[CO] Hold'."
            ),
            "unverified_game_event": "Do not state attacks, executions, or deaths not in the game ledger as facts.",
            "unknown_attacker_claim": "Attackers are not revealed. Do not state guesses as confirmed facts.",
            "unverified_dead_role": "True roles of dead players are hidden. Do not treat roles as confirmed.",
            "dead_player_action_target": "You cannot vote for, execute, or target dead players. Replace the request with a question or consideration for living players.",
            "dead_player_question": "Dead players cannot answer. Withdraw the question and ask living players instead.",
            "external_game_memory": "Past games, prior roles, and training memories cannot be used. State 'Not in record, cannot determine' for unrecorded matters.",
            "unverified_past_vote_reference": "Use only past votes from this game's vote ledger. If unrecorded, state 'Not in record, cannot determine'.",
            "repeated_discussion_topic": "Do not repeat questions or conclusions already settled in today's discussion; state new points to check.",
            "out_of_phase_action_claim": "Do not assert night abilities as already executed or scheduled during day discussion.",
            "unverified_attributed_speech": "Do not quote statements not in the history as another player's words.",
            "repeated_speech": "Do not repeat your own past speech; add new evidence or proposals.",
            "question_not_answered_first": "Answer the previous speaker directly in your first sentence.",
            "wolf_result_not_disclosed": "As true Seer, claim Seer CO and reveal unrevealed Werewolf results with target, verdict, reason, and vote proposal.",
            "seer_result_without_claim": "You mentioned divination target or result, so you must explicitly claim Seer CO in the same speech. Use format '[CO] Seer' and '[RESULT] Target=Name Result=Werewolf/Human'. If not revealing, do not state target or verdict.",
            "inconsistent_seer_result": "Do not change your previously claimed divination results; remain consistent with past claims.",
            "inconsistent_role_claim": "Do not change your previously claimed role; remain consistent with past claims.",
            "contradictory_past_vote": (
                "Match your past vote targets confirmed in the vote ledger. "
                "If asked for change reasons, refer to SELF_ACTION_MEMORY and do not fabricate reasons not in record."
            ),
            "recorded_action_denied": (
                "Your actions are recorded in SELF_ACTION_MEMORY. "
                "Do not claim 'no record'; distinguish confirmed records from current speculation."
            ),
            "past_action_question_unanswered": (
                "Check the target and reason of asked past actions against SELF_ACTION_MEMORY and answer directly. "
                "If unrecorded, state so explicitly without fabricating."
            ),
            "vague_wait_and_see": "Avoid repeated vague 'wait and see' or 'lack of information'; state specific targets, evidence, and judgments.",
        }
        if response_issue in retry_guidance:
            return f"\nPrevious response could not be accepted. {retry_guidance[response_issue]}"
        if self._must_use_seer_info(action, actor):
            target, _ = self._latest_seer_result()
            return (
                "\nPrevious response was empty, pass, or did not utilize divination results. "
                f"This time, speak specifically based on the divination result for {target}. "
                "You may utilize it without revealing the result publicly."
            )
        return (
            "\nPrevious response was empty or whitespace only. "
            f"Please provide a substantive response in {actor.response_language}."
        )

    def _short_discussion_retry_prompt(
        self,
        actor: Player,
        response_issue: str | None,
        recent_messages: tuple[Message, ...] | None = None,
        *,
        body_only: bool = False,
    ) -> str:
        public_claims = list(self.state.public_role_claims)
        recorded_seer_claims = {
            (claim.day, claim.claimant, claim.target, claim.result)
            for claim in public_claims
            if claim.claimed_role == Role.SEER.value
        }
        for claim in self.state.seer_claims:
            key = (claim.day, claim.claimant, claim.target, claim.result)
            if key not in recorded_seer_claims:
                public_claims.append(
                    PublicRoleClaim(
                        day=claim.day,
                        claimant=claim.claimant,
                        claimed_role=Role.SEER.value,
                        target=claim.target,
                        result=claim.result,
                    )
                )
        public_claims_text = [
            f"{claim.claimant} claims "
            f"{ROLE_NAMES_EN.get(Role(claim.claimed_role), claim.claimed_role)}"
            + (
                f" ({claim.target}={'Werewolf' if claim.result == 'wolf' else 'Human'})"
                if claim.target is not None and claim.result is not None
                else ""
            )
            for claim in public_claims[-4:]
        ]
        death_events = sorted(
            [
                (day, "Attack death", name)
                for day, name in self.state.attack_deaths
            ]
            + [(day, "Execution", name) for day, name in self.state.executions]
        )[-3:]
        death_summary = ", ".join(
            f"Day {day} {name} died ({cause})" for day, cause, name in death_events
        ) or "None"
        dead_players = tuple(
            player.name for player in self.state.players if not player.is_alive
        )
        latest_vote = (
            ", ".join(
                f"{voter}->{target}"
                for voter, target in self.state.vote_history[-1][2].items()
            )
            if self.state.vote_history
            else "None"
        )
        vote_ledger = (
            ", ".join(
                f"Day {vote_day} {'Runoff' if runoff else 'Regular'}: "
                + " / ".join(
                    f"{voter}->{target}" for voter, target in votes.items()
                )
                for vote_day, runoff, votes in self.state.vote_history
            )
            or "None"
        )
        conversation = (
            recent_messages
            if recent_messages is not None
            else tuple(self.state.conversation_log)
        )
        current_day_history = PromptBuilder.current_day_messages(conversation)
        recent_speech = [
            f"{message.speaker}: {message.content[:240]}"
            for message in current_day_history[-3:]
        ]
        # TASK-046: core-ledger圧縮（再要求の短縮プロンプト）では、次ターンの回答に
        # 必要不可欠な「本人への未回答質問」と「直近の反証」を必ず保持する。
        # 議論本文の論点・口調は制約せず、提示する事実を増やすだけの変更。
        unanswered_question: str | None = None
        for message in reversed(current_day_history[-8:]):
            if message.speaker == actor.name:
                continue
            content = message.content if isinstance(message.content, str) else ""
            is_question = any(
                cue in content
                for cue in ("?", "？", "なぜ", "理由は", "質問", "確認", "教えて", "聞かせて")
            )
            if is_question and PromptBuilder._is_directed_to_player(
                actor, message, current_day_history
            ):
                unanswered_question = f"{message.speaker}: {content[:200]}"
                break
        counter_arguments = [
            f"{message.speaker}: {message.content[:160]}"
            for message in current_day_history[-6:]
            if message.speaker != actor.name
            and isinstance(message.content, str)
            and any(
                cue in message.content
                for cue in (
                    "反証", "反論", "違う", "誤り", "矛盾", "counter", "disagree", "contradict"
                )
            )
        ][-2:]
        private_seer_results = (
            [
                f"{target}: {'Werewolf' if result == 'wolf' else 'Human'}"
                for _, target, result in self.state.seer_result_history[-3:]
            ]
            if actor.role is Role.SEER
            else []
        )
        if (
            actor.role is Role.SEER
            and not private_seer_results
            and self.state.seer_results
        ):
            private_seer_results = [
                f"{target}: {'Werewolf' if result == 'wolf' else 'Human'}"
                for target, result in list(self.state.seer_results.items())[-3:]
            ]
        guidance = self._discussion_retry_instruction(
            "discussion", actor, response_issue
        ).strip()
        previous_message = next(
            reversed(current_day_history),
            None,
        )
        if PromptBuilder._is_directed_to_player(
            actor,
            previous_message,
            current_day_history,
        ):
            guidance += (
                f" Answer {previous_message.speaker} directly in your first sentence."
            )
        return (
            "This is a short retry prompt. Omit old omitted dialogue; use only the following facts and recent speeches.\n"
            "[Game System Confirmed Facts]\n"
            f"- Your Player Name: {actor.name}\n"
            f"- Self-Reference: {actor.name} is yourself. Always refer to yourself in the first person (I/me/my). "
            f"Do not discuss {actor.name} in the third person.\n"
            f"- Your Role: {ROLE_NAMES_EN[actor.role]}\n"
            f"- Current Day: Day {self.state.day}\n"
            "- Current Phase: Day (Discussion)\n"
            f"- Discussion Turn: {self.state.discussion_turn}/3\n"
            f"- Alive Players: {', '.join(self.state.alive_players)}\n"
            f"- Known Dead: {', '.join(dead_players) or 'None'}\n"
            f"- Confirmed Deaths (Last 3): {death_summary}\n"
            f"- Latest Vote Result: {latest_vote}\n"
            f"- Vote History: {vote_ledger}\n"
            f"- Response Language: {actor.response_language} (Strictly {actor.response_language})\n"
            "[Theory (Inference - Not Confirmed)] Prioritize revealing Werewolf results early. "
            "A lone Seer CO with no counter-claim - especially a Day 1 CO - is placed more genuinely "
            "(真置き) the longer it persists, so doubt it only with concrete inconsistency. "
            "Faking claims is an option for wolves/madmen, not mandatory.\n"
            f"[Public Claims & Results] {'; '.join(public_claims_text) or 'None'}.\n"
            f"[Recent Speech (Unverified)] {'; '.join(recent_speech) or 'None'}.\n"
            + (
                f"[Unanswered Question To You] {unanswered_question}\n"
                if unanswered_question
                else ""
            )
            + (
                f"[Recent Counter-arguments (Unverified)] {'; '.join(counter_arguments)}.\n"
                if counter_arguments
                else ""
            )
            + "[Information Classification] Public speech and CO claims are unverified; do not treat them as confirmed facts.\n"
            "Do not use past games, training memories, or real-world info; rely strictly on this game's records. "
            "State 'Not in record, cannot determine' for unrecorded matters. Ask living players, not dead players.\n"
            + (
                f"[Private Seer Results (Seer Only)] {'; '.join(private_seer_results)}.\n"
                if private_seer_results
                else ""
            )
            + f"[Correction Guidance] {guidance}\n"
            + (
                f"CO decision is complete. Output only 3-5 sentences of discussion body text in {actor.response_language} without markers or CO lines."
                if body_only
                else "Response must follow the 2-line structure: '[CO] None/Hold/Claim' followed by '[BODY] body'. "
                "Only when revealing Seer results, add line '[RESULT] Target=<the player you actually divined> "
                "Result=Werewolf or Human' (never write the placeholder itself)."
            )
            + f" Write 3-5 sentences in [BODY] concisely covering answers to questions, new evidence/counter-arguments, "
            f"and current suspicion and vote intention strictly in {actor.response_language}."
        )

    @staticmethod
    def _escalated_num_predict(limit: int) -> int:
        """生成起因の再試行で送信する生成上限（倍、最大8192）。"""
        return max(limit, min(limit * 2, 8192))

    def _attempt_generation_guidance(
        self,
        actor: Player,
        budget_key: str,
        *,
        escalated: bool = False,
        thinking_off: bool = False,
        retry_cap: bool = False,
    ) -> str:
        """実際に送信する予定の上限・思考設定をプロンプトへ反映する。

        再要求では `_chat`側で思考offと生成上限の倍化が行われるため、
        プロンプトの指示が実際のリクエストとずれないようにする。
        `retry_cap`（タイムアウト・秘密漏洩の再要求）では実際に送る上限を
        `RETRY_NUM_PREDICT_LIMIT`へ絞るため、その値をプロンプトへ明記する。
        """
        limit, _source = resolve_action_num_predict(
            player=actor,
            cli_options=self.options,
            budget_key=budget_key,
        )
        if not thinking_off:
            thinking_enabled_for_floor, _ = resolve_thinking(actor, self.options, budget_key)
            limit, _source = apply_thinking_floor(
                limit, _source, thinking_enabled_for_floor, budget_key
            )
        effective_limit = limit
        if escalated:
            base = limit if isinstance(limit, int) and not isinstance(limit, bool) else 512
            effective_limit = self._escalated_num_predict(base)
        if retry_cap:
            effective_limit = (
                min(effective_limit, RETRY_NUM_PREDICT_LIMIT)
                if isinstance(effective_limit, int) and not isinstance(effective_limit, bool)
                else RETRY_NUM_PREDICT_LIMIT
            )
        if (
            self._slow_model(actor)
            and normalize_action_budget_key(budget_key) in {"discussion", "wolf_discussion"}
            and not retry_cap
        ):
            # タイムアウト観測モデルでは実際送信の上限も絞る（プロンプト指示と一致させる）。
            effective_limit = (
                min(effective_limit, TIMEOUT_SAFE_NUM_PREDICT)
                if isinstance(effective_limit, int) and not isinstance(effective_limit, bool)
                else TIMEOUT_SAFE_NUM_PREDICT
            )
            thinking_off = True
        reservation = self._body_reservation(
            actor,
            budget_key,
            effective_limit,
            "generation_guidance",
            False if thinking_off else None,
            retry=bool(thinking_off) or bool(retry_cap),
        )
        if reservation.applied and reservation.num_predict is not None:
            # TASK-053: 実際に送信する上限（本文予約込み）をプロンプトへ反映し、
            # 指示とリクエストのずれを防ぐ。
            effective_limit = reservation.num_predict
        guidance = self._generation_budget_guidance(
            actor,
            budget_key,
            thinking_enabled=False if thinking_off else None,
            num_predict=effective_limit,
            body_reservation=reservation if reservation.applied else None,
        )
        if retry_cap:
            guidance += (
                f"This retry runs with thinking (think) off, num_predict="
                f"{RETRY_NUM_PREDICT_LIMIT}, and only the context needed for this prompt; "
                "output one short body in a single pass.\n"
            )
        return guidance

    def _generation_budget_guidance(
        self,
        actor: Player,
        budget_key: str,
        *,
        thinking_enabled: bool | None = None,
        num_predict: int | None = None,
        body_reservation: BodyReservation | None = None,
    ) -> str:
        """行動別生成上限に対する最終回答用の残余トークンを指示する。

        再要求では実際に送信する値（思考off・増加後の上限）を受け取り、
        プロンプトの指示と実際のリクエストを一致させる。
        TASK-053: `body_reservation` を渡すと「思考用の別枠余白」と「本文最低予算」の
        内訳を指示へ加える（論点・口調・結論は固定せず、本文だけが最後まで出るようにする）。
        """
        if num_predict is None:
            num_predict, _ = resolve_action_num_predict(
                player=actor,
                cli_options=self.options,
                budget_key=budget_key,
            )
        if thinking_enabled is None:
            thinking_enabled, _ = resolve_thinking(actor, self.options, budget_key)
        notes = ""
        if thinking_enabled is False:
            notes = (
                "\nThinking (think) is off for this request. Write no thinking or preamble; "
                "output only the final answer in the requested format."
            )
        if body_reservation is not None and body_reservation.applied:
            notes += (
                f"\n[Body reserve] num_predict includes a separate allowance: about "
                f"{body_reservation.thinking_allowance_tokens} tokens are reserved for your own "
                f"reasoning and at least {body_reservation.body_min_tokens} tokens are kept for "
                "the final answer. Think as freely as you like, then write the answer so it "
                "finishes inside the limit."
            )
        if num_predict is None:
            return notes
        output_share = max(1, round(num_predict * 3 // 10))
        key = normalize_action_budget_key(budget_key)
        if key in {"discussion", "wolf_discussion"}:
            # 議論・人狼相談は根拠を説明する文量が必要なので、残余トークンを
            # 「短文」ではなく具体的な文量として指示する（TASK-024の文量緩和）。
            body_share = max(1, round(num_predict * 6 // 10))
            return (
                f"\n[Generation limit] num_predict={num_predict} tokens. Use roughly the last "
                f"{body_share} tokens for the [BODY] final answer: write 3-5 sentences covering "
                "answers to questions, confirmed facts, whom you suspect and why, and what to "
                "check next. Do not stop mid-way; output to the end."
                f"{notes}"
            )
        return (
            f"\n[Generation limit] num_predict={num_predict} tokens. Keep roughly the last "
            f"{output_share} tokens for the final answer: state the conclusion, the basis, and "
            "unconfirmed items briefly."
            f"{notes}"
        )

    def _wolf_consult_light_retry_prompt(
        self,
        actor: Player,
        recent_messages: tuple[Message, ...] | None = None,
    ) -> str:
        """Light retry for werewolf consultation."""
        partner = actor.partner
        wolf_chat = recent_messages if recent_messages is not None else ()
        recent_consult = [
            f"{message.speaker}: {message.content[:160]}"
            for message in wolf_chat[-2:]
            if message.speaker != "system"
        ]
        alive = tuple(
            name
            for name in self.state.alive_players
            if name != actor.name and name != partner
        )
        return (
            "Light retry for werewolf consultation. Previous long dialogue is not referenced.\n"
            "[Confirmed Facts]\n"
            f"- Your Player Name: {actor.name}\n"
            f"- Your Role: {ROLE_NAMES_EN[actor.role]}\n"
            f"- Werewolf Partner: {partner}\n"
            f"- Current Day: Day {self.state.day}\n"
            f"- Eligible Attack Targets: {', '.join(alive) or 'None'}\n"
            f"- Recent Consultation (Unverified): {'; '.join(recent_consult) or 'None'}\n"
            "[Instructions] Output only the following 2 items in concise "
            f"{actor.response_language}:\n"
            "1. 1 candidate to attack\n"
            "2. Candidate(s) not to attack and why\n"
            "- Write from the Werewolf camp's standpoint (村人陣営を削る目的で書く). Never write as a "
            "villager keeping village order or praising a village-side player's stability.\n"
            "Refer to yourself in the first person (I/me/my); do not use third-person self-references.\n"
            f"出力は厳密に{actor.response_language}で行ってください。"
        )

    @staticmethod
    def _result_label(result: str) -> str:
        return "Werewolf" if result == "wolf" else "Human"

    def _seer_co_correction_prompt(
        self, actor: Player, spoken_result: tuple[str, str] | None
    ) -> str:
        """Short correction prompt for missing Seer CO."""
        lines = [
            "[CO Correction] Your statement mentions divination results without a Seer CO. "
            "You must explicitly claim Seer CO in the same statement.",
            f"- You: {actor.name}",
        ]
        co_target: str | None = None
        co_result: str | None = None
        if actor.role is Role.SEER and self.state.seer_results:
            target, result = self._latest_seer_result()
            co_target, co_result = target, result
            lines.append(
                f"- Your divination result: Target={target}, Result={self._result_label(result)}"
            )
            lines.append("Publish this result together with a Seer CO.")
        elif spoken_result is not None:
            co_target, co_result = spoken_result
            lines.append(
                f"- Previously stated result: Target={spoken_result[0]}, "
                f"Result={self._result_label(spoken_result[1])}"
            )
            lines.append("Publish this result together with a Seer CO.")
        lines.extend(
            [
                f"Output strictly in {actor.response_language} using these 2 lines only without extra text:",
                "[CO] Seer",
                # TASK-043: 案内文に「Player X」のようなプレースホルダを書かない。
                # 弱いモデル（game_e2991441d6f0 Player 3）はそれを模倣して実在しない
                # 対象の判定を公開したため、実際の対象名を出す。
                f"[RESULT] Target={co_target or '<the player you actually divined>'} "
                f"Result={self._result_label(co_result) if co_result is not None else 'Werewolf or Human'}",
                "Reason and vote proposal are optional; if adding them, put [BODY] on the 3rd line.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _discussion_sentences(text: str | None) -> tuple[str, ...]:
        """文分割（None・非文字列は空_tuple扱いで正規表現呼び出しを保護する）。"""
        if text is None or not isinstance(text, str):
            return ()
        return tuple(
            sentence.strip()
            for sentence in re.split(r"(?<=[。！？!?])|[\r\n]+", text)
            if sentence.strip()
        )

    @staticmethod
    def _references_external_game_memory(text: str) -> bool:
        external_memory_patterns = (
            r"(?:過去|以前|前回|昔|別の)(?:の)?(?:人狼)?(?:ゲーム|試合|対戦|村)",
            r"(?:過去|以前|前回|昔)(?:に|の)?[^。！？\n]{0,24}"
            r"(?:人狼|狂人|占い師|村人)(?:だった|でした|経験)",
            r"(?:過去|以前|前回|昔)の役職",
            r"(?:学習|訓練)(?:時|データ|内容|の記憶)",
            r"実在(?:する|の)?(?:人物|プレイヤー)|現実(?:の|では)",
            r"(?i:\b(?:previous|past|earlier|last|other|another|a prior)\b"
            r"[^.!?\n]{0,24}\b(?:werewolf game|game|match|round|village|role)\b)",
            r"(?i:\b(?:training data|trained|learned|real[- ]world|real person|"
            r"in real life|outside this game)\b)",
        )
        return any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in external_memory_patterns
        )

    def _dead_player_question_targets(self, text: str) -> tuple[str, ...]:
        dead_players = frozenset(
            player.name for player in self.state.players if not player.is_alive
        )
        living_players = frozenset(
            player.name for player in self.state.players if player.is_alive
        )
        classifications = analyze_dead_player_references(
            text, dead_players, living_players
        )
        return tuple(
            sorted(
                {
                    c.dead_player
                    for c in classifications
                    if c.is_dead_player_question and c.dead_player
                }
            )
        )

    def _dead_player_action_targets(self, text: str) -> tuple[str, ...]:
        dead_players = frozenset(
            player.name for player in self.state.players if not player.is_alive
        )
        living_players = frozenset(
            player.name for player in self.state.players if player.is_alive
        )
        classifications = analyze_dead_player_references(
            text, dead_players, living_players
        )
        return tuple(
            sorted(
                {
                    c.dead_player
                    for c in classifications
                    if c.is_actionable_target and c.dead_player
                }
            )
        )

    def _has_unverified_past_vote_reference(self, text: str) -> bool:
        vote_history = self.state.vote_history
        player_names = tuple(player.name for player in self.state.players)
        for sentence in self._discussion_sentences(text):
            if any(
                marker in sentence
                for marker in (
                    "記録にない",
                    "台帳に記録がない",
                    "記録を確認できない",
                    "判断できない",
                )
            ) or re.search(r"(?i:not in record|cannot determine|not in the ledger)", sentence):
                continue
            previous_vote = re.search(
                r"(?:昨日|前日|前回|前の(?:ターン|日|ラウンド)?|過去の)"
                r"(?:の)?(?:投票|票)"
                r"|(?i:\b(?:yesterday|the previous day|last day|last vote|previous vote|"
                r"prior vote|past vote|yesterday's vote)\b)",
                sentence,
            )
            if previous_vote is None:
                continue
            cue = previous_vote.group(0)
            if "昨日" in cue or "前日" in cue or re.search(r"(?i:yesterday|previous day|last day)", cue):
                expected_day = self.state.day - 1
                relevant_history = [
                    votes
                    for day, _, votes in vote_history
                    if day == expected_day
                ]
                if not relevant_history:
                    return True
            else:
                relevant_history = [
                    votes for _, _, votes in vote_history
                ]
            if not relevant_history:
                return True
            recorded_votes = {
                (voter, target)
                for votes in relevant_history
                for voter, target in votes.items()
            }
            for voter in player_names:
                for target in player_names:
                    if voter == target:
                        continue
                    relation = re.search(
                        rf"{re.escape(voter)}(?:さん)?(?:は|が)"
                        rf"[^。！？!?]{{0,20}}{re.escape(target)}(?:さん)?"
                        rf"(?:に|へ)(?:投票|入れ)",
                        sentence,
                    )
                    if relation and (voter, target) not in recorded_votes:
                        return True
        return False

    def _discussion_topic_repeated(
        self, text: str, conversation_history: tuple[Message, ...] | None = None
    ) -> bool:
        history = (
            conversation_history
            if conversation_history is not None
            else tuple(self.state.conversation_log)
        )
        prior_messages = PromptBuilder.current_day_messages(
            history
        )
        prior_sentences = [
            sentence
            for message in prior_messages
            for sentence in self._discussion_sentences(message.content)
        ]
        question_cues = (
            "?",
            "？",
            "ですか",
            "ますか",
            "でしょうか",
            "聞きたい",
            "教えて",
            "説明して",
            "確認したい",
            "なぜ",
            "どうして",
            "who",
            "what",
            "why",
            "how",
            "ask",
            "question",
            "explain",
            "check",
        )
        conclusion_cues = (
            "怪しい",
            "人狼",
            "疑",
            "投票",
            "信頼",
            "可能性",
            "と考え",
            "と思い",
            "suspect",
            "suspicious",
            "werewolf",
            "vote",
            "trust",
            "possible",
            "think",
            "believe",
        )

        def category(sentence: str) -> str | None:
            lowered = sentence.lower()
            if any(cue in sentence or cue in lowered for cue in question_cues):
                return "question"
            if any(cue in sentence or cue in lowered for cue in conclusion_cues):
                return "conclusion"
            return None

        def normalized(sentence: str) -> str:
            return re.sub(r"[\W_]+", "", sentence, flags=re.UNICODE)

        seen: list[tuple[str, str, str]] = [
            (kind, normalized(sentence), sentence)
            for sentence in prior_sentences
            if (kind := category(sentence)) is not None
        ]
        for sentence in self._discussion_sentences(text):
            if re.search(r"(?:占い師)?CO|判定\s*=|(?i:\bCO\b|result\s*=)", sentence, re.IGNORECASE):
                continue
            kind = category(sentence)
            normalized_sentence = normalized(sentence)
            if kind is None or len(normalized_sentence) < 10:
                continue
            if any(
                prior_kind == kind
                and len(prior_text) >= 10
                and SequenceMatcher(None, normalized_sentence, prior_text).ratio()
                >= 0.82
                and not self._adds_new_discussion_signal(sentence, prior_sentence)
                for prior_kind, prior_text, prior_sentence in seen
            ):
                return True
            seen.append((kind, normalized_sentence, sentence))
        return False

    @staticmethod
    def _adds_new_discussion_signal(current: str, previous: str) -> bool:
        def signals(sentence: str | None) -> set[tuple[str, str]]:
            if not sentence or not isinstance(sentence, str):
                return set()
            found: set[tuple[str, str]] = set()
            vote_target = re.search(
                r"(?:投票先(?:は|を)?\s*)?(Player\s+\d+|P\d+)(?:さん)?(?:へ|に)(?:の)?(?:投票|票)"
                r"|(?:vote for|voting for|my vote is for|vote target[:=]?)\s*(Player\s+\d+|P\d+)",
                sentence,
                re.IGNORECASE,
            )
            if vote_target is not None:
                target = vote_target.group(1) or vote_target.group(2)
                if target:
                    found.add(("vote_target", target))
            for kind, cue in (
                ("evidence", r"(?:根拠|理由|具体的には|記録では|発言では)|(?i:\b(?:basis|reason|evidence|record|speech)\b)"),
                ("refutation", r"(?:反証|矛盾|ただし|しかし|一方で|訂正)|(?i:\b(?:refut|contradict|however|but|correct|revis)\w*)"),
            ):
                # cue は分岐を含むため `(?:cue)` で包む。素直に連結すると末尾の
                # capture group が英語分岐側だけで参加し、日本語分岐が一致した時に
                # group(1) が None になって TypeError になった。
                match = re.search(r"(?:" + cue + r")([^。！？!?]{1,36})", sentence)
                if match is not None:
                    value = re.sub(r"[\W_]+", "", match.group(1) or "", flags=re.UNICODE)
                    if value:
                        found.add((kind, value))
            return found

        return bool(signals(current) - signals(previous))

    @staticmethod
    def _meta_echo_sentences(sentence: str) -> list[str]:
        """本文に混入したシステムメタ・修復定型文の該当文を返す（ゲーム内発言ではない）。"""
        if not sentence or is_pass_body(sentence):
            return []
        return [
            pattern
            for pattern in META_ECHO_PATTERNS
            if re.search(pattern, sentence, re.IGNORECASE)
        ]

    @staticmethod
    def _template_placeholder_sentences(text: str | None) -> list[str]:
        """プロンプトの案内文・プレースホルダをそのまま書いた文を検出する（TASK-043）。"""
        return [
            sentence
            for sentence in OllamaActions._discussion_sentences(text)
            if any(
                re.search(pattern, sentence, re.IGNORECASE)
                for pattern in _PLACEHOLDER_PATTERNS
            )
        ]

    @staticmethod
    def _contains_template_placeholder(text: str | None) -> bool:
        return bool(OllamaActions._template_placeholder_sentences(text))

    def _strip_meta_echo(self, content: str) -> str:
        """本文からメタ発話だけの文を取り除く。CO・結果ブロックは保持する。"""
        if not content:
            return content
        structured = parse_discussion_output(content)
        body = structured.body if structured is not None else content
        kept = [
            sentence
            for sentence in self._discussion_sentences(body)
            if not self._meta_echo_sentences(sentence)
        ]
        stripped_body = " ".join(kept).strip()
        if structured is None:
            return stripped_body
        blocks = [f"[CO] {structured.claim}"]
        if structured.result:
            blocks.append(f"[RESULT] {structured.result}")
        if stripped_body:
            blocks.append(f"[BODY] {stripped_body}")
        return "\n".join(blocks)

    def _fallback_discussion_turn(
        self, actor: Player, conversation_history: tuple[Message, ...] | None
    ) -> str:
        """重複・メタ発話の除去で本文が空になった場合の決定的な代替文。

        生存者への確認点を述べる内容のある文であり、修復用のメタ定型文ではない。
        直前の発言者が自分へ質問していた場合は、その相手へ先に回答する形にする。
        """
        question_set = self.state.action_candidate_sets.get(f"question:{actor.name}")
        target = (
            question_set.fallback
            if question_set is not None
            else next(
                (name for name in self.state.alive_players if name != actor.name),
                None,
            )
        )
        if target is None:
            return "Pass"
        history = (
            conversation_history
            if conversation_history is not None
            else tuple(self.state.conversation_log)
        )
        current_day_history = PromptBuilder.current_day_messages(history)
        previous_message = current_day_history[-1] if current_day_history else None
        prefix = (
            f"To {previous_message.speaker}: "
            if previous_message is not None
            and previous_message.speaker != actor.name
            and PromptBuilder._is_directed_to_player(
                actor, previous_message, current_day_history
            )
            else ""
        )
        if prefix and is_japanese_language(actor.response_language):
            prefix = f"{previous_message.speaker}さんの発言に答えます。"
        if is_japanese_language(actor.response_language):
            return (
                f"{prefix}{target}さんの現在の投票意思と、直前の発言の根拠を確認したい。"
            )
        return (
            f"{prefix}I want to check {target}'s current voting intention and "
            "the basis of their latest speech."
        )

    _TRUNCATION_TERMINATORS = re.compile(r"[。！？!?.]\s*$")

    def _truncated_output_usable(self, response_text: str, action: str) -> bool:
        """切り詰められた応答がそのまま受理できる内容を持つか（TASK-042）。

        構造化議論応答は `[CO]` と `[BODY]`（CO決定では `[CO]` 単独）が揃うことを、
        自由文は文末が文の区切りで閉じていることを条件にする。途中で切れた文は
        再試行・フォールバック対象のままにする。
        """
        if not response_text or not isinstance(response_text, str):
            return False
        if not response_text.strip():
            return False
        if action in {"discussion", "discussion_body", "discussion_claim"}:
            if action == "discussion_claim":
                decision = parse_discussion_claim(response_text)
                return decision is not None and bool(decision.claim.strip())
            structured = parse_discussion_output(response_text)
            if structured is None:
                return False
            return bool(structured.claim.strip()) and bool(structured.body.strip())
        return bool(self._TRUNCATION_TERMINATORS.search(response_text.strip()))

    def _preserved_fallback_response(
        self,
        action: str,
        actor: Player,
        raw_response: str,
        recent_messages: tuple[Message, ...] | None = None,
    ) -> str | None:
        """フォールバックでもモデルが書いた本文を可能な限り保持する（TASK-042）。

        秘匿情報の漏洩や、再生成が必要な致命的な整合違反が残る場合は保持しない。
        切り詰め・形式・重複系の残りエラーは許容して、無内容な「Pass」より内容を残す。
        """
        if not raw_response or not isinstance(raw_response, str) or not raw_response.strip():
            return None
        if action == "wolf_discussion":
            cleaned = self._strip_meta_echo(self._final_response(raw_response)).strip()
            if not cleaned or self._meta_echo_sentences(cleaned):
                return None
            return cleaned
        structured = parse_discussion_output(raw_response)
        if action == "discussion_claim":
            decision = parse_discussion_claim(raw_response)
            if decision is None or not decision.claim.strip():
                return None
            blocks = [f"[CO] {decision.claim}"]
            if decision.result:
                blocks.append(f"[RESULT] {decision.result}")
            candidate = "\n".join(blocks)
        elif structured is not None:
            body = self._strip_meta_echo(structured.body).strip() if structured.body else ""
            if not body:
                return None
            if action == "discussion_body":
                candidate = body
            else:
                blocks = [f"[CO] {structured.claim}"]
                if structured.result:
                    blocks.append(f"[RESULT] {structured.result}")
                blocks.append(f"[BODY] {body}")
                candidate = "\n".join(blocks)
        else:
            candidate = self._strip_meta_echo(raw_response).strip()
        if not candidate or self._meta_echo_sentences(candidate):
            return None
        if self._public_response_leak_findings(action, actor, candidate):
            return None
        residual = self._discussion_response_issue(actor, candidate, recent_messages)
        tolerated: frozenset[str | None] = frozenset(
            {
                None,
                "generation_truncated",
                "discussion_output_format",
                "repeated_discussion_topic",
                "past_action_question_unanswered",
                "question_not_answered_first",
                "vague_wait_and_see",
            }
        )
        if residual in tolerated:
            return candidate
        return None

    def _consult_fallback_text(self, actor: Player) -> str:
        """人狼相談のフォールバックを応答言語の内容ある文にする（TASK-042）。

        旧来の英語定型文（"could not respond to the consultation"）は相談履歴に残り、
        以後の相談がその英語文を引用する原因になっていた。
        """
        candidates = [
            name
            for name in self.state.alive_players
            if name != actor.name and name != actor.partner
        ]
        if not candidates:
            return self._localized_text(
                actor,
                "No attack candidate is available tonight.",
                "今夜は攻撃候補がいないため、攻撃案は出さない。",
            )
        target = candidates[0]
        return self._localized_text(
            actor,
            f"Attack target: {target}\nNo strong basis is recorded yet; compare other candidates tomorrow.",
            f"Attack target: {target}\n強い根拠はまだ記録にないため、明日は他の候補も比較したい。",
        )

    def _attributed_quote_sentences(self, text: str, history: tuple[Message, ...]) -> bool:
        """記録にない発言を他プレイヤーの引用として示しているかを判定する（TASK-042）。"""
        if not isinstance(text, str) or not text:
            return False
        known_speech = tuple(message for message in history if message.speaker != "system")
        for player in self.state.players:
            attributed_quote = re.compile(
                rf"{re.escape(player.name)}(?:さん)?(?:は|が)[^。！？\n]{{0,24}}[「『\"]([^」『\"]+)[」『\"]"
            )
            for match in attributed_quote.finditer(text):
                quote = match.group(1).strip()
                if quote and not any(
                    message.speaker == player.name and quote in message.content
                    for message in known_speech
                ):
                    return True
        return False

    @staticmethod
    def _localized_text(actor: Player, english: str, japanese: str) -> str:
        """修復・代替の定型文を応答言語で返す（英語文が日本語の公開発言に混入しない）。"""
        return japanese if is_japanese_language(actor.response_language) else english

    def _repair_discussion_response(
        self,
        actor: Player,
        response: str,
        issue: str,
        conversation_history: tuple[Message, ...] | None = None,
    ) -> str | None:
        structured = parse_discussion_output(response)
        body = structured.body if structured is not None else response
        known_players = {player.name for player in self.state.players}
        repair_history = (
            conversation_history
            if conversation_history is not None
            else tuple(self.state.conversation_log)
        )
        replacement: str | None = None
        if issue == "external_game_memory":
            replacement = self._localized_text(
                actor,
                "That information is not in this game's record, so I cannot determine it.",
                "その情報はこのゲームの記録にないため、判断できません。",
            )
        elif issue == "unverified_past_vote_reference":
            replacement = self._localized_text(
                actor,
                "That past vote is not recorded in this game's ledger, so I cannot rely on it.",
                "その過去の投票はこのゲームの台帳に記録されていないため、依据にできません。",
            )
        elif issue == "unverified_dead_role":
            replacement = self._localized_text(
                actor,
                "The true role is not confirmed, so I treat that player's role as speculation.",
                "死亡したプレイヤーの真役職は未確認なので、その役職は推測として扱います。",
            )
        elif issue == "dead_player_question":
            target = next(
                (
                    name
                    for name in self._dead_player_question_targets(body)
                    if name
                ),
                None,
            )
            question_set = self.state.action_candidate_sets.get(
                f"question:{actor.name}"
            )
            alternative = (
                question_set.fallback
                if question_set is not None
                else next(
                    (
                        name
                        for name in self.state.alive_players
                        if name != actor.name
                    ),
                    None,
                )
            )
            if target is not None:
                replacement = (
                    self._localized_text(
                        actor,
                        f"{target} is dead and cannot answer, so I will ask {alternative} about what the record shows.",
                        f"{target}さんは死亡しているため答えられません。代替として{alternative}さんに記録の確認を求めます。",
                    )
                    if alternative is not None
                    else self._localized_text(
                        actor,
                        f"{target} is dead and cannot answer, so I cannot judge what is not recorded.",
                        f"{target}さんは死亡しているため答えられません。記録にない事項は判断できません。",
                    )
                )
        elif issue == "dead_player_action_target":
            target = next(
                (
                    name
                    for name in self._dead_player_action_targets(body)
                    if name
                ),
                None,
            )
            question_set = self.state.action_candidate_sets.get(
                f"question:{actor.name}"
            )
            alternative = (
                question_set.fallback
                if question_set is not None
                else next(
                    (
                        name
                        for name in self.state.alive_players
                        if name != actor.name
                    ),
                    None,
                )
            )
            if target is not None and alternative is not None:
                replacement = self._localized_text(
                    actor,
                    f"{target} is dead and cannot be targeted. My current judgment is about the living player {alternative}.",
                    f"{target}さんは死亡しているため行動対象にできません。私の現在の判断対象は生存している{alternative}さんです。",
                )
            elif alternative is not None:
                replacement = self._localized_text(
                    actor,
                    f"My current judgment is about the living player {alternative}.",
                    f"私の現在の判断対象は生存している{alternative}さんです。",
                )
            else:
                replacement = self._localized_text(
                    actor,
                    "I do not ask dead players for current actions; I check only what is recorded.",
                    "死亡したプレイヤーに現在の行動を求めません。記録された事項だけを確認します。",
                )
        elif issue in {"contradictory_past_vote", "recorded_action_denied"}:
            latest_vote = next(
                (
                    memory
                    for memory in reversed(
                        self.state.player_action_memories.get(actor.name, [])
                    )
                    if memory.action_type == "vote"
                ),
                None,
            )
            if latest_vote is not None and latest_vote.vote_target:
                replacement = self._localized_text(
                    actor,
                    f"In the action ledger, I voted for {latest_vote.vote_target}. "
                    "If the reason at that time was not recorded, I separate it from my current guess.",
                    f"行動台帳では、私は{latest_vote.vote_target}に投票しました。当時の理由が記録されていないなら、現在の推測と区別します。",
                )
            else:
                replacement = self._localized_text(
                    actor,
                    "There is no record confirming my past vote target, "
                    "so I do not assert the past target or reason.",
                    "過去の投票先を確定する記録がないため、当時の投票先や理由を断定しません。",
                )
        elif issue == "contradictory_past_action":
            latest_action = next(
                (
                    memory
                    for memory in reversed(
                        self.state.player_action_memories.get(actor.name, [])
                    )
                    if memory.action_type in {"seer", "attack"}
                    and memory.selected_target
                ),
                None,
            )
            if latest_action is not None:
                action_label = (
                    "divination target"
                    if latest_action.action_type == "seer"
                    else "attack target"
                )
                replacement = self._localized_text(
                    actor,
                    f"In the action ledger, my {action_label} on Day {latest_action.day} was "
                    f"{latest_action.selected_target}.",
                    f"行動台帳では、{latest_action.day}日目の私の"
                    + (
                        "占い対象は"
                        if latest_action.action_type == "seer"
                        else "襲撃対象は"
                    )
                    + f"{latest_action.selected_target}でした。",
                )
            else:
                replacement = self._localized_text(
                    actor,
                    "There is no record confirming my past role action, "
                    "so I do not assert the target.",
                    "私の過去の役職行動を確定する記録がないため、対象を断定しません。",
                )
        elif issue == "public_fact_denied":
            replacement = self._localized_text(
                actor,
                "I check what is recorded in the public FactLedger, and I do not claim it is "
                "unrecorded; I separate the record from my current judgment.",
                "公開事実台帳に記録されている内容を確認し、未記録だと主張はしません。記録と現在の判断を区別します。",
            )
        elif issue == "past_action_question_unanswered":
            latest_action = next(
                iter(
                    reversed(
                        self.state.player_action_memories.get(actor.name, [])
                    )
                ),
                None,
            )
            if latest_action is not None:
                reason = (
                    latest_action.reasoning_summary
                    or self._localized_text(
                        actor,
                        "the reason at that time was not recorded",
                        "当時の理由は記録されていない",
                    )
                )
                replacement = self._localized_text(
                    actor,
                    f"About my {latest_action.action_type} on Day {latest_action.day}: {reason}. "
                    "My current judgment is a guess based on the current public information.",
                    f"{latest_action.day}日目の私の{latest_action.action_type}についての回答: {reason}。"
                    "現在の判断は、現在の公開情報に基づく推測です。",
                )
        elif issue == "unknown_player_reference":
            # TASK-042: 存在しないプレイヤー名を含む文だけを取り除く。他の文の内容は
            # 保持するため、再生成→フォールバックより情報損失が少ない。
            replacement = None
        elif issue == "unverified_attributed_speech":
            # TASK-042: 記録にない引用（他プレイヤーの発言として示された架空の引用）を含む
            # 文だけを取り除く。他の文の根拠は保持する。
            replacement = None
        elif issue == "repeated_discussion_topic":
            # TASK-036: 修復用の定型文を本文へ挿さない。挿すと本文がメタ発話として
            # 公開され、他のモデルが同じ文を引用・連呼する（実機ログでの連呼の発生源）。
            replacement = None
        elif issue == META_ECHO_ISSUE:
            replacement = None
        elif issue == "third_person_self_reference":
            # 三人称の自己参照は文の内容を削らず、自分の名前を一人称へ置き換えるだけ
            # で修復する（TASK-039）。再生成や定型文の挿入は議論の自由度を削る。
            replacement = None

        if replacement is None and issue not in {
            META_ECHO_ISSUE,
            "repeated_discussion_topic",
            "third_person_self_reference",
            "unknown_player_reference",
            "unverified_attributed_speech",
            TEMPLATE_PLACEHOLDER_ISSUE,
        }:
            return None
        changed = False
        third_person_sentences = (
            self._third_person_self_sentences(
                body, actor, include_role_assertions=False
            )
            if issue == "third_person_self_reference"
            else ()
        )
        repaired_sentences: list[str] = []
        for sentence in self._discussion_sentences(body):
            should_replace = (
                issue == "external_game_memory"
                and self._references_external_game_memory(sentence)
                or issue == "unverified_past_vote_reference"
                and self._has_unverified_past_vote_reference(sentence)
                or issue == "dead_player_question"
                and bool(self._dead_player_question_targets(sentence))
                or issue == "repeated_discussion_topic"
                and self._discussion_topic_repeated(
                    sentence, conversation_history
                )
                or issue == "contradictory_past_vote"
                and bool(SelfConsistencyGuard.inspect(sentence, actor, self.state))
                or issue == "contradictory_past_action"
                and bool(SelfConsistencyGuard.inspect(sentence, actor, self.state))
                or issue == "recorded_action_denied"
                and bool(SelfConsistencyGuard.inspect(sentence, actor, self.state))
                or issue == "public_fact_denied"
                and bool(SelfConsistencyGuard.inspect(sentence, actor, self.state))
                or issue == META_ECHO_ISSUE
                and bool(self._meta_echo_sentences(sentence))
                or issue == "third_person_self_reference"
                and sentence in third_person_sentences
                or issue == "unknown_player_reference"
                and any(
                    name not in known_players
                    for name in re.findall(r"Player\s+\d+", sentence)
                )
                or issue == "unverified_attributed_speech"
                and self._attributed_quote_sentences(sentence, repair_history)
                # TASK-043: プレースホルダ・案内文の模倣を含む文だけ除去する。
                or issue == TEMPLATE_PLACEHOLDER_ISSUE
                and bool(self._template_placeholder_sentences(sentence))
            )
            if issue == "unverified_dead_role":
                should_replace = any(
                    not any(
                        hedge in sentence
                        for hedge in (
                            "かもしれない",
                            "かもしれません",
                            "可能性",
                            "推測",
                            "と思う",
                            "と考える",
                            "とは限らない",
                            "未確認",
                            "maybe",
                            "might",
                            "possibly",
                            "possible",
                            "guess",
                            "suspect",
                            "speculation",
                            "not confirmed",
                            "unverified",
                        )
                    )
                    and re.search(
                        rf"{re.escape(dead_name)}(?:さん)?(?:は|が)?[^。！？!?]{{0,16}}"
                        rf"(?:人狼|狂人|占い師|村人)(?:だった|でした|です|だ|と判明|と確定)",
                        sentence,
                    )
                    for dead_name in (
                        player.name
                        for player in self.state.players
                        if not player.is_alive
                    )
                )
            if issue == "dead_player_action_target":
                should_replace = bool(self._dead_player_action_targets(sentence))
            if should_replace:
                changed = True
                if issue == "third_person_self_reference":
                    repaired_sentences.append(
                        self._first_person_substitution(sentence, actor)
                    )
                elif replacement is not None:
                    repaired_sentences.append(replacement)
            else:
                repaired_sentences.append(sentence)
        # TASK-043: 案内文の模倣は本文の該当文だけでなく、CO行・結果行のプレースホルダも
        # 決定的に除去する（実在しない対象の判定を公開しない）。
        placeholder_dropped_claim = (
            issue == TEMPLATE_PLACEHOLDER_ISSUE
            and structured is not None
            and self._contains_template_placeholder(structured.claim)
        )
        placeholder_dropped_result = (
            issue == TEMPLATE_PLACEHOLDER_ISSUE
            and structured is not None
            and bool(structured.result)
            and self._contains_template_placeholder(structured.result)
        )
        if not changed and not placeholder_dropped_claim and not placeholder_dropped_result:
            return None
        repaired_body = " ".join(repaired_sentences)
        if not repaired_body.strip():
            # 全文が重複・メタ発話で除去された場合は、メタ文ではなく内容のある
            # ゲーム内発言へ決定的に補正する（無内容な定型文を公開しない）。
            repaired_body = self._fallback_discussion_turn(actor, conversation_history)
        if structured is None:
            return repaired_body
        blocks: list[str] = []
        if placeholder_dropped_claim:
            self._emit_placeholder_line_dropped(actor, "CO", structured.claim)
        else:
            blocks.append(f"【CO】{structured.claim}")
        if structured.result and not placeholder_dropped_result:
            blocks.append(f"【結果】{structured.result}")
        elif placeholder_dropped_result:
            # TASK-043: 幻覚の対象を含む結果行は公開しない（COと本文は保持）。
            self._emit_placeholder_line_dropped(actor, "RESULT", structured.result)
        if repaired_body:
            blocks.append(f"【本文】{repaired_body}")
        return "\n".join(blocks)

    def _emit_placeholder_line_dropped(
        self, actor: Player, block_label: str, dropped: str
    ) -> None:
        """プレースホルダ模倣の行を除去したことを診断イベントとして残す（TASK-043）。"""
        self._emit(
            "hallucinated_template_line_dropped",
            "discussion",
            actor,
            time.time(),
            reason=TEMPLATE_PLACEHOLDER_ISSUE,
            block=block_label,
            dropped_chars=len(dropped),
        )

    _THIRD_PERSON_SELF_MARKERS = (
        "人狼",
        "村人",
        "占い師",
        "狂人",
        "発言",
        "意見",
        "怪しい",
        "疑",
        "投票",
        "主張",
        "CO",
        "占",
    )

    def _third_person_self_sentences(
        self, body: str, actor: Player, *, include_role_assertions: bool = True
    ) -> tuple[str, ...]:
        """自分のプレイヤー名を三人称の主語として自分について論じる文を集める（TASK-039）。

        旧来検出は行頭だけをを見ていたため、文中の「Player 1が疑われている」型の
        自己参照を見逃していた。引用（「 」等）内の他人発言は自己参照ではない。
        `include_role_assertions=False`では役職語を含む文を除外する（再要求で
        一人称のCO・役職発言へ書き直させるため、決定的置換の対象から外す）。
        """
        if not body or not actor.name:
            return ()
        own = re.escape(actor.name)
        markers = "|".join(re.escape(marker) for marker in self._THIRD_PERSON_SELF_MARKERS)
        found: list[str] = []
        for sentence in self._discussion_sentences(body):
            match = re.search(
                rf"{own}(?:さん)?(?:は|が|も)(?=[^。！？!?]{{0,32}}(?:{markers}))",
                sentence,
            )
            if match is None:
                continue
            if any(
                quote in sentence[: match.start()] for quote in ("「", "『", '"', "'")
            ):
                continue
            if not include_role_assertions and re.search(
                r"(?:人狼|村人|占い師|狂人)", sentence
            ):
                continue
            found.append(sentence)
        return tuple(found)

    @staticmethod
    def _first_person_substitution(sentence: str, actor: Player) -> str:
        """自分のプレイヤー名を一人称へ置き換える（助詞・内容量はそのまま維持する）。"""
        if not actor.name:
            return sentence
        own = re.escape(actor.name)
        return re.sub(
            rf"{own}(?:さん)?(?P<particle>は|が|も|を|に|へ|の)?",
            lambda match: f"私{match.group('particle') or ''}",
            sentence,
        )

    def _discussion_response_issue(
        self,
        actor: Player,
        response: str | None,
        conversation_history: tuple[Message, ...] | None = None,
    ) -> str | None:
        # 本文がNoneのまま渡ると正規表現がTypeErrorになるため、空文字へ正規化する。
        if response is None:
            response = ""
        elif not isinstance(response, str):
            response = str(response)
        structured_output = parse_discussion_output(response)
        if has_discussion_output_markers(response) and structured_output is None:
            return "discussion_output_format"
        body = structured_output.body if structured_output is not None else response
        claim_content = (
            structured_output.claim
            if structured_output is not None
            and structured_output.claim not in {"none", "hold"}
            else ""
            if structured_output is not None
            else response
        )
        # 未公開を表す [RESULT] 値（`None`・空・なし等）は空へ正規化し、公開判定として
        # `seer_result_without_claim` に分類しない（TASK-041、再試行→フォールバックの抑制）。
        result_block = normalize_result_field(
            structured_output.result if structured_output is not None else ""
        )
        result_content = (
            "\n".join(part for part in (claim_content, result_block, body) if part)
            if structured_output is not None
            else response
        )
        # TASK-043: 案内文・プレースホルダの模倣（`ターゲット=プレイヤーX` / `判定=人狼/人間` 等）は、
        # 実在しない対象の占い結果として `seer_result_without_claim`（致命的）扱いにならない。
        # 該当行・文を決定的に除去できるため、再試行→フォールバックを作らない。
        if self._contains_template_placeholder(
            "\n".join(part for part in (claim_content, result_block, body) if part)
        ):
            return TEMPLATE_PLACEHOLDER_ISSUE
        known_players = {player.name for player in self.state.players}
        explicit_seer_co = is_public_seer_claim(claim_content, actor.name)
        public_result = extract_public_seer_result(result_content, tuple(known_players))

        # 占い結果を公開する応答は、本文の内容を検証する前に明示的な占い師COを要求する。
        if result_block and not explicit_seer_co:
            return "seer_result_without_claim"
        if public_result is not None and not explicit_seer_co:
            target, result = public_result
            if (
                actor.role is Role.SEER
                and self.state.seer_results.get(target) == result
                and not self._has_published_seer_result(actor.name, target, result)
            ):
                return "seer_result_without_claim"

        if body:
            language = actor.response_language.lower()
            if language in ("english", "en"):
                if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", body):
                    return "wrong_response_language"
            elif language in ("japanese", "日本語") and not re.search(
                r"[\u3040-\u30ff\u3400-\u9fff]", body
            ):
                # --language Japanese（英字表記）でも日本語出力を要求する。
                return "wrong_response_language"

        # TASK-036: 修復定型文・システムメタ語の混入は本文から除去する（再生成不要）。
        # 混入した文をそのまま公開すると、他のモデルが同じメタ文を連呼する。
        if body and self._meta_echo_sentences(body):
            return META_ECHO_ISSUE

        history = (
            conversation_history
            if conversation_history is not None
            else tuple(self.state.conversation_log)
        )
        current_day_history = PromptBuilder.current_day_messages(history)
        previous_message = current_day_history[-1] if current_day_history else None
        if (
            self.question_hint_mode == QUESTION_HINT_STRICT
            and body
            and PromptBuilder._is_directed_to_player(
                actor, previous_message, current_day_history
            )
        ):
            # strict: 1文目に直前の発言者名を要求する（旧来動作・意味的再試行あり）。
            first_sentence = re.split(r"[。！？\n]", body, maxsplit=1)[0]
            if previous_message is not None and previous_message.speaker not in first_sentence:
                return "question_not_answered_first"

        mentioned_players = re.findall(r"Player\s+\d+", response)
        if any(name not in known_players for name in mentioned_players):
            return "unknown_player_reference"
        if self._references_external_game_memory(body):
            return "external_game_memory"
        if self._has_unverified_past_vote_reference(body):
            return "unverified_past_vote_reference"
        claimed_role = extract_public_role_claim(claim_content, actor.name)
        public_claims = [
            claim
            for claim in self.state.public_role_claims
            if claim.claimant == actor.name
        ]
        if claimed_role is not None and any(
            claim.claimed_role != claimed_role for claim in public_claims
        ):
            return "inconsistent_role_claim"
        if public_result is not None:
            target, result = public_result
            if any(
                claim.claimant == actor.name
                and claim.target == target
                and claim.result is not None
                and claim.result != result
                for claim in self.state.seer_claims
            ) or any(
                claim.claimant == actor.name
                and claim.claimed_role == Role.SEER.value
                and claim.target == target
                and claim.result is not None
                and claim.result != result
                for claim in self.state.public_role_claims
            ):
                return "inconsistent_seer_result"
        if actor.role is Role.SEER:
            unpublished_wolf_results = [
                target
                for target, result in self.state.seer_results.items()
                if result == "wolf"
                and not any(
                    claim.claimant == actor.name
                    and claim.target == target
                    and claim.result == "wolf"
                    for claim in self.state.seer_claims
                )
            ]
            if unpublished_wolf_results and (
                not is_public_seer_claim(claim_content, actor.name)
                or public_result
                not in {(target, "wolf") for target in unpublished_wolf_results}
            ):
                return "wolf_result_not_disclosed"
        dead_players = {
            player.name for player in self.state.players if not player.is_alive
        }
        if self._dead_player_question_targets(body):
            return "dead_player_question"
        if self._dead_player_action_targets(body):
            return "dead_player_action_target"
        for dead_name in dead_players:
            name = re.escape(dead_name)
            for sentence in self._discussion_sentences(body):
                if re.search(
                    rf"{name}(?:さん)?(?:は|が)?[^。！？!?]{{0,16}}"
                    rf"(?:人狼|狂人|占い師|村人)(?:だった|でした|です|だ|と判明|と確定)",
                    sentence,
                ) and not any(
                    hedge in sentence
                    for hedge in (
                        "かもしれない",
                        "かもしれません",
                        "可能性",
                        "推測",
                        "と思う",
                        "と考える",
                        "とは限らない",
                        "未確認",
                    )
                ):
                    return "unverified_dead_role"

        if re.search(
            r"(?:今夜|夜に)(?:は)?(?:私は|僕は|私が|僕が)?"
            r"(?:占う|占います|襲撃する|襲撃します)",
            body,
        ):
            return "out_of_phase_action_claim"

        for player in self.state.players:
            name = re.escape(player.name)
            if actor.role is not Role.WOLF and (
                re.search(
                    rf"{name}(?:さん)?(?:が|は)[^。！？\n]{{0,20}}(?:襲撃した|襲撃しました|襲った|襲いました|襲撃者(?:です|だ))",
                    body,
                )
                or re.search(
                    rf"襲撃者は{name}(?:さん)?(?:です|だ)",
                    body,
                )
            ):
                return "unknown_attacker_claim"
            if re.search(
                rf"{name}(?:さん)?(?:は|が)?[^。！？\n]{{0,20}}襲撃され(?:た|ました)",
                body,
            ) and not any(victim == player.name for _, victim in self.state.attack_deaths):
                return "unverified_game_event"
            if re.search(
                rf"{name}(?:さん)?(?:は|が)?[^。！？\n]{{0,20}}処刑され(?:た|ました)",
                body,
            ) and not any(executed == player.name for _, executed in self.state.executions):
                return "unverified_game_event"
            if re.search(
                rf"{name}(?:さん)?(?:は|が)?[^。！？\n]{{0,20}}死亡し(?:た|ました)",
                body,
            ) and player.is_alive:
                return "unverified_game_event"

        known_speech = tuple(
            message for message in history if message.speaker != "system"
        )
        for player in self.state.players:
            attributed_quote = re.compile(
                rf"{re.escape(player.name)}(?:さん)?(?:は|が)[^。！？\n]{{0,24}}[「『\"]([^」』\"]+)[」』\"]"
            )
            for match in attributed_quote.finditer(body):
                quote = match.group(1).strip()
                if quote and not any(
                    message.speaker == player.name and quote in message.content
                    for message in known_speech
                ):
                    return "unverified_attributed_speech"

        previous_speeches = [
            message.content
            for message in known_speech
            if message.speaker == actor.name
        ]
        normalized = re.sub(r"[\W_]+", "", body, flags=re.UNICODE)
        for previous in previous_speeches:
            prior_normalized = re.sub(r"[\W_]+", "", previous, flags=re.UNICODE)
            if len(normalized) >= 20 and normalized == prior_normalized:
                return "repeated_speech"
            if min(len(normalized), len(prior_normalized)) >= 40 and (
                SequenceMatcher(None, normalized, prior_normalized).ratio() >= 0.92
            ):
                return "repeated_speech"
        if self._discussion_topic_repeated(body, history):
            return "repeated_discussion_topic"
        consistency_findings = SelfConsistencyGuard.inspect(
            body, actor, self.state, previous_message
        )
        # TASK-042: warning級の整合指摘（過去の行動理由の未回答など）は、strictモード
        # 以外では再生成の材料にしない。観測は _question_answer_check が別経路で残すため、
        # 形式合わせの再試行→フォールバックだけを減らす。
        repair_findings = tuple(
            finding for finding in consistency_findings if finding.severity == "repair"
        )
        warning_findings = tuple(
            finding for finding in consistency_findings if finding.severity == "warning"
        )
        if repair_findings:
            return repair_findings[0].category
        if warning_findings and self.question_hint_mode == QUESTION_HINT_STRICT:
            return warning_findings[0].category
        # 様子見・情報不足のみの反復を検出（パスは除外）
        if not self._is_pass(response):
            vague_patterns = [
                r"様子見",
                r"様子を",
                r"情報不足",
                r"まだわからない",
                r"判断できない",
                r"何とも言えない",
                r"様子を伺",
            ]
            import re as _re
            if any(_re.search(p, body, _re.IGNORECASE) for p in vague_patterns):
                # 単に様子見と言っているだけで新しい情報がない場合
                if len(body.strip()) < 60:
                    return "vague_wait_and_see"
        # 三人称の自己参照は他の整合チェックより後に評価する（TASK-039）。役職語を含む
        # 文は再要求で一人称へ書き直させ、役職語のない文だけ決定的に一人称へ置換する
        # （置換だけで直せる場合は再生成しない＝議論の自由度を維持）。
        if self._third_person_self_sentences(body, actor):
            return "third_person_self_reference"
        return None

    def _discussion_claim_response_issue(
        self, actor: Player, response: str
    ) -> str | None:
        decision = parse_discussion_claim(response)
        if decision is None:
                    return "discussion_claim_format"
        if decision.claim in {"none", "hold"}:
                    if decision.result:
                        return "discussion_claim_format"
                    if actor.role is Role.SEER and any(
                        result == "wolf"
                        and not self._has_published_seer_result(actor.name, target, result)
                        for target, result in self.state.seer_results.items()
                    ):
                        return "wolf_result_not_disclosed"
                    return None
        claimed_role = extract_public_role_claim(decision.claim, actor.name)
        if claimed_role is None:
                    return "discussion_claim_format"
        if decision.result:
                    if claimed_role != Role.SEER.value:
                        return "discussion_claim_format"
                    result = extract_public_seer_result(
                        decision.result,
                        tuple(player.name for player in self.state.players),
                    )
                    if result is None:
                        return "discussion_claim_format"
                    target, result_kind = result
                    if actor.role is Role.SEER and self.state.seer_results.get(target) != result_kind:
                        return "unverified_seer_result_claim"
                    if any(
                        claim.claimant == actor.name
                        and claim.target == target
                        and claim.result not in {None, result_kind}
                        for claim in self.state.seer_claims
                    ):
                        return "inconsistent_seer_result"
        elif actor.role is Role.SEER and any(
                    result == "wolf"
                    and not self._has_published_seer_result(actor.name, target, result)
                    for target, result in self.state.seer_results.items()
        ):
                    return "wolf_result_not_disclosed"
        if any(
                    claim.claimant == actor.name and claim.claimed_role != claimed_role
                    for claim in self.state.public_role_claims
        ):
                    return "inconsistent_role_claim"
        return None

    def _deterministic_discussion_claim(self, actor: Player) -> str | None:
        if actor.role is not Role.SEER:
                    return None
        target = next(
                    (
                        target
                        for target, result in self.state.seer_results.items()
                        if result == "wolf"
                        and not self._has_published_seer_result(
                            actor.name, target, result
                        )
                    ),
                    None,
        )
        if target is None:
                    return None
        return (
                    "[CO] Seer\n"
                    f"[RESULT] Target={target} Result=Werewolf"
        )

    def _latest_seer_result(self) -> tuple[str, str]:
        if self.state.seer_result_history:
            _, target, result = self.state.seer_result_history[-1]
            return target, result
        return next(reversed(self.state.seer_results.items()))

    def _has_published_seer_result(
        self, claimant: str, target: str, result: str
    ) -> bool:
        return self.state.claim_ledger.public_seer_results.get(
            (claimant, target)
        ) == result or any(
            claim.claimant == claimant
            and claim.target == target
            and claim.result == result
            for claim in self.state.seer_claims
        ) or any(
            claim.claimant == claimant
            and claim.claimed_role == Role.SEER.value
            and claim.target == target
            and claim.result == result
            for claim in self.state.public_role_claims
        )

    def _options_for(
        self,
        actor: Player,
        budget_key: str = "discussion",
        *,
        disable_thinking: bool = False,
    ) -> tuple[dict[str, object] | None, str, bool | None, str]:
        """行動別生成上限と思考生成を優先順位で解決して options を組み立てる。

        num_predict優先順位: player行動別 > CLI行動別 > 全体設定（対象選択・CO補正はキャップ適用）
        > 行動別既定値
        思考優先順位: player行動別 > CLI行動別 > player全体 > CLI全体 > 行動別既定値
        再要求（disable_thinking=True）では思考を強制的にoffにする。
        """
        key = normalize_action_budget_key(budget_key)
        num_predict, num_predict_source = resolve_action_num_predict(
            player=actor,
            cli_options=self.options,
            budget_key=key,
        )
        options = (
            {
                option_key: value
                for option_key, value in self.options.items()
                if value is not None
                and not option_key.startswith("num_predict_by_action:")
                and not option_key.startswith("thinking_by_action:")
                and option_key not in {
                    "override_player_num_ctx",
                    "dynamic_context",
                    "question_hint_mode",
                    # TASK-053: 本文予算予約のon/offは実行側の設定で、Ollamaへは送らない。
                    "body_reservation",
                    # TASK-057: プロンプト契約の選択はプロンプト構築側（state）で使う設定なので、
                    # サーバーへは送らない（送信オプションに未知のキーを混ぜない）。
                    "prompt_contract_policy",
                }
            }
            if self.options is not None
            else {}
        )
        override_player_num_ctx = bool(
            self.options and self.options.get("override_player_num_ctx")
        )
        if actor.num_ctx is not None and not override_player_num_ctx:
            options["num_ctx"] = actor.num_ctx
        elif (
            "num_ctx" not in options
            and (
                actor.recommended_num_ctx is not None
                or actor.max_num_ctx is not None
            )
        ):
            options["num_ctx"] = (
                actor.recommended_num_ctx
                if actor.recommended_num_ctx is not None
                else actor.max_num_ctx
            )
        if (
            actor.max_num_ctx is not None
            and isinstance(options.get("num_ctx"), int)
            and not isinstance(options.get("num_ctx"), bool)
        ):
            options["num_ctx"] = min(
                int(options["num_ctx"]), actor.max_num_ctx
            )
        if num_predict is not None:
            options["num_predict"] = num_predict
        # TASK-043: プレイヤー別の生成サンプリングは全体設定（CLI）より優先する。
        # 幻覚・案内文の模倣を起こすモデルは temperature を下げて安定させる。
        for option_key in GENERATION_SAMPLING_KEYS:
            player_value = getattr(actor, option_key, None)
            if player_value is not None:
                options[option_key] = player_value
        thinking_enabled, thinking_source = resolve_thinking(actor, self.options, key)
        capabilities = self.state.model_capabilities.get(actor.model, {})
        if capabilities.get("thinking") is False and thinking_enabled is True:
            # そのモデルは思考生成非対応と判明しているため、設定指定より先にoffへ落とす。
            # 同一要求の400再試行（実送信3回）を防ぐ。
            thinking_enabled = False
            thinking_source = THINKING_CAPABILITY_SOURCE
        elif capabilities.get("thinking_budget") is False and thinking_enabled is True:
            # 思考onでは思考トークンが生成上限をすべて消費して最終回答が空になったため、
            # このモデルの思考行動はoffへ落とす（offなら同じ上限で本文が生成できる）。
            thinking_enabled = False
            thinking_source = THINKING_BUDGET_SOURCE
        if (
            capabilities.get("timed_out") is True
            and key in {"discussion", "wolf_discussion"}
        ):
            # TASK-036: このモデルの議論がタイムアウトしたため、以後の思考・生成上限を
            # 動的に絞る（思考onの大上限はローカル環境で最も応答時間が長い）。
            if (
                isinstance(num_predict, int)
                and not isinstance(num_predict, bool)
                and num_predict > TIMEOUT_SAFE_NUM_PREDICT
            ):
                num_predict = TIMEOUT_SAFE_NUM_PREDICT
                num_predict_source = TIMEOUT_SAFE_NUM_PREDICT_SOURCE
                options["num_predict"] = num_predict
            if thinking_enabled is True:
                thinking_enabled = False
                thinking_source = TIMEOUT_SAFE_THINKING_SOURCE
        if disable_thinking:
            thinking_enabled = False
            thinking_source = "retry_thinking_off"
        num_predict, num_predict_source = apply_thinking_floor(
            num_predict, num_predict_source, thinking_enabled, key
        )
        # TASK-053: 初回リクエストでは「本文最低予算 + モデル別thinking余白」を分離して
        # 確保する。投票・襲撃・占いなどの短い行動は本文を先に確保し、思考が上限を
        # 食い潰しても空応答にしない。設定値より優先して引き上げるが上限 cap で抑え、
        # タイムアウト観測済みモデルでは適用しない（待ち時間優先）。再要求は対象外。
        reservation = self._body_reservation(
            actor,
            key,
            num_predict,
            num_predict_source,
            thinking_enabled,
            retry=disable_thinking,
        )
        if reservation.applied and reservation.num_predict is not None:
            num_predict = reservation.num_predict
            num_predict_source = reservation.source
        if num_predict is not None:
            # 思考下限で変更した値も実際に送信するoptionsへ反映しないと、
            # 優先元だけ表示され送信値は変わらない。
            options["num_predict"] = num_predict
        if thinking_enabled is not None:
            options["thinking"] = thinking_enabled
        return options or None, num_predict_source, thinking_enabled, thinking_source

    def _dynamic_context_enabled(self) -> bool:
        """動的コンテキスト（num_ctx・発言履歴予算）を適用するか（既定on）。"""
        if self.options is None:
            return True
        return self.options.get("dynamic_context") is not False

    def _slow_model(self, actor: Player) -> bool:
        """このモデルで過去にタイムアウトが観測されているか（TASK-036）。"""
        return bool(
            self.state.model_capabilities.get(actor.model, {}).get("timed_out") is True
        )

    @staticmethod
    def _request_token_estimate(
        messages: list[dict[str, object]], num_predict: object
    ) -> int:
        """送信プロンプトの実測文字数と生成上限から、必要なコンテキストトーク数を概算する。"""
        return request_token_estimate(messages, num_predict)

    def _fit_request_context(
        self,
        options: dict[str, object],
        messages: list[dict[str, object]],
        actor: Player,
        *,
        thinking: bool = False,
        thinking_margin: int | None = None,
    ) -> tuple[dict[str, object], int | None, bool]:
        """必要なコンテキストだけ確保する動的num_ctx（TASK-036）。

        実測プロンプトサイズ + 生成上限（思考トークンの残余を含む）+ 余白 + thinking分
        （TASK-046）を必要量とし、必要量の2倍以上の過大割り当てだけを縮小する。逆に必要量
        不足の場合は、プレイヤー設定の `max_num_ctx` まで拡張する（設定の上限を超えない）。
        概算は日本語混在では大きめ（安全側）になる。
        TASK-053: `thinking_margin` にはモデル×アクション別の実測値を使い、思考を全く
        使わないモデルのコンテキストを過大確保しない（節約と応答時間の短縮）。
        """
        return fit_dynamic_context(
            options,
            messages,
            ceiling=actor.max_num_ctx,
            thinking=thinking,
            thinking_margin=thinking_margin,
        )

    def _dynamic_evidence_budget(
        self,
        actor: Player,
        budget_key: str,
        prompt_ctx: PromptContext,
    ) -> int | None:
        """割り当てコンテキストから逆算した発言履歴の文字数予算（TASK-036）。

        発言履歴が割り当てコンテキストに収まらない場合だけ、優先度の低い発言を削って
        精度（確定扱いの発言・直近・現在の質問・公開CO台帳）を守りながら応答時間を短縮する。
        タイムアウト観測モデルでは予算をさらに絞る。
        """
        if not self._dynamic_context_enabled() or prompt_ctx.diagnostics is None:
            return None
        options, _num_predict_source, thinking_enabled, _thinking_source = (
            self._options_for(actor, budget_key)
        )
        allocatable = options.get("num_ctx") if options else None
        if not isinstance(allocatable, int) or isinstance(allocatable, bool):
            # TASK-046: num_ctx未指定（サーバー側設定・既定値に委ねる）では、解決済み
            # 実効コンテキスト上限を割り当てとして発言履歴予算の逆算に使う。
            allocatable = self._resolve_context_budget(actor).limit
        if not isinstance(allocatable, int) or isinstance(allocatable, bool):
            return None
        generation = options.get("num_predict") if options else None
        if not isinstance(generation, int) or isinstance(generation, bool):
            generation = DEFAULT_ACTION_NUM_PREDICT.get(
                normalize_action_budget_key(budget_key), 512
            )
        diagnostics = prompt_ctx.diagnostics
        evidence_chars = int(diagnostics.get("evidence_char_count") or 0)
        raw_chars = len(prompt_ctx.system_prompt) + len(prompt_ctx.user_prompt)
        fixed_tokens = math.ceil(max(0, raw_chars - evidence_chars) * CONTEXT_TOKENS_PER_CHAR)
        # TASK-046: thinking on のリクエストは思考消費分を別枠で差し引き、本文が
        # 空・切り詰めされる入力を削る（発言の自由度は削らない：履歴の優先除去のみ）。
        # TASK-053: 余白はモデル×アクション別の実測値を使う。思考を使わないモデルでは
        # 0 に近い値になり、発言履歴へ回せるぶんだけコンテキストを増やす（節約と両立）。
        thinking_reserve = self._thinking_margin_for(actor, budget_key, bool(thinking_enabled))
        available_input_tokens = (
            int(allocatable) - int(generation) - CONTEXT_MARGIN_TOKENS - thinking_reserve
        )
        budget = int(
            (available_input_tokens - fixed_tokens) / CONTEXT_TOKENS_PER_CHAR
        )
        budget = min(CONTEXT_EVIDENCE_MAX_CHARS, max(CONTEXT_EVIDENCE_MIN_CHARS, budget))
        if self._slow_model(actor):
            budget = int(budget * TIMEOUT_EVIDENCE_RATIO)
        if budget >= evidence_chars:
            return None
        return budget

    def _record_model_timeout(
        self, actor: Player, action: str | None, started_at: float | None
    ) -> None:
        """タイムアウトをモデル能力情報へ記録し、以後の議論・人狼相談を絞って送る。"""
        capabilities = self.state.model_capabilities.setdefault(actor.model, {})
        first_detection = capabilities.get("timed_out") is not True
        capabilities["timed_out"] = True
        if first_detection and action is not None and started_at is not None:
            self._emit(
                "model_timeout_downgrade",
                action,
                actor,
                started_at,
                reason="model_timeout_observed",
                num_predict=TIMEOUT_SAFE_NUM_PREDICT,
                thinking_enabled=False,
                evidence_ratio=TIMEOUT_EVIDENCE_RATIO,
            )

    def _fit_retry_budget(
        self,
        options: dict[str, object],
        num_predict_source: str,
        messages: list[dict[str, object]],
        *,
        respect_explicit_num_ctx: bool = False,
        effective_limit: int | None = None,
    ) -> tuple[dict[str, object], str, bool, bool]:
        """タイムアウト・秘密漏洩の再要求で、送信済みの短いプロンプトに見合う上限へ絞る。

        思考offでも `num_predict` の大上限や大 `num_ctx` の再送は、ローカル環境で
        タイムアウトや数分の応答時間を作る（TASK-035）。本文だけ求める再要求では
        上限を `RETRY_NUM_PREDICT_LIMIT` に、コンテキストを実際のプロンプトサイズに見合う値へ
        縮小する。`override_player_num_ctx`（CLIで明示指定した比較実験）では尊重する。
        TASK-046: 解決済み実効上限 `effective_limit` を超えるnum_ctxは再試行でも送らない。
        """
        fitted = dict(options)
        num_predict_applied = False
        num_ctx_applied = False
        current_predict = fitted.get("num_predict")
        if (
            isinstance(current_predict, int)
            and not isinstance(current_predict, bool)
            and current_predict > RETRY_NUM_PREDICT_LIMIT
        ):
            fitted["num_predict"] = RETRY_NUM_PREDICT_LIMIT
            num_predict_source = RETRY_NUM_PREDICT_SOURCE
            num_predict_applied = True
        requested_ctx = fitted.get("num_ctx")
        if (
            isinstance(requested_ctx, int)
            and not isinstance(requested_ctx, bool)
            and not respect_explicit_num_ctx
        ):
            prompt_chars = sum(
                len(str(message.get("content", ""))) for message in messages
            )
            fitted_ctx = max(
                RETRY_NUM_CTX_MIN_FIT,
                math.ceil(prompt_chars * RETRY_NUM_CTX_TOKENS_PER_CHAR)
                + RETRY_NUM_PREDICT_LIMIT
                + RETRY_NUM_CTX_MARGIN_TOKENS,
            )
            if isinstance(effective_limit, int) and not isinstance(effective_limit, bool):
                fitted_ctx = min(fitted_ctx, int(effective_limit))
                if requested_ctx > int(effective_limit):
                    # 実効上限を超える設定値の再送は打ち切り（超過分の証拠を落とさないため
                    # 入力も上限内に圧縮済み）。
                    fitted["num_ctx"] = fitted_ctx
                    num_ctx_applied = True
            # 設定値を大幅に上回る割り当てだけを縮小し、わずかな差は設定どおり維持する。
            if requested_ctx >= fitted_ctx * 2:
                fitted["num_ctx"] = fitted_ctx
                num_ctx_applied = True
        return fitted, num_predict_source, num_predict_applied, num_ctx_applied

    def _chat_with_claim_tools(
        self,
        actor: Player,
        messages: list[dict[str, object]],
        action: str,
        *,
        increase_generation: bool,
        started_at: float,
        request_number: int = 1,
        request_reason: str = "initial",
        tool_based: bool = False,
        budget_key: str = "discussion",
        disable_thinking: bool = False,
        retry_budget_fit: bool = False,
    ) -> ChatResponse:
        # ツール非対応と判明したモデルは、以後ツールを送らず文章CO経路だけを使う。
        capabilities = self.state.model_capabilities.get(actor.model, {})
        tools = (
            [CLAIM_ROLE_TOOL]
            if action == "discussion"
            and capabilities.get("claim_role_tool", True)
            and callable(getattr(self.client, "chat_with_metadata", None))
            else None
        )
        try:
            response = self._chat(
                actor,
                messages,
                increase_generation=increase_generation,
                tools=tools,
                action=action,
                started_at=started_at,
                request_number=request_number,
                request_reason=request_reason,
                tool_based=tool_based,
                budget_key=budget_key,
                disable_thinking=disable_thinking,
                retry_budget_fit=retry_budget_fit,
            )
        except OllamaError as error:
            if tools is None or not self._is_tool_call_unsupported(error):
                raise
            self._record_tool_unsupported(actor, action, started_at)
            tools = None
            response = self._chat(
                actor,
                messages,
                increase_generation=increase_generation,
                action=action,
                started_at=started_at,
                request_number=request_number,
                request_reason=request_reason,
                tool_based=False,
                budget_key=budget_key,
                disable_thinking=disable_thinking,
                retry_budget_fit=retry_budget_fit,
            )
        for _ in range(2):
            if not response.tool_calls:
                return response
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            }
                        }
                        for call in response.tool_calls
                    ],
                }
            )
            for call in response.tool_calls:
                result = self._handle_public_tool_call(
                    actor, call, action, started_at
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": call.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            response = self._chat(
                actor,
                messages,
                tools=tools,
                action=action,
                started_at=started_at,
                request_number=request_number,
                request_reason=request_reason,
                tool_based=tool_based,
                budget_key=budget_key,
                disable_thinking=disable_thinking,
                retry_budget_fit=retry_budget_fit,
            )
        if response.tool_calls:
            return ChatResponse(
                content="",
                attempts=response.attempts,
                done_reason=response.done_reason,
                content_length=0,
                thinking_present=response.thinking_present,
                eval_count=response.eval_count,
                eval_duration_ms=response.eval_duration_ms,
            )
        return response

    @staticmethod
    def _is_tool_call_unsupported(error: OllamaError) -> bool:
        message = error.message.casefold()
        return any(
            marker in message
            for marker in (
                "does not support tools",
                "doesn't support tools",
                "tool calling is not supported",
                "tool calls are not supported",
                "tools are not supported",
                "ツールに対応していません",
                "ツール呼び出しに対応していません",
            )
        )

    @staticmethod
    def _is_thinking_unsupported(error: OllamaError) -> bool:
        """Ollamaが思考生成（think）非対応を理由に400を返したかどうか。"""
        message = error.message.casefold()
        return any(
            marker in message
            for marker in (
                "does not support thinking",
                "doesn't support thinking",
                "thinking is not supported",
                "thinking not supported",
                "does not support reasoning",
                "thinkingモード",
                "思考生成に対応していません",
                "思考に対応していません",
            )
        )

    def _record_thinking_unsupported(
        self,
        actor: Player,
        action: str | None,
        started_at: float | None,
    ) -> None:
        """思考生成非対応を能力情報へ記録し、以後の行動ではthink=falseだけを送る。"""
        capabilities = self.state.model_capabilities.setdefault(actor.model, {})
        first_detection = capabilities.get("thinking") is not False
        capabilities["thinking"] = False
        if first_detection and action is not None and started_at is not None:
            self._emit(
                "thinking_unavailable",
                action,
                actor,
                started_at,
                reason="model_does_not_support_thinking",
            )

    def _record_generation_response(
        self,
        actor: Player,
        budget_key: str,
        action: str | None,
        started_at: float | None,
        request_options: dict[str, object],
        request_number: int,
        request_reason: str,
        thinking_enabled: bool | None,
        send_started: float,
        response: ChatResponse,
    ) -> ChatResponse:
        """送信結果をモデル×アクション別の生成プロファイルへ加算する（TASK-053）。

        本文の有無・thinkingの有無・打ち切り（done_reason=length）・実測生成トークン・
        1送信あたりの待ち時間を集計し、`llm_generation_attempt` として記録する。
        議論の論点・口調には一切干渉せず、観測だけを残す。
        """

        key = normalize_action_budget_key(budget_key)
        profile = self._profile_for(actor, key)
        body_text = self._final_response(response.content).strip()
        elapsed_seconds = max(0.0, time.perf_counter() - send_started)
        thinking_present = bool(response.thinking_present) or self._has_thinking(
            response.content
        )
        empty = not body_text
        thinking_only = empty and thinking_present
        profile.record_attempt(
            request_number=request_number,
            body_char_count=len(body_text),
            thinking_present=thinking_present,
            done_reason=response.done_reason,
            eval_count=response.eval_count,
            elapsed_seconds=elapsed_seconds,
            empty=empty and not thinking_only,
            thinking_only=thinking_only,
        )
        body_budget = inspect_body_budget(
            profile,
            key,
            request_options.get("num_predict"),
            thinking_enabled=bool(thinking_enabled) if thinking_enabled is not None else None,
        )
        if thinking_only:
            # 「thinking余白＋本文最低予算を予約していたのに空になった」かだけを見る。
            # 余白を積んでいない送信でたまたま本文枠が残ったケースは数えない
            # （予約の成否自体は body_budget の body_covered で検証する）。
            allowance_tokens = body_budget.get("thinking_allowance_tokens")
            if (
                body_budget["body_covered"]
                and isinstance(allowance_tokens, int)
                and not isinstance(allowance_tokens, bool)
                and allowance_tokens > 0
            ):
                profile.covered_thinking_only_count += 1
            if response.done_reason == "length" and action is not None and started_at is not None:
                # 思考が生成上限を食い潰して本文が空（TASK-053: まず余白を拡大する）。
                self._record_thinking_budget_exhausted(
                    actor,
                    action,
                    started_at,
                    {
                        "eval_count": response.eval_count,
                        "num_predict": request_options.get("num_predict"),
                        "content_length": len(body_text),
                    },
                )
        if action is not None and started_at is not None:
            self._emit(
                "llm_generation_attempt",
                action,
                actor,
                started_at,
                request_number=request_number,
                request_reason=request_reason,
                action_budget_key=key,
                request_elapsed_seconds=round(elapsed_seconds, 3),
                thinking_requested=bool(thinking_enabled) if thinking_enabled is not None else None,
                thinking_present=thinking_present,
                body_char_count=len(body_text),
                body_empty=empty,
                thinking_only_response=thinking_only,
                done_reason=response.done_reason,
                eval_count=response.eval_count,
                num_predict=request_options.get("num_predict"),
                body_budget=body_budget,
                generation_profile=profile.serialize(),
            )
        return response

    def _record_generation_failure(
        self,
        actor: Player,
        budget_key: str,
        action: str | None,
        started_at: float | None,
        request_options: dict[str, object],
        request_number: int,
        request_reason: str,
        thinking_enabled: bool | None,
        send_started: float,
        error: OllamaError,
    ) -> None:
        """通信失敗をプロファイルへ記録する（タイムアウト・接続・HTTP）。"""

        key = normalize_action_budget_key(budget_key)
        profile = self._profile_for(actor, key)
        elapsed_seconds = max(0.0, time.perf_counter() - send_started)
        timed_out = error.kind == "timeout"
        profile.record_attempt(
            request_number=request_number,
            elapsed_seconds=elapsed_seconds,
            empty=True,
            timed_out=timed_out,
            transport_error=not timed_out,
        )
        if action is not None and started_at is not None:
            self._emit(
                "llm_generation_attempt",
                action,
                actor,
                started_at,
                request_number=request_number,
                request_reason=request_reason,
                action_budget_key=key,
                request_elapsed_seconds=round(elapsed_seconds, 3),
                thinking_requested=bool(thinking_enabled) if thinking_enabled is not None else None,
                thinking_present=False,
                body_char_count=0,
                body_empty=True,
                thinking_only_response=False,
                done_reason=None,
                eval_count=None,
                num_predict=request_options.get("num_predict"),
                generation_issue=self._transport_failure_reason(error),
                generation_profile=profile.serialize(),
            )

    def _record_thinking_budget_exhausted(
        self,
        actor: Player,
        action: str,
        started_at: float,
        metadata: dict[str, str | int | bool | None],
    ) -> None:
        """思考が生成上限を食い潰した観測を、まず「モデル別thinking余白の拡大」で解消する。

        TASK-053: 実機ログ game_088e49f0690c では投票（gpt-oss）と人狼相談（gemma系）の
        思考が num_predict をすべて消費し、本文が空のまま done_reason=length になった。
        従来は即座にそのモデルの思考行動を off にしていたため推論の自由を削ったので、
        ここではまず別枠の thinking 余白を拡大して本文を確保する（送信上限は
        `reserve_body_budget` が吊り上げる）。拡大を尽くしても同じ行動で空が続く場合のみ、
        安全側として思考 off へ切り替える。
        """
        capabilities = self.state.model_capabilities.setdefault(actor.model, {})
        if capabilities.get("thinking_budget") is False:
            # 既に off 判定済み（毎ターン同じ記録を増やさない）。
            return
        budget_key = normalize_action_budget_key(action)
        profile = self._profile_for(actor, budget_key)
        sent = metadata.get("num_predict")
        allowance, widened = profile.widen_allowance(
            num_predict_sent=(
                int(sent)
                if isinstance(sent, int) and not isinstance(sent, bool)
                else None
            ),
            max_allowance=min(
                MAX_THINKING_ALLOWANCE_TOKENS,
                BODY_RESERVATION_MAX_NUM_PREDICT.get(
                    budget_key, MAX_THINKING_ALLOWANCE_TOKENS
                ),
            ),
        )
        keep_thinking = widened or profile.exhaustion_count < THINKING_OFF_AFTER_EXHAUSTIONS
        self._emit(
            "thinking_budget_exhausted",
            action,
            actor,
            started_at,
            reason=THINKING_BUDGET_EXHAUSTED_REASON,
            eval_count=metadata.get("eval_count"),
            num_predict=metadata.get("num_predict"),
            content_length=metadata.get("content_length"),
            thinking_off_switched=not keep_thinking,
            thinking_kept=keep_thinking,
            thinking_allowance_tokens=allowance,
            thinking_allowance_widened=bool(widened),
            thinking_exhaustion_count=profile.exhaustion_count,
            body_min_tokens=BODY_MIN_TOKENS_BY_ACTION.get(budget_key),
        )
        if keep_thinking:
            return
        capabilities["thinking_budget"] = False

    def _record_tool_unsupported(
        self, actor: Player, action: str, started_at: float
    ) -> None:
        capabilities = self.state.model_capabilities.setdefault(actor.model, {})
        first_detection = capabilities.get("claim_role_tool") is not False
        capabilities["claim_role_tool"] = False
        if first_detection:
            self._emit(
                "claim_tool_unavailable",
                action,
                actor,
                started_at,
                reason="model_does_not_support_tools",
            )

    def _handle_public_tool_call(
        self,
        actor: Player,
        call: ToolCall,
        action: str,
        started_at: float,
    ) -> dict[str, str]:
        if call.name != "claim_role":
            return {
                "status": "rejected",
                "message": "This public tool is not available.",
            }
        if action != "discussion" or not actor.is_alive:
            self._emit(
                "claim_role_rejected",
                action,
                actor,
                started_at,
                rejection_reason=(
                    "dead_player" if not actor.is_alive else "invalid_phase"
                ),
            )
            return {
                "status": "rejected",
                "message": "Claims cannot be published in the current phase.",
            }

        def reject(message: str, reason: str) -> dict[str, str]:
            self._emit(
                "claim_role_rejected",
                action,
                actor,
                started_at,
                rejection_reason=reason,
            )
            return {"status": "rejected", "message": message}

        args = call.arguments
        role = args.get("role")
        if not isinstance(role, str) or role not in {item.value for item in Role}:
            return reject("Invalid role value.", "invalid_role")
        # TASK-055実装案2: allow_bluff_role_claim=False の設定では、保有していない役職のCOを
        # 受理しない。既定（許可）ではブラフCOも自由な公開主張として通すため、この分岐は入らない。
        if (
            not getattr(self.state, "allow_bluff_role_claim", True)
            and role != actor.role.value
        ):
            self.state.role_provenance.record_disallowed_claim(
                day=self.state.day,
                claimant=actor.name,
                claimed_role=role,
                source=SOURCE_PUBLIC_CLAIM,
                discussion_turn=self.state.discussion_turn,
            )
            self._emit(
                "role_claim_disallowed",
                action,
                actor,
                started_at,
                body_preserved=True,
                role_line_dropped=True,
                interpretation=PUBLIC_LINE_LABEL,
            )
            return reject(
                "Only the true holder of a role may publish that claim under this rule. "
                "Your discussion body stays yours.",
                "bluff_role_claim_disallowed",
            )
        target = args.get("target")
        result = args.get("result")
        reason = args.get("reason")
        vote_proposal = args.get("vote_proposal")
        for field_name, value, max_length in (
            ("target", target, 128),
            ("reason", reason, 500),
            ("vote_proposal", vote_proposal, 300),
        ):
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > max_length
            ):
                return reject(
                    f"Invalid {field_name} value.",
                    f"invalid_{field_name}",
                )
        if result is not None and (
            not isinstance(result, str) or result not in {"wolf", "human"}
        ):
            return reject("Invalid result value.", "invalid_result")
        if role == Role.SEER.value:
            if (target is None) != (result is None):
                return reject(
                    "Both target and result are required for a divination result.",
                    "incomplete_seer_result",
                )
            if target is not None and target not in {
                player.name for player in self.state.players
            }:
                return reject(
                    "That target player does not exist.",
                    "unknown_target",
                )
        elif target is not None or result is not None:
            return reject(
                "Divination results can only be published with a Seer claim.",
                "non_seer_result",
            )

        existing_claims = [
            claim
            for claim in self.state.public_role_claims
            if claim.claimant == actor.name
        ]
        previous_seer_claims = [
            claim
            for claim in self.state.seer_claims
            if claim.claimant == actor.name
        ]
        prior_role = next(
            (
                claim.claimed_role
                for claim in existing_claims
            ),
            Role.SEER.value if previous_seer_claims else None,
        )
        if prior_role is not None and prior_role != role:
            return reject(
                "This contradicts your previous role claim.",
                "inconsistent_role_claim",
            )
        previous_result = next(
            (
                claim.result
                for claim in [*existing_claims, *previous_seer_claims]
                if role == Role.SEER.value
                and claim.target == target
                and claim.result is not None
            ),
            None,
        )
        if previous_result is not None and previous_result != result:
            return reject(
                "This contradicts your previously published divination result.",
                "inconsistent_seer_result",
            )
        duplicate = next(
            (
                claim
                for claim in existing_claims
                if claim.claimed_role == role
                and claim.target == target
                and claim.result == result
            ),
            None,
        )
        if duplicate is not None or (
            role == Role.SEER.value
            and target is not None
            and any(
                claim.target == target and claim.result == result
                for claim in previous_seer_claims
            )
        ):
            self._emit(
                "claim_role_rejected",
                action,
                actor,
                started_at,
                rejection_reason="duplicate_claim",
            )
            return {
                "status": "already_published",
                "message": "The same public claim is already recorded.",
            }

        # TASK-055: 先に占い結果の公開IDを決め、CO行には「本人の公開主張」であることを示す
        # IDだけを付ける。エンジンが発行した実際の判定（RoleEvidence）との対応は
        # role_provenance 台帳（管理者ログ専用）だけが持つ。
        seer_refs: list[str] = []
        if role == Role.SEER.value:
            seer_index_before = len(self.state.seer_claims)
            if not previous_seer_claims:
                self.state.seer_claims.append(
                    PublicSeerClaim(day=self.state.day, claimant=actor.name)
                )
            if target is not None and result is not None:
                self.state.seer_claims.append(
                    PublicSeerClaim(
                        day=self.state.day,
                        claimant=actor.name,
                        target=target,
                        result=result,
                    )
                )
            seer_refs = [
                f"[seer_claim:{index}]"
                for index in range(seer_index_before + 1, len(self.state.seer_claims) + 1)
            ]
        claim_ref = f"[claim:{len(self.state.public_role_claims) + 1}]"
        self.state.public_role_claims.append(
            PublicRoleClaim(
                day=self.state.day,
                claimant=actor.name,
                claimed_role=role,
                target=target,
                result=result,
                reason=reason.strip() if isinstance(reason, str) else None,
                vote_proposal=(
                    vote_proposal.strip()
                    if isinstance(vote_proposal, str)
                    else None
                ),
                source=SOURCE_PUBLIC_CLAIM,
                evidence_ids=tuple([claim_ref, *seer_refs]),
            )
        )
        self.state.role_provenance.record_claim(
            claim_ref=claim_ref,
            day=self.state.day,
            claimant=actor.name,
            claimed_role=role,
            source=SOURCE_PUBLIC_CLAIM,
            actor_role=actor.role.value,
            target=target,
            result=result,
            discussion_turn=self.state.discussion_turn,
        )
        self._emit(
            "claim_role_published",
            action,
            actor,
            started_at,
            claimed_role=role,
            target=target,
            claimed_result=result,
            reason=reason,
            vote_proposal=vote_proposal,
            source="tool",
            claim_ref=claim_ref,
            interpretation=PUBLIC_LINE_LABEL,
        )
        return {
            "status": "published",
            "message": "Your public claim has been recorded.",
        }

    def _chat(
        self,
        actor: Player,
        messages: list[dict[str, object]],
        increase_generation: bool = False,
        tools: list[dict[str, object]] | None = None,
        action: str | None = None,
        started_at: float | None = None,
        request_number: int = 1,
        request_reason: str = "initial",
        tool_based: bool = False,
        budget_key: str = "discussion",
        disable_thinking: bool = False,
        retry_budget_fit: bool = False,
    ) -> ChatResponse:
        options, num_predict_source, thinking_enabled, thinking_source = self._options_for(
            actor, budget_key, disable_thinking=disable_thinking
        )
        if increase_generation:
            current_limit = options.get("num_predict", 512) if options else 512
            if isinstance(current_limit, int) and not isinstance(current_limit, bool):
                options = dict(options or {})
                options["num_predict"] = max(current_limit, min(current_limit * 2, 8192))
                num_predict_source = "retry_generation_boost"
        override_player_num_ctx = bool(
            self.options and self.options.get("override_player_num_ctx")
        )
        # TASK-046: env・Modelfile・/api/ps・設定から実効コンテキスト予算の採用値を
        # 単一値として解決（モデル別キャッシュ・初回のみeffective_context_resolutionイベント記録）。
        context_resolution = self._resolve_context_budget(actor, action, started_at)
        # TASK-036: 動的コンテキスト。過大割り当てを実測サイズへ縮小し、不足時は
        # max_num_ctxまで拡張する。タイムアウト・漏洩の再要求（retry_budget_fit）では
        # 既に要求サイズへ絞るため、ここでは適用しない。CLIの明示上書きでは尊重する。
        num_ctx_before_dynamic_fit = options.get("num_ctx") if options else None
        dynamic_context_applied = False
        context_needed_tokens: int | None = None
        if (
            self._dynamic_context_enabled()
            and not override_player_num_ctx
            and not retry_budget_fit
        ):
            options, context_needed_tokens, dynamic_context_applied = (
                self._fit_request_context(
                    options or {},
                    messages,
                    actor,
                    thinking=bool(thinking_enabled),
                    # TASK-053: モデル×アクション別の実測thinking余白だけで確保する。
                    thinking_margin=self._thinking_margin_for(
                        actor, budget_key, bool(thinking_enabled)
                    ),
                )
            )
        num_ctx_before_fit = options.get("num_ctx") if options else None
        retry_num_predict_applied = False
        retry_num_ctx_applied = False
        if retry_budget_fit:
            options, num_predict_source, retry_num_predict_applied, retry_num_ctx_applied = (
                self._fit_retry_budget(
                    options or {},
                    num_predict_source,
                    messages,
                    respect_explicit_num_ctx=override_player_num_ctx,
                    effective_limit=context_resolution.limit,
                )
            )
        original_timeout = getattr(self.client, "timeout", None)
        effective_timeout = (
            actor.timeout
            if actor.timeout is not None
            else original_timeout
        )
        if original_timeout is not None and actor.timeout is not None:
            self.client.timeout = actor.timeout
        max_retries = getattr(self.client, "max_retries", 2)
        timeout_retries = getattr(self.client, "timeout_retries", max_retries)
        context_diagnostics: dict[str, object] = {}
        if callable(getattr(self.client, "context_diagnostics", None)):
            context_diagnostics = self.client.context_diagnostics(actor.model)
        requested_num_ctx = options.get("num_ctx") if options else None
        if actor.num_ctx is not None and not override_player_num_ctx:
            num_ctx_source = (
                "player_setting_capped"
                if actor.max_num_ctx is not None
                and actor.num_ctx > actor.max_num_ctx
                else "player_setting"
            )
        elif (
            actor.recommended_num_ctx is not None
            and (self.options is None or self.options.get("num_ctx") is None)
        ):
            num_ctx_source = "player_recommended_setting"
        elif (
            actor.max_num_ctx is not None
            and actor.num_ctx is None
            and (self.options is None or self.options.get("num_ctx") is None)
        ):
            num_ctx_source = "player_maximum_setting"
        elif self.options is not None and self.options.get("num_ctx") is not None:
            num_ctx_source = (
                "cli_setting_capped"
                if actor.max_num_ctx is not None
                and isinstance(self.options.get("num_ctx"), int)
                and self.options["num_ctx"] > actor.max_num_ctx
                else "cli_setting"
            )
        elif requested_num_ctx is not None:
            num_ctx_source = "api_request"
        else:
            num_ctx_source = "ollama_service_default_or_environment"
        setting_num_ctx_source = num_ctx_source
        if dynamic_context_applied:
            num_ctx_source = DYNAMIC_CONTEXT_SOURCE
        if retry_num_ctx_applied:
            num_ctx_source = RETRY_NUM_CTX_SOURCE

        def emit_request_configuration(
            request_options: dict[str, object],
            thinking_value: bool | None,
            thinking_label: str,
            reason: str,
        ) -> None:
            """実際に送信するリクエスト設定をログ・CLIへ反映する。"""
            if action is None or started_at is None:
                return
            self._emit(
                "llm_request_configuration",
                action,
                actor,
                started_at,
                request_number=request_number,
                request_reason=reason,
                tool_based=tool_based,
                request_options=request_options,
                request_timeout_seconds=effective_timeout,
                num_ctx_source=num_ctx_source,
                requested_num_ctx=request_options.get("num_ctx"),
                timeout_retries=timeout_retries,
                retry_budget_fit=bool(retry_num_predict_applied or retry_num_ctx_applied),
                num_ctx_before_retry_fit=num_ctx_before_fit if retry_num_ctx_applied else None,
                dynamic_context_fit=bool(dynamic_context_applied),
                num_ctx_before_dynamic_fit=(
                    num_ctx_before_dynamic_fit if dynamic_context_applied else None
                ),
                num_ctx_setting_source=setting_num_ctx_source,
                context_needed_tokens=context_needed_tokens,
                system_prompt_char_count=sum(
                    len(str(message.get("content", "")))
                    for message in messages
                    if message.get("role") == "system"
                ),
                user_prompt_char_count=sum(
                    len(str(message.get("content", "")))
                    for message in messages
                    if message.get("role") == "user"
                ),
                raw_prompt_char_count=sum(
                    len(str(message.get("content", ""))) for message in messages
                ),
                num_predict=request_options.get("num_predict"),
                num_predict_source=num_predict_source,
                action_budget_key=normalize_action_budget_key(budget_key),
                thinking_enabled=thinking_value,
                thinking_source=thinking_label,
                max_retries=max_retries,
                context_diagnostics=context_diagnostics,
                # TASK-046: 実効コンテキスト上限の単一値解決と、入力/出力/thinking/余白の
                # 予算構成分解をリクエストごとに記録する（何を落としたかの検証基盤）。
                effective_context_limit=context_resolution.limit,
                context_budget_source=context_resolution.source,
                context_limit_conflict=context_resolution.conflict,
                # 由来別名称（TASK-046レビュー指摘）: 設定由来 / サーバー観測値 / 実際に送る値。
                # effective_requested_num_ctx は fit・再要求fit適用後の送信num_ctx（requested_num_ctxと同値）。
                configured_context_limit=context_resolution.configured,
                server_observed_context_limit=context_resolution.server_observed,
                effective_requested_num_ctx=request_options.get("num_ctx"),
                context_budget_breakdown=context_budget_breakdown(
                    messages,
                    request_options.get("num_predict"),
                    thinking_enabled=bool(thinking_value),
                    effective_limit=context_resolution.limit,
                    # TASK-053: モデル別実測のthinking余白を予算構成分解へ反映する。
                    thinking_margin=self._thinking_margin_for(
                        actor, budget_key, bool(thinking_value)
                    ),
                ),
                thinking_margin_tokens=self._thinking_margin_for(
                    actor, budget_key, bool(thinking_value)
                ),
                # TASK-053: 実際に送る生成上限が「本文最低予算 + thinking余白」を保てているかの
                # 検査（モデル×アクション別の空応答率・thinking消費率の比較基盤）。
                body_budget=inspect_body_budget(
                    self._thinking_profiles.get(
                        actor.model, normalize_action_budget_key(budget_key)
                    ),
                    normalize_action_budget_key(budget_key),
                    request_options.get("num_predict"),
                    thinking_enabled=bool(thinking_value),
                ),
            )

        emit_request_configuration(
            options or {}, thinking_enabled, thinking_source, request_reason
        )

        def send(request_options: dict[str, object] | None) -> ChatResponse:
            if hasattr(self.client, "_last_chat_response"):
                delattr(self.client, "_last_chat_response")
            if tools is not None and callable(
                getattr(self.client, "chat_with_metadata", None)
            ):
                response = self.client.chat_with_metadata(
                    actor.model,
                    messages,
                    options=request_options,
                    tools=tools,
                )
            elif request_options is not None:
                content = self._normalize_response_content(
                    self.client.chat(actor.model, messages, options=request_options)
                )
                metadata = getattr(self.client, "_last_chat_response", None)
                if isinstance(metadata, ChatResponse):
                    response = metadata
                else:
                    response = ChatResponse(
                        content=content,
                        attempts=max(1, getattr(self.client, "_last_attempts", 1)),
                        content_length=len(content),
                        thinking_present=self._has_thinking(content),
                    )
            else:
                content = self._normalize_response_content(self.client.chat(actor.model, messages))
                metadata = getattr(self.client, "_last_chat_response", None)
                if isinstance(metadata, ChatResponse):
                    response = metadata
                else:
                    response = ChatResponse(
                        content=content,
                        attempts=max(1, getattr(self.client, "_last_attempts", 1)),
                        content_length=len(content),
                        thinking_present=self._has_thinking(content),
                    )
            after_diagnostics: dict[str, object] = {}
            if callable(getattr(self.client, "context_diagnostics", None)):
                after_diagnostics = self.client.context_diagnostics(actor.model)
            effective_context = after_diagnostics.get("ollama_ps_context")
            return replace(
                response,
                requested_num_ctx=(
                    request_options.get("num_ctx")
                    if request_options
                    and isinstance(request_options.get("num_ctx"), int)
                    and not isinstance(request_options.get("num_ctx"), bool)
                    else None
                ),
                effective_num_ctx=(
                    effective_context
                    if isinstance(effective_context, int)
                    and not isinstance(effective_context, bool)
                    else None
                ),
                num_ctx_source=num_ctx_source,
            )

        def send_and_record(request_options: dict[str, object] | None) -> ChatResponse:
            """1送信を計測し、モデル×アクション別の生成プロファイルへ加算する（TASK-053）。"""

            send_started = time.perf_counter()
            try:
                response = send(request_options)
            except OllamaError as error:
                self._record_generation_failure(
                    actor,
                    budget_key,
                    action,
                    started_at,
                    request_options or {},
                    request_number,
                    request_reason,
                    thinking_enabled,
                    send_started,
                    error,
                )
                raise
            return self._record_generation_response(
                actor,
                budget_key,
                action,
                started_at,
                request_options or {},
                request_number,
                request_reason,
                thinking_enabled,
                send_started,
                response,
            )

        try:
            try:
                return send_and_record(options)
            except OllamaError as error:
                if error.kind == "timeout":
                    # TASK-036: タイムアウトをモデル別能力情報へ記録し、以後の議論・
                    # 人狼相談の思考・生成上限・発言予算を動的に絞る。
                    self._record_model_timeout(actor, action, started_at)
                # 思考生成非対応のモデルでは400が返る。同じ要求を再送せず、
                # think=offで1回だけ再送し、能力情報を記録して以後の行動もoffにする。
                if not self._is_thinking_unsupported(error) or not (
                    options and options.get("thinking") is True
                ):
                    raise
                self._record_thinking_unsupported(actor, action, started_at)
                fallback_options = dict(options or {})
                fallback_options["thinking"] = False
                emit_request_configuration(
                    fallback_options,
                    False,
                    THINKING_CAPABILITY_SOURCE,
                    f"{request_reason} (resent with think=off; the model does not support thinking)",
                )
                return send_and_record(fallback_options)
        finally:
            if original_timeout is not None:
                self.client.timeout = original_timeout

    @staticmethod
    def _normalize_response_content(content: object) -> str:
        """クライアント応答本文を文字列へ正規化する（Noneは空文字）。

        Ollamaが本文を返さなかった場合、Noneのまま `len()` や正規表現へ渡すと
        TypeError（"expected string or bytes-like object, got 'NoneType'"）になる。
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return str(content)

    @staticmethod
    def _has_thinking(response: str | None) -> bool:
        if response is None or not isinstance(response, str):
            return False
        return bool(_THINK_BLOCK.search(response) or _UNCLOSED_THINK_BLOCK.search(response))

    @staticmethod
    def _is_context_overflow(error: OllamaError) -> bool:
        message = error.message.casefold()
        return any(
            marker in message
            for marker in (
                "context length",
                "context window",
                "maximum context",
                "prompt is too long",
                "input length",
                "num_ctx",
                "コンテキスト長",
                "コンテキストが長",
            )
        )

    @staticmethod
    def _transport_failure_reason(error: OllamaError) -> str:
        return {
            "timeout": "timeout",
            "connection": "communication_failure",
            "http": "communication_failure",
            "invalid_response": "invalid_ollama_response",
        }.get(error.kind, "communication_failure")

    def _start_action(self, action: str, actor: Player) -> float:
        started_at = time.perf_counter()
        self._emit(
            "llm_action_started",
            action,
            actor,
            started_at,
        )
        return started_at

    def _complete_action(
        self,
        action: ActionKind | str,
        actor: Player,
        started_at: float,
        semantic_retries: int,
        transport_attempts: int,
        fallback_used: bool,
        selected_target: str | None = None,
        content: str | None = None,
        error: OllamaError | None = None,
        failure_reason: str | None = None,
        failure_reasons: tuple[str, ...] = (),
        response_metadata: dict[str, str | int | bool | None] | None = None,
        content_preserved: bool = False,
        recovered_content: bool = False,
        repair_applied: bool = False,
        fatal_safety_fallback: bool = False,
        validation_severity: str = "ok",
    ) -> None:
        self._emit(
            "llm_action_completed",
            action,
            actor,
            started_at,
            elapsed_seconds=round(time.perf_counter() - started_at, 3),
            semantic_retries=semantic_retries,
            transport_retries=max(0, transport_attempts - (semantic_retries + 1)),
            total_retries=semantic_retries
            + max(0, transport_attempts - (semantic_retries + 1)),
            fallback_used=fallback_used,
            content_preserved=content_preserved,
            recovered_content=recovered_content,
            repair_applied=repair_applied,
            fatal_safety_fallback=fatal_safety_fallback,
            validation_severity=validation_severity,
            selected_target=selected_target,
            content=content,
            error=error.to_dict() if error is not None else None,
            failure_reason=failure_reason,
            failure_reasons=failure_reasons,
            failure_classes=tuple(
                dict.fromkeys(
                    failure_class
                    for failure_reason in failure_reasons
                    if (failure_class := classify_failure_reason(failure_reason))
                    is not None
                )
            ),
            input_token_count=(
                response_metadata.get("prompt_eval_count")
                if response_metadata is not None
                else None
            ),
            output_token_count=(
                response_metadata.get("eval_count")
                if response_metadata is not None
                else None
            ),
            done_reason=(
                response_metadata.get("done_reason")
                if response_metadata is not None
                else None
            ),
            requested_num_ctx=(
                response_metadata.get("requested_num_ctx")
                if response_metadata is not None
                else None
            ),
            effective_num_ctx=(
                response_metadata.get("effective_num_ctx")
                if response_metadata is not None
                else None
            ),
            num_ctx_source=(
                response_metadata.get("num_ctx_source")
                if response_metadata is not None
                else None
            ),
            response_metadata=response_metadata,
        )

    def _emit(
        self,
        event_type: str,
        action: ActionKind | str,
        actor: Player,
        started_at: float,
        **details: object,
    ) -> None:
        if self.event_callback is None:
            return
        action_name = action.value if isinstance(action, ActionKind) else action
        if event_type == "llm_action_parse_failed" and "failure_class" not in details:
            reason = details.get("reason")
            details["failure_class"] = (
                classify_failure_reason(reason)
                if isinstance(reason, str)
                else None
            )
        phase_labels = {
            "night_wolf": "夜・人狼相談",
            "night_seer": "夜・占い",
            "day_discussion": "昼・議論",
            "voting": "投票",
        }
        self.event_callback(
            {
                "type": event_type,
                "day": self.state.day,
                "phase": self.state.phase.value,
                "phase_label": phase_labels.get(
                    self.state.phase.value, self.state.phase.value
                ),
                "player": actor.name,
                "model": actor.model,
                "action": action_name,
                "discussion_claim_mode": self.discussion_claim_mode,
                "private": action_name
                in {"wolf_discussion", ActionKind.ATTACK.value, ActionKind.SEER.value},
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                **details,
            }
        )

    @staticmethod
    def _label(action: ActionKind) -> str:
        return {
            ActionKind.ATTACK: "Attack target",
            ActionKind.SEER: "Seer target",
            ActionKind.VOTE: "Vote",
        }[action]

    def _player(self, name: str) -> Player:
        return next(player for player in self.state.players if player.name == name)
