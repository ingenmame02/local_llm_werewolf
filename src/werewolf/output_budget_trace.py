"""TASK-063: 出力予算・thinking・timeoutを「プロンプト論理の失敗」と分離するトレース。

実機（game_* で Player 1 を `llm-jp-4.1-33b-thinking-fixed` に変更した試合）では、
17アクション中5件の失敗が thinking-only 4件と timeout 1件に分かれた。強化モデルの採用
自体は妥当でも、モデル固有の thinking 挙動と出力予算が、論理状態の保持とは別の障害に
なっており、両方を同じ指標（回答率・仮説矛盾率）で見ると原因を取り違える。

本モジュールはLLMを呼ばない純粋な分類・集計だけを持つ。口調・論点・推理順には干渉せず、
thinking を一律に無効化することもしない（失敗の切り分けと予算の上限値の提案のみ）。

分類カテゴリ（相互排他的な「最初の不足原因」）:
- `ok`: 本文ありで正常生成
- `thinking_budget`: 思考が生成上限を食い潰して本文が空（done_reason=length／thinking本文のみ）
- `generation_cap`: 思考以外の理由で打ち切られ本文が不完全
- `timeout`: 送信タイムアウト（応答なし）
- `transport`: 接続・HTTPなどの通信失敗
- `output_format`: 本文は出たが指示形式を満たさない（CO/BODY/RESULT の形など）
- `prompt_logic`: 内容面の問題（未回答・根拠欠落・既出論点の再掲など＝議論の質）

`prompt_logic` は今回の分類では自動判定せず、呼び出し側（llm_actions）が形式チェックや
質問応答チェックの結果を受け取って記録する。これにより「モデルが考えられていないのか、
考えているが論理が弱いのか」をログで分離できる。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

# ---------------------------------------------------------------------------
# カテゴリ（文字列はログ・CLIでそのまま使う。命名は安定させる）
# ---------------------------------------------------------------------------

CATEGORY_OK = "ok"
CATEGORY_THINKING_BUDGET = "thinking_budget"
CATEGORY_GENERATION_CAP = "generation_cap"
CATEGORY_TIMEOUT = "timeout"
CATEGORY_TRANSPORT = "transport"
CATEGORY_OUTPUT_FORMAT = "output_format"
CATEGORY_PROMPT_LOGIC = "prompt_logic"

FAILURE_CATEGORIES: tuple[str, ...] = (
    CATEGORY_THINKING_BUDGET,
    CATEGORY_GENERATION_CAP,
    CATEGORY_TIMEOUT,
    CATEGORY_TRANSPORT,
    CATEGORY_OUTPUT_FORMAT,
    CATEGORY_PROMPT_LOGIC,
)
ALL_CATEGORIES: tuple[str, ...] = (CATEGORY_OK,) + FAILURE_CATEGORIES

# 「モデルの出力予算・通信」側か「プロンプト論理・形式」側かの切り分け（評価用ラベル）
BUDGET_BOUND_CATEGORIES: frozenset[str] = frozenset(
    {CATEGORY_THINKING_BUDGET, CATEGORY_GENERATION_CAP, CATEGORY_TIMEOUT, CATEGORY_TRANSPORT}
)
LOGIC_BOUND_CATEGORIES: frozenset[str] = frozenset(
    {CATEGORY_OUTPUT_FORMAT, CATEGORY_PROMPT_LOGIC}
)

RETRY_BODY_PRESERVED = "preserved"
RETRY_BODY_LOST = "lost"
RETRY_BODY_NOT_APPLICABLE = "not_applicable"

# ---------------------------------------------------------------------------
# thinking-fixed系モデルの行動別予算（TASK-063）
# ---------------------------------------------------------------------------
# 名前に "thinking" を含み、かつ "fixed" を含むモデル（例: llm-jp-4.1-33b-thinking-fixed）は
# 「thinkingを出すことが確定している」部類で、思考量がほぼ固定分だけ発生する。汎用の
# 適応余白（TASK-053）だと本文が压迫されるため、行動別の「本文最低量」と「thinking上限」を
# 別表で持つ。ここで決めるのはトークン予算だけで、推理の手順・論点・口調は決めない。
THINKING_FIXED_NAME_TOKENS: tuple[str, ...] = ("thinking", "fixed")


def is_thinking_fixed_model(model: object) -> bool:
    """モデル名が thinking-fixed 系かどうか（小文字化して部分文字列で判定する）。"""
    name = str(model or "").lower()
    if not name:
        return False
    compact = name.replace("_", "-")
    return all(token in compact for token in THINKING_FIXED_NAME_TOKENS)


# 行動別の本文最低トークン（強化モデルでは議論本文を先に確保する）。TASK-053の
# BODY_MIN_TOKENS_BY_ACTION より議論・CO補正を厚くし、短い行動（投票・占い）は据え置く。
FIXED_MODEL_BODY_MIN_TOKENS_BY_ACTION: dict[str, int] = {
    "discussion": 832,
    "wolf_discussion": 512,
    "co_correction": 320,
    "vote": 192,
    "attack": 160,
    "seer": 160,
}
# 行動別の thinking 上限トークン。これを超えて思考へ回さない（=本文が空になるのを防ぐ）。
# 「thinking を切る」設定ではなく、同じ生成上限の中で思考に割り当てられる上限である。
FIXED_MODEL_THINKING_CAP_TOKENS_BY_ACTION: dict[str, int] = {
    "discussion": 2048,
    "wolf_discussion": 1280,
    "co_correction": 768,
    "vote": 640,
    "attack": 640,
    "seer": 640,
}


def fixed_model_caps(model: object, budget_key: str) -> dict[str, int] | None:
    """thinking-fixed系モデルなら行動別（本文最低量・thinking上限）を返す。"""
    if not is_thinking_fixed_model(model):
        return None
    key = str(budget_key or "discussion")
    body_min = FIXED_MODEL_BODY_MIN_TOKENS_BY_ACTION.get(key)
    cap = FIXED_MODEL_THINKING_CAP_TOKENS_BY_ACTION.get(key)
    if body_min is None or cap is None:
        return None
    return {"body_min_tokens": int(body_min), "thinking_cap_tokens": int(cap)}


# ---------------------------------------------------------------------------
# 1送信の分類
# ---------------------------------------------------------------------------
def _int_or_none(value: object) -> int | None:
    return None if isinstance(value, bool) or not isinstance(value, int) else int(value)


# llm_actions の応答問題名を、出力形式側とプロンプト論理側へ写すための対応表。
# 「本文は出ているが形が違う」なら output_format、「内容は出るが議論として弱い」なら
# prompt_logic に分ける（thinking不足・timeoutとは別枠で数える目的）。
FORMAT_ISSUE_NAMES: frozenset[str] = frozenset(
    {
        "discussion_output_format",
        "seer_result_without_claim",
        "wolf_result_not_disclosed",
        "inconsistent_seer_result",
        "inconsistent_role_claim",
        "template_placeholder_left",
        "marker_missing",
    }
)
LOGIC_ISSUE_NAMES: frozenset[str] = frozenset(
    {
        "question_not_answered_first",
        "past_action_question_unanswered",
        "repeated_discussion_topic",
        "low_novelty",
        "duplicate_claim",
        "unsupported_reason",
    }
)


def issue_category(issue: object) -> str | None:
    """応答問題名を output_format / prompt_logic のいずれかへ分類する（不明なら None）。"""
    name = str(issue or "")
    if not name:
        return None
    if name in FORMAT_ISSUE_NAMES:
        return CATEGORY_OUTPUT_FORMAT
    if name in LOGIC_ISSUE_NAMES:
        return CATEGORY_PROMPT_LOGIC
    return None


def classify_attempt(
    *,
    thinking_only: bool = False,
    timed_out: bool = False,
    transport_error: bool = False,
    body_char_count: int = 0,
    thinking_present: bool = False,
    done_reason: str | None = None,
    eval_count: object = None,
    num_predict: object = None,
    format_issue: str | None = None,
    logic_issue: str | None = None,
) -> str:
    """1送信の結果を単一カテゴリへ分類する（優先順位は通信→timeout→生成量→形式→論理）。

    - 本文があり、形式・論理の問題も無いなら `ok`。
    - 本文が空で thinking がある（または done_reason=length）なら `thinking_budget`。
    - 本文が空で思考証拠が無い場合は生成上限 (`generation_cap`) か timeout/transport へ寄せる。
    - 本文ありでも形式崩れなら `output_format`（内容は出ている＝思考は機能している）。
    """
    if transport_error and not timed_out:
        return CATEGORY_TRANSPORT
    if timed_out:
        return CATEGORY_TIMEOUT
    body_chars = int(body_char_count or 0)
    if body_chars <= 0:
        if thinking_only or thinking_present:
            return CATEGORY_THINKING_BUDGET
        if str(done_reason or "") == "length":
            predicted = _int_or_none(num_predict)
            generated = _int_or_none(eval_count)
            # 生成上限に達して空（思考証拠なし）→ 上限そのものが足りない
            if predicted is not None and generated is not None and generated >= predicted:
                return CATEGORY_GENERATION_CAP
            return CATEGORY_THINKING_BUDGET
        return CATEGORY_GENERATION_CAP
    if format_issue:
        return CATEGORY_OUTPUT_FORMAT
    if logic_issue:
        return CATEGORY_PROMPT_LOGIC
    if str(done_reason or "") == "length":
        return CATEGORY_GENERATION_CAP
    return CATEGORY_OK


@dataclass(frozen=True)
class AttemptRow:
    """送信1件のトレース行（モデル・行動・入力推定・thinking・生成結果）。"""

    model: str
    budget_key: str
    action: str
    request_number: int
    category: str
    body_char_count: int = 0
    thinking_requested: bool | None = None
    thinking_present: bool = False
    done_reason: str | None = None
    eval_count: int | None = None
    num_predict: int | None = None
    prompt_char_count: int | None = None
    est_input_tokens: int | None = None
    num_ctx: int | None = None
    elapsed_seconds: float = 0.0
    short_retry: bool = False
    body_min_tokens: int | None = None
    thinking_cap_tokens: int | None = None

    def serialize(self) -> dict[str, object]:
        data: dict[str, object] = {
            "model": self.model,
            "action_budget_key": self.budget_key,
            "action": self.action,
            "request_number": int(self.request_number),
            "category": self.category,
            "budget_bound": self.category in BUDGET_BOUND_CATEGORIES,
            "logic_bound": self.category in LOGIC_BOUND_CATEGORIES,
            "body_char_count": int(self.body_char_count),
            "thinking_present": bool(self.thinking_present),
            "elapsed_seconds": round(float(self.elapsed_seconds), 3),
        }
        for key in (
            "thinking_requested",
            "done_reason",
            "eval_count",
            "num_predict",
            "prompt_char_count",
            "est_input_tokens",
            "num_ctx",
            "body_min_tokens",
            "thinking_cap_tokens",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.short_retry:
            data["short_retry"] = True
        return data


@dataclass
class ActionOutcome:
    """1アクションの最終結果（再要求で本文を失ったか、元の本文を保持できたか）。"""

    model: str
    budget_key: str
    action: str
    attempts: int = 0
    categories: tuple[str, ...] = ()
    final_body_char_count: int = 0
    short_retry_used: bool = False
    # 初回応答に本文があったのに再要求で書き直したなら preserved、初回が thinking-only 等で
    # 本文が無く後続で出したなら lost（最初の思考は応答として残っていない）
    body_preservation: str = RETRY_BODY_NOT_APPLICABLE

    def serialize(self) -> dict[str, object]:
        counts: dict[str, int] = {}
        for category in self.categories:
            counts[category] = counts.get(category, 0) + 1
        return {
            "model": self.model,
            "action_budget_key": self.budget_key,
            "action": self.action,
            "attempts": int(self.attempts),
            "attempt_categories": list(self.categories),
            "failure_categories": [c for c in self.categories if c != CATEGORY_OK],
            "final_body_char_count": int(self.final_body_char_count),
            "body_generated": bool(self.final_body_char_count),
            "short_retry_used": bool(self.short_retry_used),
            "body_preservation": self.body_preservation,
            "category_counts": counts,
        }


def _row_matches(row: AttemptRow, key: tuple[str, str, str]) -> bool:
    return row.model == key[0] and row.budget_key == key[1] and row.action == key[2]


@dataclass
class OutputBudgetTrace:
    """モデル×アクション別の出力予算トレース（送信行と行動単位の結果を集計する）。

    目的は失敗理由を ①出力予算・thinking・timeout/通信、②出力形式、③プロンプト論理
    （議論の質）へ分離すること。口調・論点・推理順には干渉せず、thinkingを一律に
    無効化することもしない（予算の上限提案と記録のみ）。
    """

    rows: list[AttemptRow] = field(default_factory=list)
    outcomes: dict[tuple[str, str, str], ActionOutcome] = field(default_factory=dict)

    def record_attempt(
        self,
        *,
        model: str,
        budget_key: str,
        action: str,
        request_number: int,
        category: str | None = None,
        body_char_count: int = 0,
        thinking_requested: bool | None = None,
        thinking_present: bool = False,
        timed_out: bool = False,
        transport_error: bool = False,
        thinking_only: bool = False,
        done_reason: str | None = None,
        eval_count: object = None,
        num_predict: object = None,
        prompt_char_count: object = None,
        est_input_tokens: object = None,
        num_ctx: object = None,
        elapsed_seconds: float = 0.0,
        short_retry: bool = False,
        format_issue: str | None = None,
        logic_issue: str | None = None,
    ) -> AttemptRow:
        """送信1件を単一カテゴリへ分類して記録する（category未指定なら自動判定）。"""
        resolved = category or classify_attempt(
            thinking_only=thinking_only,
            timed_out=timed_out,
            transport_error=transport_error,
            body_char_count=body_char_count,
            thinking_present=thinking_present,
            done_reason=done_reason,
            eval_count=eval_count,
            num_predict=num_predict,
            format_issue=format_issue,
            logic_issue=logic_issue,
        )
        caps = fixed_model_caps(model, budget_key)
        row = AttemptRow(
            model=str(model),
            budget_key=str(budget_key),
            action=str(action),
            request_number=max(1, int(request_number or 1)),
            category=str(resolved),
            body_char_count=int(body_char_count or 0),
            thinking_requested=thinking_requested,
            thinking_present=bool(thinking_present),
            done_reason=done_reason,
            eval_count=_int_or_none(eval_count),
            num_predict=_int_or_none(num_predict),
            prompt_char_count=_int_or_none(prompt_char_count),
            est_input_tokens=_int_or_none(est_input_tokens),
            num_ctx=_int_or_none(num_ctx),
            elapsed_seconds=float(elapsed_seconds or 0.0),
            short_retry=bool(short_retry),
            body_min_tokens=(caps or {}).get("body_min_tokens"),
            thinking_cap_tokens=(caps or {}).get("thinking_cap_tokens"),
        )
        self.rows.append(row)
        key = (row.model, row.budget_key, row.action)
        outcome = self.outcomes.get(key)
        if outcome is None:
            outcome = ActionOutcome(model=row.model, budget_key=row.budget_key, action=row.action)
            self.outcomes[key] = outcome
        outcome.attempts += 1
        outcome.categories = (*outcome.categories, row.category)
        if row.short_retry:
            outcome.short_retry_used = True
        # 本文が出た送信があれば、その時点で「失ったか保持したか」を確定させる（追加の
        # フックが無くても行動単位の結果が揃う）。後から本文なし行が来ても結果は据え置く。
        if row.body_char_count > 0:
            self.record_final_body(
                model=row.model,
                budget_key=row.budget_key,
                action=row.action,
                final_body_char_count=max(
                    outcome.final_body_char_count, row.body_char_count
                ),
            )
        return row

    def record_final_body(
        self,
        *,
        model: str,
        budget_key: str,
        action: str,
        final_body_char_count: int,
    ) -> str:
        """アクション成立時の最終本文から、再要求で本文を失ったかを判定して記録する。

        - 初回送信に本文があり、やり直しでも本文がある → `preserved`（形式・論理の修正のみ）
        - 初回が thinking-only / timeout / 空で、後続の再要求で本文を出した → `lost`
          （最初の思考は応答として残らず、本文生成機会を1回消費したことを示す）
        - 再要求が無かった → `not_applicable`
        """
        key = (str(model), str(budget_key), str(action))
        outcome = self.outcomes.get(key)
        if outcome is None:
            outcome = ActionOutcome(model=key[0], budget_key=key[1], action=key[2])
            self.outcomes[key] = outcome
        outcome.final_body_char_count = int(final_body_char_count or 0)
        rows = [row for row in self.rows if _row_matches(row, key)]
        if len(rows) <= 1:
            outcome.body_preservation = RETRY_BODY_NOT_APPLICABLE
        elif not outcome.final_body_char_count:
            outcome.body_preservation = RETRY_BODY_LOST
        elif rows and rows[0].body_char_count > 0 and all(
            row.category == CATEGORY_OK for row in rows[:-1]
        ):
            outcome.body_preservation = RETRY_BODY_PRESERVED
        elif any(row.category != CATEGORY_OK for row in rows[:-1]):
            outcome.body_preservation = (
                RETRY_BODY_PRESERVED if rows and rows[0].body_char_count > 0 else RETRY_BODY_LOST
            )
        else:
            outcome.body_preservation = RETRY_BODY_PRESERVED
        return outcome.body_preservation

    def annotate_last_attempt(
        self,
        *,
        model: str,
        budget_key: str,
        action: str,
        format_issue: str | None = None,
        logic_issue: str | None = None,
    ) -> AttemptRow | None:
        """直近の送信行に「形式・論理」の問題を関連付け、カテゴリを更新する（追記しない）。

        本文は出ていない送信（thinking_budget / timeout など）は上書きしない（不足の原因は
        そちら）。本文がある送信に対してのみ、output_format / prompt_logic へ書き換える。
        """
        key = (str(model), str(budget_key), str(action))
        rows = [row for row in self.rows if _row_matches(row, key)]
        if not rows:
            return None
        row = rows[-1]
        issue = format_issue or logic_issue
        if not issue or row.body_char_count <= 0:
            return None
        category = CATEGORY_OUTPUT_FORMAT if format_issue else CATEGORY_PROMPT_LOGIC
        updated = replace(row, category=category)
        index = self.rows.index(row)
        self.rows[index] = updated
        outcome = self.outcomes.get(key)
        if outcome is not None:
            categories = list(outcome.categories)
            if categories:
                categories[-1] = updated.category
            outcome.categories = tuple(categories)
        return updated

    def model_rows(self, model: str) -> list[AttemptRow]:
        return [row for row in self.rows if row.model == str(model)]

    def metrics(self) -> dict[str, object]:
        """モデル別内訳（本文生成率・カテゴリ別件数・平均入力推定・短縮再要求など）。"""
        per_model: dict[str, dict[str, object]] = {}
        for model in sorted({row.model for row in self.rows}):
            rows = self.model_rows(model)
            counts = {category: 0 for category in ALL_CATEGORIES}
            for row in rows:
                counts[row.category] = counts.get(row.category, 0) + 1
            attempts = len(rows)
            outcomes = [o for o in self.outcomes.values() if o.model == model]
            prompt_chars = [int(r.prompt_char_count) for r in rows if r.prompt_char_count]
            elapsed = sum(float(r.elapsed_seconds) for r in rows)
            per_model[model] = {
                "attempts": int(attempts),
                "actions": len(outcomes),
                "category_counts": counts,
                "body_generation_rate": (
                    round(sum(1 for r in rows if r.body_char_count > 0) / attempts, 4)
                    if attempts
                    else 0.0
                ),
                "budget_bound_failures": sum(
                    1 for r in rows if r.category in BUDGET_BOUND_CATEGORIES
                ),
                "logic_bound_failures": sum(
                    1 for r in rows if r.category in LOGIC_BOUND_CATEGORIES
                ),
                "thinking_only_count": counts[CATEGORY_THINKING_BUDGET],
                "timeout_count": counts[CATEGORY_TIMEOUT],
                "transport_error_count": counts[CATEGORY_TRANSPORT],
                "generation_cap_count": counts[CATEGORY_GENERATION_CAP],
                "output_format_count": counts[CATEGORY_OUTPUT_FORMAT],
                "prompt_logic_count": counts[CATEGORY_PROMPT_LOGIC],
                "short_retry_count": sum(1 for r in rows if r.short_retry),
                "avg_prompt_char_count": (
                    round(sum(prompt_chars) / len(prompt_chars), 1) if prompt_chars else 0.0
                ),
                "max_prompt_char_count": int(max(prompt_chars, default=0)),
                "avg_elapsed_seconds": round(elapsed / attempts, 3) if attempts else 0.0,
                "body_lost_on_retry": sum(
                    1 for o in outcomes if o.body_preservation == RETRY_BODY_LOST
                ),
                "body_preserved_on_retry": sum(
                    1 for o in outcomes if o.body_preservation == RETRY_BODY_PRESERVED
                ),
                "thinking_fixed_model": is_thinking_fixed_model(model),
            }
        totals = {category: 0 for category in ALL_CATEGORIES}
        for row in self.rows:
            totals[row.category] = totals.get(row.category, 0) + 1
        return {
            "output_budget_attempts": len(self.rows),
            "output_budget_actions": len(self.outcomes),
            "output_budget_categories": totals,
            "output_budget_body_lost_total": sum(
                1 for o in self.outcomes.values() if o.body_preservation == RETRY_BODY_LOST
            ),
            "output_budget_body_preserved_total": sum(
                1 for o in self.outcomes.values() if o.body_preservation == RETRY_BODY_PRESERVED
            ),
            "output_budget_by_model": per_model,
        }

    def serialize_public(self) -> dict[str, object]:
        """公開ログ用（モデル別内訳と行動単位の結果）。発言本文は載せない。"""
        return {
            "metrics": self.metrics(),
            "outcomes": [
                outcome.serialize()
                for outcome in sorted(
                    self.outcomes.values(),
                    key=lambda o: (o.model, o.budget_key, o.action),
                )
            ],
        }


__all__ = [
    "ALL_CATEGORIES",
    "ActionOutcome",
    "AttemptRow",
    "BUDGET_BOUND_CATEGORIES",
    "CATEGORY_GENERATION_CAP",
    "CATEGORY_OK",
    "CATEGORY_OUTPUT_FORMAT",
    "CATEGORY_PROMPT_LOGIC",
    "CATEGORY_THINKING_BUDGET",
    "CATEGORY_TIMEOUT",
    "CATEGORY_TRANSPORT",
    "FIXED_MODEL_BODY_MIN_TOKENS_BY_ACTION",
    "FIXED_MODEL_THINKING_CAP_TOKENS_BY_ACTION",
    "LOGIC_BOUND_CATEGORIES",
    "OutputBudgetTrace",
    "RETRY_BODY_LOST",
    "RETRY_BODY_NOT_APPLICABLE",
    "RETRY_BODY_PRESERVED",
    "THINKING_FIXED_NAME_TOKENS",
    "FORMAT_ISSUE_NAMES",
    "LOGIC_ISSUE_NAMES",
    "classify_attempt",
    "fixed_model_caps",
    "is_thinking_fixed_model",
    "issue_category",
]
