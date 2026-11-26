"""TASK-065: thinking・timeout・生成上限の処置を「モデル×行動×直近失敗履歴」へ限定する。

実機ログ `game_425755ce9a83` では、Player 8 の議論で thinking が生成上限を枯らし、
Player 1 の投票で thinking-only になり、Player 3 は出力上限で切れた。同じモデルの別の
行動（占い・襲撃・CO補正）は本文を出せていたのに、旧来は timeout や thinking 枯渇を
**モデル単位の恒久フラグ**に保存したため、投票で失敗すると以後の議論まで思考off・
生成上限縮小が及んだ（「投票で詰まっただけなのに議論も絞る」不整合）。

本モジュールはLLMを呼ばない純粋な窓管理だけを持つ。方策は2つ:

- `action`（既定）: 処置はその行動にだけ当てる。別行動は自由なまま。同じ kind が
  2つの行動で観測されたときだけモデル全体へ広げる（恒久ではなく証拠の合算）
- `model`: 旧来動作（モデル単位）。A/B比較のための逃げ道

どちらも解除できる（直近の成功が続けば縮小・off を戻す）。推理の手順・結論・口調・
話題には干渉せず、本文が得られた場合は内容を保持する。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

# TASK-063 と同じカテゴリ語彙を使う（ログ上の名前を揃える）
CATEGORY_OK = "ok"
CATEGORY_THINKING_BUDGET = "thinking_budget"
CATEGORY_GENERATION_CAP = "generation_cap"
CATEGORY_TIMEOUT = "timeout"
CATEGORY_TRANSPORT = "transport"
CATEGORY_OUTPUT_FORMAT = "output_format"
CATEGORY_PROMPT_LOGIC = "prompt_logic"

# 「出力予算・通信」側の失敗だけで縮小を検討する（形式・論理の失敗では絞らない）
BUDGET_BOUND_CATEGORIES: frozenset[str] = frozenset(
    {
        CATEGORY_THINKING_BUDGET,
        CATEGORY_GENERATION_CAP,
        CATEGORY_TIMEOUT,
        CATEGORY_TRANSPORT,
    }
)

OUTPUT_STABILIZATION_SCOPE_ACTION = "action"
OUTPUT_STABILIZATION_SCOPE_MODEL = "model"
OUTPUT_STABILIZATION_SCOPES: tuple[str, ...] = (
    OUTPUT_STABILIZATION_SCOPE_ACTION,
    OUTPUT_STABILIZATION_SCOPE_MODEL,
)
OUTPUT_STABILIZATION_SCOPE_DEFAULT = OUTPUT_STABILIZATION_SCOPE_ACTION


def normalize_output_stabilization_scope(value: object) -> str:
    """設定値を安定化スコープへ正規化する（未知値は既定の action）。"""
    if isinstance(value, str) and value.strip():
        candidate = value.strip().lower()
        if candidate in OUTPUT_STABILIZATION_SCOPES:
            return candidate
    return OUTPUT_STABILIZATION_SCOPE_DEFAULT


def output_stabilization_scope_label(scope: object) -> str:
    """ログ用ラベル（scoped_to_model_action / whole_model_downgrade）。"""
    if normalize_output_stabilization_scope(scope) == OUTPUT_STABILIZATION_SCOPE_MODEL:
        return "whole_model_downgrade"
    return "scoped_to_model_action"


# 直近送信の内訳を見る窓（実機ログの連続失敗は2〜3件で固まった）
FAILURE_WINDOW_SIZE = 4
# 同じ kind がこの数の行動で観測されたらモデル全体の証拠とする（恒久ではなく合算）
MODEL_WIDE_DOWNGRADE_ACTIONS = 2
# この数だけ本文付き成功が続いたら縮小・offを解除する（自由な思考を戻す）
RELAX_AFTER_SUCCESS_STREAK = 3

DOWNGRADE_TIMEOUT_FIT = "timeout_fit"
DOWNGRADE_THINKING_OFF = "thinking_off"
DOWNGRADE_KINDS: tuple[str, ...] = (DOWNGRADE_TIMEOUT_FIT, DOWNGRADE_THINKING_OFF)


@dataclass
class ActionStabilityRecord:
    """1モデル×1行動の直近失敗窓と、処置の適用状態（決定的・LL不使用）。"""

    budget_key: str
    window: deque[str] = field(default_factory=lambda: deque(maxlen=FAILURE_WINDOW_SIZE))
    success_streak: int = 0
    budget_bound_streak: int = 0
    downgrades: set[str] = field(default_factory=set)
    applied_events: int = 0
    relaxed_events: int = 0

    def record(self, category: object) -> None:
        key = str(category or CATEGORY_OK)
        self.window.append(key)
        if key == CATEGORY_OK:
            self.success_streak += 1
            self.budget_bound_streak = 0
        elif key in BUDGET_BOUND_CATEGORIES:
            self.budget_bound_streak += 1
            self.success_streak = 0
        else:
            # 形式・論理側は双方のリセットだけ（縮小の材料にしない）
            self.success_streak = 0
            self.budget_bound_streak = 0

    @property
    def should_relax(self) -> bool:
        return bool(self.downgrades) and self.success_streak >= RELAX_AFTER_SUCCESS_STREAK

    @property
    def budget_bound_in_window(self) -> int:
        return sum(1 for item in self.window if item in BUDGET_BOUND_CATEGORIES)

    def snapshot(self) -> dict[str, object]:
        return {
            "action_budget_key": self.budget_key,
            "recent_categories": list(self.window),
            "budget_bound_in_window": self.budget_bound_in_window,
            "budget_bound_streak": self.budget_bound_streak,
            "success_streak": self.success_streak,
            "active_downgrades": sorted(self.downgrades),
            "downgrade_events": self.applied_events,
            "relaxed_events": self.relaxed_events,
        }


class OutputStabilityTracker:
    """モデル×行動別の直近失敗窓と、恒久化を避ける解除条件を持つ台帳（LL不使用）。"""

    def __init__(self, scope: object = OUTPUT_STABILIZATION_SCOPE_DEFAULT) -> None:
        self.scope = normalize_output_stabilization_scope(scope)
        self._records: dict[tuple[str, str], ActionStabilityRecord] = {}

    # --- 観測 -----------------------------------------------------------
    def record(self, model: str, budget_key: str, category: object) -> dict[str, object]:
        """1送信の結果カテゴリを窓へ積む（縮小の可否は適用側が決める）。"""
        entry = self._entry(str(model or ""), str(budget_key or "discussion"))
        entry.record(category)
        return entry.snapshot()

    def _entry(self, model: str, budget_key: str) -> ActionStabilityRecord:
        key = (model, budget_key)
        entry = self._records.get(key)
        if entry is None:
            entry = ActionStabilityRecord(budget_key=budget_key)
            self._records[key] = entry
        return entry

    def streaks(self, model: str, budget_key: str) -> tuple[int, int]:
        """(連続成功, 予算側に連続で偏った数)。継続・解除の判断材料。"""
        entry = self._entry(str(model or ""), str(budget_key or "discussion"))
        return entry.success_streak, entry.budget_bound_streak

    # --- 処置の適用状態 -------------------------------------------------
    def apply_downgrade(self, model: str, budget_key: str, kind: str) -> dict[str, object]:
        """縮小・offを適用した記録を残す（kind は timeout_fit / thinking_off）。"""
        entry = self._entry(str(model or ""), str(budget_key or "discussion"))
        first_application = kind not in entry.downgrades
        entry.downgrades.add(kind)
        entry.applied_events += 1
        payload = entry.snapshot()
        payload["first_application"] = bool(first_application)
        payload["scope"] = self.scope
        payload["model_wide_actions"] = sorted(self.actions_with(model, kind))
        return payload

    def actions_with(self, model: str, kind: str) -> set[str]:
        """その kind が適用済みの行動名集合（モデル全体へ広げる根拠の合算に使う）。"""
        model_name = str(model or "")
        return {
            key[1]
            for key, entry in self._records.items()
            if key[0] == model_name and kind in entry.downgrades
        }

    def model_wide_evidence(self, model: str, kind: str) -> bool:
        """別行動でも同じ処置が決まったか（モデル単位で絞る根拠の合算）。"""
        return len(self.actions_with(model, kind)) >= MODEL_WIDE_DOWNGRADE_ACTIONS

    def has_downgrade(self, model: str, budget_key: str, kind: str) -> bool:
        """その行動に処置が掛かっているか（スコープは呼び出し側の判断に任せる）。"""
        return kind in self._entry(str(model or ""), str(budget_key or "discussion")).downgrades

    def relax(self, model: str, budget_key: str) -> dict[str, object] | None:
        """成功が続いていれば処置を解く。該当がなければ None を返す。"""
        entry = self._entry(str(model or ""), str(budget_key or "discussion"))
        if not entry.should_relax:
            return None
        cleared = sorted(entry.downgrades)
        entry.downgrades.clear()
        entry.relaxed_events += 1
        payload = entry.snapshot()
        payload["cleared_downgrades"] = cleared
        payload["scope"] = self.scope
        return payload

    def serialize(self) -> dict[str, object]:
        """ログ用の逐表（処置中の行動と直近窓のみ。発言内容は含まれない）。"""
        rows: dict[str, dict[str, object]] = {}
        for (model, budget_key), entry in self._records.items():
            rows.setdefault(model, {})[budget_key] = entry.snapshot()
        return {"scope": self.scope, "models": rows}


