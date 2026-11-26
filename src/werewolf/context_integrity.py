"""TASK-065: 実際に使われるコンテキスト長の証拠を4つに分け、送信前に気づけるようにする。

実機ログ `game_425755ce9a83` では Player 1（`llm-jp-4.1-33b-thinking-fixed`）の設定が
`num_ctx=16384` だったのに対し、そのモデルの Modelfile とプロセス環境変数は 4096 だった。
採用値は 16384 のまま送信され、通信失敗・約360秒の timeout・思考予算枯渇が起きた後で
初めて「設定値とサーバー観測値が違う」と分かった（TASK-046 の解決イベントは初回送信時に
しか出ないため、遅れて届く）。本モジュールはLLMを呼ばない純粋な評価だけで、
設定（configured）／サーバー観測（server_observed）／ロード実測（loaded）／送信要求
（requested）を分けて保持し、ゲーム開始前に警告として出せる形にする。

設計方針（TASK-065の思想そのまま）:
- 節約: 議論本文へ余分な制約文を足さない。整合チェックは運用面の数値と警告だけ
- 一貫性: 「どの値を根拠に送ったか」を1つの値に潰さず、由来ごとに残す
- 合理的な判断: 縮退する場合は安全側の値を示すだけで、決め打ちはしない
- 自由: 方策は off / warn（既定＝挙動不変）/ cap（安全値へ縮退）の3つ。
  `num_ctx` は API リクエストで Modelfile を上書きできるため、cap は明示的な選択時だけ

数値だけが出力であり、役職・占い結果・襲撃対象などの秘密情報は一切含まれない。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 方策
# ---------------------------------------------------------------------------

CONTEXT_INTEGRITY_POLICY_OFF = "off"
CONTEXT_INTEGRITY_POLICY_WARN = "warn"
CONTEXT_INTEGRITY_POLICY_CAP = "cap"
CONTEXT_INTEGRITY_POLICIES: tuple[str, ...] = (
    CONTEXT_INTEGRITY_POLICY_OFF,
    CONTEXT_INTEGRITY_POLICY_WARN,
    CONTEXT_INTEGRITY_POLICY_CAP,
)
CONTEXT_INTEGRITY_POLICY_DEFAULT = CONTEXT_INTEGRITY_POLICY_WARN


def normalize_context_integrity_policy(value: object) -> str:
    """設定値を整合チェックの方策へ正規化する（未知値は既定の warn）。"""
    if isinstance(value, str) and value.strip():
        candidate = value.strip().lower()
        if candidate in CONTEXT_INTEGRITY_POLICIES:
            return candidate
    return CONTEXT_INTEGRITY_POLICY_DEFAULT


def context_integrity_policy_label(policy: object) -> str:
    """ログ用のラベル（observe_only / warn_before_send / safe_value_adoption）。"""
    normalized = normalize_context_integrity_policy(policy)
    if normalized == CONTEXT_INTEGRITY_POLICY_OFF:
        return "integrity_check_disabled"
    if normalized == CONTEXT_INTEGRITY_POLICY_CAP:
        return "safe_value_adoption"
    return "warn_before_send"


# ---------------------------------------------------------------------------
# 不一致カテゴリ（相互排他的ではなく「どの証拠と割れたか」の一覧）
# ---------------------------------------------------------------------------

KIND_ALIGNED = "aligned"
KIND_CONFIGURED_ABOVE_SERVER = "configured_above_server_observed"
KIND_CONFIGURED_ABOVE_LOADED = "configured_above_loaded_context"
KIND_LOADED_UNAVAILABLE = "loaded_context_unavailable"
KIND_NO_EVIDENCE = "no_context_evidence"
KIND_SAFE_VALUE_APPLIED = "safe_value_applied"

CONTEXT_INTEGRITY_KINDS: tuple[str, ...] = (
    KIND_ALIGNED,
    KIND_CONFIGURED_ABOVE_SERVER,
    KIND_CONFIGURED_ABOVE_LOADED,
    KIND_LOADED_UNAVAILABLE,
    KIND_NO_EVIDENCE,
    KIND_SAFE_VALUE_APPLIED,
)


def _as_positive_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = int(text)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


EVIDENCE_SOURCE_PS = "api_ps_loaded"
EVIDENCE_SOURCE_MODELFILE = "modelfile"
EVIDENCE_SOURCE_PROCESS_ENV = "process_env"


@dataclass(frozen=True)
class ContextIntegrityCheck:
    """1モデル分のコンテキスト整合チェック結果（数値のみ。秘密情報は含まれない）。"""

    model: str
    policy: str
    configured: int | None = None
    configured_source: str = ""
    server_observed: int | None = None
    server_observed_source: str = ""
    loaded_context: int | None = None
    modelfile_num_ctx: int | None = None
    process_env_num_ctx: int | None = None
    max_num_ctx: int | None = None
    requested_num_ctx: int | None = None
    adopted_num_ctx: int | None = None
    safe_value: int | None = None
    mismatch: bool = False
    applied: bool = False
    kinds: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_fields(self) -> dict[str, object]:
        """イベント・state・ログへ載せる中立フィールド（由来を1つの値へ潰さない）。"""
        return {
            "model": self.model,
            "context_integrity_policy": self.policy,
            "context_integrity_policy_label": context_integrity_policy_label(self.policy),
            "configured_context_limit": self.configured,
            "configured_context_source": self.configured_source,
            "server_observed_context_limit": self.server_observed,
            "server_observed_context_source": self.server_observed_source,
            "loaded_context_from_ps": self.loaded_context,
            "modelfile_num_ctx": self.modelfile_num_ctx,
            "process_env_num_ctx": self.process_env_num_ctx,
            "max_num_ctx_setting": self.max_num_ctx,
            "requested_num_ctx": self.requested_num_ctx,
            "adopted_num_ctx": self.adopted_num_ctx,
            "safe_context_value": self.safe_value,
            "context_integrity_mismatch": bool(self.mismatch),
            "context_integrity_kinds": list(self.kinds),
            "context_integrity_applied": bool(self.applied),
            "context_integrity_notes": list(self.notes),
        }

    def warning_lines(self) -> tuple[str, ...]:
        """端末へ出す運用警告（プロンプト・発言本文には絶対に載せない）。"""
        lines: list[str] = []
        if KIND_CONFIGURED_ABOVE_SERVER in self.kinds and self.server_observed:
            configured_note = (
                f"num_ctx 設定 {self.configured}"
                if self.configured
                else f"自動拡張の天井 max_num_ctx {self.max_num_ctx}"
            )
            action = (
                f"--context-integrity cap のため {self.safe_value} へ縮退して送ります"
                if self.applied
                else "送信前に --num-ctx / Modelfile / OLLAMA_CONTEXT_LENGTH を確認してください"
            )
            lines.append(
                f"{self.model}: {configured_note}"
                f"（{self.configured_source or 'player_setting'}）がサーバー観測値 "
                f"{self.server_observed}（{self.server_observed_source}）を超えています。{action}"
            )
        if KIND_CONFIGURED_ABOVE_LOADED in self.kinds and self.configured and self.loaded_context:
            lines.append(
                f"{self.model}: ロード済み実測 {self.loaded_context} より大きい値"
                f"（{self.configured}）を要求しています。窓を超える入力は切り詰められます"
            )
        if KIND_LOADED_UNAVAILABLE in self.kinds:
            lines.append(
                f"{self.model}: /api/ps でロード済みコンテキストを確認できません"
                "（未ロードのため実測は初回送信後にしか出ません）"
            )
        if KIND_NO_EVIDENCE in self.kinds:
            lines.append(
                f"{self.model}: サーバー側のコンテキスト証拠が取れません"
                "（num_ctx 未指定の運用では Ollama の既定値に依存します）"
            )
        return tuple(lines)


def assess_context_integrity(
    model: str,
    *,
    configured: object = None,
    configured_source: str = "",
    ps_loaded_context: object = None,
    modelfile_num_ctx: object = None,
    process_env_num_ctx: object = None,
    max_num_ctx: object = None,
    policy: object = CONTEXT_INTEGRITY_POLICY_DEFAULT,
) -> ContextIntegrityCheck:
    """設定値・サーバー観測・ロード実測を突き合わせ、採用値と警告を決める（純関数）。

    `configured`（APIリクエストの num_ctx）は Modelfile を上書きできる最も強い根拠だが、
    観測値を超える場合は mismatch として分ける。縮退（cap）は明示的に選ばれたときだけ行う。
    """
    normalized_policy = normalize_context_integrity_policy(policy)
    configured_value = _as_positive_int(configured)
    loaded_value = _as_positive_int(ps_loaded_context)
    modelfile_value = _as_positive_int(modelfile_num_ctx)
    env_value = _as_positive_int(process_env_num_ctx)
    max_value = _as_positive_int(max_num_ctx)

    evidence: list[tuple[str, int]] = []
    if loaded_value is not None:
        evidence.append((EVIDENCE_SOURCE_PS, loaded_value))
    if modelfile_value is not None:
        evidence.append((EVIDENCE_SOURCE_MODELFILE, modelfile_value))
    if env_value is not None:
        evidence.append((EVIDENCE_SOURCE_PROCESS_ENV, env_value))

    observed: int | None = None
    observed_source = ""
    for source in (EVIDENCE_SOURCE_PS, EVIDENCE_SOURCE_MODELFILE, EVIDENCE_SOURCE_PROCESS_ENV):
        matches = [value for key, value in evidence if key == source]
        if matches:
            observed = max(matches)
            observed_source = source
            break

    requested = configured_value if configured_value is not None else observed
    kinds: list[str] = []
    mismatch = False
    ceiling = max_value if configured_value is None and max_value is not None else None
    comparison_target = configured_value if configured_value is not None else ceiling
    if (
        comparison_target is not None
        and observed is not None
        and comparison_target > observed
    ):
        mismatch = True
        kinds.append(KIND_CONFIGURED_ABOVE_SERVER)
        if loaded_value is not None and comparison_target > loaded_value:
            kinds.append(KIND_CONFIGURED_ABOVE_LOADED)
    if loaded_value is None:
        kinds.append(KIND_LOADED_UNAVAILABLE)
    if not evidence:
        kinds.append(KIND_NO_EVIDENCE)
    if not kinds:
        kinds.append(KIND_ALIGNED)

    safe_value: int | None = None
    if (
        mismatch
        and requested is not None
        and observed is not None
        and requested > observed
    ):
        safe_value = max(1, min(requested, observed))
    adopted = requested
    applied = False
    if (
        normalized_policy == CONTEXT_INTEGRITY_POLICY_CAP
        and safe_value is not None
        and requested is not None
        and safe_value < requested
    ):
        adopted = safe_value
        applied = True
    return ContextIntegrityCheck(
        model=str(model or ""),
        policy=normalized_policy,
        configured=configured_value,
        configured_source=str(configured_source or ""),
        server_observed=observed,
        server_observed_source=observed_source,
        loaded_context=loaded_value,
        modelfile_num_ctx=modelfile_value,
        process_env_num_ctx=env_value,
        max_num_ctx=max_value,
        requested_num_ctx=requested,
        adopted_num_ctx=adopted,
        safe_value=safe_value,
        mismatch=mismatch,
        applied=applied,
        kinds=tuple(kinds),
    )

