"""TASK-053: モデル×アクション別のthinking消費プロファイルと本文予算分離。

最新ゲーム game_088e49f0690c では投票（gpt-oss）と人狼相談（gemma系）で
thinking-only 応答が発生し、思考トークンが生成上限（`num_predict`）を食い潰して
本文が空のまま打ち切られていた。従来は「そのモデルの思考行動を一律 off」へ落として
いたため、モデルの推論の自由を削り、同じ上限では本文量も増えなかった。

本モジュールはLLMを呼ばない純粋な計測・予算計算だけを扱う。

1. モデル×アクション別の観測集計（要求数・本文長・thinking有無・空応答/打ち切り/
   タイムアウト・待ち時間・本文が空で打ち切られた応答の実測生成トークン =
   thinking消費量の下限）。
2. 初回リクエスト向けの予算分離: `本文最低予算(body_min) + モデル別thinking余白`。
   thinking証拠のない行動では余白を追加しない（コンテキストと応答時間の節約）。
3. 余白の適応更新: 本文が空のまま打ち切られたら実測値を上振れ補正した余白へ拡大し、
   拡大した後も空になった場合のみ上流の thinking off 判定（能力情報）へ委ねる。
4. 送信する num_predict が本文予算を保持できているかの検査（評価ログ用）。
"""

import math
from dataclasses import dataclass, field

# 全局の既定thinking余白（TASK-046で安全側として採用した値）。モデル別の観測値が
# ある場合はそちらを優先するため、これ自体を一律には増やさない。
DEFAULT_THINKING_ALLOWANCE_TOKENS: int = 1024
MIN_THINKING_ALLOWANCE_TOKENS: int = 256
MAX_THINKING_ALLOWANCE_TOKENS: int = 3072
# 本文予約で引き上げる生成上限の行動別キャップ。ローカル環境では生成上限の過大化が
# 待ち時間に直結するため、再要求（倍化・フォールバック）より安い範囲に抑える。
BODY_RESERVATION_MAX_NUM_PREDICT: dict[str, int] = {
    "discussion": 6144,
    "wolf_discussion": 3072,
    "vote": 1792,
    "attack": 1536,
    "seer": 1536,
    "co_correction": 1280,
}
# 本文最低トークン（解析に必要な形式 + 短い理由）。短い行動は本文を先に確保する。
BODY_MIN_TOKENS_BY_ACTION: dict[str, int] = {
    "discussion": 640,
    "wolf_discussion": 448,
    "vote": 192,
    "attack": 160,
    "seer": 160,
    "co_correction": 256,
}
# done_reason=length の打ち切り応答では、思考は上限を「少なくともその分」消費して
# 止まっている。実測値は下限なので上振れ分を補正する倍率。
THINKING_UNDERESTIMATE_RATIO: float = 1.25
# 本文余白の拡大で何回まで耐え、その後に thinking off 化（能力判定）へ移行するか。
# 1回目の不足は「余白が足りなかっただけ」の可能性が高く、即offでは推論の自由を削る
# ため、まず余白を拡大する（TASK-053の目的）。
THINKING_OFF_AFTER_EXHAUSTIONS: int = 2

# `_options_for` が予約を適用したことを示す解決元ラベル（ログ・CLI表示用）。
RESERVED_BODY_SOURCE = "model_action_body_reservation"
RESERVATION_SKIPPED_RETRY = "retry_request"
RESERVATION_SKIPPED_DISABLED = "body_reservation_disabled"
RESERVATION_SKIPPED_EVIDENCE = "no_thinking_evidence"
RESERVATION_SKIPPED_HEADROOM = "configured_already_covers_body"
# タイムアウト観測済みモデルでは本文予約を適用しない（応答時間の悪化を避ける）。
RESERVATION_SKIPPED_TIMEOUT = "model_timeout_fit"

EMPTY_CONTENT_REASON = "empty_content"
THINKING_ONLY_REASON = "thinking_only_response"
GENERATION_TRUNCATED_REASON = "generation_truncated"
TIMEOUT_REASON = "timeout"


def _clamped_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return int(value)


@dataclass
class ModelActionProfile:
    """1モデル×1行動キーの生成観測プロファイル（送信のたびに更新する）。"""

    model: str
    budget_key: str
    requests: int = 0
    first_requests: int = 0
    retry_requests: int = 0
    body_present_count: int = 0
    empty_content_count: int = 0
    thinking_only_count: int = 0
    # done_reason=length で打ち切られた回数（本文がある場合も含む生成量の証拠）
    length_stop_count: int = 0
    timeout_count: int = 0
    transport_error_count: int = 0
    thinking_present_count: int = 0
    # 本文が空のまま打ち切られた応答で実測された生成トークンの最大値。
    # 「ここまで思考へ使われて本文が0になった」という下限証拠として使う。
    thinking_floor_tokens: int = 0
    thinking_token_total: int = 0
    thinking_token_samples: int = 0
    body_char_total: int = 0
    elapsed_seconds_total: float = 0.0
    first_attempt_empty_count: int = 0
    exhaustion_count: int = 0
    widened_count: int = 0
    # thinking余白（本文最低予算込み）を積んでいたのに、思考だけで空になった送信数。
    # 「予約が足りていないか」を検出する徴候として使う（予約の成否自体は body_budget で測る）。
    covered_thinking_only_count: int = 0
    # 適応拡大した余白（None=観測から自動算出）。widen_allowance() で更新する。
    allowance_tokens: int | None = None
    allowance_history: list[int] = field(default_factory=list)

    @property
    def empty_response_count(self) -> int:
        return self.empty_content_count + self.thinking_only_count

    @property
    def empty_response_rate(self) -> float:
        return round(self.empty_response_count / self.requests, 4) if self.requests else 0.0

    @property
    def thinking_only_rate(self) -> float:
        return round(self.thinking_only_count / self.requests, 4) if self.requests else 0.0

    @property
    def first_attempt_empty_rate(self) -> float:
        if not self.first_requests:
            return 0.0
        return round(self.first_attempt_empty_count / self.first_requests, 4)

    @property
    def truncated_rate(self) -> float:
        return round(self.length_stop_count / self.requests, 4) if self.requests else 0.0

    @property
    def thinking_share_rate(self) -> float:
        """応答に思考が含まれた割合（モデル×アクション別のthinking消費率）。"""
        return (
            round(self.thinking_present_count / self.requests, 4) if self.requests else 0.0
        )

    @property
    def avg_elapsed_seconds(self) -> float:
        if not self.requests:
            return 0.0
        return round(self.elapsed_seconds_total / self.requests, 3)

    @property
    def avg_body_chars(self) -> float:
        present = self.body_present_count
        return round(self.body_char_total / present, 1) if present else 0.0

    @property
    def avg_thinking_tokens(self) -> float:
        if not self.thinking_token_samples:
            return 0.0
        return round(self.thinking_token_total / self.thinking_token_samples, 1)

    @property
    def has_thinking_evidence(self) -> bool:
        return self.thinking_present_count > 0 or self.thinking_floor_tokens > 0

    def record_attempt(
        self,
        *,
        request_number: int = 1,
        body_char_count: int = 0,
        thinking_present: bool = False,
        done_reason: str | None = None,
        eval_count: int | None = None,
        elapsed_seconds: float = 0.0,
        empty: bool = False,
        thinking_only: bool = False,
        timed_out: bool = False,
        transport_error: bool = False,
    ) -> None:
        """1送信ぶんの観測をプロファイルへ加算する（通信失敗時も呼ぶ）。"""

        key = _clamped_int(request_number, 1)
        self.requests += 1
        if key <= 1:
            self.first_requests += 1
        else:
            self.retry_requests += 1
        chars = max(0, _clamped_int(body_char_count))
        self.body_char_total += chars
        if chars > 0:
            self.body_present_count += 1
        if thinking_present:
            self.thinking_present_count += 1
        if done_reason == "length":
            self.length_stop_count += 1
        tokens = _clamped_int(eval_count, -1)
        if tokens >= 0:
            self.thinking_token_total += tokens
            self.thinking_token_samples += 1
        if timed_out:
            self.timeout_count += 1
        if transport_error:
            self.transport_error_count += 1
        if empty:
            self.empty_content_count += 1
            if key <= 1:
                self.first_attempt_empty_count += 1
        if thinking_only:
            self.thinking_only_count += 1
            if key <= 1:
                self.first_attempt_empty_count += 1
        # 本文が空で打ち切られた送信の実測生成量は、思考が消費した下限トークン。
        if empty or thinking_only:
            if done_reason == "length" and tokens >= 0 and tokens > self.thinking_floor_tokens:
                self.thinking_floor_tokens = tokens
            elif self.thinking_floor_tokens == 0 and tokens > 0:
                self.thinking_floor_tokens = tokens
        self.elapsed_seconds_total += max(0.0, float(elapsed_seconds))

    def widen_allowance(
        self,
        *,
        num_predict_sent: int | None = None,
        max_allowance: int = MAX_THINKING_ALLOWANCE_TOKENS,
    ) -> tuple[int, bool]:
        """thinking予算不足のたびに本文余白を拡大する（off化は最終手段）。

        返り値は (新しい余白トークン, 実際に変化してoff化余地が残ったか)。
        上限に達した場合は変化なしとなり、上流で thinking off 判定（能力情報）へ進む。
        """

        self.exhaustion_count += 1
        current = _clamped_int(self.allowance_tokens)
        base = (
            current
            if current > 0
            else max(
                MIN_THINKING_ALLOWANCE_TOKENS,
                int(math.ceil(self.thinking_floor_tokens * THINKING_UNDERESTIMATE_RATIO)),
                DEFAULT_THINKING_ALLOWANCE_TOKENS,
            )
        )
        candidate = min(max_allowance, max(base * 2, base + MIN_THINKING_ALLOWANCE_TOKENS))
        sent = _clamped_int(num_predict_sent)
        if sent > 0:
            # 送信上限そのものを跨ぐような余白にはできない（本文0のままでは意味がない）。
            candidate = min(candidate, max(MIN_THINKING_ALLOWANCE_TOKENS, sent - 1))
        if candidate <= current:
            return current, False
        self.allowance_tokens = candidate
        self.allowance_history.append(candidate)
        self.widened_count += 1
        return candidate, True

    def serialize(self) -> dict[str, object]:
        return {
            "model": self.model,
            "action_budget_key": self.budget_key,
            "requests": self.requests,
            "first_requests": self.first_requests,
            "retry_requests": self.retry_requests,
            "body_present_count": self.body_present_count,
            "empty_content_count": self.empty_content_count,
            "thinking_only_count": self.thinking_only_count,
            "length_stop_count": self.length_stop_count,
            "timeout_count": self.timeout_count,
            "transport_error_count": self.transport_error_count,
            "thinking_present_count": self.thinking_present_count,
            "thinking_floor_tokens": self.thinking_floor_tokens,
            "avg_thinking_tokens": self.avg_thinking_tokens,
            "exhaustion_count": self.exhaustion_count,
            "widened_count": self.widened_count,
            "covered_thinking_only_count": self.covered_thinking_only_count,
            "allowance_tokens": self.allowance_tokens,
            "empty_response_rate": self.empty_response_rate,
            "thinking_only_rate": self.thinking_only_rate,
            "first_attempt_empty_rate": self.first_attempt_empty_rate,
            "truncated_rate": self.truncated_rate,
            "thinking_share_rate": self.thinking_share_rate,
            "avg_elapsed_seconds": self.avg_elapsed_seconds,
            "avg_body_chars": self.avg_body_chars,
        }


class ThinkingProfiles:
    """モデル×アクション別のプロファイル集（ゲーム中にのみ存在する状態）。"""

    def __init__(self) -> None:
        self._profiles: dict[str, dict[str, ModelActionProfile]] = {}

    def get(self, model: str, budget_key: str) -> ModelActionProfile | None:
        return self._profiles.get(model, {}).get(budget_key)

    def get_or_create(self, model: str, budget_key: str) -> ModelActionProfile:
        actions = self._profiles.setdefault(model, {})
        profile = actions.get(budget_key)
        if profile is None:
            profile = ModelActionProfile(model=model, budget_key=budget_key)
            actions[budget_key] = profile
        return profile

    def record(self, model: str, budget_key: str, **attempt: object) -> ModelActionProfile:
        profile = self.get_or_create(model, budget_key)
        profile.record_attempt(**attempt)  # type: ignore[arg-type]
        return profile

    def models(self) -> tuple[str, ...]:
        return tuple(self._profiles.keys())

    def snapshot(self) -> dict[str, dict[str, dict[str, object]]]:
        return {
            model: {
                budget_key: profile.serialize()
                for budget_key, profile in actions.items()
            }
            for model, actions in self._profiles.items()
        }


def thinking_allowance(
    profile: ModelActionProfile | None,
    *,
    thinking_enabled: bool | None,
    default_margin: int = DEFAULT_THINKING_ALLOWANCE_TOKENS,
    max_allowance: int = MAX_THINKING_ALLOWANCE_TOKENS,
) -> int:
    """送信時に確保すべき thinking 余白（モデル別の別予算）。

    - 適応拡大済みの指定値があればそれを最優先する。
    - 打ち切り実測（floor）があれば上振れ補正した値を使う。
    - 証拠がなく thinking off の場合は 0 を返す（余白を積まずコンテキストを節約）。
    """

    if profile is None:
        return int(default_margin) if thinking_enabled is True else 0
    specified = _clamped_int(profile.allowance_tokens)
    if specified > 0:
        return min(max_allowance, specified)
    floor = _clamped_int(profile.thinking_floor_tokens)
    if floor > 0:
        allowance = max(
            MIN_THINKING_ALLOWANCE_TOKENS,
            int(math.ceil(floor * THINKING_UNDERESTIMATE_RATIO)),
        )
        return min(max_allowance, allowance)
    if thinking_enabled is True:
        return int(default_margin)
    return 0


@dataclass(frozen=True)
class BodyReservation:
    """初回リクエストの num_predict に本文予算と thinking 余白を分離した結果。"""

    num_predict: int | None
    source: str
    applied: bool
    skip_reason: str | None
    body_min_tokens: int
    thinking_allowance_tokens: int
    required_tokens: int
    cap_tokens: int | None
    increased_by: int

    def serialize(self) -> dict[str, object]:
        return {
            "num_predict": self.num_predict,
            "num_predict_source": self.source,
            "reserved": bool(self.applied),
            "skip_reason": self.skip_reason,
            "body_min_tokens": self.body_min_tokens,
            "thinking_allowance_tokens": self.thinking_allowance_tokens,
            "required_generation_tokens": self.required_tokens,
            "reservation_cap_tokens": self.cap_tokens,
            "increased_by_tokens": self.increased_by,
        }


def reserve_body_budget(
    num_predict: object,
    num_predict_source: str,
    profile: ModelActionProfile | None,
    budget_key: str,
    *,
    thinking_enabled: bool | None,
    body_min: int | None = None,
    allowance: int | None = None,
    cap: int | None = None,
    enabled: bool = True,
    retry_request: bool = False,
) -> BodyReservation:
    """本文最低予算 + thinking余白を確保できる生成上限へ初回リクエストを引き上げる。

    player/CLIの設定値より優先して引き上げるが、上限 cap で待ち時間増を抑える
    （再要求の倍化・フォールバックより安い）。設定値が既に要求量を満たす場合は
    何もしない（下げない）。再要求は本文専用に絞るため対象外。
    thinking の証拠がない行動では余白を追加しない。
    """

    body_minimum = (
        _clamped_int(body_min)
        if body_min is not None
        else _clamped_int(BODY_MIN_TOKENS_BY_ACTION.get(budget_key))
    )
    reserved_allowance = (
        _clamped_int(allowance)
        if allowance is not None
        else thinking_allowance(profile, thinking_enabled=thinking_enabled)
    )
    required = body_minimum + reserved_allowance
    limit_value = _clamped_int(cap if cap is not None else BODY_RESERVATION_MAX_NUM_PREDICT.get(budget_key))

    def build(
        value: int | None, applied: bool, reason: str | None, increased: int
    ) -> BodyReservation:
        return BodyReservation(
            num_predict=value,
            source=RESERVED_BODY_SOURCE if applied else num_predict_source,
            applied=applied,
            skip_reason=reason,
            body_min_tokens=body_minimum,
            thinking_allowance_tokens=reserved_allowance,
            required_tokens=required,
            cap_tokens=limit_value or None,
            increased_by=increased,
        )

    if not enabled:
        return build(
            num_predict if isinstance(num_predict, int) and not isinstance(num_predict, bool) else None,
            False,
            RESERVATION_SKIPPED_DISABLED,
            0,
        )
    if retry_request:
        return build(
            num_predict if isinstance(num_predict, int) and not isinstance(num_predict, bool) else None,
            False,
            RESERVATION_SKIPPED_RETRY,
            0,
        )
    has_evidence = bool(profile and profile.has_thinking_evidence) or reserved_allowance > 0
    if required <= 0 or (not has_evidence and thinking_enabled is not True):
        return build(
            num_predict if isinstance(num_predict, int) and not isinstance(num_predict, bool) else None,
            False,
            RESERVATION_SKIPPED_EVIDENCE,
            0,
        )
    if num_predict is None or isinstance(num_predict, bool):
        # 設定未指定（サービス既定任せ）では、証拠があるときだけ明示値として設定する。
        target = min(required, limit_value) if limit_value > 0 else required
        return build(target, True, None, target)
    configured = int(num_predict)
    target = min(required, limit_value) if limit_value > 0 else required
    if target <= configured:
        return build(configured, False, RESERVATION_SKIPPED_HEADROOM, 0)
    return build(target, True, None, target - configured)


def inspect_body_budget(
    profile: ModelActionProfile | None,
    budget_key: str,
    num_predict: object,
    *,
    thinking_enabled: bool | None,
    allowance: int | None = None,
) -> dict[str, object]:
    """実際に送る num_predict が「本文 + thinking」を保持できているかの検査。"""

    body_minimum = _clamped_int(BODY_MIN_TOKENS_BY_ACTION.get(budget_key))
    reserved_allowance = (
        _clamped_int(allowance)
        if allowance is not None
        else thinking_allowance(profile, thinking_enabled=thinking_enabled)
    )
    required = body_minimum + reserved_allowance
    value = (
        int(num_predict)
        if isinstance(num_predict, int) and not isinstance(num_predict, bool)
        else None
    )
    return {
        "num_predict": value,
        "body_min_tokens": body_minimum,
        "thinking_allowance_tokens": reserved_allowance,
        "required_generation_tokens": required,
        "headroom_tokens": (value - required) if value is not None else None,
        "body_covered": bool(value is not None and value >= required),
        "thinking_floor_tokens": _clamped_int(profile.thinking_floor_tokens) if profile else 0,
    }
