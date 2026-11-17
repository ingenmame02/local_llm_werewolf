import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


# 行動別生成上限（num_predict）で扱う行動キー。議論は文量が必要で、
# 対象選択とCO補正は解析に必要な最小形式を優先する。
ACTION_BUDGET_KEYS: tuple[str, ...] = (
    "discussion",
    "wolf_discussion",
    "attack",
    "seer",
    "vote",
    "co_correction",
)

PERSONALITY_PROFILES: dict[str, str] = {
    "balanced": "落ち着いた調子で、結論と質問のバランスを取る。",
    "analytical": "関連する事実を一つか二つ整理してから、根拠と結論を述べる。",
    "questioning": "必要な場合に具体的な質問を一つ置き、回答との一貫性を確かめる。",
    "assertive": "暫定的な見解を根拠とともに簡潔に示し、証拠がないことは断定しない。",
    "cautious": "不確実性があれば短く示し、別の可能性を一つまで検討する。",
    "consensus": "意見の一致点と争点を一つずつ整理し、次の確認点を示す。",
    "intuitive": "発言や投票の目立つ変化を仮説として示し、観察した根拠を添える。",
}

# TASK-044: 個性プロファイルの制限を廃する「自由発言モード」。
# 話し方の preset を一切適用せず、口調・構成・注目点をモデル自身の判断に任せる。
# 役職の目的・勝敗条件・行動制約・確定事実・出力形式はこれまで通り適用する。
FREE_PERSONALITY_PROFILE = "free"

# 設定ファイル・CLIで自由モードを指す別名。JSONの null も自由扱いにする。
PERSONALITY_PROFILE_ALIASES: dict[str, str] = {
    "": FREE_PERSONALITY_PROFILE,
    "free": FREE_PERSONALITY_PROFILE,
    "none": FREE_PERSONALITY_PROFILE,
    "null": FREE_PERSONALITY_PROFILE,
    "off": FREE_PERSONALITY_PROFILE,
    "no_personality": FREE_PERSONALITY_PROFILE,
    "natural": FREE_PERSONALITY_PROFILE,
    "custom": FREE_PERSONALITY_PROFILE,
}

# CLI・評価スクリプトの choices に使うキー（既定プロファイル + 自由モード）。
PERSONALITY_CHOICE_KEYS: tuple[str, ...] = (*PERSONALITY_PROFILES, FREE_PERSONALITY_PROFILE)


def normalize_personality_profile(value: object, label: str) -> str:
    """Validate a personality profile and return its canonical key.

    ``None`` / ``""`` / aliases (``none``, ``off``, ``natural``, ...) resolve to
    the free mode ``free``, which applies no speaking-style preset at all.
    """
    if value is None:
        return FREE_PERSONALITY_PROFILE
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in PERSONALITY_PROFILES:
            return normalized
        if normalized in PERSONALITY_PROFILE_ALIASES:
            return PERSONALITY_PROFILE_ALIASES[normalized]
    raise ValueError(
        f"{label} must be one of {', '.join(PERSONALITY_CHOICE_KEYS)} "
        "(use null or 'none' for the free expression mode)."
    )


def is_free_personality(profile: object) -> bool:
    """True when no speaking-style preset is applied to this player."""
    return isinstance(profile, str) and profile.strip().casefold() == FREE_PERSONALITY_PROFILE


# TASK-043: プレイヤー別の生成サンプリング設定。幻覚・案内文の模倣を起こすモデルは
# temperature を下げて安定させ、全体設定（CLI）よりプレイヤー設定を優先する。
GENERATION_SAMPLING_KEYS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "repeat_penalty",
)

# 許容範囲（key: (下限, 上限, 整数必須)）。Ollamaのsamplerの範囲と一致させる。
GENERATION_SAMPLING_BOUNDS: dict[str, tuple[float, float, bool]] = {
    "temperature": (0.0, 2.0, False),
    "top_p": (0.0, 1.0, False),
    "top_k": (0, 400, True),
    "repeat_penalty": (0.0, 2.0, False),
}


def validate_generation_option(value: object, key: str, label: str) -> float | int:
    """プレイヤー別の生成サンプリング値を許容範囲で検証する。"""
    low, high, requires_integer = GENERATION_SAMPLING_BOUNDS[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} for {key} must be a number.")
    if not math.isfinite(value):
        raise ValueError(f"{label} for {key} must be a finite number.")
    if requires_integer and int(value) != value:
        raise ValueError(f"{label} for {key} must be an integer.")
    if not low <= value <= high:
        raise ValueError(
            f"{label} for {key} must be between {low} and {high} (got {value})."
        )
    return int(value) if requires_integer else float(value)


# プレイヤー設定・CLI設定が未指定時に使う行動別デフォルト上限。
# 議論は「質問回答・確認できる事実・疑う根拠・次の確認点」を説明する文量が必要なので、
# 思考トークンの消費を考慮した残余を確保する（TASK-024で文量緩和）。
# 思考系モデル（deepseek-r1, qwen3, gpt-oss, gemma4等）は思考トークンがnum_predictを
# 消費するため、議論・人狼相談の既定値を引き上げ、本文生成のための残余を確保する。
DEFAULT_ACTION_NUM_PREDICT: dict[str, int] = {
    "discussion": 2560,
    "wolf_discussion": 1400,
    # TASK-042: 対象選択は「対象行 + 短い理由」の2行構成になったため、切り詰め防止の
    # 残余を確保する（実機ログでは vote/attack の 256 トークンが切り詰め原因になった）。
    "attack": 640,
    "seer": 640,
    "vote": 640,
    # CO補正は「【CO】」「【結果】」「【本文】」の最小3行を出力できる量が必要。
    # 思考系モデルでは思考トークンが消費されるため、1行形式より広い残余を持つ。
    # COの根拠説明を含めるため最低512トークンを確保。
    "co_correction": 512,
}

# 最終回答が1行で済む行動（対象選択・CO補正）。大きい全体設定を持つ思考系モデルでも、
# これらの行動では解析に必要な最小形式だけ生成し、無駄な応答時間を抑える。
MINIMAL_OUTPUT_ACTION_KEYS: frozenset[str] = frozenset(
    {"attack", "seer", "vote", "co_correction"}
)

# 全体設定（num_predict）から行動別上限へ降ろすときのキャップ。
# 思考系のモデルが空応答・切り詰めを起こさないよう、既定値より広い残余を残す。
# TASK-042: 実機ログでは投票・占い・襲撃の切り詰め（generation_truncated）が
# フォールバックの主因になったため、キャップを引き上げる。
MINIMAL_ACTION_MAX_NUM_PREDICT: dict[str, int] = {
    "attack": 1024,
    "seer": 1024,
    "vote": 1024,
    "co_correction": 640,
}

# 思考生成（thinking）を行動別に有効にする既定方針。
# Ollamaへはリクエスト毎の `think` として送る。襲撃・占い・投票・CO補正は
# 最終回答が1行（CO補正は最小3行）で済むため、思考トークンが num_predict を
# 消費して空応答・切り詰めを起こさないよう既定では off にする。
DEFAULT_THINKING_BY_ACTION: dict[str, bool] = {
    "discussion": True,
    "wolf_discussion": True,
    "attack": False,
    "seer": False,
    "vote": False,
    "co_correction": False,
}

# 思考設定の解決優先順位。
THINKING_RESOLUTION_PRIORITY: tuple[str, ...] = (
    "player_by_action",
    "cli_by_action",
    "player_global",
    "cli_global",
    "default_by_action",
)

# モデルが思考生成非対応と判明した時に使う解決元ラベル。
# 設定で on と指定されていても、実際に400を返すモデルでは off へ落とす。
THINKING_CAPABILITY_SOURCE = "model_capability"

# 思考onの生成上限を引き上げたことを示す解決元ラベル。
THINKING_NUM_PREDICT_SOURCE = "thinking_allowance"

# 思考onで思考が生成上限を消費して切り詰めたモデルの解決元ラベル。
# 以後の同一モデルの思考行動は off へ自動で切り替える。
THINKING_BUDGET_SOURCE = "thinking_budget_exhausted"

# 思考onの行動で、生成上限を引き上げた後の残余下限。思考トークンがnum_predictを
# 消費するため、本文生成のための最低残余を確保する（TASK-024）。
# 議論は根拠説明が必要なため、人狼相談より広い残余を持つ。
THINKING_AFFORDABLE_NUM_PREDICT: dict[str, int] = {
    "discussion": 4096,
    "wolf_discussion": 2048,
}

# 思考予算不足の判定理由（イベント・ログ表示用）。
THINKING_BUDGET_EXHAUSTED_REASON = "thinking_consumed_generation_budget"

# 表示用の思考設定ラベル（CLI・評価スクリプトで共有する）。
THINKING_DEFAULT_LABELS: dict[str, str] = {
    "discussion": "議論(思考on)",
    "wolf_discussion": "人狼相談(思考on)",
    "attack": "襲撃(思考off)",
    "seer": "占い(思考off)",
    "vote": "投票(思考off)",
    "co_correction": "CO補正(思考off)",
}


def validate_thinking_flag(value: object, label: str) -> bool:
    """Accept real booleans and short boolean-ish strings for thinking flags."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "on", "yes", "1"}:
            return True
        if normalized in {"false", "off", "no", "0"}:
            return False
    raise ValueError(f"{label} must be a boolean (true/false).")


def validate_thinking_by_action(value: object, label: str) -> dict[str, bool]:
    """Validate an action key -> thinking mapping and return a plain dict."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object of action keys to true/false.")
    parsed: dict[str, bool] = {}
    for key, thinking in value.items():
        if not isinstance(key, str) or key not in ACTION_BUDGET_KEYS:
            raise ValueError(
                f"{label} contains unsupported action key {key!r}; "
                f"expected one of {', '.join(ACTION_BUDGET_KEYS)}."
            )
        parsed[key] = validate_thinking_flag(thinking, f"{label} value for {key}")
    return parsed


def parse_thinking_by_action_pairs(
    values: Sequence[str], label: str = "thinking-by-action"
) -> dict[str, bool]:
    """Parse CLI ``action=true|false`` pairs into per-action thinking flags."""
    parsed: dict[str, bool] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key.strip():
            raise ValueError(
                f"{label} entry {value!r} must be formatted as action=true/false."
            )
        key = key.strip()
        if key not in ACTION_BUDGET_KEYS:
            raise ValueError(
                f"{label} contains unsupported action key {key!r}; "
                f"expected one of {', '.join(ACTION_BUDGET_KEYS)}."
            )
        parsed[key] = validate_thinking_flag(raw, f"{label} value for {key}")
    return parsed


# 表示用の行動別ラベル（CLI・評価スクリプトで共有する）。
ACTION_BUDGET_DEFAULT_LABELS: dict[str, str] = {
    "discussion": "議論(1536)",
    "wolf_discussion": "人狼相談(768)",
    "attack": "襲撃(256)",
    "seer": "占い(256)",
    "vote": "投票(256)",
    "co_correction": "CO補正(256)",
}


@dataclass(frozen=True)
class PlayerConfig:
    name: str
    model: str
    num_ctx: int | None = None
    timeout: float | None = None
    num_predict: int | None = None
    response_language: str = "日本語"
    num_predict_by_action: Mapping[str, int] | None = None
    thinking: bool | None = None
    thinking_by_action: Mapping[str, bool] | None = None
    personality_profile: str = "balanced"
    max_num_ctx: int | None = None
    recommended_num_ctx: int | None = None
    context_strategy: str | None = None
    claim_coaching: bool = False
    # TASK-043: プレイヤー別の生成サンプリング（未指定なら CLI / モデル既定値を使う）。
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None

    def __post_init__(self) -> None:
        # TASK-044: 個性プロファイルはここで検証する（自由モード free を含む）。
        # frozen なので値の正規化はせず、不正値は ValueError だけ返す
        # （load_player_configs と Player側で正規化する）。
        normalize_personality_profile(
            self.personality_profile, f"Player {self.name} personality_profile"
        )
        # TASK-043: プレイヤー別の生成サンプリングはここで範囲検証する
        # （JSONローダー以外から直接作られた設定も同じ基準を通す）。frozenなので
        # 値の正規化はせず、範囲外はValueErrorだけ返す。
        for key in GENERATION_SAMPLING_KEYS:
            value = getattr(self, key)
            if value is None:
                continue
            validate_generation_option(value, key, f"Player {self.name}")


def validate_action_num_predict(
    value: object, label: str
) -> dict[str, int]:
    """Validate an action key -> num_predict mapping and return a plain dict."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object of action keys to positive integers.")
    parsed: dict[str, int] = {}
    for key, num_predict in value.items():
        if not isinstance(key, str) or key not in ACTION_BUDGET_KEYS:
            raise ValueError(
                f"{label} contains unsupported action key {key!r}; "
                f"expected one of {', '.join(ACTION_BUDGET_KEYS)}."
            )
        if isinstance(num_predict, bool) or not isinstance(num_predict, int) or num_predict <= 0:
            raise ValueError(f"{label} value for {key} must be a positive integer.")
        parsed[key] = num_predict
    return parsed


def parse_action_num_predict_pairs(values: Sequence[str], label: str = "action-num-predict") -> dict[str, int]:
    """Parse CLI ``key=value`` pairs into validated action generation budgets."""
    parsed: dict[str, int] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"{label} entry {value!r} must be formatted as action=num_predict.")
        key = key.strip()
        if key not in ACTION_BUDGET_KEYS:
            raise ValueError(
                f"{label} contains unsupported action key {key!r}; "
                f"expected one of {', '.join(ACTION_BUDGET_KEYS)}."
            )
        try:
            num_predict = int(raw)
        except ValueError as error:
            raise ValueError(f"{label} value for {key} must be a positive integer.") from error
        if num_predict <= 0:
            raise ValueError(f"{label} value for {key} must be a positive integer.")
        parsed[key] = num_predict
    return parsed


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "players.json"
)


def load_player_configs(
    config_path: str | Path | None = None,
) -> tuple[PlayerConfig, ...]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with path.open(encoding="utf-8") as config_file:
        data = json.load(config_file)

    if not isinstance(data, dict) or not isinstance(data.get("players"), list):
        raise ValueError("Player configuration must contain a 'players' list.")

    players: list[PlayerConfig] = []
    for index, entry in enumerate(data["players"]):
        if not isinstance(entry, dict):
            raise ValueError(f"Player configuration at index {index} must be an object.")
        name = entry.get("name")
        model = entry.get("model")
        num_ctx = entry.get("num_ctx")
        max_num_ctx = entry.get("max_num_ctx")
        recommended_num_ctx = entry.get("recommended_num_ctx")
        context_strategy = entry.get("context_strategy")
        timeout = entry.get("timeout")
        num_predict = entry.get("num_predict")
        response_language = entry.get("response_language", "日本語")
        num_predict_by_action = entry.get("num_predict_by_action")
        thinking = entry.get("thinking")
        thinking_by_action = entry.get("thinking_by_action")
        personality_profile = entry.get("personality_profile", "balanced")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Player name at index {index} must be a non-empty string.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Player model at index {index} must be a non-empty string.")
        personality_profile = normalize_personality_profile(
            personality_profile,
            f"Player personality_profile at index {index}",
        )
        if num_ctx is not None and (
            not isinstance(num_ctx, int) or isinstance(num_ctx, bool) or num_ctx <= 0
        ):
            raise ValueError(
                f"Player num_ctx at index {index} must be a positive integer."
            )
        for key, context_value in (
            ("max_num_ctx", max_num_ctx),
            ("recommended_num_ctx", recommended_num_ctx),
        ):
            if context_value is not None and (
                not isinstance(context_value, int)
                or isinstance(context_value, bool)
                or context_value <= 0
            ):
                raise ValueError(
                    f"Player {key} at index {index} must be a positive integer."
                )
        if (
            max_num_ctx is not None
            and recommended_num_ctx is not None
            and recommended_num_ctx > max_num_ctx
        ):
            raise ValueError(
                f"Player recommended_num_ctx at index {index} cannot exceed max_num_ctx."
            )
        if context_strategy is not None and context_strategy not in {
            "full_transcript",
            "structured_ledger",
            "hybrid_retrieval",
            "round_transcript_memory",
        }:
            raise ValueError(
                f"Player context_strategy at index {index} must be "
                "'full_transcript', 'structured_ledger', 'hybrid_retrieval', "
                "or 'round_transcript_memory'."
            )
        if timeout is not None and (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(
                f"Player timeout at index {index} must be a positive number."
            )
        if num_predict is not None and (
            not isinstance(num_predict, int)
            or isinstance(num_predict, bool)
            or num_predict <= 0
        ):
            raise ValueError(
                f"Player num_predict at index {index} must be a positive integer."
            )
        if not isinstance(response_language, str) or not response_language.strip():
            raise ValueError(
                f"Player response_language at index {index} must be a non-empty string."
            )
        if num_predict_by_action is not None:
            num_predict_by_action = validate_action_num_predict(
                num_predict_by_action,
                f"Player num_predict_by_action at index {index}",
            )
        if thinking is not None:
            thinking = validate_thinking_flag(
                thinking, f"Player thinking at index {index}"
            )
        if thinking_by_action is not None:
            thinking_by_action = validate_thinking_by_action(
                thinking_by_action,
                f"Player thinking_by_action at index {index}",
            )
        sampling: dict[str, float | int | None] = {}
        for key in GENERATION_SAMPLING_KEYS:
            value = entry.get(key)
            if value is None:
                sampling[key] = None
                continue
            sampling[key] = validate_generation_option(
                value, key, f"Player {key} at index {index}"
            )
        claim_coaching = entry.get("claim_coaching", False)
        players.append(
            PlayerConfig(
                name=name,
                model=model,
                num_ctx=num_ctx,
                max_num_ctx=max_num_ctx,
                recommended_num_ctx=recommended_num_ctx,
                context_strategy=context_strategy,
                timeout=float(timeout) if timeout is not None else None,
                num_predict=num_predict,
                response_language=response_language.strip(),
                num_predict_by_action=num_predict_by_action,
                thinking=thinking,
                thinking_by_action=thinking_by_action,
                personality_profile=personality_profile,
                claim_coaching=claim_coaching,
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                top_k=sampling["top_k"],
                repeat_penalty=sampling["repeat_penalty"],
            )
        )

    if len(players) != 8:
        raise ValueError(f"Expected exactly 8 player configurations, got {len(players)}.")
    if len({player.name for player in players}) != len(players):
        raise ValueError("Player names must be unique.")

    return tuple(players)
