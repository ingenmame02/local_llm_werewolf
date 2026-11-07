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

# プレイヤー設定・CLI設定が未指定時に使う行動別デフォルト上限。
# 議論は「質問回答・確認できる事実・疑う根拠・次の確認点」を説明する文量が必要なので、
# 思考トークンの消費を考慮した残余を確保する（TASK-024で文量緩和）。
DEFAULT_ACTION_NUM_PREDICT: dict[str, int] = {
    "discussion": 1536,
    "wolf_discussion": 768,
    "attack": 256,
    "seer": 256,
    "vote": 256,
    # CO補正は「【CO】」「【結果】」「【本文】」の最小3行を出力できる量が必要。
    # 思考系モデルでは思考トークンが消費されるため、1行形式より広い残余を持つ。
    "co_correction": 256,
}

# 最終回答が1行で済む行動（対象選択・CO補正）。大きい全体設定を持つ思考系モデルでも、
# これらの行動では解析に必要な最小形式だけ生成し、無駄な応答時間を抑える。
MINIMAL_OUTPUT_ACTION_KEYS: frozenset[str] = frozenset(
    {"attack", "seer", "vote", "co_correction"}
)

# 全体設定（num_predict）から行動別上限へ降ろすときのキャップ。
# 思考系のモデルが空応答・切り詰めを起こさないよう、既定値より広い残余を残す。
MINIMAL_ACTION_MAX_NUM_PREDICT: dict[str, int] = {
    "attack": 768,
    "seer": 768,
    "vote": 768,
    "co_correction": 384,
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

# 思考onで生成する行動では、思考トークンが num_predict を消費するため、
# 最終回答の残余を確保する下限を適用する。実機ログでは思考だけが生成上限を
# すべて消費し、contentが空のまま done_reason=length になった（TASK-024）。
# 下限は「行動別既定値を使った場合」のみ適用し、player・CLIの明示設定は上書きしない。
THINKING_AFFORDABLE_NUM_PREDICT: dict[str, int] = {
    "discussion": 3072,
    "wolf_discussion": 1536,
}

# 思考onの生成上限を引き上げたことを示す解決元ラベル。
THINKING_NUM_PREDICT_SOURCE = "thinking_allowance"

# 思考onで思考が生成上限を消費して切り詰めたモデルの解決元ラベル。
# 以後の同一モデルの思考行動は off へ自動で切り替える。
THINKING_BUDGET_SOURCE = "thinking_budget_exhausted"

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
        if (
            not isinstance(personality_profile, str)
            or personality_profile not in PERSONALITY_PROFILES
        ):
            raise ValueError(
                f"Player personality_profile at index {index} must be one of "
                f"{', '.join(PERSONALITY_PROFILES)}."
            )
        if num_ctx is not None and (
            not isinstance(num_ctx, int) or isinstance(num_ctx, bool) or num_ctx <= 0
        ):
            raise ValueError(
                f"Player num_ctx at index {index} must be a positive integer."
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
        players.append(
            PlayerConfig(
                name=name,
                model=model,
                num_ctx=num_ctx,
                timeout=float(timeout) if timeout is not None else None,
                num_predict=num_predict,
                response_language=response_language.strip(),
                num_predict_by_action=num_predict_by_action,
                thinking=thinking,
                thinking_by_action=thinking_by_action,
                personality_profile=personality_profile,
            )
        )

    if len(players) != 8:
        raise ValueError(f"Expected exactly 8 player configurations, got {len(players)}.")
    if len({player.name for player in players}) != len(players):
        raise ValueError("Player names must be unique.")

    return tuple(players)
