import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlayerConfig:
    name: str
    model: str
    num_ctx: int | None = None
    timeout: float | None = None
    num_predict: int | None = None
    response_language: str = "日本語"


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
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Player name at index {index} must be a non-empty string.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Player model at index {index} must be a non-empty string.")
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
        players.append(
            PlayerConfig(
                name=name,
                model=model,
                num_ctx=num_ctx,
                timeout=float(timeout) if timeout is not None else None,
                num_predict=num_predict,
                response_language=response_language.strip(),
            )
        )

    if len(players) != 8:
        raise ValueError(f"Expected exactly 8 player configurations, got {len(players)}.")
    if len({player.name for player in players}) != len(players):
        raise ValueError("Player names must be unique.")

    return tuple(players)
