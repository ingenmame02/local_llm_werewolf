import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlayerConfig:
    name: str
    model: str
    num_ctx: int | None = None


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
        players.append(PlayerConfig(name=name, model=model, num_ctx=num_ctx))

    if len(players) != 8:
        raise ValueError(f"Expected exactly 8 player configurations, got {len(players)}.")
    if len({player.name for player in players}) != len(players):
        raise ValueError("Player names must be unique.")

    return tuple(players)
