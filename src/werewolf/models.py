import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from .config import ACTION_BUDGET_KEYS, PERSONALITY_PROFILES


class Role(str, Enum):
    WOLF = "wolf"
    MADMAN = "madman"
    SEER = "seer"
    VILLAGER = "villager"


class Phase(str, Enum):
    NIGHT_WOLF = "night_wolf"
    NIGHT_SEER = "night_seer"
    MORNING = "morning"
    DAY_DISCUSSION = "day_discussion"
    VOTING = "voting"
    ENDED = "ended"


@dataclass
class Player:
    name: str
    model: str
    role: Role
    is_alive: bool = True
    partner: str | None = None
    num_ctx: int | None = None
    timeout: float | None = None
    num_predict: int | None = None
    response_language: str = "日本語"
    num_predict_by_action: Mapping[str, int] | None = None
    thinking: bool | None = None
    thinking_by_action: Mapping[str, bool] | None = None
    personality_profile: str = "balanced"

    def __post_init__(self) -> None:
        if self.num_ctx is not None and (
            not isinstance(self.num_ctx, int)
            or isinstance(self.num_ctx, bool)
            or self.num_ctx <= 0
        ):
            raise ValueError("Player num_ctx must be a positive integer.")
        if self.timeout is not None and (
            not isinstance(self.timeout, (int, float))
            or isinstance(self.timeout, bool)
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("Player timeout must be a positive number.")
        if self.num_predict is not None and (
            not isinstance(self.num_predict, int)
            or isinstance(self.num_predict, bool)
            or self.num_predict <= 0
        ):
            raise ValueError("Player num_predict must be a positive integer.")
        if not isinstance(self.response_language, str) or not self.response_language.strip():
            raise ValueError("Player response_language cannot be empty.")
        if (
            not isinstance(self.personality_profile, str)
            or self.personality_profile not in PERSONALITY_PROFILES
        ):
            raise ValueError(
                "Player personality_profile must be one of "
                f"{', '.join(PERSONALITY_PROFILES)}."
            )
        if self.num_predict_by_action is not None:
            if not isinstance(self.num_predict_by_action, Mapping):
                raise ValueError(
                    "Player num_predict_by_action must be a mapping of action keys "
                    "to positive integers."
                )
            for key, num_predict in self.num_predict_by_action.items():
                if key not in ACTION_BUDGET_KEYS:
                    raise ValueError(
                        f"Player num_predict_by_action contains unsupported action key {key!r}."
                    )
                if isinstance(num_predict, bool) or not isinstance(num_predict, int) or num_predict <= 0:
                    raise ValueError(
                        f"Player num_predict_by_action value for {key} must be a positive integer."
                    )
        if self.thinking is not None and not isinstance(self.thinking, bool):
            raise ValueError("Player thinking must be a boolean.")
        if self.thinking_by_action is not None:
            if not isinstance(self.thinking_by_action, Mapping):
                raise ValueError(
                    "Player thinking_by_action must be a mapping of action keys "
                    "to booleans."
                )
            for key, thinking in self.thinking_by_action.items():
                if key not in ACTION_BUDGET_KEYS:
                    raise ValueError(
                        f"Player thinking_by_action contains unsupported action key {key!r}."
                    )
                if not isinstance(thinking, bool):
                    raise ValueError(
                        f"Player thinking_by_action value for {key} must be a boolean."
                    )


@dataclass(frozen=True)
class Message:
    speaker: str
    content: str


@dataclass(frozen=True)
class PublicSeerClaim:
    day: int
    claimant: str
    target: str | None = None
    result: str | None = None


@dataclass(frozen=True)
class PublicRoleClaim:
    day: int
    claimant: str
    claimed_role: str
    target: str | None = None
    result: str | None = None
    reason: str | None = None
    vote_proposal: str | None = None


@dataclass
class GameState:
    players: list[Player]
    day: int = 1
    phase: Phase = Phase.NIGHT_WOLF
    conversation_log: list[Message] = field(default_factory=list)
    wolf_chat_log: list[Message] = field(default_factory=list)
    seer_results: dict[str, str] = field(default_factory=dict)
    seer_result_history: list[tuple[int, str, str]] = field(default_factory=list)
    votes: dict[str, str] = field(default_factory=dict)
    vote_history: list[tuple[int, bool, dict[str, str]]] = field(default_factory=list)
    attack_deaths: list[tuple[int, str]] = field(default_factory=list)
    executions: list[tuple[int, str]] = field(default_factory=list)
    winner: str | None = None
    seed: int | None = None
    pending_attack: str | None = None
    discussion_turn: int = 0
    vote_phase_results: dict[str, dict[str, str]] = field(default_factory=dict)
    completed_vote_phases: set[str] = field(default_factory=set)
    voting_completed_day: int | None = None
    seer_claims: list[PublicSeerClaim] = field(default_factory=list)
    public_role_claims: list[PublicRoleClaim] = field(default_factory=list)
    winner_reason: str | None = None
    winner_counts: dict[str, int] = field(default_factory=dict)
    # ゲーム中に判明したモデル能力（例：{"model-x": {"claim_role_tool": False}}）。
    model_capabilities: dict[str, dict[str, bool]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.day < 0:
            raise ValueError("Day cannot be negative.")
        if not 0 <= self.discussion_turn <= 3:
            raise ValueError("Discussion turn must be between zero and three.")
        names = [player.name for player in self.players]
        if len(names) != len(set(names)):
            raise ValueError("Player names must be unique.")

    @property
    def alive_players(self) -> list[str]:
        return [player.name for player in self.players if player.is_alive]

    def mark_dead(self, player_name: str) -> None:
        player = next((p for p in self.players if p.name == player_name), None)
        if player is None:
            raise KeyError(f"Unknown player: {player_name}")
        if not player.is_alive:
            raise ValueError(f"Player is already dead: {player_name}")
        player.is_alive = False
