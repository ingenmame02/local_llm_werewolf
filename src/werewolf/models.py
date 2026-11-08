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
    max_num_ctx: int | None = None
    recommended_num_ctx: int | None = None
    context_strategy: str | None = None

    def __post_init__(self) -> None:
        if self.num_ctx is not None and (
            not isinstance(self.num_ctx, int)
            or isinstance(self.num_ctx, bool)
            or self.num_ctx <= 0
        ):
            raise ValueError("Player num_ctx must be a positive integer.")
        for key, context_value in (
            ("max_num_ctx", self.max_num_ctx),
            ("recommended_num_ctx", self.recommended_num_ctx),
        ):
            if context_value is not None and (
                not isinstance(context_value, int)
                or isinstance(context_value, bool)
                or context_value <= 0
            ):
                raise ValueError(f"Player {key} must be a positive integer.")
        if (
            self.max_num_ctx is not None
            and self.recommended_num_ctx is not None
            and self.recommended_num_ctx > self.max_num_ctx
        ):
            raise ValueError(
                "Player recommended_num_ctx cannot exceed max_num_ctx."
            )
        if self.context_strategy is not None and self.context_strategy not in {
            "full_transcript",
            "structured_ledger",
            "hybrid_retrieval",
        }:
            raise ValueError("Player context_strategy is not supported.")
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
class ClaimRecord:
    claimant: str
    claimed_role: str
    first_published_day: int
    first_published_turn: int
    latest_published_day: int
    latest_published_turn: int
    updateable: bool = True


@dataclass
class ClaimLedger:
    public_claims: dict[str, ClaimRecord] = field(default_factory=dict)
    public_seer_results: dict[tuple[str, str], str] = field(default_factory=dict)


@dataclass(frozen=True)
class LedgerFact:
    day: int
    kind: str
    description: str
    visibility: str = "public"
    actor: str | None = None


@dataclass
class FactLedger:
    confirmed_facts: list[LedgerFact] = field(default_factory=list)
    public_claims: list[PublicRoleClaim] = field(default_factory=list)
    model_inferences: list[LedgerFact] = field(default_factory=list)


@dataclass
class ReasoningMemo:
    confirmed_facts: list[str] = field(default_factory=list)
    public_claims: list[str] = field(default_factory=list)
    suspicion_ranking: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    next_targets: list[str] = field(default_factory=list)
    private_role_facts: list[str] = field(default_factory=list)
    updates: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ActionCandidateSet:
    action: str
    actor: str
    day: int
    candidates: tuple[str, ...]
    fallback: str | None = None

    def __post_init__(self) -> None:
        if len(self.candidates) != len(set(self.candidates)):
            raise ValueError("Action candidates must be unique.")
        if self.fallback is not None and self.fallback not in self.candidates:
            raise ValueError("Action fallback must be one of the candidates.")


@dataclass(frozen=True)
class LedgerSnapshot:
    boundary: str
    action: str
    actor: str
    day: int
    public_claims: tuple[tuple[str, str, int, int], ...]
    public_seer_results: tuple[tuple[str, str, str], ...]
    seer_results: tuple[tuple[str, str], ...]
    dead_players: tuple[str, ...]
    votes: tuple[tuple[str, str], ...]


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
    claim_ledger: ClaimLedger = field(default_factory=ClaimLedger)
    fact_ledger: FactLedger = field(default_factory=FactLedger)
    action_candidate_sets: dict[str, ActionCandidateSet] = field(default_factory=dict)
    discussion_claim_checks: set[tuple[int, str]] = field(default_factory=set)
    ledger_snapshots: list[LedgerSnapshot] = field(default_factory=list)
    # ゲーム中に判明したモデル能力（例：{"model-x": {"claim_role_tool": False}}）。
    model_capabilities: dict[str, dict[str, bool]] = field(default_factory=dict)
    reasoning_memos: dict[str, ReasoningMemo] = field(default_factory=dict)
    reasoning_memory_enabled: bool = True

    def __post_init__(self) -> None:
        if self.day < 0:
            raise ValueError("Day cannot be negative.")
        if not 0 <= self.discussion_turn <= 3:
            raise ValueError("Discussion turn must be between zero and three.")
        names = [player.name for player in self.players]
        if len(names) != len(set(names)):
            raise ValueError("Player names must be unique.")
        for claim in self.public_role_claims:
            self.claim_ledger.public_claims.setdefault(
                claim.claimant,
                ClaimRecord(
                    claimant=claim.claimant,
                    claimed_role=claim.claimed_role,
                    first_published_day=claim.day,
                    first_published_turn=0,
                    latest_published_day=claim.day,
                    latest_published_turn=0,
                ),
            )
            if claim.target is not None and claim.result is not None:
                self.claim_ledger.public_seer_results[
                    (claim.claimant, claim.target)
                ] = claim.result
        for claim in self.seer_claims:
            if claim.target is not None and claim.result is not None:
                self.claim_ledger.public_seer_results[
                    (claim.claimant, claim.target)
                ] = claim.result

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

    def refresh_reasoning_memos(self) -> None:
        """Refresh actor-specific notes using public state and actor-owned secrets only."""
        public_facts = [
            f"[fact:{fact.day}:{index}] {fact.description}"
            for index, fact in enumerate(self.fact_ledger.confirmed_facts, 1)
            if fact.visibility == "public"
        ]
        public_facts.extend(
            f"[vote:{day}:{'runoff' if runoff else 'initial'}:{voter}] "
            f"{voter} voted for {target}"
            for day, runoff, votes in self.vote_history
            for voter, target in votes.items()
        )
        public_facts.extend(
            f"[death:attack:{day}:{player}] {player} died in an attack"
            for day, player in self.attack_deaths
        )
        public_facts.extend(
            f"[death:execution:{day}:{player}] {player} was executed"
            for day, player in self.executions
        )
        public_claims = [
            f"[claim:{index}] Day {claim.day}: {claim.claimant} publicly claimed "
            f"{claim.claimed_role}"
            + (
                f"; {claim.target}={claim.result}"
                if claim.target is not None and claim.result is not None
                else ""
            )
            + " (unverified)"
            for index, claim in enumerate(self.public_role_claims, 1)
        ]
        public_claims.extend(
            f"[seer_claim:{index}] Day {claim.day}: {claim.claimant} publicly "
            f"claimed {claim.target}={claim.result} (unverified)"
            for index, claim in enumerate(self.seer_claims, 1)
            if claim.target is not None and claim.result is not None
        )
        votes_by_target: dict[str, list[str]] = {}
        for day, runoff, votes in self.vote_history:
            ballot = "runoff" if runoff else "initial"
            for voter, target in votes.items():
                votes_by_target.setdefault(target, []).append(
                    f"[vote:{day}:{ballot}:{voter}]"
                )
        suspicion_ranking = [
            f"{target}: {len(sources)} recorded vote(s); evidence {', '.join(sources)}"
            for target, sources in sorted(
                votes_by_target.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
        ]
        speech_records: list[tuple[int, Message]] = []
        for message in self.conversation_log:
            if message.speaker != "system":
                speech_records.append((len(speech_records) + 1, message))

        for player in self.players:
            unresolved: list[str] = []
            for position, (speech_id, message) in enumerate(speech_records):
                if message.speaker == player.name:
                    continue
                is_question = (
                    "?" in message.content
                    or "？" in message.content
                    or any(
                        cue in message.content
                        for cue in ("質問", "聞きたい", "確認したい", "教えて")
                    )
                )
                addresses_player = (
                    player.name in message.content
                )
                later_reply = any(
                    later.speaker == player.name
                    for _, later in speech_records[position + 1 :]
                )
                if is_question and addresses_player and not later_reply:
                    unresolved.append(
                        f"[speech:{speech_id}] {message.speaker}からの質問（未回答）"
                    )

            next_targets = [
                target
                for target in (
                    item.split(":", 1)[0] for item in suspicion_ranking
                )
                if target != player.name and target in self.alive_players
            ][:3]
            if not next_targets:
                next_targets = [
                    name for name in self.alive_players if name != player.name
                ][:3]

            private_facts: list[str] = []
            if player.role is Role.SEER:
                private_facts = [
                    f"[private_seer:{index}] {target}={result}"
                    for index, (target, result) in enumerate(
                        self.seer_results.items(), 1
                    )
                ]
            elif player.role is Role.WOLF and player.partner is not None:
                private_facts = [f"[private_wolf_partner] {player.partner}"]

            previous = self.reasoning_memos.get(player.name)
            current_parts = {
                "confirmed_facts": public_facts,
                "public_claims": public_claims,
                "suspicion_ranking": suspicion_ranking,
                "unresolved_questions": unresolved,
                "next_targets": next_targets,
                "private_role_facts": private_facts,
            }
            updates = (
                ["推理メモを初期化"]
                if previous is None
                else [
                    f"{key}を更新"
                    for key, value in current_parts.items()
                    if getattr(previous, key) != value
                ]
            )
            self.reasoning_memos[player.name] = ReasoningMemo(
                **current_parts,
                updates=updates or ["前回から変更なし"],
            )

    def snapshot_ledgers(
        self, boundary: str, action: str, actor: str
    ) -> LedgerSnapshot:
        snapshot = LedgerSnapshot(
            boundary=boundary,
            action=action,
            actor=actor,
            day=self.day,
            public_claims=tuple(
                sorted(
                    (
                        claim.claimant,
                        claim.claimed_role,
                        claim.first_published_day,
                        claim.first_published_turn,
                    )
                    for claim in self.claim_ledger.public_claims.values()
                )
            ),
            public_seer_results=tuple(
                sorted(
                    (claimant, target, result)
                    for (claimant, target), result
                    in self.claim_ledger.public_seer_results.items()
                )
            ),
            seer_results=tuple(sorted(self.seer_results.items())),
            dead_players=tuple(
                player.name for player in self.players if not player.is_alive
            ),
            votes=tuple(sorted(self.votes.items())),
        )
        self.ledger_snapshots.append(snapshot)
        return snapshot
