import random
import re
from typing import Callable, Protocol

from .models import (
    GameState,
    Message,
    Phase,
    Player,
    PublicRoleClaim,
    PublicSeerClaim,
    Role,
)

_CLAIMED_ROLE_NAMES = {
    "占い師": "seer",
    "人狼": "wolf",
    "狂人": "madman",
    "村人": "villager",
}


def extract_public_role_claim(content: str) -> str | None:
    role_pattern = "|".join(_CLAIMED_ROLE_NAMES)
    match = re.search(
        rf"^\s*(?:(?:私|僕|俺|自分|わたし|ぼく)(?:は|が)?\s*)?"
        rf"({role_pattern})(?:\s*CO(?:です)?|\s*(?:です|だ|として)?)"
        rf"(?=[。\s、,，／/]|$)",
        content,
    )
    return _CLAIMED_ROLE_NAMES[match.group(1)] if match is not None else None


def is_public_seer_claim(content: str) -> bool:
    return extract_public_role_claim(content) == Role.SEER.value


def extract_public_seer_result(
    content: str, player_names: tuple[str, ...]
) -> tuple[str, str] | None:
    target = next(
        (
            name
            for name in sorted(player_names, key=len, reverse=True)
            if re.search(
                rf"(?:占い)?対象\s*[：:]\s*{re.escape(name)}(?:さん)?(?:\s|[／/、。]|$)",
                content,
            )
        ),
        None,
    )
    result = re.search(r"(?:判定|占い結果|結果)\s*[：:]\s*(人狼|人間)", content)
    if target is None or result is None:
        return None
    return target, "wolf" if result.group(1) == "人狼" else "human"


class ActionPolicy(Protocol):
    def wolf_discuss(
        self, wolf: str, partner: str, history: tuple[Message, ...]
    ) -> str: ...

    def choose_attack(self, wolves: tuple[str, ...], candidates: tuple[str, ...]) -> str: ...

    def choose_seer_target(
        self, seer: str, candidates: tuple[str, ...], results: tuple[tuple[str, str], ...]
    ) -> str: ...

    def discuss(
        self, player: str, day: int, alive: tuple[str, ...], history: tuple[Message, ...]
    ) -> str: ...

    def vote(
        self, voter: str, candidates: tuple[str, ...], day: int, runoff: bool
    ) -> str: ...


class MockActions:
    """Deterministic policy for exercising the game without an LLM."""

    def wolf_discuss(
        self, wolf: str, partner: str, history: tuple[Message, ...]
    ) -> str:
        return f"{wolf} confers with {partner}."

    def choose_attack(self, wolves: tuple[str, ...], candidates: tuple[str, ...]) -> str:
        return candidates[0]

    def choose_seer_target(
        self, seer: str, candidates: tuple[str, ...], results: tuple[tuple[str, str], ...]
    ) -> str:
        return candidates[0]

    def discuss(
        self, player: str, day: int, alive: tuple[str, ...], history: tuple[Message, ...]
    ) -> str:
        return f"Day {day}: {player} speaks."

    def vote(
        self, voter: str, candidates: tuple[str, ...], day: int, runoff: bool
    ) -> str:
        eligible = tuple(candidate for candidate in candidates if candidate != voter)
        if not eligible:
            raise ValueError(f"No eligible vote targets for {voter}.")
        return eligible[0]


class GameEngine:
    def __init__(
        self,
        state: GameState,
        actions: ActionPolicy,
        seed: int | None = None,
        event_callback: Callable[[dict[str, object]], None] | None = None,
    ):
        self.state = state
        self.actions = actions
        self._rng = random.Random(state.seed if seed is None else seed)
        self.event_callback = event_callback

    def run(self) -> GameState:
        self.check_winner()
        while self.state.winner is None:
            self.run_round()
        return self.state

    def run_round(self) -> None:
        if self.state.winner is not None or self.state.phase is Phase.ENDED:
            raise RuntimeError("Cannot run a round after the game has ended.")
        self.run_wolf_phase()
        if self.state.winner is not None:
            return
        self.run_seer_phase()
        self.resolve_morning()
        if self.state.winner is not None:
            return
        self.run_discussion()
        self.run_voting()

    def run_wolf_phase(self) -> None:
        self.state.phase = Phase.NIGHT_WOLF
        wolves = self._living_by_role(Role.WOLF)
        if not wolves:
            self.check_winner()
            return
        if self.state.day == 1:
            self.state.pending_attack = None
            return

        if len(wolves) == 2:
            for wolf in wolves:
                partner = next(player for player in wolves if player is not wolf)
                content = self.actions.wolf_discuss(
                    wolf.name, partner.name, tuple(self.state.wolf_chat_log)
                )
                self.state.wolf_chat_log.append(Message(wolf.name, content))

        candidates = tuple(
            player.name
            for player in self.state.players
            if player.is_alive and player.role is not Role.WOLF
        )
        if candidates:
            target = self.actions.choose_attack(
                tuple(wolf.name for wolf in wolves), candidates
            )
            self.state.pending_attack = (
                target if target in candidates else self._rng.choice(candidates)
            )
        else:
            self.state.pending_attack = None

    def run_seer_phase(self) -> None:
        self.state.phase = Phase.NIGHT_SEER
        seers = self._living_by_role(Role.SEER)
        if not seers:
            return

        seer = seers[0]
        candidates = tuple(player.name for player in self.state.players if player.is_alive)
        if not candidates:
            return
        target = self.actions.choose_seer_target(
            seer.name, candidates, tuple(self.state.seer_results.items())
        )
        if target not in candidates:
            target = self._rng.choice(candidates)
        target_player = self._player(target)
        result = "wolf" if target_player.role is Role.WOLF else "human"
        self.state.seer_results[target] = result
        self.state.seer_result_history.append((self.state.day, target, result))

    def resolve_morning(self) -> None:
        self.state.phase = Phase.MORNING
        victim = self.state.pending_attack
        self.state.pending_attack = None
        if victim is not None and self._player(victim).is_alive:
            self.state.mark_dead(victim)
            self.state.attack_deaths.append((self.state.day, victim))
            self.state.conversation_log.append(
                Message("system", f"{victim} was attacked during the night.")
            )
        self.check_winner()
        if self.state.winner is None:
            self.state.phase = Phase.DAY_DISCUSSION

    def run_discussion(self) -> None:
        self.state.phase = Phase.DAY_DISCUSSION
        for turn in range(1, 4):
            self.state.discussion_turn = turn
            for player in self.state.players:
                if not player.is_alive:
                    continue
                content = self.actions.discuss(
                    player.name,
                    self.state.day,
                    tuple(self.state.alive_players),
                    tuple(self.state.conversation_log),
                )
                self.state.conversation_log.append(Message(player.name, content))
                self._record_public_role_claim(player.name, content)
        self.state.phase = Phase.VOTING

    def run_voting(self) -> None:
        if (
            self.state.phase is not Phase.VOTING
            or self.state.voting_completed_day == self.state.day
        ):
            return

        alive = tuple(self.state.alive_players)
        if not alive:
            self.check_winner()
            return

        day = self.state.day
        initial_phase_id = f"day-{day}-initial"
        runoff_phase_id = f"day-{day}-runoff"
        initial_already_completed = (
            initial_phase_id in self.state.completed_vote_phases
        )
        first_votes = self._collect_votes(
            alive, alive, runoff=False, phase_id=initial_phase_id, attempt=1
        )
        tied = self._top_voted(first_votes)
        final_votes = first_votes
        if not initial_already_completed:
            self.state.vote_history.append((day, False, dict(first_votes)))
        if len(tied) > 1:
            runoff_already_completed = (
                runoff_phase_id in self.state.completed_vote_phases
            )
            final_votes = self._collect_votes(
                alive,
                tuple(tied),
                runoff=True,
                phase_id=runoff_phase_id,
                attempt=2,
            )
            if not runoff_already_completed:
                self.state.vote_history.append((day, True, dict(final_votes)))
            tied = self._top_voted(final_votes)
        executed = tied[0] if len(tied) == 1 else self._rng.choice(tied)

        self.state.votes = final_votes
        self.state.voting_completed_day = day
        self.state.mark_dead(executed)
        self.state.executions.append((day, executed))
        self.state.conversation_log.append(
            Message("system", f"{executed} was voted out.")
        )
        self.check_winner()
        if self.state.winner is None:
            self.state.day += 1
            self.state.phase = Phase.NIGHT_WOLF
        self._emit_voting_event(
            "voting_phase_ended",
            voting_phase_id=f"day-{day}",
            day=day,
            executed=executed,
            next_phase=self.state.phase.value,
        )

    def check_winner(self) -> str | None:
        living_wolves = sum(
            player.is_alive and player.role is Role.WOLF for player in self.state.players
        )
        living_madmen = sum(
            player.is_alive and player.role is Role.MADMAN
            for player in self.state.players
        )
        living_village = sum(
            player.is_alive and player.role in (Role.SEER, Role.VILLAGER)
            for player in self.state.players
        )
        self.state.winner_counts = {
            "living_wolves": living_wolves,
            "living_madmen": living_madmen,
            "living_village": living_village,
        }
        previous_winner = self.state.winner
        if living_wolves == 0:
            self.state.winner = "village"
            self.state.winner_reason = "no_living_wolves"
        else:
            if living_wolves >= living_village:
                self.state.winner = "werewolf"
                self.state.winner_reason = "living_wolves_reached_village_parity"

        if self.state.winner is not None:
            self.state.phase = Phase.ENDED
            if previous_winner is None and self.event_callback is not None:
                self.event_callback(
                    {
                        "type": "winner_decided",
                        "winner": self.state.winner,
                        "reason": self.state.winner_reason,
                        "counts": dict(self.state.winner_counts),
                    }
                )
        return self.state.winner

    def _record_public_role_claim(self, speaker: str, content: str) -> None:
        claimed_role = extract_public_role_claim(content)
        if claimed_role is None:
            return

        names = tuple(player.name for player in self.state.players)
        existing = next(
            (
                claim
                for claim in self.state.public_role_claims
                if claim.claimant == speaker and claim.claimed_role == claimed_role
            ),
            None,
        )
        prior_role = next(
            (
                claim.claimed_role
                for claim in self.state.public_role_claims
                if claim.claimant == speaker
            ),
            (
                Role.SEER.value
                if any(claim.claimant == speaker for claim in self.state.seer_claims)
                else None
            ),
        )
        if prior_role is not None and prior_role != claimed_role:
            self._emit_public_claim(
                "claim_role_conflict",
                day=self.state.day,
                claimant=speaker,
                claimed_role=claimed_role,
                previous_role=prior_role,
            )
            return
        reason_match = re.search(r"(?:理由|根拠)\s*[：:]\s*([^。\n]+)", content)
        vote_match = re.search(
            r"(?:投票提案|次の提案)\s*[：:]\s*([^。\n]+)", content
        )
        if existing is None:
            self.state.public_role_claims.append(
                PublicRoleClaim(
                    day=self.state.day,
                    claimant=speaker,
                    claimed_role=claimed_role,
                    reason=reason_match.group(1).strip() if reason_match else None,
                    vote_proposal=vote_match.group(1).strip() if vote_match else None,
                )
            )
            self._emit_public_claim(
                "claim_role_published",
                day=self.state.day,
                claimant=speaker,
                claimed_role=claimed_role,
                source="text_compatibility",
            )

        if claimed_role != Role.SEER.value:
            return

        has_claimed = any(
            claim.claimant == speaker for claim in self.state.seer_claims
        )
        if not has_claimed:
            self.state.seer_claims.append(
                PublicSeerClaim(day=self.state.day, claimant=speaker)
            )
            self._emit_public_claim(
                "seer_claim",
                day=self.state.day,
                claimant=speaker,
            )

        result = extract_public_seer_result(content, names)
        if result is None:
            return
        target, claimed_result = result
        previous_result = next(
            (
                claim.result
                for claim in self.state.public_role_claims
                if claim.claimant == speaker
                and claim.claimed_role == Role.SEER.value
                and claim.target == target
                and claim.result is not None
            ),
            next(
                (
                    claim.result
                    for claim in self.state.seer_claims
                    if claim.claimant == speaker
                    and claim.target == target
                    and claim.result is not None
                ),
                None,
            ),
        )
        if previous_result is not None and previous_result != claimed_result:
            self._emit_public_claim(
                "claim_role_conflict",
                day=self.state.day,
                claimant=speaker,
                claimed_role=claimed_role,
                target=target,
                claimed_result=claimed_result,
                previous_result=previous_result,
            )
            return
        if any(
            claim.claimant == speaker
            and claim.target == target
            and claim.result == claimed_result
            for claim in self.state.seer_claims
        ):
            return
        self.state.seer_claims.append(
            PublicSeerClaim(
                day=self.state.day,
                claimant=speaker,
                target=target,
                result=claimed_result,
            )
        )
        self.state.public_role_claims.append(
            PublicRoleClaim(
                day=self.state.day,
                claimant=speaker,
                claimed_role=Role.SEER.value,
                target=target,
                result=claimed_result,
                reason=reason_match.group(1).strip() if reason_match else None,
                vote_proposal=vote_match.group(1).strip() if vote_match else None,
            )
        )
        self._emit_public_claim(
            "seer_result_claim",
            day=self.state.day,
            claimant=speaker,
            target=target,
            claimed_result=claimed_result,
        )

    def _emit_public_claim(self, event_type: str, **details: object) -> None:
        if self.event_callback is not None:
            self.event_callback({"type": event_type, **details})

    def _collect_votes(
        self,
        voters: tuple[str, ...],
        candidates: tuple[str, ...],
        runoff: bool,
        phase_id: str | None = None,
        attempt: int | None = None,
    ) -> dict[str, str]:
        day = self.state.day
        ballot_id = phase_id or f"day-{day}-{'runoff' if runoff else 'initial'}"
        vote_attempt = attempt or (2 if runoff else 1)
        votes = self.state.vote_phase_results.setdefault(ballot_id, {})
        if ballot_id in self.state.completed_vote_phases:
            return dict(votes)
        self._emit_voting_event(
            "voting_phase_started",
            voting_phase_id=ballot_id,
            day=day,
            attempt=vote_attempt,
            voters=list(voters),
            candidates=list(candidates),
        )
        living = set(self.state.alive_players)
        requested_voters = set(voters)
        if len(requested_voters) != len(voters):
            raise ValueError(f"Duplicate voters supplied for {ballot_id}.")
        if set(votes) - requested_voters:
            raise ValueError(f"Unexpected recorded voters for {ballot_id}.")
        for voter in voters:
            if voter in votes:
                continue
            eligible = tuple(
                candidate
                for candidate in candidates
                if candidate != voter and candidate in living
            )
            if not eligible:
                raise ValueError(f"No eligible living vote targets for {voter}.")
            self._emit_voting_event(
                "vote_started",
                voting_phase_id=ballot_id,
                day=day,
                voter=voter,
                attempt=vote_attempt,
            )
            choice = self.actions.vote(voter, candidates, self.state.day, runoff)
            selected = choice if choice in eligible else self._rng.choice(eligible)
            votes[voter] = selected
            self._emit_voting_event(
                "vote_completed",
                voting_phase_id=ballot_id,
                day=day,
                voter=voter,
                attempt=vote_attempt,
                target=selected,
                fallback_used=choice not in eligible,
            )
        self.state.completed_vote_phases.add(ballot_id)
        self._emit_voting_event(
            "voting_ballot_completed",
            voting_phase_id=ballot_id,
            day=day,
            attempt=vote_attempt,
            votes=dict(votes),
        )
        return dict(votes)

    def _emit_voting_event(self, event_type: str, **details: object) -> None:
        if self.event_callback is None:
            return
        self.event_callback(
            {"type": event_type, "phase": Phase.VOTING.value, **details}
        )

    @staticmethod
    def _top_voted(votes: dict[str, str]) -> list[str]:
        counts: dict[str, int] = {}
        for target in votes.values():
            counts[target] = counts.get(target, 0) + 1
        maximum = max(counts.values())
        return [target for target, count in counts.items() if count == maximum]

    def _living_by_role(self, role: Role) -> list[Player]:
        return [
            player
            for player in self.state.players
            if player.is_alive and player.role is role
        ]

    def _player(self, name: str) -> Player:
        return next(player for player in self.state.players if player.name == name)
