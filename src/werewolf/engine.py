import random
from typing import Protocol

from .models import GameState, Message, Phase, Player, Role


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
    def __init__(self, state: GameState, actions: ActionPolicy, seed: int | None = None):
        self.state = state
        self.actions = actions
        self._rng = random.Random(state.seed if seed is None else seed)

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
        self.state.phase = Phase.VOTING

    def run_voting(self) -> None:
        self.state.phase = Phase.VOTING
        alive = tuple(self.state.alive_players)
        if not alive:
            self.check_winner()
            return

        first_votes = self._collect_votes(alive, alive, runoff=False)
        tied = self._top_voted(first_votes)
        final_votes = first_votes
        self.state.vote_history.append((self.state.day, False, dict(first_votes)))
        if len(tied) > 1:
            final_votes = self._collect_votes(alive, tied, runoff=True)
            self.state.vote_history.append((self.state.day, True, dict(final_votes)))
            tied = self._top_voted(final_votes)
        executed = tied[0] if len(tied) == 1 else self._rng.choice(tied)

        self.state.votes = final_votes
        self.state.mark_dead(executed)
        self.state.executions.append((self.state.day, executed))
        self.state.conversation_log.append(
            Message("system", f"{executed} was voted out.")
        )
        self.check_winner()
        if self.state.winner is None:
            self.state.day += 1
            self.state.phase = Phase.NIGHT_WOLF

    def check_winner(self) -> str | None:
        living_wolves = sum(
            player.is_alive and player.role is Role.WOLF for player in self.state.players
        )
        if living_wolves == 0:
            self.state.winner = "village"
        else:
            wolf_faction = sum(
                player.is_alive and player.role in (Role.WOLF, Role.MADMAN)
                for player in self.state.players
            )
            village_faction = sum(
                player.is_alive and player.role in (Role.SEER, Role.VILLAGER)
                for player in self.state.players
            )
            if wolf_faction >= village_faction:
                self.state.winner = "werewolf"

        if self.state.winner is not None:
            self.state.phase = Phase.ENDED
        return self.state.winner

    def _collect_votes(
        self, voters: tuple[str, ...], candidates: tuple[str, ...], runoff: bool
    ) -> dict[str, str]:
        votes: dict[str, str] = {}
        living = set(self.state.alive_players)
        for voter in voters:
            eligible = tuple(
                candidate
                for candidate in candidates
                if candidate != voter and candidate in living
            )
            if not eligible:
                raise ValueError(f"No eligible living vote targets for {voter}.")
            choice = self.actions.vote(voter, candidates, self.state.day, runoff)
            votes[voter] = choice if choice in eligible else self._rng.choice(eligible)
        return votes

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
