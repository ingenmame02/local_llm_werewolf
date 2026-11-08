import random
import re
from dataclasses import dataclass
from typing import Callable, Protocol

from .models import (
    ActionCandidateSet,
    ClaimRecord,
    GameState,
    LedgerFact,
    Message,
    Phase,
    Player,
    PublicRoleClaim,
    PublicSeerClaim,
    Role,
)
from .discussion_format import parse_discussion_claim, parse_discussion_output

_CLAIMED_ROLE_NAMES = {
    "占い師": "seer",
    "人狼": "wolf",
    "狂人": "madman",
    "村人": "villager",
}


@dataclass(frozen=True)
class _ClaimPublication:
    role_published: bool = False
    result_published: bool = False
    duplicate_suppressed: bool = False


def extract_public_role_claim(content: str) -> str | None:
    role_pattern = "|".join(_CLAIMED_ROLE_NAMES)
    match = re.match(
        rf"^\s*(?:対抗\s*)?"
        rf"(?:(?:私|僕|俺|自分|わたし|ぼく)(?:は|が)?\s*)?"
        rf"(?P<role>{role_pattern})",
        content,
    )
    if match is None:
        return None

    remainder = content[match.end() :]
    explicit_claim = re.match(
        r"\s*(?:CO(?:です|します)?|です|だ|として"
        r"(?:名乗(?:る|ります|りました)|CO(?:です|します)?))"
        r"(?=\s*[:：、,，／/。]|$|\s+(?:対象|判定|結果))",
        remainder,
    )
    claim_separator = re.match(r"\s*(?:[:：、,，／/。]|$)", remainder)
    if explicit_claim is None and claim_separator is None:
        return None
    return _CLAIMED_ROLE_NAMES[match.group("role")]


def is_public_seer_claim(content: str) -> bool:
    return extract_public_role_claim(content) == Role.SEER.value


def extract_public_seer_result(
    content: str, player_names: tuple[str, ...]
) -> tuple[str, str] | None:
    names = sorted(player_names, key=len, reverse=True)
    target = next(
        (
            name
            for name in names
            if re.search(
                rf"(?:占い)?対象\s*[：:=＝]\s*{re.escape(name)}"
                rf"(?:さん)?(?:\s|[／/、。,，]|$)",
                content,
            )
        ),
        None,
    )
    result = re.search(
        r"(?:判定|占い結果|結果)\s*[：:=＝]\s*(人狼|人間)",
        content,
    )
    if target is not None and result is not None:
        return target, "wolf" if result.group(1) == "人狼" else "human"

    for name in names:
        natural_result = re.search(
            rf"{re.escape(name)}(?:さん)?\s*(?:は|が)\s*"
            rf"(?:(?:占い結果|判定)で\s*)?(人狼|人間)"
            rf"(?:と(?:判定(?:された|です|でした)?)?)?",
            content,
        )
        if natural_result is not None:
            return name, "wolf" if natural_result.group(1) == "人狼" else "human"
    return None


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
            candidate_set = ActionCandidateSet(
                action="attack",
                actor=wolves[0].name,
                day=self.state.day,
                candidates=candidates,
                fallback=candidates[0],
            )
            self.state.action_candidate_sets["attack"] = candidate_set
            self.state.snapshot_ledgers("before", "attack", wolves[0].name)
            target = self.actions.choose_attack(
                tuple(wolf.name for wolf in wolves), candidate_set.candidates
            )
            self.state.snapshot_ledgers("after", "attack", wolves[0].name)
            self.state.pending_attack = (
                target if target in candidate_set.candidates else candidate_set.fallback
            )
        else:
            self.state.pending_attack = None

    def run_seer_phase(self) -> None:
        self.state.phase = Phase.NIGHT_SEER
        seers = self._living_by_role(Role.SEER)
        if not seers:
            return

        seer = seers[0]
        candidates = tuple(
            player.name
            for player in self.state.players
            if player.is_alive
            and player.name != seer.name
            and player.name not in self.state.seer_results
        )
        candidate_set = ActionCandidateSet(
            action="seer",
            actor=seer.name,
            day=self.state.day,
            candidates=candidates,
            fallback=candidates[0] if candidates else None,
        )
        self.state.action_candidate_sets["seer"] = candidate_set
        if not candidates:
            return
        self.state.snapshot_ledgers("before", "seer", seer.name)
        target = self.actions.choose_seer_target(
            seer.name,
            candidate_set.candidates,
            tuple(self.state.seer_results.items()),
        )
        self.state.snapshot_ledgers("after", "seer", seer.name)
        if target not in candidate_set.candidates:
            target = candidate_set.fallback
        if target is None:
            return
        target_player = self._player(target)
        result = "wolf" if target_player.role is Role.WOLF else "human"
        self.state.seer_results[target] = result
        self.state.seer_result_history.append((self.state.day, target, result))
        self.state.fact_ledger.confirmed_facts.append(
            LedgerFact(
                day=self.state.day,
                kind="seer_result",
                description=f"{target}: {result}",
                visibility="seer",
                actor=seer.name,
            )
        )

    def resolve_morning(self) -> None:
        self.state.phase = Phase.MORNING
        victim = self.state.pending_attack
        self.state.pending_attack = None
        if victim is not None and self._player(victim).is_alive:
            self.state.mark_dead(victim)
            self.state.attack_deaths.append((self.state.day, victim))
            self.state.fact_ledger.confirmed_facts.append(
                LedgerFact(
                    day=self.state.day,
                    kind="attack_death",
                    description=f"{victim} was attacked during the night.",
                )
            )
            self.state.conversation_log.append(
                Message("system", f"{victim} was attacked during the night.")
            )
        self.check_winner()
        if self.state.winner is None:
            self.state.conversation_log.append(
                Message("system", f"Day {self.state.day} begins.")
            )
            self.state.phase = Phase.DAY_DISCUSSION

    def run_discussion(self) -> None:
        self.state.phase = Phase.DAY_DISCUSSION
        for turn in range(1, 4):
            self.state.discussion_turn = turn
            for player in self.state.players:
                if not player.is_alive:
                    continue
                question_candidates = tuple(
                    name for name in self.state.alive_players if name != player.name
                )
                self.state.action_candidate_sets[f"question:{player.name}"] = (
                    ActionCandidateSet(
                        action="question",
                        actor=player.name,
                        day=self.state.day,
                        candidates=question_candidates,
                        fallback=question_candidates[0] if question_candidates else None,
                    )
                )
                stage_claim = getattr(
                    self.actions, "should_stage_discussion_claim", None
                )
                choose_claim = getattr(
                    self.actions, "choose_discussion_claim", None
                )
                discuss_body = getattr(self.actions, "discuss_body", None)
                has_staged_interface = (
                    callable(stage_claim)
                    and callable(choose_claim)
                    and callable(discuss_body)
                    and getattr(self.actions, "discussion_claim_mode", "legacy")
                    != "legacy"
                )
                if has_staged_interface:
                    if stage_claim(player.name):
                        self.state.snapshot_ledgers(
                            "before", "discussion_claim", player.name
                        )
                        claim_response = choose_claim(
                            player.name,
                            self.state.day,
                            tuple(self.state.alive_players),
                            tuple(self.state.conversation_log),
                        )
                        self.state.snapshot_ledgers(
                            "after", "discussion_claim", player.name
                        )
                        decision = parse_discussion_claim(claim_response)
                        if decision is not None and decision.claim not in {"なし", "保留"}:
                            public_claim = "\n".join(
                                part
                                for part in (decision.claim, decision.result)
                                if part
                            )
                            publication = self._record_public_role_claim(
                                player.name,
                                public_claim,
                                claim_text=decision.claim,
                            )
                            public_lines = []
                            if publication.role_published:
                                public_lines.append(decision.claim)
                            if publication.result_published and decision.result:
                                public_lines.append(f"【結果】{decision.result}")
                            if (
                                publication.duplicate_suppressed
                                and not publication.result_published
                            ):
                                public_lines = []
                            if public_lines:
                                self.state.conversation_log.append(
                                    Message(player.name, "\n".join(public_lines))
                                )
                    self.state.snapshot_ledgers(
                        "before", "discussion_body", player.name
                    )
                    body = discuss_body(
                        player.name,
                        self.state.day,
                        tuple(self.state.alive_players),
                        tuple(self.state.conversation_log),
                    )
                    self.state.snapshot_ledgers(
                        "after", "discussion_body", player.name
                    )
                    if body:
                        self.state.conversation_log.append(Message(player.name, body))
                        self._record_model_inference(player.name, body)
                else:
                    self.state.snapshot_ledgers("before", "discussion", player.name)
                    content = self.actions.discuss(
                        player.name,
                        self.state.day,
                        tuple(self.state.alive_players),
                        tuple(self.state.conversation_log),
                    )
                    self.state.snapshot_ledgers("after", "discussion", player.name)
                    structured_output = parse_discussion_output(content)
                    if structured_output is None:
                        publication = self._record_public_role_claim(
                            player.name, content
                        )
                        body = (
                            self._remove_duplicate_claim(content)
                            if publication.duplicate_suppressed
                            else content
                        )
                        if body:
                            self.state.conversation_log.append(
                                Message(player.name, body)
                            )
                            self._record_model_inference(player.name, body)
                    else:
                        publication = _ClaimPublication()
                        if structured_output.claim not in {"なし", "保留"}:
                            claim_parts = [
                                part
                                for part in (
                                    structured_output.claim,
                                    structured_output.result,
                                    structured_output.body,
                                )
                                if part
                            ]
                            publication = self._record_public_role_claim(
                                player.name,
                                "\n".join(claim_parts),
                                claim_text=structured_output.claim,
                            )
                        body = structured_output.body
                        if publication.duplicate_suppressed:
                            body = self._remove_duplicate_claim(body)
                        if body:
                            self.state.conversation_log.append(
                                Message(player.name, body)
                            )
                            self._record_model_inference(player.name, body)
                if self.state.reasoning_memory_enabled:
                    self.state.refresh_reasoning_memos()
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
        self.state.fact_ledger.confirmed_facts.append(
            LedgerFact(
                day=day,
                kind="execution",
                description=f"{executed} was voted out.",
            )
        )
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

    def _record_public_role_claim(
        self, speaker: str, content: str, claim_text: str | None = None
    ) -> _ClaimPublication:
        claimed_role = extract_public_role_claim(
            claim_text if claim_text is not None else content
        )
        if claimed_role is None:
            return _ClaimPublication()

        names = tuple(player.name for player in self.state.players)
        ledger_claim = self.state.claim_ledger.public_claims.get(speaker)
        prior_role = ledger_claim.claimed_role if ledger_claim is not None else next(
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
        correction = bool(
            re.search(r"(?:訂正|訂正します|誤りでした|撤回します)", content)
        )
        role_published = ledger_claim is None
        duplicate_suppressed = ledger_claim is not None and prior_role == claimed_role
        if prior_role is not None and prior_role != claimed_role:
            if not correction or (
                ledger_claim is not None and not ledger_claim.updateable
            ):
                self._emit_public_claim(
                    "claim_role_conflict",
                    day=self.state.day,
                    claimant=speaker,
                    claimed_role=claimed_role,
                    previous_role=prior_role,
                )
                return _ClaimPublication()
            role_published = True
            if ledger_claim is not None:
                ledger_claim.claimed_role = claimed_role
                ledger_claim.latest_published_day = self.state.day
                ledger_claim.latest_published_turn = self.state.discussion_turn
        elif ledger_claim is None:
            ledger_claim = ClaimRecord(
                claimant=speaker,
                claimed_role=claimed_role,
                first_published_day=self.state.day,
                first_published_turn=self.state.discussion_turn,
                latest_published_day=self.state.day,
                latest_published_turn=self.state.discussion_turn,
            )
            self.state.claim_ledger.public_claims[speaker] = ledger_claim
        if duplicate_suppressed:
            self._emit_public_claim(
                "duplicate_claim_suppressed",
                day=self.state.day,
                discussion_turn=self.state.discussion_turn,
                claimant=speaker,
                claimed_role=claimed_role,
            )
        reason_match = re.search(
            r"(?:理由|根拠)\s*[：:=＝]\s*([^、,，。\n]+)", content
        )
        vote_match = re.search(
            r"(?:投票提案|次の提案)\s*[：:=＝]\s*([^。\n]+)", content
        )
        if role_published:
            self.state.public_role_claims.append(
                PublicRoleClaim(
                    day=self.state.day,
                    claimant=speaker,
                    claimed_role=claimed_role,
                    reason=(
                        reason_match.group(1).strip()
                        if reason_match
                        else "訂正"
                        if correction
                        else None
                    ),
                    vote_proposal=vote_match.group(1).strip() if vote_match else None,
                )
            )
            self.state.fact_ledger.public_claims.append(
                self.state.public_role_claims[-1]
            )
            self._emit_public_claim(
                "claim_role_corrected"
                if correction and prior_role is not None
                else "claim_role_published",
                day=self.state.day,
                claimant=speaker,
                claimed_role=claimed_role,
                source="text_compatibility",
            )

        if claimed_role != Role.SEER.value:
            return _ClaimPublication(
                role_published=role_published,
                duplicate_suppressed=duplicate_suppressed,
            )

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
            return _ClaimPublication(
                role_published=role_published,
                duplicate_suppressed=duplicate_suppressed,
            )
        target, claimed_result = result
        if target == speaker:
            self._emit_public_claim(
                "illegal_seer_claim_target",
                day=self.state.day,
                claimant=speaker,
                target=target,
                reason="self_target",
            )
            return _ClaimPublication(
                role_published=role_published,
                duplicate_suppressed=duplicate_suppressed,
            )
        result_key = (speaker, target)
        previous_result = self.state.claim_ledger.public_seer_results.get(result_key)
        result_published = previous_result is None or (
            previous_result != claimed_result and correction
        )
        if previous_result is not None and previous_result != claimed_result and not correction:
            self._emit_public_claim(
                "claim_role_conflict",
                day=self.state.day,
                claimant=speaker,
                claimed_role=claimed_role,
                target=target,
                claimed_result=claimed_result,
                previous_result=previous_result,
            )
            return _ClaimPublication(
                role_published=role_published,
                duplicate_suppressed=duplicate_suppressed,
            )
        if not result_published:
            return _ClaimPublication(
                role_published=role_published,
                duplicate_suppressed=duplicate_suppressed,
            )
        self.state.claim_ledger.public_seer_results[result_key] = claimed_result
        if ledger_claim is not None:
            ledger_claim.latest_published_day = self.state.day
            ledger_claim.latest_published_turn = self.state.discussion_turn
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
                reason=(
                    reason_match.group(1).strip()
                    if reason_match
                    else "訂正"
                    if correction
                    else None
                ),
                vote_proposal=vote_match.group(1).strip() if vote_match else None,
            )
        )
        self.state.fact_ledger.public_claims.append(
            self.state.public_role_claims[-1]
        )
        self._emit_public_claim(
            "seer_result_corrected"
            if correction and previous_result is not None
            else "seer_result_claim",
            day=self.state.day,
            claimant=speaker,
            target=target,
            claimed_result=claimed_result,
        )
        return _ClaimPublication(
            role_published=role_published,
            result_published=True,
            duplicate_suppressed=duplicate_suppressed,
        )

    def _remove_duplicate_claim(self, content: str) -> str:
        role = extract_public_role_claim(content)
        if role is None:
            return content.strip()
        lines = content.splitlines()
        if len(lines) > 1 and extract_public_role_claim(lines[0]) == role:
            return "\n".join(lines[1:]).strip()
        sentence = re.search(r"[。！？!?]", content)
        if sentence is not None and extract_public_role_claim(
            content[: sentence.end()]
        ) == role:
            return content[sentence.end() :].strip()
        return ""

    def _record_model_inference(self, speaker: str, content: str) -> None:
        if content.strip() and content.strip() != "パス":
            self.state.fact_ledger.model_inferences.append(
                LedgerFact(
                    day=self.state.day,
                    kind="unverified_model_statement",
                    description=content.strip(),
                    actor=speaker,
                )
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
            candidate_set = ActionCandidateSet(
                action=f"vote:{ballot_id}",
                actor=voter,
                day=day,
                candidates=eligible,
                fallback=eligible[0],
            )
            self.state.action_candidate_sets[f"vote:{ballot_id}:{voter}"] = (
                candidate_set
            )
            self._emit_voting_event(
                "vote_started",
                voting_phase_id=ballot_id,
                day=day,
                voter=voter,
                attempt=vote_attempt,
            )
            self.state.snapshot_ledgers("before", candidate_set.action, voter)
            choice = self.actions.vote(
                voter, candidate_set.candidates, self.state.day, runoff
            )
            self.state.snapshot_ledgers("after", candidate_set.action, voter)
            selected = (
                choice
                if choice in candidate_set.candidates
                else candidate_set.fallback
            )
            if selected is None:
                raise RuntimeError(f"No deterministic vote fallback for {voter}.")
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
        for voter, target in votes.items():
            self.state.fact_ledger.confirmed_facts.append(
                LedgerFact(
                    day=day,
                    kind="vote",
                    description=f"{voter} voted for {target}.",
                )
            )
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
