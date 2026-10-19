import pytest

from werewolf import (
    GameEngine,
    GameState,
    MockActions,
    Phase,
    Player,
    Role,
)
from werewolf.engine import extract_public_role_claim, extract_public_seer_result


def make_state(*roles: Role, seed: int = 11) -> GameState:
    players = [
        Player(name=f"P{index}", model=f"model-{index}", role=role)
        for index, role in enumerate(roles)
    ]
    return GameState(players=players, seed=seed)


class ScriptedActions(MockActions):
    def __init__(self) -> None:
        self.wolf_discussions = 0
        self.seer_actions = 0
        self.runoff_voters: list[str] = []

    def wolf_discuss(self, wolf: str, partner: str, history) -> str:
        self.wolf_discussions += 1
        return super().wolf_discuss(wolf, partner, history)

    def choose_attack(self, wolves, candidates) -> str:
        return candidates[0]

    def choose_seer_target(self, seer, candidates, results) -> str:
        self.seer_actions += 1
        return candidates[0]

    def vote(self, voter, candidates, day, runoff) -> str:
        if runoff:
            self.runoff_voters.append(voter)
            return "P0" if voter == "P2" else "P2"
        return {
            "P0": "P1",
            "P1": "P0",
            "P2": "P3",
            "P3": "P2",
        }[voter]


class AlwaysVoteWolf(MockActions):
    def vote(self, voter, candidates, day, runoff) -> str:
        return "P1" if voter == "P0" else "P0"


class AlwaysVoteForSelf(MockActions):
    def vote(self, voter, candidates, day, runoff) -> str:
        return voter


def test_village_wins_when_the_last_wolf_is_executed() -> None:
    state = make_state(Role.WOLF, Role.SEER, Role.VILLAGER)
    state.phase = Phase.VOTING

    GameEngine(state, AlwaysVoteWolf()).run_voting()

    assert state.winner == "village"
    assert state.phase is Phase.ENDED
    assert not state.players[0].is_alive


def test_madman_is_excluded_from_winner_parity_counts() -> None:
    state = make_state(Role.WOLF, Role.MADMAN, Role.VILLAGER, Role.VILLAGER)

    winner = GameEngine(state, MockActions()).check_winner()

    assert winner is None
    assert state.winner_counts == {
        "living_wolves": 1,
        "living_madmen": 1,
        "living_village": 2,
    }
    assert state.winner_reason is None


def test_werewolf_wins_at_parity_without_counting_madmen() -> None:
    state = make_state(Role.WOLF, Role.MADMAN, Role.VILLAGER)

    winner = GameEngine(state, MockActions()).check_winner()

    assert winner == "werewolf"
    assert state.winner_reason == "living_wolves_reached_village_parity"
    assert state.winner_counts == {
        "living_wolves": 1,
        "living_madmen": 1,
        "living_village": 1,
    }
    assert state.phase is Phase.ENDED


def test_village_wins_when_no_wolves_remain_even_if_madman_is_alive() -> None:
    state = make_state(Role.MADMAN, Role.VILLAGER)

    winner = GameEngine(state, MockActions()).check_winner()

    assert winner == "village"
    assert state.winner_reason == "no_living_wolves"


def test_first_night_skips_wolf_discussion_and_attack_but_keeps_seer_action() -> None:
    class FirstNightActions(MockActions):
        wolf_discussions = 0
        attacks = 0
        seer_actions = 0

        def wolf_discuss(self, wolf, partner, history):
            self.wolf_discussions += 1
            return super().wolf_discuss(wolf, partner, history)

        def choose_attack(self, wolves, candidates):
            self.attacks += 1
            return super().choose_attack(wolves, candidates)

        def choose_seer_target(self, seer, candidates, results):
            self.seer_actions += 1
            return "P2"

    state = make_state(
        Role.WOLF, Role.WOLF, Role.SEER, Role.VILLAGER, Role.VILLAGER
    )
    actions = FirstNightActions()
    engine = GameEngine(state, actions)

    engine.run_wolf_phase()

    assert state.pending_attack is None
    assert actions.wolf_discussions == 0
    assert actions.attacks == 0

    engine.run_seer_phase()
    engine.resolve_morning()

    assert actions.seer_actions == 1
    assert state.attack_deaths == []
    assert state.day == 1
    assert state.phase is Phase.DAY_DISCUSSION
    assert state.seer_results == {"P2": "human"}


def test_second_night_runs_normal_wolf_attack() -> None:
    class AttackActions(MockActions):
        attacks = 0

        def choose_attack(self, wolves, candidates):
            self.attacks += 1
            return candidates[0]

    state = make_state(Role.WOLF, Role.SEER, Role.VILLAGER)
    state.day = 2
    actions = AttackActions()

    GameEngine(state, actions).run_wolf_phase()

    assert actions.attacks == 1
    assert state.pending_attack == "P1"


def test_discussion_records_public_seer_claim_and_report_without_truth_leak() -> None:
    state = make_state(Role.SEER, Role.WOLF, Role.VILLAGER)

    class ClaimActions(MockActions):
        def discuss(self, player, day, alive, history):
            if player == "P0" and not any(
                message.speaker == player for message in history
            ):
                return (
                    "私は占い師です。対象：P1／判定：人狼／根拠：占い結果。"
                    "次の提案：P1に投票したいです。"
                )
            if player == "P1" and not any(
                message.speaker == player for message in history
            ):
                return (
                    "占い師CO。対象：P2／判定：人間／根拠：占い結果。"
                    "次の提案：P2は一旦保留したいです。"
                )
            return "まだ判断を保留します。"

    events: list[dict[str, object]] = []
    GameEngine(state, ClaimActions(), event_callback=events.append).run_discussion()

    assert [
        (claim.day, claim.claimant, claim.target, claim.result)
        for claim in state.seer_claims
    ] == [
        (1, "P0", None, None),
        (1, "P0", "P1", "wolf"),
        (1, "P1", None, None),
        (1, "P1", "P2", "human"),
    ]
    assert [
        (claim.claimant, claim.claimed_role, claim.target, claim.result)
        for claim in state.public_role_claims
    ] == [
        ("P0", "seer", None, None),
        ("P0", "seer", "P1", "wolf"),
        ("P1", "seer", None, None),
        ("P1", "seer", "P2", "human"),
    ]
    assert [event["type"] for event in events] == [
        "claim_role_published",
        "seer_claim",
        "seer_result_claim",
        "claim_role_published",
        "seer_claim",
        "seer_result_claim",
    ]
    assert events[0]["claimant"] == "P0"
    assert events[2]["target"] == "P1"
    assert events[2]["claimed_result"] == "wolf"
    assert events[3]["claimant"] == "P1"
    assert events[5]["target"] == "P2"
    assert events[5]["claimed_result"] == "human"
    assert all("role" not in event for event in events)


@pytest.mark.parametrize(
    ("content", "expected_role"),
    [
        ("私は占い師", "seer"),
        ("占い師CO：対象=Player 1、判定=人狼", "seer"),
        ("対抗占い師CO：対象=Player 1、判定=人間", "seer"),
        ("占い師として名乗る", "seer"),
        ("狂人CO：理由=対抗します", "madman"),
        ("私は村人です。", "villager"),
    ],
)
def test_public_role_claim_parser_accepts_natural_language_and_template_variants(
    content: str, expected_role: str
) -> None:
    assert extract_public_role_claim(content) == expected_role


def test_structured_seer_report_parser_accepts_equals_delimiters() -> None:
    assert extract_public_seer_result(
        "占い師CO：対象=Player 2、判定=人狼、理由=夜の占い、投票提案=投票",
        ("Player 1", "Player 2"),
    ) == ("Player 2", "wolf")


def test_text_compatible_claim_records_conflicts_without_overwriting_prior_claim() -> None:
    state = make_state(Role.VILLAGER, Role.WOLF, Role.SEER)

    class ConflictingClaimActions(MockActions):
        def discuss(self, player, day, alive, history):
            if player != "P0":
                return "状況を確認します。"
            if not any(message.speaker == player for message in history):
                return "私は占い師です。"
            return "私は村人です。"

    events: list[dict[str, object]] = []
    GameEngine(
        state,
        ConflictingClaimActions(),
        event_callback=events.append,
    ).run_discussion()

    assert [
        claim.claimed_role
        for claim in state.public_role_claims
        if claim.claimant == "P0"
    ] == ["seer"]
    assert any(event["type"] == "claim_role_conflict" for event in events)


def test_madman_is_not_an_attacker_but_can_be_a_target() -> None:
    state = make_state(
        Role.WOLF, Role.WOLF, Role.MADMAN, Role.SEER, Role.VILLAGER, Role.VILLAGER
    )
    state.day = 2

    class RecordingAttackActions(MockActions):
        wolves: tuple[str, ...] = ()
        candidates: tuple[str, ...] = ()

        def choose_attack(self, wolves, candidates) -> str:
            self.wolves = wolves
            self.candidates = candidates
            return candidates[0]

    actions = RecordingAttackActions()
    GameEngine(state, actions).run_wolf_phase()

    assert actions.wolves == ("P0", "P1")
    assert "P2" in actions.candidates
    assert state.pending_attack == "P2"
    assert len(state.wolf_chat_log) == 2


def test_tied_vote_repeats_only_between_tied_candidates() -> None:
    state = make_state(Role.WOLF, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER)
    state.phase = Phase.VOTING
    actions = ScriptedActions()

    GameEngine(state, actions).run_voting()

    assert actions.runoff_voters == ["P0", "P1", "P2", "P3"]
    assert state.players[2].is_alive is False
    assert state.phase is Phase.NIGHT_WOLF
    assert state.votes == {"P0": "P2", "P1": "P2", "P2": "P0", "P3": "P2"}
    assert len(state.vote_history) == 2
    assert state.vote_history[0][1] is False
    assert state.vote_history[1][1] is True
    assert state.executions == [(1, "P2")]


def test_each_voter_is_processed_once_and_completed_voting_is_idempotent() -> None:
    state = GameState(
        players=[
            Player(f"Player {index}", f"model-{index}", role)
            for index, role in enumerate(
                (Role.WOLF, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER,
                 Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, Role.SEER),
                start=1,
            )
        ],
        phase=Phase.VOTING,
        seed=4,
    )

    class CountVotes(MockActions):
        calls: list[tuple[str, bool]] = []

        def vote(self, voter, candidates, day, runoff) -> str:
            self.calls.append((voter, runoff))
            return "Player 1" if voter != "Player 1" else "Player 2"

    actions = CountVotes()
    events: list[dict[str, object]] = []
    engine = GameEngine(state, actions, event_callback=events.append)

    engine.run_voting()
    first_result = (state.day, state.phase, dict(state.votes), len(state.executions))
    engine.run_voting()

    assert [voter for voter, runoff in actions.calls if not runoff] == [
        f"Player {index}" for index in range(1, 9)
    ]
    assert all(not runoff for _, runoff in actions.calls)
    assert len(actions.calls) == 8
    assert state.voting_completed_day == 1
    assert (state.day, state.phase, dict(state.votes), len(state.executions)) == first_result
    assert sum(event["type"] == "voting_phase_ended" for event in events) == 1


def test_still_tied_runoff_is_limited_to_one_vote_per_living_player() -> None:
    state = make_state(Role.WOLF, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER)
    state.phase = Phase.VOTING

    class AlwaysTie(MockActions):
        calls: list[tuple[str, bool]] = []

        def vote(self, voter, candidates, day, runoff) -> str:
            self.calls.append((voter, runoff))
            return "P1" if voter in {"P0", "P2"} else "P2"

    actions = AlwaysTie()
    GameEngine(state, actions).run_voting()

    assert len(actions.calls) == 8
    assert all(actions.calls.count((voter, runoff)) == 1 for voter, runoff in actions.calls)
    assert [runoff for _, runoff in actions.calls].count(True) == 4
    assert len(state.vote_history) == 2
    assert len(state.executions) == 1


def test_self_votes_are_replaced_deterministically_in_initial_and_runoff_votes() -> None:
    def collect_votes(seed: int) -> tuple[dict[str, str], dict[str, str]]:
        state = make_state(
            Role.WOLF, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, seed=seed
        )
        engine = GameEngine(state, AlwaysVoteForSelf())
        first_votes = engine._collect_votes(
            ("P0", "P1", "P2", "P3"),
            ("P0", "P1", "P2", "P3"),
            runoff=False,
        )
        runoff_votes = engine._collect_votes(
            ("P0", "P1", "P2", "P3"),
            ("P1", "P2"),
            runoff=True,
        )
        return first_votes, runoff_votes

    first_votes, runoff_votes = collect_votes(seed=23)
    assert (first_votes, runoff_votes) == collect_votes(seed=23)
    assert all(voter != target for voter, target in first_votes.items())
    assert all(voter != target for voter, target in runoff_votes.items())
    assert set(runoff_votes.values()) <= {"P1", "P2"}


def test_mock_actions_never_votes_for_self() -> None:
    assert MockActions().vote("P0", ("P0", "P1"), day=1, runoff=False) == "P1"


def test_discussion_pass_is_recorded_as_a_public_conversation_turn() -> None:
    state = make_state(Role.WOLF, Role.SEER, Role.VILLAGER)
    discussion_turns: list[int] = []

    class PassDiscussion(MockActions):
        def discuss(self, player, day, alive, history) -> str:
            discussion_turns.append(state.discussion_turn)
            return "パス"

    GameEngine(state, PassDiscussion()).run_discussion()

    discussion_messages = [
        message for message in state.conversation_log if message.speaker != "system"
    ]
    assert len(discussion_messages) == 3 * len(state.alive_players)
    assert all(message.content == "パス" for message in discussion_messages)
    assert discussion_turns == [1] * 3 + [2] * 3 + [3] * 3
    assert state.discussion_turn == 3
    assert state.phase is Phase.VOTING


def test_still_tied_runoff_uses_repeatable_seeded_tiebreak() -> None:
    first = make_state(Role.WOLF, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER)
    second = make_state(Role.WOLF, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER)
    first.phase = second.phase = Phase.VOTING

    class PersistentlyTied(ScriptedActions):
        def vote(self, voter, candidates, day, runoff) -> str:
            if runoff:
                self.runoff_voters.append(voter)
            return "P1" if voter in ("P0", "P2") else "P2"

    GameEngine(first, PersistentlyTied()).run_voting()
    GameEngine(second, PersistentlyTied()).run_voting()

    first_executed = [player.name for player in first.players if not player.is_alive]
    second_executed = [player.name for player in second.players if not player.is_alive]
    assert first_executed == second_executed
    assert first_executed[0] in {"P1", "P2"}


def test_dead_seer_does_not_act_and_single_wolf_skips_consultation() -> None:
    state = make_state(Role.WOLF, Role.SEER, Role.VILLAGER, Role.VILLAGER)
    state.day = 2
    actions = ScriptedActions()
    engine = GameEngine(state, actions)

    engine.run_wolf_phase()
    engine.run_seer_phase()
    engine.resolve_morning()
    assert not state.players[1].is_alive
    assert actions.seer_actions == 1

    engine.run_wolf_phase()
    engine.run_seer_phase()

    assert actions.wolf_discussions == 0
    assert actions.seer_actions == 1


def test_mock_policy_can_complete_a_full_game_without_external_llm() -> None:
    from werewolf import create_game

    state = create_game(seed=42)

    result = GameEngine(state, MockActions()).run()

    assert result.winner in {"village", "werewolf"}
    assert result.phase is Phase.ENDED
    assert all(not player.is_alive or player.name in result.alive_players for player in result.players)
