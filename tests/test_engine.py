from werewolf import GameEngine, GameState, MockActions, Phase, Player, Role


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


def test_madman_counts_for_werewolf_parity() -> None:
    state = make_state(Role.WOLF, Role.MADMAN, Role.VILLAGER)

    winner = GameEngine(state, MockActions()).check_winner()

    assert winner == "werewolf"
    assert state.phase is Phase.ENDED


def test_madman_is_not_an_attacker_but_can_be_a_target() -> None:
    state = make_state(
        Role.WOLF, Role.WOLF, Role.MADMAN, Role.SEER, Role.VILLAGER, Role.VILLAGER
    )

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
