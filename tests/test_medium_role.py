from werewolf.action_parser import ActionKind, ActionParser
from werewolf.engine import GameEngine
from werewolf.models import GameState, Player, Role
from werewolf.prompts import PromptBuilder


class _StubActions:
    def __init__(self, chosen: str) -> None:
        self.chosen = chosen
        self.called = 0

    def choose_medium_target(self, medium, candidates, results):
        self.called += 1
        return self.chosen


def test_medium_role_included_in_village_winner_count() -> None:
    players = [
        Player(name="Player 1", model="m", role=Role.WOLF, is_alive=True),
        Player(name="Player 2", model="m", role=Role.MEDIUM, is_alive=True),
        Player(name="Player 3", model="m", role=Role.VILLAGER, is_alive=True),
    ]
    state = GameState(players=players)
    engine = GameEngine(state=state, actions=None)
    assert engine.check_winner() is None


def test_medium_system_prompt_and_action_parser() -> None:
    actor = Player(name="Player 2", model="m", role=Role.MEDIUM, response_language="日本語")
    sys_prompt = PromptBuilder.build_system_prompt(actor)
    assert "Medium" in sys_prompt or "霊媒師" in sys_prompt
    assert (
        ActionParser.parse(
            "Medium target: Player 4", ActionKind.MEDIUM, ("Player 4", "Player 5")
        )
        == "Player 4"
    )


def test_medium_phase_skips_day1_and_inspects_dead_on_day2() -> None:
    players = [
        Player(name="Player 1", model="m", role=Role.WOLF, is_alive=False),
        Player(name="Player 2", model="m", role=Role.MEDIUM, is_alive=True),
        Player(name="Player 3", model="m", role=Role.VILLAGER, is_alive=False),
        Player(name="Player 4", model="m", role=Role.VILLAGER, is_alive=True),
    ]
    state = GameState(
        players=players,
        day=1,
        executions=[(1, "Player 1")],
        attack_deaths=[(1, "Player 3")],
    )
    actions = _StubActions("Player 1")
    engine = GameEngine(state=state, actions=actions)

    engine.run_medium_phase()
    assert actions.called == 0
    assert state.medium_results == {}

    state.day = 2
    engine.run_medium_phase()
    assert actions.called == 1
    assert state.medium_results == {"Player 1": "wolf"}
    assert state.medium_result_history == [(2, "Player 1", "wolf")]
