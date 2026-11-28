import json

from werewolf.action_parser import ActionKind, ActionParser
from werewolf.models import Player, Role
from werewolf.ollama import OllamaClient
from werewolf.prompts import PromptBuilder


def test_action_parser_parses_json_target() -> None:
    candidates = ("Player 1", "Player 2", "Player 3")
    assert (
        ActionParser.parse('{"target": "Player 2"}', ActionKind.VOTE, candidates)
        == "Player 2"
    )
    assert (
        ActionParser.parse('{"target": "プレイヤー3"}', ActionKind.VOTE, candidates)
        == "Player 3"
    )


def test_ollama_client_passes_format_top_level() -> None:
    captured: list[dict[str, object]] = []

    def fake_transport(request, timeout):
        captured.append(json.loads(request.data.decode("utf-8")))
        return json.dumps({"message": {"content": '{"target": "Player 2"}'}}).encode(
            "utf-8"
        )

    client = OllamaClient(transport=fake_transport)
    schema = {
        "type": "object",
        "properties": {"target": {"type": "string", "enum": ["Player 2", "Player 3"]}},
        "required": ["target"],
    }
    res = client.chat(
        "test-model",
        [{"role": "user", "content": "vote"}],
        options={"thinking": False, "format": schema},
    )
    assert res == '{"target": "Player 2"}'
    assert captured[0]["think"] is False
    assert captured[0]["format"] == schema
    assert "format" not in captured[0].get("options", {})


def test_build_surgical_repair_prompt_keeps_previous_output_and_allowlist() -> None:
    actor = Player(name="Player 1", model="test-model", role=Role.VILLAGER, response_language="日本語")
    ctx = PromptBuilder.build_surgical_repair_prompt(
        actor=actor,
        previous_response="Player 9 が怪しいと思います。Player 2 の意見には賛成です。",
        issue_guidance="存在しないプレイヤーへの言及を削除してください。",
        alive_players=("Player 1", "Player 2", "Player 3"),
    )
    assert "対象にできる生存者: Player 1, Player 2, Player 3" in ctx.user_prompt
    assert "Player 9 が怪しいと思います。" in ctx.user_prompt
    assert "指摘事項:" in ctx.user_prompt
