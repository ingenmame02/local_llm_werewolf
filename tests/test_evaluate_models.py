import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.evaluate_models import (
    PromptTestScenario,
    evaluate_scenario,
    generate_scenarios,
)
from werewolf import ActionKind, OllamaClient, OllamaError, Player, Role


def test_generate_scenarios_covers_four_actions() -> None:
    player = Player(name="Player 1", model="CALM3-22B", role=Role.VILLAGER)
    all_names = tuple(f"Player {i}" for i in range(1, 9))

    scenarios = generate_scenarios(player, all_names)
    assert len(scenarios) == 4
    action_types = [s.action_type for s in scenarios]
    assert action_types == ["discussion", "vote", "attack", "seer"]


def test_evaluate_scenario_success() -> None:
    client = MagicMock(spec=OllamaClient)
    client.chat.return_value = "投票：Player 2"

    player = Player(name="Player 1", model="CALM3-22B", role=Role.VILLAGER)
    all_names = tuple(f"Player {i}" for i in range(1, 9))
    scenarios = generate_scenarios(player, all_names)

    vote_scenario = next(s for s in scenarios if s.action_type == "vote")
    result = evaluate_scenario(client, "CALM3-22B", vote_scenario, {"temperature": 0.7})

    assert result["model"] == "CALM3-22B"
    assert result["action_type"] == "vote"
    assert result["parsed_target"] == "Player 2"
    assert result["is_format_compliant"] is True
    assert result["is_parse_valid"] is True
    assert result["error"] is None
    assert result["retry_count"] == 0


def test_evaluate_scenario_invalid_target_format_reprompts_and_fails() -> None:
    client = MagicMock(spec=OllamaClient)
    # 1回目：形式不遵守、2回目：依然として形式不遵守
    client.chat.side_effect = ["誰にしようかな。Player 2かな。", "やっぱり決められない。"]

    player = Player(name="Player 1", model="CALM3-22B", role=Role.VILLAGER)
    all_names = tuple(f"Player {i}" for i in range(1, 9))
    scenarios = generate_scenarios(player, all_names)

    vote_scenario = next(s for s in scenarios if s.action_type == "vote")
    result = evaluate_scenario(client, "CALM3-22B", vote_scenario, {"temperature": 0.7})

    assert result["parsed_target"] is None
    assert result["is_format_compliant"] is False
    assert result["is_parse_valid"] is False
    assert result["status"] == "FAILED"
    assert result["fallback_used"] is True
    assert result["retry_count"] == 1
    assert result["semantic_retries"] == 1
    assert result["transport_retries"] == 0
    assert result["raw_responses"] == ["誰にしようかな。Player 2かな。", "やっぱり決められない。"]
    assert client.chat.call_count == 2


def test_evaluate_scenario_reprompt_success() -> None:
    client = MagicMock(spec=OllamaClient)
    # 1回目：形式不遵守、2回目：再プロンプトで正解
    client.chat.side_effect = ["迷っています", "投票：Player 2"]

    player = Player(name="Player 1", model="CALM3-22B", role=Role.VILLAGER)
    all_names = tuple(f"Player {i}" for i in range(1, 9))
    scenarios = generate_scenarios(player, all_names)

    vote_scenario = next(s for s in scenarios if s.action_type == "vote")
    result = evaluate_scenario(client, "CALM3-22B", vote_scenario, {"temperature": 0.7})

    assert result["parsed_target"] == "Player 2"
    assert result["is_format_compliant"] is True
    assert result["is_parse_valid"] is True
    assert result["status"] == "OK"
    assert result["fallback_used"] is False
    assert result["retry_count"] == 1
    assert result["semantic_retries"] == 1
    assert result["transport_retries"] == 0
    assert result["raw_responses"] == ["迷っています", "投票：Player 2"]


def test_evaluate_scenario_markdown_decoration_attack() -> None:
    client = MagicMock(spec=OllamaClient)
    client.chat.return_value = "さっさと終わらせよう。\n\n**襲撃対象：Player 4**"

    player = Player(name="Player 1", model="CALM3-22B", role=Role.VILLAGER)
    all_names = tuple(f"Player {i}" for i in range(1, 9))
    scenarios = generate_scenarios(player, all_names)

    attack_scenario = next(s for s in scenarios if s.action_type == "attack")
    result = evaluate_scenario(client, "CALM3-22B", attack_scenario, {"temperature": 0.7})

    assert result["parsed_target"] == "Player 4"
    assert result["is_format_compliant"] is True
    assert result["is_parse_valid"] is True
    assert result["status"] == "OK"
    assert result["retry_count"] == 0


def test_evaluate_scenario_ollama_error_handled_gracefully() -> None:
    client = MagicMock(spec=OllamaClient)
    client.chat.side_effect = OllamaError(kind="timeout", message="Request timed out", attempts=3)

    player = Player(name="Player 1", model="CALM3-22B", role=Role.VILLAGER)
    all_names = tuple(f"Player {i}" for i in range(1, 9))
    scenarios = generate_scenarios(player, all_names)

    disc_scenario = next(s for s in scenarios if s.action_type == "discussion")
    result = evaluate_scenario(client, "CALM3-22B", disc_scenario, {"temperature": 0.7})

    assert result["raw_response"] is None
    assert result["raw_responses"] == []
    assert result["error"]["kind"] == "timeout"
    assert result["retry_count"] == 2
    assert result["transport_retries"] == 2
    assert result["semantic_retries"] == 0
    assert result["is_format_compliant"] is False
    assert result["is_parse_valid"] is False
    assert result["status"] == "FAILED"


def test_cli_execution_with_mock_transport(tmp_path: Path, monkeypatch) -> None:
    # CLI 全体のモック実行テスト
    test_output_dir = tmp_path / "eval_out"
    config_file = tmp_path / "players_mock.json"

    # 8プレイヤーのテスト用コンフィグ作成
    config_file.write_text(
        json.dumps(
            {
                "players": [
                    {"name": f"P{i}", "model": f"model-{i}"}
                    for i in range(1, 9)
                ]
            }
        ),
        encoding="utf-8",
    )

    # OllamaClient.chat をモック
    def mock_chat(self, model: str, messages: list[dict[str, str]], options=None) -> str:
        last_user = messages[-1]["content"]
        if "選択可能な候補:" in last_user:
            # 候補行から最初の候補を取得
            candidates_line = [l for l in last_user.splitlines() if "選択可能な候補:" in l][0]
            first_cand = candidates_line.split(":", 1)[1].split(",")[0].strip()
            return f"投票：{first_cand}"
        elif "選択可能な襲撃対象：" in last_user:
            candidates_line = [l for l in last_user.splitlines() if "選択可能な襲撃対象：" in l][0]
            first_cand = candidates_line.split("：", 1)[1].split(",")[0].strip()
            return f"**襲撃対象：{first_cand}**"
        elif "占い対象：" in last_user:
            return "占い対象：P2"
        return "こんにちは。私は村人です。"

    monkeypatch.setattr(OllamaClient, "chat", mock_chat)

    import scripts.evaluate_models as eval_mod

    test_args = [
        "evaluate_models.py",
        "--models-file",
        str(config_file),
        "--output",
        str(test_output_dir),
        "--rounds",
        "1",
        "--timeout",
        "10",
        "--temperature",
        "0.7",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    eval_mod.main()

    # summary.json が生成されているか
    summary_file = test_output_dir / "summary.json"
    assert summary_file.exists()

    with summary_file.open("r", encoding="utf-8") as f:
        summary_data = json.load(f)

    assert "models" in summary_data
    assert summary_data["transport_retries"] == 0
    assert summary_data["semantic_retries"] == 0
    assert summary_data["total_retries"] == 0
    assert summary_data["fallback_count"] == 0
    assert len(summary_data["models"]) == 8

    for i in range(1, 9):
        m = f"model-{i}"
        assert m in summary_data["models"]
        model_stat = summary_data["models"][m]
        assert model_stat["total_actions"] == 4
        assert model_stat["parse_success_rate"] == 1.0
        assert model_stat["format_compliance_rate"] == 1.0
        assert model_stat["error_count"] == 0
        assert model_stat["transport_retries"] == 0
        assert model_stat["semantic_retries"] == 0
        assert model_stat["total_retries"] == 0
        assert model_stat["fallback_count"] == 0

        # jsonl ファイルの存在確認
        jsonl_file = test_output_dir / f"{m}.jsonl"
        assert jsonl_file.exists()
        lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 4
        record = json.loads(lines[0])
        assert "raw_responses" in record
        assert "transport_retries" in record
        assert "semantic_retries" in record
        assert "total_retries" in record
        assert "fallback_used" in record
