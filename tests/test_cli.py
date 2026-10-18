import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from werewolf import (
    ActionKind,
    ActionParser,
    GameEngine,
    GameState,
    MockActions,
    OllamaActions,
    OllamaClient,
    OllamaError,
    Phase,
    Role,
    build_parser,
    create_game,
    run_cli_game,
)


def test_cli_parser_options() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--mode", "ollama",
        "--models-file", "config/players.json",
        "--seed", "99",
        "--timeout", "60.0",
        "--temperature", "0.5",
        "--num-ctx", "2048",
        "--log-dir", "custom_logs",
    ])
    assert args.mode == "ollama"
    assert args.models_file == Path("config/players.json")
    assert args.seed == 99
    assert args.timeout == 60.0
    assert args.temperature == 0.5
    assert args.num_ctx == 2048
    assert args.log_dir == Path("custom_logs")


def test_mock_actions_completes_game_via_cli(tmp_path: Path, capsys) -> None:
    log_dir = tmp_path / "logs"
    args = build_parser().parse_args([
        "--mode", "mock",
        "--seed", "0",
        "--log-dir", str(log_dir),
    ])

    exit_code = run_cli_game(args)
    assert exit_code == 0

    captured = capsys.readouterr().out
    assert "人狼ゲーム CLI" in captured
    assert "ゲーム終了！" in captured
    assert "勝者: 村人陣営 (village)" in captured
    assert "JSONLログ保存先:" in captured

    # ログファイルの検査
    log_files = list(log_dir.glob("game_*.jsonl"))
    assert len(log_files) == 1
    log_file = log_files[0]
    lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").strip().splitlines()]

    assert lines[0]["type"] == "game_init"
    assert lines[0]["mode"] == "mock"
    assert lines[0]["seed"] == 0

    assert lines[-1]["type"] == "game_result"
    assert lines[-1]["winner"] == "village"
    assert lines[-1]["phase"] == Phase.ENDED.value
    assert len(lines[-1]["players"]) == 8

    voting_records = [line for line in lines if line["type"] == "voting_result"]
    assert voting_records
    assert all(
        voter != target
        for record in voting_records
        for voter, target in record["votes"].items()
    )


def test_village_and_werewolf_wins_with_mock_actions(tmp_path: Path) -> None:
    # seed=0 は village 勝利
    log_dir_village = tmp_path / "village_logs"
    args_village = build_parser().parse_args([
        "--mode", "mock",
        "--seed", "0",
        "--log-dir", str(log_dir_village),
    ])
    assert run_cli_game(args_village) == 0
    village_result = json.loads(list(log_dir_village.glob("game_*.jsonl"))[0].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert village_result["winner"] == "village"

    # seed=1 は werewolf 勝利
    log_dir_wolf = tmp_path / "wolf_logs"
    args_wolf = build_parser().parse_args([
        "--mode", "mock",
        "--seed", "1",
        "--log-dir", str(log_dir_wolf),
    ])
    assert run_cli_game(args_wolf) == 0
    wolf_result = json.loads(list(log_dir_wolf.glob("game_*.jsonl"))[0].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert wolf_result["winner"] == "werewolf"


def test_identical_seed_reproducibility(tmp_path: Path) -> None:
    log_dir1 = tmp_path / "run1"
    log_dir2 = tmp_path / "run2"

    args1 = build_parser().parse_args(["--mode", "mock", "--seed", "123", "--log-dir", str(log_dir1)])
    args2 = build_parser().parse_args(["--mode", "mock", "--seed", "123", "--log-dir", str(log_dir2)])

    assert run_cli_game(args1) == 0
    assert run_cli_game(args2) == 0

    res1 = json.loads(list(log_dir1.glob("game_*.jsonl"))[0].read_text(encoding="utf-8").strip().splitlines()[-1])
    res2 = json.loads(list(log_dir2.glob("game_*.jsonl"))[0].read_text(encoding="utf-8").strip().splitlines()[-1])

    assert res1["winner"] == res2["winner"]
    assert res1["day"] == res2["day"]
    assert res1["alive_players"] == res2["alive_players"]
    assert [p["role"] for p in res1["players"]] == [p["role"] for p in res2["players"]]


def test_ollama_offline_error_handled_gracefully_and_saves_log(tmp_path: Path, monkeypatch, capsys) -> None:
    # 接続不可（ping 失敗）をシミュレート
    monkeypatch.setattr(OllamaClient, "ping", lambda self: False)

    log_dir = tmp_path / "error_logs"
    args = build_parser().parse_args([
        "--mode", "ollama",
        "--seed", "42",
        "--log-dir", str(log_dir),
    ])

    exit_code = run_cli_game(args)
    # エラー時は終了コード1でログ保存
    assert exit_code == 1

    captured = capsys.readouterr().out
    assert "エラー" in captured
    assert "ログを保存しました:" in captured

    log_files = list(log_dir.glob("game_*.jsonl"))
    assert len(log_files) == 1
    last_line = json.loads(log_files[0].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert last_line["type"] == "game_result"
    assert "接続できません" in str(last_line["error"])


def test_ollama_mode_invalid_response_uses_fallback_and_records_trace(tmp_path: Path, monkeypatch) -> None:
    # Ollama が形式不正な回答（またはMarkdown装飾）を返した場合の動作を検証
    call_count = 0

    def mock_chat(self, model, messages, options=None):
        nonlocal call_count
        call_count += 1
        last_user = messages[-1]["content"]
        if "投票" in last_user:
            # 不正な形式を返し、再プロンプト後も不正回答 -> フォールバック発動
            return "誰にも投票したくありません。"
        elif "襲撃" in last_user:
            # Markdown装飾された正常な回答
            return "**襲撃対象：Player 1**"
        elif "占い" in last_user:
            return "占い対象：Player 2"
        return "こんにちは。村人の発言です。"

    monkeypatch.setattr(OllamaClient, "chat", mock_chat)

    log_dir = tmp_path / "trace_logs"
    args = build_parser().parse_args([
        "--mode", "ollama",
        "--seed", "42",
        "--log-dir", str(log_dir),
    ])

    exit_code = run_cli_game(args)
    assert exit_code == 0

    log_file = list(log_dir.glob("game_*.jsonl"))[0]
    last_line = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert last_line["type"] == "game_result"
    assert len(last_line["traces"]) > 0

    # 投票で fallback_used が True になっているトレースが存在することを確認
    vote_traces = [t for t in last_line["traces"] if t["action"] == "vote"]
    assert len(vote_traces) > 0
    assert any(t["fallback_used"] is True for t in vote_traces)
