import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from werewolf import (
    ActionKind,
    ActionParser,
    ChatResponse,
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
from werewolf.cli import GameLogger


def test_cli_parser_options() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--mode", "ollama",
        "--models-file", "config/players.json",
        "--seed", "99",
        "--timeout", "60.0",
        "--temperature", "0.5",
        "--num-ctx", "2048",
        "--num-predict", "256",
        "--log-dir", "custom_logs",
        "--show-private",
    ])
    assert args.mode == "ollama"
    assert args.models_file == Path("config/players.json")
    assert args.seed == 99
    assert args.timeout == 60.0
    assert args.temperature == 0.5
    assert args.num_ctx == 2048
    assert args.num_predict == 256
    assert args.log_dir == Path("custom_logs")
    assert args.show_private is True
    assert parser.parse_args([]).num_ctx is None
    private_args = parser.parse_args(["--private-log"])
    assert private_args.private_log is True


def test_game_logger_aggregates_retries_fallbacks_and_latency_by_model(tmp_path: Path) -> None:
    logger = GameLogger("stats", tmp_path, {})
    logger.append_record(
        {
            "type": "llm_action_parse_failed",
            "model": "slow-model",
            "reason": "empty_content",
        }
    )
    logger.append_record(
        {
            "type": "llm_action_completed",
            "model": "slow-model",
            "total_retries": 2,
            "fallback_used": True,
            "elapsed_seconds": 12.5,
            "error": {"kind": "timeout"},
        }
    )
    logger.append_record(
        {
            "type": "llm_action_completed",
            "model": "slow-model",
            "total_retries": 0,
            "fallback_used": False,
            "elapsed_seconds": 1.25,
            "error": None,
        }
    )
    logger.finalize(create_game(seed=4))

    result = json.loads(logger.log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert result["model_stats"]["slow-model"] == {
        "actions": 2,
        "retries": 2,
        "fallback_count": 1,
        "fallback_rate": 0.5,
        "timeout_count": 1,
        "empty_response_count": 1,
        "empty_response_rate": 0.5,
        "truncated_count": 0,
        "parse_failure_count": 1,
        "elapsed_seconds": 13.75,
        "failure_reasons": {"empty_content": 1},
    }


def test_game_logger_records_model_tool_capability(tmp_path: Path) -> None:
    logger = GameLogger("capability", tmp_path, {})
    state = create_game(seed=5)
    state.model_capabilities["werewolf-calm3-22b-8k"] = {"claim_role_tool": False}

    logger.finalize(state)

    result = json.loads(logger.log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert result["type"] == "game_result"
    assert result["model_capabilities"] == {
        "werewolf-calm3-22b-8k": {"claim_role_tool": False}
    }


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
    assert "seed" not in lines[0]

    assert lines[-1]["type"] == "game_result"
    assert lines[-1]["winner"] == "village"
    assert lines[-1]["winner_reason"] == "no_living_wolves"
    assert lines[-1]["winner_counts"]["living_wolves"] == 0
    assert "living_madmen" in lines[-1]["winner_counts"]
    assert "living_village" in lines[-1]["winner_counts"]
    assert lines[-1]["phase"] == Phase.ENDED.value
    assert len(lines[-1]["players"]) == 8
    assert any(line["type"] == "first_night_no_attack" for line in lines)
    assert "初夜のため人狼の相談・襲撃はありません" in captured

    voting_records = [line for line in lines if line["type"] == "voting_result"]
    assert voting_records
    assert all(
        voter != target
        for record in voting_records
        for voter, target in record["votes"].items()
    )
    vote_events = [
        line
        for line in lines
        if line["type"] in {"voting_phase_started", "vote_started", "vote_completed",
                            "voting_ballot_completed", "voting_phase_ended"}
    ]
    completed_votes = [line for line in vote_events if line["type"] == "vote_completed"]
    assert completed_votes
    assert len({
        (line["voting_phase_id"], line["voter"])
        for line in completed_votes
    }) == len(completed_votes)
    ended_phase_ids = [
        line["voting_phase_id"]
        for line in vote_events
        if line["type"] == "voting_phase_ended"
    ]
    assert len(ended_phase_ids) == len(set(ended_phase_ids))


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

    args1 = build_parser().parse_args([
        "--mode", "mock", "--seed", "123", "--log-dir", str(log_dir1),
        "--private-log",
    ])
    args2 = build_parser().parse_args([
        "--mode", "mock", "--seed", "123", "--log-dir", str(log_dir2),
        "--private-log",
    ])

    assert run_cli_game(args1) == 0
    assert run_cli_game(args2) == 0

    private1 = next(log_dir1.glob("game_*.private.jsonl"))
    private2 = next(log_dir2.glob("game_*.private.jsonl"))
    res1 = json.loads(private1.read_text(encoding="utf-8").strip().splitlines()[-1])
    res2 = json.loads(private2.read_text(encoding="utf-8").strip().splitlines()[-1])

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
    assert last_line["error_occurred"] is True


def test_ollama_mode_invalid_response_uses_fallback_and_records_trace(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    # Ollama が形式不正な回答（またはMarkdown装飾）を返した場合の動作を検証
    call_count = 0

    def mock_chat(self, model, messages, options=None, tools=None):
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
        "--private-log",
    ])

    exit_code = run_cli_game(args)
    assert exit_code == 0

    log_file = next(
        path for path in log_dir.glob("game_*.jsonl")
        if not path.name.endswith(".private.jsonl")
    )
    last_line = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert last_line["type"] == "game_result"
    private_file = next(log_dir.glob("game_*.private.jsonl"))
    private_last_line = json.loads(
        private_file.read_text(encoding="utf-8").strip().splitlines()[-1]
    )
    assert len(private_last_line["traces"]) > 0

    # 投票で fallback_used が True になっているトレースが存在することを確認
    vote_traces = [t for t in private_last_line["traces"] if t["action"] == "vote"]
    assert len(vote_traces) > 0
    assert any(t["fallback_used"] is True for t in vote_traces)
    output = capsys.readouterr().out
    assert "応答を解析できません。再要求します。" in output
    assert "フォールバック 有" in output


@pytest.mark.parametrize(
    ("private_option", "show_private", "expect_private_log"),
    [(None, False, False), ("--show-private", True, True), ("--private-log", False, True)],
)
def test_ollama_cli_streams_ordered_action_events_and_gates_private_text(
    tmp_path: Path,
    monkeypatch,
    capsys,
    private_option: str | None,
    show_private: bool,
    expect_private_log: bool,
) -> None:
    private_wolf_text = "WOLF_PRIVATE_MARKER"
    attack_response = "ATTACK_RAW_RESPONSE_MARKER"
    seer_response = "SEER_RAW_RESPONSE_MARKER"
    vote_response = "VOTE_RAW_RESPONSE_MARKER"

    def mock_chat(self, model, messages, options=None, tools=None):
        prompt = messages[-1]["content"]
        if "これまでの相談内容" in prompt:
            return private_wolf_text
        if "襲撃対象：" in prompt:
            line = next(line for line in prompt.splitlines() if "選択可能な襲撃対象：" in line)
            return f"{attack_response}\n襲撃対象：{line.split('：', 1)[1].split(',', 1)[0].strip()}"
        if "占い対象：" in prompt:
            line = next(
                line
                for line in prompt.splitlines()
                if "選択可能な占い対象：" in line
            )
            return f"{seer_response}\n占い対象：{line.split('：', 1)[1].split(',', 1)[0].strip()}"
        if "選択可能な候補:" in prompt:
            line = next(line for line in prompt.splitlines() if "選択可能な候補:" in line)
            return f"{vote_response}\n投票：{line.split(':', 1)[1].split(',', 1)[0].strip()}"
        return "公開発言です。PUBLIC_DAY_SPEECH"

    monkeypatch.setattr(OllamaClient, "ping", lambda self: True)
    monkeypatch.setattr(OllamaClient, "chat", mock_chat)
    monkeypatch.setattr(
        OllamaClient,
        "chat_with_metadata",
        lambda self, model, messages, options=None, tools=None: ChatResponse(
            content=mock_chat(self, model, messages, options=options, tools=tools),
            attempts=1,
        ),
    )
    log_dir = tmp_path / "event_logs"
    argv = [
        "--mode", "ollama",
        "--seed", "42",
        "--log-dir", str(log_dir),
    ]
    if private_option:
        argv.append(private_option)

    assert run_cli_game(build_parser().parse_args(argv)) == 0

    output = capsys.readouterr().out
    assert "[Day 2][夜・人狼相談][Player " in output
    assert "応答待ち..." in output
    assert "PUBLIC_DAY_SPEECH" in output
    assert (private_wolf_text in output) is show_private

    public_log = next(
        path for path in log_dir.glob("game_*.jsonl")
        if not path.name.endswith(".private.jsonl")
    )
    public_text = public_log.read_text(encoding="utf-8")
    records = [
        json.loads(line)
        for line in public_text.splitlines()
    ]
    assert [record["sequence"] for record in records] == list(range(len(records)))
    assert "seed" not in records[0]
    assert "raw_responses" not in public_text
    assert "role" not in records[-1]["players"][0]
    assert "partner" not in records[-1]["players"][0]
    assert "wolf_chat_log" not in records[-1]
    assert "seer_results" not in records[-1]
    public_attack = next(
        record for record in records if record["type"] == "action_pending_attack"
    )
    public_seer = next(
        record for record in records if record["type"] == "seer_results"
    )
    assert "target" not in public_attack
    assert "results" not in public_seer
    for action in ("attack", "seer"):
        public_action_results = [
            record
            for record in records
            if record["type"] == "llm_action_completed"
            and record["action"] == action
        ]
        assert public_action_results
        assert all(
            "selected_target" not in record and "content" not in record
            for record in public_action_results
        )
    assert all(
        secret not in public_text
        for secret in (
            private_wolf_text,
            attack_response,
            seer_response,
            vote_response,
        )
    )
    wolf_events = [
        record for record in records
        if record["type"] == "llm_action_completed"
        and record["action"] == "wolf_discussion"
    ]
    assert wolf_events
    assert wolf_events[0]["private"] is True
    assert wolf_events[0]["fallback_used"] is False
    assert any(record["type"] == "llm_action_started" for record in records)
    assert any(record["type"] == "llm_action_parse_failed" for record in records)
    assert ("占い結果（管理者表示）" in output) is show_private

    output_completion_order = [
        line for line in output.splitlines()
        if "応答待ち..." in line or "完了 (" in line
    ]
    assert output_completion_order
    assert next(
        index for index, line in enumerate(output_completion_order)
        if "応答待ち..." in line
    ) < next(
        index for index, line in enumerate(output_completion_order) if "完了 (" in line
    )

    private_logs = list(log_dir.glob("game_*.private.jsonl"))
    assert bool(private_logs) is expect_private_log
    if expect_private_log:
        private_log = private_logs[0]
        private_text = private_log.read_text(encoding="utf-8")
        private_records = [
            json.loads(line) for line in private_text.splitlines()
        ]
        assert [record["sequence"] for record in private_records] == list(
            range(len(private_records))
        )
        for secret in (
            private_wolf_text,
            attack_response,
            seer_response,
            vote_response,
        ):
            assert secret in private_text
        assert "seed" in private_records[0]
        assert all("role" in player for player in private_records[-1]["players"])
        assert "wolf_chat_log" in private_records[-1]
        assert "seer_results" in private_records[-1]
        private_attack = next(
            record
            for record in private_records
            if record["type"] == "action_pending_attack"
        )
        assert "target" not in public_attack
        assert private_attack["target"] in private_records[-1]["alive_players"] + [
            player["name"]
            for player in private_records[-1]["players"]
            if not player["is_alive"]
        ]
        private_seer = next(
            record
            for record in private_records
            if record["type"] == "seer_results"
        )
        assert "results" not in public_seer
        assert private_seer["results"]


def test_ctrl_c_records_current_llm_action_before_finalizing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(OllamaClient, "ping", lambda self: True)

    def interrupt_chat(self, model, messages, options=None, tools=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(OllamaClient, "chat", interrupt_chat)
    log_dir = tmp_path / "interrupt_logs"
    args = build_parser().parse_args([
        "--mode", "ollama",
        "--seed", "42",
        "--log-dir", str(log_dir),
    ])

    with pytest.raises(SystemExit) as exit_error:
        run_cli_game(args)

    assert exit_error.value.code == 130
    assert "中断した処理" in capsys.readouterr().out
    records = [
        json.loads(line)
        for line in next(log_dir.glob("game_*.jsonl"))
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    interrupted = next(
        record for record in records if record["type"] == "llm_action_interrupted"
    )
    assert interrupted["active_action"]["player"] in {
        player["name"] for player in records[-1]["players"]
    }
    assert interrupted["active_action"]["action"] == "seer"
    assert records[-1]["error_occurred"] is True
