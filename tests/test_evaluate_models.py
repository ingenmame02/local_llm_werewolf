import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.evaluate_models import (
    PromptTestScenario,
    build_parser,
    evaluate_scenario,
    extract_discussion_signals,
    generate_scenarios,
    summarize_discussion_signals,
    select_player_configs,
)
from werewolf import ActionKind, OllamaClient, OllamaError, Player, PlayerConfig, Role


def test_generate_scenarios_covers_five_actions_including_seer_co() -> None:
    player = Player(name="Player 1", model="CALM3-22B", role=Role.VILLAGER)
    all_names = tuple(f"Player {i}" for i in range(1, 9))

    scenarios = generate_scenarios(player, all_names)
    assert len(scenarios) == 5
    action_types = [s.action_type for s in scenarios]
    assert action_types == [
        "discussion",
        "vote",
        "attack",
        "seer",
        "discussion_seer_co",
    ]
    co_scenario = next(s for s in scenarios if s.action_type == "discussion_seer_co")
    assert co_scenario.expected_action_kind is None
    assert co_scenario.player_names == all_names
    assert "未公開の人狼判定" in co_scenario.prompt_ctx.user_prompt
    assert all(s.personality_profile == player.personality_profile for s in scenarios)
    assert next(s for s in scenarios if s.action_type == "seer").scenario_role == "seer"
    seer_scenario = next(s for s in scenarios if s.action_type == "seer")
    assert "Player 2" not in seer_scenario.candidates
    assert "Player 2" in seer_scenario.prompt_ctx.user_prompt
    assert co_scenario.scenario_role == "seer"


def test_seer_target_benchmark_records_invalid_target_rate() -> None:
    player = Player(name="Player 1", model="model-a", role=Role.VILLAGER)
    names = tuple(f"Player {index}" for index in range(1, 9))
    scenario = next(
        scenario
        for scenario in generate_scenarios(player, names)
        if scenario.action_type == "seer"
    )
    client = MagicMock(spec=OllamaClient)
    client.chat.side_effect = ["占い対象：Player 2", "占い対象：Player 3"]

    result = evaluate_scenario(client, player.model, scenario, {"num_ctx": 4096})

    assert result["invalid_target"] is True
    assert result["is_parse_valid"] is True
    assert result["failure_reasons"] == ["invalid_target"]


def test_discussion_signals_measure_targets_not_just_vocabulary() -> None:
    response = (
        "【CO】なし\n"
        "【結果】なし\n"
        "【本文】Player 2さんに投票の理由を質問します。\n"
        "Player 3は発言が変わったので疑っています。\n"
        "Player 3への投票を提案します。"
    )

    signals = extract_discussion_signals(
        response, tuple(f"Player {index}" for index in range(1, 9))
    )

    assert signals == {
        "question_targets": ["Player 2"],
        "suspicion_targets": ["Player 3"],
        "suspicion_reasons": [
            "discussion_behavior",
            "inconsistent_statements",
        ],
        "vote_proposals": ["Player 3"],
    }
    summary = summarize_discussion_signals(
        [{"discussion_signals": signals}, {"discussion_signals": signals}]
    )
    assert summary["question_targets"] == {
        "target_counts": {"Player 2": 2},
        "distinct_targets": 1,
    }
    assert summary["suspicion_targets"]["target_counts"] == {"Player 3": 2}
    assert summary["suspicion_reasons"]["reason_counts"] == {
        "discussion_behavior": 2,
        "inconsistent_statements": 2,
    }


def test_seer_co_scenario_measures_co_establishment_and_short_correction() -> None:
    player = Player(name="Player 1", model="CALM3-22B", role=Role.VILLAGER)
    all_names = tuple(f"Player {i}" for i in range(1, 9))
    scenario = next(
        s
        for s in generate_scenarios(player, all_names)
        if s.action_type == "discussion_seer_co"
    )

    # 1回目：結果のみでCOなし → 短いCO補正 → 2回目：最小形式のCO＋結果
    client = MagicMock(spec=OllamaClient)
    client.chat.side_effect = [
        "【CO】なし\n【本文】Player 2は人狼と判定されました。",
        "【CO】占い師\n【結果】対象=Player 2 判定=人狼\n【本文】Player 2への投票を提案します。",
    ]

    result = evaluate_scenario(client, "CALM3-22B", scenario, {"num_ctx": 4096})

    assert result["is_parse_valid"] is True
    assert result["co_established"] is True
    assert result["seer_result_published"] is True
    assert result["fallback_used"] is False
    assert result["semantic_retries"] == 1
    retry_prompt = client.chat.call_args_list[1].kwargs["messages"][-1]["content"]
    assert "同じ発言内で明示的に占い師COしてください" in retry_prompt
    assert "【CO】占い師" in retry_prompt

    # 自然文のCOも成立として扱う
    legacy_client = MagicMock(spec=OllamaClient)
    legacy_client.chat.return_value = "私は占い師です。対象=Player 2、判定=人狼。"
    legacy_result = evaluate_scenario(
        legacy_client, "CALM3-22B", scenario, {"num_ctx": 4096}
    )
    assert legacy_result["co_established"] is True
    assert legacy_result["seer_result_published"] is True
    assert legacy_result["is_parse_valid"] is True
    assert legacy_result["discussion_format"] == "legacy"

    # COが成立しなければフォールバック扱いになる
    failed_client = MagicMock(spec=OllamaClient)
    failed_client.chat.side_effect = [
        "【CO】なし\n【本文】まだ決めかねます。",
        "【CO】なし\n【本文】まだ決めかねます。",
    ]
    failed_result = evaluate_scenario(
        failed_client, "CALM3-22B", scenario, {"num_ctx": 4096}
    )
    assert failed_result["co_established"] is False
    assert failed_result["is_parse_valid"] is False
    assert failed_result["fallback_used"] is True
    assert failed_result["status"] == "FAILED"


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


@pytest.mark.parametrize(
    "model",
    [
        "CALM3-22B-Chat-GGUF:latest",
        "deepseek-r1:32b",
        "gemma4:26b",
        "glm-4.7-flash:latest",
        "gpt-oss-20b-262000:latest",
        "mistral-small:24b-262000",
        "muse-glimmer:latest",
        "qwen3.8:27b",
    ],
)
def test_discussion_formats_are_measured_for_all_eight_configured_models(
    model: str,
) -> None:
    client = MagicMock(spec=OllamaClient)
    player = Player(name="Player 1", model=model, role=Role.VILLAGER)
    all_names = tuple(f"Player {i}" for i in range(1, 9))
    scenario = next(
        item
        for item in generate_scenarios(player, all_names)
        if item.action_type == "discussion"
    )
    responses = (
        "【CO】なし\n【本文】発言を確認して投票先を考えます。",
        "私は村人です。発言を確認して投票先を考えます。",
        "【CO】なし\n【本文】新しい発言を確認します。",
    )

    results = []
    for response in responses:
        client.chat.return_value = response
        results.append(evaluate_scenario(client, model, scenario, {"num_ctx": 4096}))

    assert [result["discussion_format"] for result in results] == [
        "two_block",
        "legacy",
        "two_block",
    ]
    assert all(result["is_format_compliant"] for result in results)
    assert all(result["is_parse_valid"] for result in results)


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
        elif "選択可能な占い対象：" in last_user:
            candidates_line = [
                line for line in last_user.splitlines() if "選択可能な占い対象：" in line
            ][0]
            first_cand = candidates_line.split("：", 1)[1].split(",")[0].strip()
            return f"占い対象：{first_cand}"
        elif "未公開の人狼判定" in last_user:
            # 占い師CO評価シナリオ：最小形式のCOと結果を返す
            return "【CO】占い師\n【結果】対象=P2 判定=人狼\n【本文】P2への投票を提案します。"
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
        assert model_stat["total_actions"] == 5
        assert model_stat["parse_success_rate"] == 1.0
        assert model_stat["format_compliance_rate"] == 1.0
        assert model_stat["error_count"] == 0
        assert model_stat["transport_retries"] == 0
        assert model_stat["semantic_retries"] == 0
        assert model_stat["total_retries"] == 0
        assert model_stat["fallback_count"] == 0
        assert model_stat["seer_co_total"] == 1
        assert model_stat["seer_co_established"] == 1
        assert model_stat["seer_co_established_rate"] == 1.0
        assert model_stat["seer_co_result_published_rate"] == 1.0
        assert model_stat["personality_profile"] == "balanced"
        assert model_stat["seer_target_invalid_rate"] == 0.0
        assert "question_targets" in model_stat["discussion_signal_distribution"]

        # jsonl ファイルの存在確認
        jsonl_file = test_output_dir / f"{m}.jsonl"
        assert jsonl_file.exists()
        lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5
        record = json.loads(lines[0])
        assert "raw_responses" in record
        assert "transport_retries" in record
        assert "semantic_retries" in record
        assert "total_retries" in record
        assert "fallback_used" in record
        assert record["personality_profile"] == "balanced"
        assert "discussion_signals" in record

def test_evaluate_scenario_doubles_generation_budget_on_generation_failure() -> None:
    """TASK-024: 評価ハーネスも本番と同じく生成起因の再試行でnum_predictを倍へ増やす。"""
    from werewolf import ChatResponse

    client = MagicMock(spec=OllamaClient)
    client.chat.side_effect = [
        ChatResponse(
            content="",
            attempts=1,
            done_reason="length",
            content_length=0,
            thinking_present=True,
            eval_count=256,
        ),
        ChatResponse(
            content="投票：Player 2",
            attempts=1,
            done_reason="stop",
            content_length=len("投票：Player 2"),
        ),
    ]
    player = Player(name="Player 1", model="CALM3-22B", role=Role.VILLAGER)
    all_names = tuple(f"Player {i}" for i in range(1, 9))
    vote_scenario = next(
        s for s in generate_scenarios(player, all_names) if s.action_type == "vote"
    )

    result = evaluate_scenario(client, "CALM3-22B", vote_scenario, {"num_predict": 256})

    assert result["is_parse_valid"] is True
    assert result["status"] == "OK"
    assert result["request_attempts"] == 2
    assert result["generation_escalated"] is True
    assert result["num_predict_by_attempt"] == [256, 512]
    assert result["failure_reasons"] == ["generation_truncated"]
    assert result["budget_exhausted"] is True
    assert result["done_reason"] == "stop"
    # 再要求は失敗理由別の短い修正指示だけを送る
    retry_prompt = client.chat.call_args_list[1].kwargs["messages"][-1]["content"]
    assert "生成上限で途切れました" in retry_prompt
    assert len(retry_prompt) < len(
        client.chat.call_args_list[0].kwargs["messages"][-1]["content"]
    )


def test_evaluate_scenario_disables_thinking_on_retry() -> None:
    """TASK-024: 評価ハーネスも本番と同じく再要求で思考off・上限倍にする。"""
    client = MagicMock(spec=OllamaClient)
    # 1回目：空応答（思考で枯渇）、2回目：思考offの短い再要求で正解
    responses = ["", "投票：Player 2"]
    sent_options: list[dict[str, object]] = []

    def chat(model, messages, options=None):
        # 同じoptions辞書が再要求で変更されるため、試行ごとにコピーを記録する
        sent_options.append(dict(options or {}))
        return responses.pop(0)

    client.chat.side_effect = chat

    player = Player(name="Player 1", model="CALM3-22B", role=Role.VILLAGER)
    all_names = tuple(f"Player {i}" for i in range(1, 9))
    scenario = next(
        s for s in generate_scenarios(player, all_names) if s.action_type == "vote"
    )

    result = evaluate_scenario(
        client,
        "CALM3-22B",
        scenario,
        {"temperature": 0.7, "num_predict": 256, "thinking": True},
    )

    assert result["status"] == "OK"
    assert result["parsed_target"] == "Player 2"
    assert result["failure_reasons"] == ["empty_content"]
    assert result["thinking_enabled"] is True
    assert result["thinking_by_attempt"] == [True, False]
    assert result["thinking_disabled_on_retry"] is True

    assert sent_options[0] == {
        "temperature": 0.7,
        "num_predict": 256,
        "thinking": True,
    }
    assert sent_options[1]["thinking"] is False
    # 生成起因の再試行では生成上限を倍へ増やす
    assert sent_options[1]["num_predict"] == 512
    assert client.chat.call_count == 2


def test_evaluate_scenario_keeps_thinking_off_for_minimal_actions() -> None:
    client = MagicMock(spec=OllamaClient)
    client.chat.return_value = "投票：Player 2"

    player = Player(name="Player 1", model="CALM3-22B", role=Role.VILLAGER)
    all_names = tuple(f"Player {i}" for i in range(1, 9))
    scenario = next(
        s for s in generate_scenarios(player, all_names) if s.action_type == "vote"
    )

    result = evaluate_scenario(client, "CALM3-22B", scenario, {"thinking": False})

    assert result["status"] == "OK"
    assert result["thinking_enabled"] is False
    assert result["thinking_by_attempt"] == [False]
    assert result["thinking_disabled_on_retry"] is False
    assert client.chat.call_count == 1


def test_evaluate_scenario_thinking_unsupported_400_falls_back_without_semantic_retry() -> None:
    """TASK-024: 評価ハーネスは思考非対応の400を意味的再試行ではなく能力切替として扱う。"""
    client = MagicMock(spec=OllamaClient)
    sent_options: list[dict[str, object]] = []

    def chat(model, messages, options=None):
        sent_options.append(dict(options or {}))
        if sent_options[-1].get("thinking") is True:
            raise OllamaError(
                "http",
                "model does not support thinking",
                attempts=1,
                status_code=400,
            )
        return "投票：Player 2"

    client.chat.side_effect = chat

    player = Player(name="Player 1", model="CALM3-22B", role=Role.VILLAGER)
    all_names = tuple(f"Player {i}" for i in range(1, 9))
    scenario = next(
        s for s in generate_scenarios(player, all_names) if s.action_type == "vote"
    )
    capabilities: dict[str, bool] = {}

    result = evaluate_scenario(
        client,
        "CALM3-22B",
        scenario,
        {"temperature": 0.7, "num_predict": 256, "thinking": True},
        capabilities,
    )

    assert result["status"] == "OK"
    assert result["parsed_target"] == "Player 2"
    assert result["error"] is None
    assert result["semantic_retries"] == 0
    assert result["thinking_capability_fallback"] is True
    assert result["failure_reasons"] == []
    assert capabilities == {"CALM3-22B": False}
    assert len(sent_options) == 2
    assert sent_options[0]["thinking"] is True
    assert sent_options[1]["thinking"] is False
    # 同じ上限のまま（生成起因の倍化はしない）
    assert sent_options[1]["num_predict"] == 256

    # 判明後は最初からoffで送り、400を再度発生させない
    client.chat.side_effect = lambda model, messages, options=None: (
        sent_options.append(dict(options or {})) or "投票：Player 2"
    )
    second = evaluate_scenario(
        client,
        "CALM3-22B",
        scenario,
        {"temperature": 0.7, "num_predict": 256, "thinking": True},
        capabilities,
    )
    assert second["status"] == "OK"
    # thinking_enabledは設定値（True）のまま、実際の送信はoff
    assert second["thinking_by_attempt"] == [False]
    assert second["thinking_capability_fallback"] is True
    assert sent_options[-1]["thinking"] is False


def test_select_player_configs_keeps_all_models_without_filter() -> None:
    configs = [
        PlayerConfig(name=f"Player {index}", model=f"werewolf-{slug}-8k")
        for index, slug in enumerate(
            ["calm3-22b", "muse-glimmer", "mistral-small", "glm-4.7-flash"], start=1
        )
    ]

    assert select_player_configs(configs, []) == configs
    # 空白のみの指定は絞り込みなし扱いで全モデルを残す
    assert select_player_configs(configs, ["", "   "]) == configs


def test_select_player_configs_filters_pilot_models_by_model_or_name() -> None:
    """TASK-024: 8モデル一括の前に、1〜2モデルだけ評価対象へ絞れる。"""
    configs = [
        PlayerConfig(name=f"Player {index}", model=f"werewolf-{slug}-8k")
        for index, slug in enumerate(
            ["calm3-22b", "muse-glimmer", "mistral-small", "glm-4.7-flash"], start=1
        )
    ]

    by_model = select_player_configs(configs, ["muse"])
    assert [cfg.model for cfg in by_model] == ["werewolf-muse-glimmer-8k"]

    # 大文字小文字無視・モデル名とプレイヤー名の両方で一致する
    mixed = select_player_configs(configs, ["GLM-4.7", "Player 1"])
    assert [cfg.model for cfg in mixed] == [
        "werewolf-calm3-22b-8k",
        "werewolf-glm-4.7-flash-8k",
    ]

    with pytest.raises(ValueError, match="matched no configured model"):
        select_player_configs(configs, ["nonexistent-model"])

    # 空白のみの指定は絞り込みなし扱いで全モデルを残す
    assert select_player_configs(configs, ["   "]) == configs


def test_evaluation_parser_accepts_model_filter_and_action_budgets() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--models",
            "muse-glimmer",
            "--num-predict-by-action",
            "discussion=2048",
            "--thinking-by-action",
            "vote=false",
        ]
    )

    assert args.models == ["muse-glimmer"]
    assert args.num_predict_by_action == ["discussion=2048"]
    assert args.thinking_by_action == ["vote=false"]
    assert args.thinking == "auto"
