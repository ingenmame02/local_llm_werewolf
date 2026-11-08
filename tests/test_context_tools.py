from scripts import benchmark_context
from scripts.benchmark_context import ResourceSampler
from scripts.diagnose_ollama_context import collect_context_report
from scripts.evaluate_models import PromptTestScenario, generate_scenarios
from werewolf import ActionKind, Player, Role
from werewolf.prompts import PromptContext


def test_context_diagnostic_report_records_request_and_model_values() -> None:
    class DiagnosticClient:
        def context_diagnostics(self, model):
            return {
                "modelfile_num_ctx": 4096,
                "ollama_ps_context": 8192,
                "ollama_ps_processor": "100% GPU",
            }

    report = collect_context_report(
        ("model-x", "model-x"),
        8192,
        client=DiagnosticClient(),
    )

    assert report["api_request_num_ctx"] == 8192
    assert report["api_request_num_ctx_source"] == "diagnostic_argument"
    assert report["models"] == {
        "model-x": {
            "modelfile_num_ctx": 4096,
            "ollama_ps_context": 8192,
            "ollama_ps_processor": "100% GPU",
        }
    }


def test_resource_summary_compares_cpu_memory_and_gpu_samples() -> None:
    sampler = ResourceSampler()
    sampler.samples = [
        {
            "system_cpu_percent": 25.0,
            "system_memory_available_bytes": 800,
            "ollama_cpu_percent": 45.0,
            "ollama_rss_bytes": 1200,
            "gpu_utilization_percent": 60.0,
            "gpu_memory_used_mib": 2048.0,
        },
        {
            "system_cpu_percent": 50.0,
            "system_memory_available_bytes": 600,
            "ollama_cpu_percent": 75.0,
            "ollama_rss_bytes": 1600,
            "gpu_utilization_percent": 90.0,
            "gpu_memory_used_mib": 4096.0,
        },
    ]

    summary = sampler.summary()

    assert summary["sample_count"] == 2
    assert summary["system_cpu_percent_average"] == 37.5
    assert summary["ollama_rss_bytes_max"] == 1600
    assert summary["gpu_utilization_percent_max"] == 90.0
    assert summary["gpu_memory_used_mib_max"] == 4096.0


def test_context_scenarios_have_repeatable_question_and_ledger_evidence() -> None:
    actor = Player("Player 1", "model-x", Role.VILLAGER)
    names = tuple(f"Player {index}" for index in range(1, 9))
    scenarios = generate_scenarios(actor, names, "hybrid_retrieval")
    discussion = next(item for item in scenarios if item.action_type == "discussion")

    assert discussion.question_expected is True
    assert discussion.questioner_name == "Player 2"
    assert "Player 1さん、Player 3への昨日の投票理由" in discussion.prompt_ctx.user_prompt
    assert "Player 2 voted for Player 3." in discussion.prompt_ctx.user_prompt
    assert discussion.prompt_ctx.diagnostics["transcript_event_count"] == 22
    assert discussion.prompt_ctx.diagnostics["excluded_event_ranges"]


def test_context_benchmark_records_filtered_actions_and_generation_metrics(
    monkeypatch, tmp_path
) -> None:
    config = Player(name="Player 1", model="model-x", role=Role.VILLAGER)
    scenarios = [
        PromptTestScenario(
            action_type="discussion",
            prompt_ctx=PromptContext("system", "user"),
            expected_action_kind=None,
            candidates=(),
            actor_name=config.name,
        ),
        PromptTestScenario(
            action_type="vote",
            prompt_ctx=PromptContext("system", "vote"),
            expected_action_kind=ActionKind.VOTE,
            candidates=(),
            actor_name=config.name,
        ),
    ]

    class Sampler:
        def start(self):
            pass

        def sample_now(self):
            return {"system_memory_available_bytes": 1000}

        def stop(self):
            return {"sample_count": 1}

    class Client:
        def __init__(self, **kwargs):
            pass

        def context_diagnostics(self, model):
            return {
                "ollama_ps_context": 8192,
                "ollama_ps_processor": "100% GPU",
                "modelfile_num_ctx": 8192,
            }

    requests = []
    scenario_strategies = []

    def evaluate(client, model, scenario, options):
        requests.append((model, scenario.action_type, options))
        return {
            "discussion_format": "two_block",
            "is_format_compliant": True,
            "is_parse_valid": False,
            "fallback_used": True,
            "elapsed_seconds": 1.5,
            "retry_count": 1,
            "request_attempts": 2,
            "num_predict": 128,
            "num_predict_by_attempt": [64, 128],
            "failure_reasons": ["discussion_output_format"],
            "done_reason": "stop",
            "thinking_present": False,
            "question_answered": None,
            "new_evidence_categories": [],
            "discussion_signals": {"vote_proposals": ["Player 2"]},
            "prompt_eval_count": 120,
            "eval_count": 30,
            "error": None,
        }

    def generate(_actor, _names, strategy, reasoning_memory=False):
        scenario_strategies.append(strategy)
        return scenarios

    monkeypatch.setattr(benchmark_context, "load_player_configs", lambda _: [config])
    monkeypatch.setattr(benchmark_context, "select_player_configs", lambda c, f: c)
    monkeypatch.setattr(benchmark_context, "generate_scenarios", generate)
    monkeypatch.setattr(benchmark_context, "evaluate_scenario", evaluate)
    monkeypatch.setattr(benchmark_context, "OllamaClient", Client)
    monkeypatch.setattr(benchmark_context, "ResourceSampler", Sampler)

    reports = benchmark_context.run_context_benchmark(
        tmp_path / "players.json",
        (8192, 12288),
        1,
        10.0,
        0.1,
        "http://localhost:11434",
        ("model-x",),
        ("discussion",),
        ("full_transcript", "structured_ledger"),
        (False,),
    )

    assert requests == [
        ("model-x", "discussion", {"temperature": 0.1, "num_ctx": 8192}),
        ("model-x", "discussion", {"temperature": 0.1, "num_ctx": 8192}),
        ("model-x", "discussion", {"temperature": 0.1, "num_ctx": 12288}),
        ("model-x", "discussion", {"temperature": 0.1, "num_ctx": 12288}),
    ]
    assert [report["context_strategy"] for report in reports] == [
        "full_transcript",
        "structured_ledger",
        "full_transcript",
        "structured_ledger",
    ]
    assert scenario_strategies == [
        "full_transcript",
        "structured_ledger",
        "full_transcript",
        "structured_ledger",
    ]
    assert reports[0]["context_expansion_comparison"] is None
    comparison = reports[2]["context_expansion_comparison"]["model-x"]
    assert comparison["reference_context"] == 8192
    assert len(comparison["persistent_failures"]) == 1
    assert comparison["improved_failures"] == []
    assert comparison["adoption_criteria_met"] is False
    for report in reports:
        model_report = report["models"]["model-x"]
        assert report["action_types"] == ["discussion"]
        assert model_report["fallback_rate"] == 1.0
        assert model_report["results"][0]["prompt_chars"] == len("systemuser")
        assert model_report["results"][0]["input_tokens"] == 120
        assert model_report["results"][0]["prompt_eval_count"] == 120
        assert model_report["results"][0]["output_tokens"] == 30
        assert model_report["results"][0]["output_token_count"] == 30
        assert model_report["results"][0]["effective_num_ctx"] == 8192
        assert model_report["results"][0]["memory_usage"]["system_memory_available_bytes"] == 1000
        assert model_report["vote_intent_specificity_rate"] == 1.0
        assert model_report["question_answer_rate"] is None
