from scripts import benchmark_context
from scripts.benchmark_context import ResourceSampler
from scripts.diagnose_ollama_context import collect_context_report
from scripts.evaluate_models import PromptTestScenario
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

    def evaluate(client, model, scenario, options):
        requests.append((model, scenario.action_type, options))
        return {
            "discussion_format": "two_block",
            "is_format_compliant": True,
            "is_parse_valid": False,
            "fallback_used": True,
            "elapsed_seconds": 1.5,
            "retry_count": 1,
            "failure_reasons": ["discussion_output_format"],
            "done_reason": "stop",
            "prompt_eval_count": 120,
            "eval_count": 30,
            "error": None,
        }

    monkeypatch.setattr(benchmark_context, "load_player_configs", lambda _: [config])
    monkeypatch.setattr(benchmark_context, "select_player_configs", lambda c, f: c)
    monkeypatch.setattr(benchmark_context, "generate_scenarios", lambda *_: scenarios)
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
    )

    assert requests == [
        ("model-x", "discussion", {"temperature": 0.1, "num_ctx": 8192}),
        ("model-x", "discussion", {"temperature": 0.1, "num_ctx": 12288}),
    ]
    for report in reports:
        model_report = report["models"]["model-x"]
        assert report["action_types"] == ["discussion"]
        assert model_report["fallback_rate"] == 1.0
        assert model_report["results"][0]["prompt_chars"] == len("systemuser")
        assert model_report["results"][0]["input_tokens"] == 120
        assert model_report["results"][0]["output_tokens"] == 30
