from scripts.benchmark_context import ResourceSampler
from scripts.diagnose_ollama_context import collect_context_report


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
