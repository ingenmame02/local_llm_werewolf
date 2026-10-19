#!/usr/bin/env python3
"""Compare serial Ollama prompt evaluations across context sizes."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    import psutil
except ImportError:
    psutil = None

from evaluate_models import evaluate_scenario, generate_scenarios
from werewolf import OllamaClient, Player, Role, load_player_configs


class ResourceSampler:
    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, int | float | None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, int | float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return self.summary()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        sample: dict[str, int | float | None] = {
            "system_cpu_percent": None,
            "system_memory_available_bytes": None,
            "ollama_cpu_percent": None,
            "ollama_rss_bytes": None,
            "gpu_utilization_percent": None,
            "gpu_memory_used_mib": None,
        }
        if psutil is not None:
            sample["system_cpu_percent"] = psutil.cpu_percent(interval=None)
            sample["system_memory_available_bytes"] = psutil.virtual_memory().available
            ollama_processes = []
            for process in psutil.process_iter(("name", "memory_info")):
                try:
                    name = (process.info["name"] or "").casefold()
                    if name in {"ollama", "ollama.exe"}:
                        ollama_processes.append(process)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if ollama_processes:
                process_cpu = 0.0
                rss_bytes = 0
                for process in ollama_processes:
                    try:
                        process_cpu += process.cpu_percent(interval=None)
                        rss_bytes += process.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                sample["ollama_cpu_percent"] = process_cpu
                sample["ollama_rss_bytes"] = rss_bytes
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            result = None
        if result is not None and result.returncode == 0:
            readings: list[tuple[float, float]] = []
            for line in result.stdout.splitlines():
                values = line.split(",")
                if len(values) != 2:
                    continue
                try:
                    readings.append((float(values[0].strip()), float(values[1].strip())))
                except ValueError:
                    continue
            if readings:
                sample["gpu_utilization_percent"] = max(value[0] for value in readings)
                sample["gpu_memory_used_mib"] = sum(value[1] for value in readings)
        self.samples.append(sample)

    def summary(self) -> dict[str, int | float | None]:
        result: dict[str, int | float | None] = {
            "sample_count": len(self.samples),
            "psutil_available": psutil is not None,
            "nvidia_smi_available": any(
                sample["gpu_utilization_percent"] is not None
                for sample in self.samples
            ),
        }
        for key in (
            "system_cpu_percent",
            "system_memory_available_bytes",
            "ollama_cpu_percent",
            "ollama_rss_bytes",
            "gpu_utilization_percent",
            "gpu_memory_used_mib",
        ):
            values = [
                float(sample[key])
                for sample in self.samples
                if sample[key] is not None
            ]
            result[f"{key}_average"] = (
                round(statistics.fmean(values), 2) if values else None
            )
            result[f"{key}_max"] = round(max(values), 2) if values else None
        return result


def run_context_benchmark(
    models_file: Path,
    contexts: tuple[int, ...],
    rounds: int,
    timeout: float,
    temperature: float,
    base_url: str,
) -> list[dict[str, Any]]:
    configs = load_player_configs(models_file)
    player_names = tuple(config.name for config in configs)
    client = OllamaClient(base_url=base_url, timeout=timeout)
    reports: list[dict[str, Any]] = []
    for context_length in contexts:
        sampler = ResourceSampler()
        sampler.start()
        model_records: dict[str, list[dict[str, Any]]] = {
            config.model: [] for config in configs
        }
        model_diagnostics: dict[str, dict[str, object]] = {}
        for round_index in range(1, rounds + 1):
            for config in configs:
                actor = Player(
                    name=config.name,
                    model=config.model,
                    role=Role.VILLAGER,
                    response_language=config.response_language,
                )
                for scenario in generate_scenarios(actor, player_names):
                    result = evaluate_scenario(
                        client,
                        config.model,
                        scenario,
                        {
                            "temperature": temperature,
                            "num_ctx": context_length,
                        },
                    )
                    model_records[config.model].append(
                        {
                            "round": round_index,
                            "action_type": scenario.action_type,
                            "discussion_format": result.get("discussion_format"),
                            "is_format_compliant": result["is_format_compliant"],
                            "is_parse_valid": result["is_parse_valid"],
                            "elapsed_seconds": result["elapsed_seconds"],
                            "retry_count": result["retry_count"],
                            "error_kind": (
                                result["error"].get("kind")
                                if result["error"] is not None
                                else None
                            ),
                        }
                    )
                model_diagnostics[config.model] = client.context_diagnostics(
                    config.model
                )
        resource_summary = sampler.stop()
        model_summary: dict[str, dict[str, Any]] = {}
        for model, records in model_records.items():
            diagnostic = model_diagnostics[model]
            count = len(records)
            discussion_records = [
                record for record in records if record["action_type"] == "discussion"
            ]
            model_summary[model] = {
                "scenario_count": count,
                "discussion_two_block_rate": (
                    round(
                        sum(
                            record["discussion_format"] == "two_block"
                            for record in discussion_records
                        )
                        / len(discussion_records),
                        4,
                    )
                    if discussion_records
                    else None
                ),
                "format_compliance_rate": round(
                    sum(record["is_format_compliant"] for record in records) / count,
                    4,
                ),
                "parse_success_rate": round(
                    sum(record["is_parse_valid"] for record in records) / count,
                    4,
                ),
                "average_elapsed_seconds": round(
                    statistics.fmean(
                        record["elapsed_seconds"] for record in records
                    ),
                    4,
                ),
                "retry_count": sum(record["retry_count"] for record in records),
                "loaded_context": diagnostic["ollama_ps_context"],
                "loaded_processor": diagnostic["ollama_ps_processor"],
                "modelfile_num_ctx": diagnostic["modelfile_num_ctx"],
                "results": records,
            }
        reports.append(
            {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "num_ctx_request": context_length,
                "num_parallel_expected": 1,
                "num_parallel_process_environment": os.environ.get(
                    "OLLAMA_NUM_PARALLEL"
                ),
                "resource_metrics": resource_summary,
                "models": model_summary,
            }
        )
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare model quality, latency, and resource use by Ollama num_ctx."
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        default=REPO_ROOT / "config" / "players.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "evaluation-results" / "context-benchmark.jsonl",
    )
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=[4096, 8192],
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--base-url", default="http://localhost:11434")
    args = parser.parse_args()
    if any(context <= 0 for context in args.context_lengths):
        parser.error("context lengths must be positive")
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if psutil is None:
        print(
            "注意: psutilがありません。CPU/メモリ使用量は未計測です。"
            "計測には pip install -e .[benchmark] を実行してください。",
            file=sys.stderr,
        )
    reports = run_context_benchmark(
        args.models_file,
        tuple(args.context_lengths),
        args.rounds,
        args.timeout,
        args.temperature,
        args.base_url,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output_file:
        for report in reports:
            output_file.write(json.dumps(report, ensure_ascii=False) + "\n")
    print(f"{len(reports)}件の設定比較を保存しました: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
