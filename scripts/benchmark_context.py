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

from evaluate_models import evaluate_scenario, generate_scenarios, select_player_configs
from werewolf import OllamaClient, Player, Role, load_player_configs

CONTEXT_ADOPTION_CRITERIA = {
    "minimum_discussion_scenarios": 10,
    "minimum_absolute_quality_metric_gain": 0.1,
    "quality_metrics": [
        "question_answer_rate",
        "novel_evidence_category_rate",
        "vote_intent_specificity_rate",
    ],
    "parse_success_must_not_regress": True,
    "maximum_average_latency_ratio": 1.25,
}


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

    def sample_now(self) -> dict[str, int | float | None]:
        return self._sample()

    def _sample(self) -> dict[str, int | float | None]:
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
        return sample

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
    model_filters: tuple[str, ...],
    action_types: tuple[str, ...],
    context_strategies: tuple[str, ...] = ("hybrid_retrieval",),
    reasoning_memory_modes: tuple[bool, ...] = (False, True),
) -> list[dict[str, Any]]:
    all_configs = load_player_configs(models_file)
    configs = select_player_configs(all_configs, list(model_filters))
    if len(configs) > 2:
        raise ValueError(
            "Context pilot selected more than two configured players; narrow --models."
        )
    player_names = tuple(config.name for config in all_configs)
    client = OllamaClient(base_url=base_url, timeout=timeout)
    reports: list[dict[str, Any]] = []
    configurations = [
        (context_length, context_strategy, reasoning_memory)
        for context_length in contexts
        for context_strategy in context_strategies
        for reasoning_memory in reasoning_memory_modes
    ]
    for context_length, context_strategy, reasoning_memory in configurations:
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
                for scenario in generate_scenarios(
                    actor,
                    player_names,
                    context_strategy,
                    reasoning_memory=reasoning_memory,
                ):
                    if scenario.action_type not in action_types:
                        continue
                    prompt_char_count = (
                        len(scenario.prompt_ctx.system_prompt)
                        + len(scenario.prompt_ctx.user_prompt)
                    )
                    result = evaluate_scenario(
                        client,
                        config.model,
                        scenario,
                        {
                            "temperature": temperature,
                            "num_ctx": context_length,
                        },
                    )
                    diagnostic = client.context_diagnostics(config.model)
                    memory_usage = sampler.sample_now()
                    model_records[config.model].append(
                        {
                            "round": round_index,
                            "action_type": scenario.action_type,
                            "discussion_format": result.get("discussion_format"),
                            "is_format_compliant": result["is_format_compliant"],
                            "is_parse_valid": result["is_parse_valid"],
                            "parse_success": result["is_parse_valid"],
                            "fallback_used": result["fallback_used"],
                            "elapsed_seconds": result["elapsed_seconds"],
                            "requested_num_ctx": context_length,
                            "effective_num_ctx": diagnostic["ollama_ps_context"],
                            "num_ctx_source": "benchmark_request",
                            "reasoning_memory_enabled": reasoning_memory,
                            "retry_count": result["retry_count"],
                            "request_attempts": result["request_attempts"],
                            "num_predict": result["num_predict"],
                            "num_predict_by_attempt": result[
                                "num_predict_by_attempt"
                            ],
                            "failure_reasons": result["failure_reasons"],
                            "thinking_present": result["thinking_present"],
                            "question_expected": scenario.question_expected,
                            "question_answered": result["question_answered"],
                            "new_evidence_categories": result[
                                "new_evidence_categories"
                            ],
                            "discussion_signals": result["discussion_signals"],
                            "vote_intent_specific": bool(
                                result["discussion_signals"].get("vote_proposals")
                            ),
                            "context_diagnostics": scenario.prompt_ctx.diagnostics,
                            "prompt_chars": prompt_char_count,
                            "input_tokens": result["prompt_eval_count"],
                            "prompt_eval_count": result["prompt_eval_count"],
                            "output_tokens": result["eval_count"],
                            "output_token_count": result["eval_count"],
                            "done_reason": result["done_reason"],
                            "memory_usage": memory_usage,
                            "context_length": context_length,
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
            question_records = [
                record
                for record in discussion_records
                if record["question_expected"]
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
                "fallback_rate": round(
                    sum(record["fallback_used"] for record in records) / count,
                    4,
                ),
                "parse_success_rate": round(
                    sum(record["is_parse_valid"] for record in records) / count,
                    4,
                ),
                "question_answer_rate": (
                    round(
                        sum(record["question_answered"] is True for record in question_records)
                        / len(question_records),
                        4,
                    )
                    if question_records
                    else None
                ),
                "novel_evidence_category_rate": (
                    round(
                        sum(bool(record["new_evidence_categories"]) for record in discussion_records)
                        / len(discussion_records),
                        4,
                    )
                    if discussion_records
                    else None
                ),
                "vote_intent_specificity_rate": (
                    round(
                        sum(record["vote_intent_specific"] for record in discussion_records)
                        / len(discussion_records),
                        4,
                    )
                    if discussion_records
                    else None
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
                "context_strategy": context_strategy,
                "reasoning_memory_enabled": reasoning_memory,
                "model_filters": list(model_filters),
                "action_types": list(action_types),
                "rounds": rounds,
                "num_parallel_expected": 1,
                "seed": 42,
                "num_parallel_process_environment": os.environ.get(
                    "OLLAMA_NUM_PARALLEL"
                ),
                "resource_metrics": resource_summary,
                "models": model_summary,
            }
        )
    reference_context = min(contexts)
    reports_by_configuration = {
        (
            report["context_strategy"],
            report["reasoning_memory_enabled"],
            report["num_ctx_request"],
        ): report
        for report in reports
    }
    for report in reports:
        reference = reports_by_configuration.get(
            (
                report["context_strategy"],
                report["reasoning_memory_enabled"],
                reference_context,
            )
        )
        report["context_adoption_criteria"] = CONTEXT_ADOPTION_CRITERIA
        if reference is None or report["num_ctx_request"] == reference_context:
            report["context_expansion_comparison"] = None
            continue
        comparisons: dict[str, Any] = {}
        for model, current_summary in report["models"].items():
            baseline_summary = reference["models"].get(model)
            if baseline_summary is None:
                continue
            baseline_results = {
                (record["round"], record["action_type"]): record
                for record in baseline_summary["results"]
            }
            current_results = {
                (record["round"], record["action_type"]): record
                for record in current_summary["results"]
            }
            improved: list[dict[str, Any]] = []
            persistent: list[dict[str, Any]] = []
            newly_failed: list[dict[str, Any]] = []
            for scenario_key in sorted(baseline_results.keys() & current_results.keys()):
                baseline_result = baseline_results[scenario_key]
                current_result = current_results[scenario_key]
                baseline_failed = not baseline_result["is_parse_valid"]
                current_failed = not current_result["is_parse_valid"]
                details = {
                    "round": scenario_key[0],
                    "action_type": scenario_key[1],
                    "baseline_failure_reasons": baseline_result["failure_reasons"],
                    "expanded_failure_reasons": current_result["failure_reasons"],
                }
                if baseline_failed and not current_failed:
                    improved.append(details)
                elif baseline_failed and current_failed:
                    persistent.append(details)
                elif not baseline_failed and current_failed:
                    newly_failed.append(details)

            quality_deltas = {
                metric: (
                    round(current_summary[metric] - baseline_summary[metric], 4)
                    if current_summary[metric] is not None
                    and baseline_summary[metric] is not None
                    else None
                )
                for metric in CONTEXT_ADOPTION_CRITERIA["quality_metrics"]
            }
            discussion_count = sum(
                record["action_type"] == "discussion"
                for record in current_summary["results"]
            )
            baseline_latency = baseline_summary["average_elapsed_seconds"]
            current_latency = current_summary["average_elapsed_seconds"]
            latency_ratio = (
                round(current_latency / baseline_latency, 4)
                if baseline_latency > 0
                else None
            )
            quality_gain = any(
                delta is not None
                and delta
                >= CONTEXT_ADOPTION_CRITERIA[
                    "minimum_absolute_quality_metric_gain"
                ]
                for delta in quality_deltas.values()
            )
            criteria_met = (
                discussion_count
                >= CONTEXT_ADOPTION_CRITERIA["minimum_discussion_scenarios"]
                and quality_gain
                and current_summary["parse_success_rate"]
                >= baseline_summary["parse_success_rate"]
                and latency_ratio is not None
                and latency_ratio
                <= CONTEXT_ADOPTION_CRITERIA["maximum_average_latency_ratio"]
            )
            comparisons[model] = {
                "reference_context": reference_context,
                "evaluated_context": report["num_ctx_request"],
                "improved_failures": improved,
                "persistent_failures": persistent,
                "new_failures": newly_failed,
                "quality_metric_deltas": quality_deltas,
                "discussion_scenario_count": discussion_count,
                "average_latency_ratio": latency_ratio,
                "adoption_criteria_met": criteria_met,
            }
        report["context_expansion_comparison"] = comparisons
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
        default=[8192, 12288, 16384, 32768],
    )
    parser.add_argument(
        "--context-strategies",
        nargs="+",
        choices=(
            "full_transcript",
            "structured_ledger",
            "hybrid_retrieval",
        ),
        default=[
            "full_transcript",
            "structured_ledger",
            "hybrid_retrieval",
        ],
    )
    parser.add_argument(
        "--reasoning-memory",
        choices=("off", "on", "both"),
        default="both",
        help="Compare without/with deterministic player reasoning memos.",
    )
    parser.add_argument(
        "--models",
        action="append",
        required=True,
        metavar="MODEL_OR_NAME",
        help="Model/player substring filter. Supply one or two filters to limit the pilot.",
    )
    parser.add_argument(
        "--actions",
        action="append",
        choices=("discussion", "vote", "attack", "seer", "discussion_seer_co"),
        default=None,
        help="Scenario type to run (repeatable); defaults to discussion only.",
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
    if len(args.models) > 2:
        parser.error("Use at most two --models filters for the context pilot.")
    action_types = tuple(args.actions or ["discussion"])
    reasoning_memory_modes = (
        (False, True)
        if args.reasoning_memory == "both"
        else (args.reasoning_memory == "on",)
    )
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
        tuple(args.models),
        action_types,
        tuple(args.context_strategies),
        reasoning_memory_modes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output_file:
        for report in reports:
            output_file.write(json.dumps(report, ensure_ascii=False) + "\n")
    print(f"{len(reports)}件の設定比較を保存しました: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
