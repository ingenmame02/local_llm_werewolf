#!/usr/bin/env python3
"""Record Ollama model and runtime context settings without changing a model."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf import OllamaClient, load_player_configs


def collect_context_report(
    models: tuple[str, ...],
    api_request_num_ctx: int | None,
    client: OllamaClient | None = None,
) -> dict[str, Any]:
    ollama = client or OllamaClient()
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "api_request_num_ctx": api_request_num_ctx,
        "api_request_num_ctx_source": (
            "diagnostic_argument" if api_request_num_ctx is not None else "omitted"
        ),
        "models": {
            model: ollama.context_diagnostics(model)
            for model in dict.fromkeys(models)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record API, Modelfile, environment, and loaded Ollama context values."
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        default=REPO_ROOT / "config" / "players.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "config" / "ollama_context.json",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="API request value to document; omitted values are not sent to Ollama.",
    )
    args = parser.parse_args()
    if args.num_ctx is not None and args.num_ctx <= 0:
        parser.error("--num-ctx must be positive")

    configs = load_player_configs(args.models_file)
    report = collect_context_report(
        tuple(config.model for config in configs),
        args.num_ctx,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"診断結果を保存しました: {args.output}")
    for model, values in report["models"].items():
        print(
            f"{model}: Modelfile num_ctx={values['modelfile_num_ctx']}, "
            f"ollama ps CONTEXT={values['ollama_ps_context']}, "
            f"PROCESSOR={values['ollama_ps_processor']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
