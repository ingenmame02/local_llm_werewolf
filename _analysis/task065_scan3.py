"""TASK-065配線位置の特定（読み取り専用・LL不使用）。UTF-8で結果を書き出す。"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "werewolf"
OUT = ROOT / "_analysis" / "task065_wiring_out.txt"


def grep(path: Path, pattern: str, limit: int = 60) -> list[str]:
    rx = re.compile(pattern)
    lines = io.open(path, encoding="utf-8").read().splitlines()
    hits = [
        f"{i + 1}: {line.strip()[:140]}" for i, line in enumerate(lines) if rx.search(line)
    ]
    return hits[:limit], len(lines)


def main() -> None:
    chunks: list[str] = []
    targets = {
        "cli.py": [
            r'add_argument\("--(prompt-layout|continuity|day-timeline|context-integrity|timeline-intent|name-normalization|thinking|body-reservation)',
            r"context_strategy|--log-dir|logger\.|GameLogger\(",
            r"actions\s*=|OllamaActions|client\.ping|base_url",
            r'"avg_|ledger_metrics\[|print\(f"',
        ],
        "llm_actions.py": [
            r'options\["num_ctx"\]|"num_ctx"',
            r"context_diagnostics|process_env_context|resolve_effective_context_limit|_context_budgets",
            r"def _emit|def _record_model_timeout|def _slow_model|thinking_budget\"\]",
            r"def _transport_failure_reason",
            r"answers_previous_question|_is_directed_to_player",
            r"LEDGER_REPAIR_ISSUES|topic_ledger|claim_ledger|public_timeline",
            r"import (context_budget|output_budget_trace|thinking_profile)",
        ],
        "engine.py": [
            r"answers_previous_question|register_topic|claim_ledger\.|topic_ledger\.|fact_ledger\.|hypothesis_ledger\.",
            r"inspect_public_body|_timeline_reference_fields|observation_metrics",
        ],
        "prompts.py": [
            r"def _is_directed_to_player|Player\\\\s|プレイヤー",
            r"pending_speakers|未発言|まだ発言",
        ],
        "discussion_format.py": [r"def answers_previous_question"],
    }
    for name, patterns in targets.items():
        text_blocks = [f"===== {name}"]
        for pattern in patterns:
            hits, total = grep(SRC / name, pattern)
            text_blocks.append(f"--- [{pattern}] total_lines={total} hits={len(hits)}")
            text_blocks.extend(hits)
        chunks.append("\n".join(text_blocks))

    # import health check of the new modules
    sys.path.insert(0, str(SRC.parent))
    try:
        from werewolf import context_integrity, name_resolution, output_stability  # noqa: F401

        imports_ok = True
    except Exception as exc:  # noqa: BLE001
        imports_ok = False
        chunks.append(f"IMPORT ERROR: {exc!r}")
    if imports_ok:
        chunks.append("new modules import OK")
    OUT.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
