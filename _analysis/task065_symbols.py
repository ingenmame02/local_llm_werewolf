"""TASK-065実装前の配線確認: 既存コードの該当シンボル行番号をUTF-8で書き出す（読み取り専用）。

コンテキスト整合チェック・thinking安定化・意図分離・名前正規化の各フックを、
既存のどの関数・イベントキーへ繋ぐかを確定するために使う。
"""
import io
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "werewolf"
OUT = ROOT / "_analysis" / "task065_symbols.txt"

TARGETS = {
    "ollama.py": [r"def context_diagnostics", r"ollama_ps_context", r"modelfile_num_ctx", r"def ping"],
    "context_budget.py": [r"^def |^class |limit|conflict"],
    "llm_actions.py": [
        r"def _resolve_context_budget",
        r"effective_context_resolution",
        r"def _options_for",
        r"timed_out",
        r"thinking_budget_disabled",
        r"def _record_generation_failure",
        r"output_budget_trace|OutputBudgetTrace",
        r"def _short_retry_prompt|short_retry",
        r"LEDGER_REPAIR_ISSUES",
        r"unknown_player_reference",
        r"unresolvable_references",
    ],
    "engine.py": [
        r"future_event_reference",
        r"inspect_public_body",
        r"unknown_player_reference",
        r"ledger_repair",
        r"def record_speech|speech_id",
        r"def _diagnostic",
    ],
    "cli.py": [
        r"add_argument\(\"--day-timeline\"",
        r"add_argument\(\"--continuity\"",
        r"future_event_references",
        r"timeline_",
        r"ledger_repairs",
        r"unknown_player_reference",
        r"model_stats",
        r"action_profiles",
    ],
    "day_timeline.py": [
        r"^def |^class |^[A-Z_]+[:=]",
        r"findings|category",
    ],
}


def main() -> None:
    chunks: list[str] = []
    for name, patterns in TARGETS.items():
        path = SRC / name
        lines = io.open(path, encoding="utf-8").read().splitlines()
        blocks: list[str] = []
        for pattern in patterns:
            rx = re.compile(pattern)
            hits = [
                f"{index + 1}: {line.strip()[:120]}"
                for index, line in enumerate(lines)
                if rx.search(line)
            ]
            blocks.append(f"--- [{pattern}] hits={len(hits)}\n" + "\n".join(hits[:40]))
        chunks.append(f"===== {name} ({len(lines)} lines)\n" + "\n".join(blocks))
    OUT.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
