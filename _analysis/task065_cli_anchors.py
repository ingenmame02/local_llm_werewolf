"""TASK-065配線位置の特定（cli.py）。読み取り専用・UTF-8出力。"""
import io
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "src" / "werewolf" / "cli.py"
OUT = ROOT / "_analysis" / "task065_cli_anchors.txt"

PATTERNS = [
    r"add_argument\(\"--prompt-layout\"",
    r"add_argument\(\"--continuity\"",
    r"add_argument\(\"--day-timeline\"",
    r"getattr\(args, \"continuity\"",
    r"\"continuity_policy\": continuity_policy",
    r"policy_label",
    r"timeline_category_counts",
    r"_timeline_public_path_counts",
    r"def summary_metrics",
    r"print_summary",
    r"def print_banner",
    r"client\.ping\(\)",
    r"def build_parser|subparsers|add_parser",
    r"thinking_budget_fallbacks",
    r"\"ledger_repairs\"",
]


def main() -> None:
    lines = io.open(CLI, encoding="utf-8").read().splitlines()
    chunks: list[str] = [f"total_lines={len(lines)}"]
    for pattern in PATTERNS:
        rx = re.compile(pattern)
        hits = [
            f"{i + 1}: {lines[i][:120]}"
            for i in range(len(lines))
            if rx.search(lines[i])
        ]
        chunks.append(f"--- [{pattern}] hits={len(hits)}")
        chunks.extend(hits[:14])
    OUT.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
