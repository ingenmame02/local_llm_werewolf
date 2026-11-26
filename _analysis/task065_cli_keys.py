"""TASK-065の集計キー（cli.py）をUTF-8で書き出す（読み取り専用）。README記述用の正確なキー名確認。"""
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_analysis" / "task065_cli_keys.txt"
CLI = ROOT / "src" / "werewolf" / "cli.py"

NEEDLES = (
    "intent",
    "question_acts",
    "speculation",
    "past_attribution",
    "variant",
    "unresolved_name",
    "integrity",
    "stabiliz",
    "relaxation",
    "timeout_fit",
)


def main() -> None:
    lines = io.open(CLI, encoding="utf-8").read().splitlines()
    picked = [
        f"{i + 1}: {line.strip()[:140]}"
        for i, line in enumerate(lines)
        if any(n in line for n in NEEDLES) and '"' in line
    ]
    OUT.write_text("\n".join(picked) + "\n", encoding="utf-8")
    print("wrote", OUT, len(picked))


if __name__ == "__main__":
    main()
