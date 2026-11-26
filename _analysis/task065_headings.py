"""TASK-065: TASKS.md の見出し行番号を UTF-8 で書き出す（読み取り専用）。

TASK-065 の「実装案」「完了条件」の該当範囲を特定するために使う。
"""
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "TASKS.md"
OUT = ROOT / "_analysis" / "task065_headings.txt"


def main() -> None:
    lines = io.open(TASKS, encoding="utf-8").read().splitlines()
    hits = [
        f"{index + 1}: {line.strip()[:160]}"
        for index, line in enumerate(lines)
        if line.startswith("#") or "TASK-065" in line
    ]
    OUT.write_text("\n".join(hits) + "\n", encoding="utf-8")
    print("total_lines", len(lines), "wrote", OUT)


if __name__ == "__main__":
    main()
