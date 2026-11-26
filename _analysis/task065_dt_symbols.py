"""TASK-065: day_timeline.py の検出関数と意図分離フック位置を確認する（読み取り専用）。"""
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "werewolf" / "day_timeline.py"
OUT = ROOT / "_analysis" / "task065_dt_symbols.txt"

KEYWORDS = ("def ", "class ", "FUTURE_SPEECH", "SUBTYPE_", "hedge", "HEDGE", "mention", "unspoken")


def main() -> None:
    lines = io.open(SRC, encoding="utf-8").read().splitlines()
    picked = []
    for index, line in enumerate(lines):
        if any(word in line for word in KEYWORDS):
            picked.append(f"{index + 1}: {line.strip()[:120]}")
    OUT.write_text("\n".join(picked) + "\n", encoding="utf-8")
    print("wrote", OUT, len(picked), "lines")


if __name__ == "__main__":
    main()
