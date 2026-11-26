"""TASK-065診断: name_resolution.py の該当範囲を repr で確認する（読み取り専用）。"""
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "werewolf" / "name_resolution.py"
OUT = ROOT / "_analysis" / "task065_nr_view.txt"


def main() -> None:
    lines = io.open(SRC, encoding="utf-8").readlines()
    parts: list[str] = []
    for start, stop, label in ((58, 80, "fold"), (240, 260, "orphan"), (310, len(lines), "tail")):
        parts.append(f"===== {label} lines {start + 1}-{stop}")
        for index in range(start, min(stop, len(lines))):
            parts.append(f"{index + 1}| {lines[index]!r}")
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
