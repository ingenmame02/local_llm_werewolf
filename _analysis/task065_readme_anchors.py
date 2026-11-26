"""TASK-065 README挿入位置の特定（読み取り専用・UTF-8出力）。"""
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_analysis" / "task065_readme_anchors.txt"
README = ROOT / "README.md"

NEEDLES = (
    "| `--language",
    "| `--thinking ",
    "| `--num-ctx",
    "- 日をまたぐ論理状態",
    "- 引用の時点分類",
    "- 役職能力の境界",
    "- 発言時点の時系列明示",
)


def main() -> None:
    lines = io.open(README, encoding="utf-8").read().splitlines()
    hits = []
    for i, line in enumerate(lines):
        text = line.strip()
        for needle in NEEDLES:
            if text.startswith(needle.rstrip("`")) or text.startswith(needle):
                hits.append(f"{i + 1}: {text[:90]}")
                break
    OUT.write_text("\n".join(hits) + f"\nTOTAL {len(lines)}\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
