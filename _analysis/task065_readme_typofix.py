"""TASK-065 README軽微修正（読み取り専用診断＋1置換のみ・UTF-8安全）。"""
import io
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

PATTERN = re.compile("\u767a\u8a00(?:as|\uff41\uff53|a s|\\s){0,2}\u6839\u64da\u306e\u65ad\u5b9a\u3060\u305f")
REPLACEMENT = "\u767a\u8a00\u3092\u6839\u64da\u306b\u3057\u305f\u65ad\u5b9a\u3060\u305f"


def main() -> None:
    with io.open(README, encoding="utf-8", newline="") as handle:
        text = handle.read()
    matches = PATTERN.findall(text)
    print("typo_matches:", [repr(m) for m in matches])
    if not matches:
        return
    fixed = PATTERN.sub(REPLACEMENT, text)
    with io.open(README, "w", encoding="utf-8", newline="") as handle:
        handle.write(fixed)
    print("replaced", len(matches), "occurrence(s)")


if __name__ == "__main__":
    main()
