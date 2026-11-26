"""TASK-065 README誤記の最終修正（実ファイルの実測codepointに合わせて置換する）。"""
import io
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

PATTERN = re.compile(
    "\u767a\u8a00"          # 発言
    "[^\\u3000-\\u9fff]{0,3}"  # 発音記号・アスキー表記など（as / ａｓ 等）
    "\u6839\u64da\u306e\u65ad\u5b9a\u3060\u3051"  # 根拠の断定け?? -> real: の断定だけ
)
PATTERN2 = re.compile(
    "\u767a\u8a00[^\\u3000-\\u9fff]{0,3}\u6839\u64da\u306e\u65ad\u5b9a\u3060\u3051"
)
# 実際の綴りは「根拠の断定だけ」（だけ = U+3060 U+3051... ではなく U+3060/U+3051 は だ/け）
REPLACEMENT = "\u767a\u8a00\u3092\u6839\u64da\u306b\u3057\u305f\u65ad\u5b9a\u3060\u3051"


def main() -> None:
    with io.open(README, encoding="utf-8", newline="") as handle:
        text = handle.read()
    hits = [repr(m.group(0)) for m in PATTERN2.finditer(text)]
    print("pattern_hits:", len(hits), hits[:3])
    if not hits:
        print("no match (already fine?)")
        return
    fixed = PATTERN2.sub(REPLACEMENT, text)
    with io.open(README, "w", encoding="utf-8", newline="") as handle:
        handle.write(fixed)
    print("replaced:", len(hits))


if __name__ == "__main__":
    main()
