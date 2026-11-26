"""TASK-065 README誤記修正（README内の該当文のみを安全に置換する）。"""
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def main() -> None:
    with io.open(README, encoding="utf-8", newline="") as handle:
        text = handle.read()
    bad = "\u904e\u53bb\u306e\u767a\u8a00as\u6839\u64da\u306e\u65ad\u5b9a\u3060\u305f"
    good = "\u904e\u53bb\u306e\u767a\u8a00\u3092\u6839\u64da\u306b\u3057\u305f\u65ad\u5b9a\u3060\u305f"
    n = text.count(bad)
    print("count_before:", n)
    if n == 0:
        print("nothing to replace")
        return
    fixed = text.replace(bad, good)
    with io.open(README, "w", encoding="utf-8", newline="") as handle:
        handle.write(fixed)
    print("replaced:", n)


if __name__ == "__main__":
    main()
