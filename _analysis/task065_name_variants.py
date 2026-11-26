"""TASK-065: 基準ログのプレイヤー名表記ゆれを拾う（読み取り専用・LL不使用）。

`unknown_player_reference` は `Player\\s+\\d+` を実在名照合するため、`Player1` のような
表記ゆれそのものは拾われない。では何が引っかかったのか——元本文の実在しない番号と、
公開発言中の全表記（`Player N` / `PlayerN` / `player n` / 全角 / `P{N}` / プレイヤーN）
を数え、正規化で吸収できる余地を測る。
"""
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "game_425755ce9a83.private.jsonl"
PUBLIC_LOG = ROOT / "logs" / "game_425755ce9a83.jsonl"


def rows(path):
    with io.open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


VARIANTS = [
    ("canonical", re.compile(r"Player\s+(\d+)")),
    ("no_space", re.compile(r"[Pp][Ll][Aa][Yy][Ee][Rr](\d+)")),
    ("fullwidth", re.compile(r"ｐｌａｙｅｒ(\d+)", re.IGNORECASE)),
    ("p_short", re.compile(r"\bP(\d+)\b")),
    ("katakana", re.compile(r"プレイヤー\s*(\d+)")),
]


def main() -> None:
    data = list(rows(LOG))
    known = {f"Player {i}" for i in range(1, 9)}

    print("== ledger_repair(unknown_player_reference) full records ==")
    for rec in data:
        if rec.get("type") == "ledger_repair":
            keep = {
                k: v
                for k, v in rec.items()
                if k
                not in {
                    "prompt",
                    "messages",
                    "system_prompt",
                    "user_prompt",
                }
            }
            print(json.dumps(keep, ensure_ascii=False)[:1500])
            print("---")

    print("\n== raw responses / bodies with non-canonical spellings ==")
    resp_keys = Counter()
    hits = []
    for rec in data:
        for key, value in rec.items():
            if not isinstance(value, str):
                continue
            if key in {"note", "reason", "type", "player", "model", "action", "phase_label"}:
                continue
            text = value
            if len(text) < 4:
                continue
            found = []
            for label, pattern in VARIANTS:
                for m in pattern.finditer(text):
                    num = int(m.group(1))
                    found.append((label, m.group(0), num))
            if not found:
                continue
            resp_keys[(rec.get("type"), key)] += 1
            bad_num = [t for t in found if t[2] not in range(1, 9)]
            non_canon = [t for t in found if t[0] != "canonical"]
            unknown = [t for t in found if f"Player {t[2]}" not in known]
            if bad_num or non_canon or unknown:
                hits.append((rec.get("type"), rec.get("player"), rec.get("action"), key, found[:8]))
    print("keys with player mentions:", dict(list(resp_keys.items())[:12]))
    print(f"anomalous hits={len(hits)}")
    for h in hits[:25]:
        print(h)

    print("\n== canonical vs variant totals over public discussion bodies ==")
    pub = list(rows(PUBLIC_LOG))
    canon = Counter()
    variant_only = Counter()
    for rec in pub:
        if rec.get("type") != "discussion_message":
            continue
        body = rec.get("body") or rec.get("content") or ""
        nums = {int(m.group(1)) for m in VARIANTS[0][1].finditer(body)}
        other = set()
        for label, pattern in VARIANTS[1:]:
            other |= {int(m.group(1)) for m in pattern.finditer(body)}
        canon.update(nums)
        for n in other - nums:
            variant_only[n] += 1
    print("canonical mention counts:", dict(sorted(canon.items())))
    print("variant-only (missed by canonical regex):", dict(sorted(variant_only.items())))

    print("\n== per-model unknown-number mentions in raw responses ==")
    bad = Counter()
    for rec in data:
        if rec.get("type") not in {"llm_action_completed", "llm_generation_attempt"}:
            continue
        for key, value in rec.items():
            if isinstance(value, str) and "Player" in value:
                for m in re.finditer(r"[Pp]layer\s*(\d+)", value):
                    num = int(m.group(1))
                    if num < 1 or num > 8:
                        bad[(rec.get("player"), rec.get("action"), m.group(0))] += 1
    print(dict(bad))

    print("\n== exact tokens matched by the detector regex (Player\\s+\\d+) ==")
    detector = re.compile(r"Player\s+\d+")
    mismatched = Counter()
    examples = {}
    for rec in data:
        if rec.get("type") not in {"llm_action_completed", "llm_generation_attempt"}:
            continue
        for key, value in rec.items():
            if not isinstance(value, str):
                continue
            for match in detector.findall(value):
                if match not in known:
                    mismatched[(rec.get("player"), rec.get("action"), repr(match))] += 1
                    examples.setdefault(repr(match), value[:300])
    for key, count in mismatched.most_common(20):
        print(count, key)
        print("   e.g.", examples[key[2]][:160].replace("\n", "\\n"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
