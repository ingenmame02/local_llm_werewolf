"""TASK-065: 全イベント文字列からプレイヤー名不一致を総ざらい（読み取り専用）。

`unknown_player_reference` の判定は `Player\\s+\\d+` を実在名集合と照合する。
表記ゆれ（`Player1` / 全角数字 / カタカナ）を正規化した上で、
- 不一致（未解決参照）になるトークン
- 質問・依頼行為なのに speech_from_unspoken_player へ分類された文
を洗い出す。
"""
import io
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "game_425755ce9a83.private.jsonl"

DETECT = re.compile(r"Player\s+\d+")


def walk(value, path=""):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from walk(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk(item, f"{path}[{index}]")


def norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def main() -> None:
    events = []
    with io.open(LOG, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))

    known = {f"Player {i}" for i in range(1, 9)}
    print("== unknown tokens anywhere (raw exact match) ==")
    counter = Counter()
    for rec in events:
        for path, text in walk(rec):
            for token in DETECT.findall(text):
                if token not in known:
                    counter[(rec.get("type"), path, token)] += 1
    for key, count in counter.most_common(40):
        print(count, key)

    print("\n== unknown tokens after NFKC normalization (what a canonical matcher sees) ==")
    fixed = Counter()
    still_bad = Counter()
    for rec in events:
        for path, text in walk(rec):
            folded = norm(text)
            for token in DETECT.findall(folded):
                if token in known:
                    fixed[(rec.get("type"), token)] += 1
                else:
                    still_bad[(rec.get("type"), repr(token))] += 1
    print("recoverable via NFKC:", dict(fixed))
    print("still unresolved:", dict(still_bad))

    print("\n== the Day 2 flagged sentence (timeline finding body) ==")
    for rec in events:
        findings = rec.get("future_event_reference_findings") or []
        for f in findings:
            print(rec.get("type"), rec.get("player"), "day", rec.get("day"))
            print("   excerpt:", norm(f.get("sentence_excerpt", "")))
            print("   subtype:", f.get("subtype"), "cause:", f.get("cause"))

    print("\n== bodies with request/question phrasing toward unspoken players ==")
    REQ = re.compile(
        r"(説明を?(?:求め|聞)|回答を?(?:求め|期待)|教えて|聞かせて|コメントを?(?:求め|期待)|説明責任)"
    )
    ATTR = re.compile(r"(言った|言い|発言|述べ|主張|公表|答えた|回答|指摘|名乗)")
    for rec in events:
        if rec.get("type") != "discussion_message":
            continue
        body = norm(rec.get("body") or rec.get("content") or "")
        for sent in re.split(r"[。！？\n]", body):
            if REQ.search(sent) and ATTR.search(sent):
                print(rec.get("player"), "|", sent[:110])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
