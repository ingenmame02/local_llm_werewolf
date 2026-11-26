"""TASK-065: unknown_player_reference の実原因を本文単位で特定する（読み取り専用・LL不使用）。

`game_425755ce9a83` の ledger_repair(reason=unknown_player_reference) 4件の元になった本文で、
判定式 `Player\\s+\\d+` が実在名集合と不一致とみなしたトークンを repr で列挙する。
併せて NFKC 正規化後に解決できるか（＝表記ゆれが原因か）を分ける。
"""
import io
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "game_425755ce9a83.private.jsonl"
OUT = ROOT / "_analysis" / "task065_unresolved_out.txt"

DETECT = re.compile(r"Player\s+\d+")
LOOSE = re.compile(
    r"(?i)(?:player|p\s*laye[rr]|プレイヤー)\s*[-_]?\s*(\d+)|(?<![0-9A-Za-z])P(\d+)(?![0-9])"
)


def walk(value, path=""):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from walk(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from walk(item, f"{path}[{i}]")


def fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def main() -> None:
    events = []
    with io.open(LOG, encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                events.append(json.loads(raw))
    known = {f"Player {i}" for i in range(1, 9)}
    lines: list[str] = []

    repair_bodies: list[tuple[str, str]] = []
    repairs = [e for e in events if e.get("type") == "ledger_repair"]
    lines.append(f"ledger_repairs={len(repairs)}")
    for rec in repairs:
        lines.append(
            f"- reason={rec.get('reason')} player={rec.get('player')} model={rec.get('model')}"
            f" preserved={rec.get('body_preserved')} seq={rec.get('sequence')}"
        )

    lines.append("\n== detector mismatches in generated/discussion bodies (repr) ==")
    bad = Counter()
    fixed_by_nfkc = Counter()
    kinds = Counter()
    for rec in events:
        if rec.get("type") not in {"llm_action_completed", "discussion_message"}:
            continue
        body_keys = [k for k in ("content", "response", "raw_response", "body") if isinstance(rec.get(k), str)]
        for key in body_keys:
            text = rec[key]
            for token in DETECT.findall(text):
                if token not in known:
                    bad[(rec.get("player"), repr(token))] += 1
                    if fold(token) in {fold(n) for n in known}:
                        fixed_by_nfkc[repr(token)] += 1
    for (player, token), count in bad.most_common(40):
        lines.append(f"{count}x player={player} token={token}")
    lines.append(f"total_mismatch={sum(bad.values())} nfkc_recoverable={sum(fixed_by_nfkc.values())}")

    lines.append("\n== mention label variants present in bodies ==")
    for rec in events:
        if rec.get("type") != "discussion_message":
            continue
        text = rec.get("content") or ""
        for match in LOOSE.finditer(text):
            raw = match.group(0)
            digit = int((match.group(1) or match.group(2) or "0").translate(str.maketrans("０１２３４５６７８９", "0123456789")))
            kind = (
                "canonical"
                if raw == f"Player {digit}"
                else "nbsp_or_no_space"
                if re.fullmatch(r"(?i)player\s*\d+", raw) and not raw.startswith(f"Player {digit}")
                else "katakana"
                if "プレイヤー" in raw
                else "p_short"
            )
            kinds[(kind, "known" if f"Player {digit}" in known else "unknown")] += 1
    for (kind, status), count in kinds.most_common():
        lines.append(f"{count}x kind={kind} target={status}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
