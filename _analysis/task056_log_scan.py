"""TASK-056 背景確認: logs/game_15b1bdf65773.jsonl の Day 1 時系列と未来参照の疑いを洗う."""
import io
import json
import re
import sys
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "logs" / "game_15b1bdf65773.jsonl"

rows = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
print("rows", len(rows))

speeches = [r for r in rows if r.get("type") == "discussion_message"]
for row in speeches:
    body = str(row.get("content") or "").replace("\n", " ")
    print(
        f"seq={row.get('sequence')} d{row.get('day')} ph={row.get('phase')} "
        f"{row.get('speaker')}: {body[:150]}"
    )

print("\n=== day1 first Player 4 speech full ===")
for row in speeches:
    if row.get("day") == 1 and row.get("speaker") == "Player 4":
        print("seq", row.get("sequence"), repr(str(row.get("content"))[:1200]))
        break

future_pat = re.compile(
    r"(?:vote[dd]?|voted|executed|exiled|killed|attacked|died|was out|投票した|投票結果|処刑|襲撃|死んだ|言っていた|主張した|明かした|COした)",
    re.IGNORECASE,
)
print("\n=== future-ish mentions per Day1 speech (vs day's vote/night state) ===")
first_speech_day1 = [r for r in speeches if r.get("day") == 1]
for row in first_speech_day1[:6]:
    body = str(row.get("content") or "")
    hits = future_pat.findall(body)
    if hits:
        print("seq", row.get("sequence"), row.get("speaker"), sorted(set(h.lower() for h in hits)))
