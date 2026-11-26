"""TASK-065: 基準ログ game_425755ce9a83 の未来参照文と該当本文を UTF-8 で書き出す。

意図分離（過去帰属／これから質問する行為／これからの行動推測）の語彙設計に使う。
読み取り専用・LL不使用。
"""
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "game_425755ce9a83.private.jsonl"
OUT = ROOT / "_analysis" / "task065_sentences.txt"


def main() -> None:
    lines: list[str] = []
    with io.open(LOG, encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            rec = json.loads(raw)
            for finding in rec.get("future_event_reference_findings") or []:
                lines.append(
                    "[FINDING] {} day={} cat={} subtype={} cause={}\n  sentence: {}".format(
                        rec.get("player"),
                        rec.get("day"),
                        finding.get("category"),
                        finding.get("subtype"),
                        finding.get("cause"),
                        finding.get("sentence_excerpt"),
                    )
                )
            if rec.get("type") == "discussion_message":
                body = rec.get("body") or rec.get("content") or ""
                if "プレイヤー6" in body or "Player6" in body:
                    lines.append(
                        "[BODY] {} day={}\n{}".format(rec.get("player"), rec.get("day"), body)
                    )
    OUT.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT, len(lines), "blocks")


if __name__ == "__main__":
    main()
