"""TASK-066: モックログから再生成コンテキストの記録が出ているか確認する簡易チェッカー。"""
from __future__ import annotations

import glob
import json
import sys


def check(directory: str) -> None:
    files = sorted(glob.glob(f"{directory}/*.jsonl"))
    print(directory, len(files))
    for path in files:
        init_policy = None
        msg_stages: dict[str, int] = {}
        named_events: dict[str, int] = {}
        summary_retry_keys: list[str] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                kind = str(row.get("type"))
                if kind == "game_init":
                    init_policy = row.get("retry_context_policy")
                elif kind == "discussion_message":
                    stage = str(row.get("body_source_stage"))
                    msg_stages[stage] = msg_stages.get(stage, 0) + 1
                else:
                    named_events[kind] = named_events.get(kind, 0) + 1
                if kind == "game_result":
                    metrics = row.get("ledger_metrics") or {}
                    summary_retry_keys = sorted(
                        key for key in metrics if key.startswith("retry_")
                    )
        print(
            f"  {path.split(chr(92))[-1]} policy={init_policy}"
            f" body_source_stage={msg_stages}"
        )
        print(
            "    retry events:",
            {
                name: count
                for name, count in named_events.items()
                if "retry" in name or "compression" in name
            },
        )
        print(
            "    all event types:",
            {name: count for name, count in sorted(named_events.items())},
        )
        print(f"    result retry keys: {summary_retry_keys[:14]}")


if __name__ == "__main__":
    for directory in sys.argv[1:] or ["_analysis"]:
        check(directory)
