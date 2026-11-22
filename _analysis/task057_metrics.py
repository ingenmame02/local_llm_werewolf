"""TASK-057 のローカルA/Bログから契約指標を読み出して並べる（実機Ollamaは使わない）。"""
import glob
import json
from pathlib import Path

REPO = Path(r"c:\Users\yukic\local_llm_werewolf")
LOG_DIR = REPO / "_analysis" / "logs_task057"
KEYS = (
    "avg_prompt_user_chars",
    "max_prompt_user_chars",
    "avg_prompt_core_chars",
    "avg_prompt_optional_chars",
    "dedup_references_total",
    "dedup_saved_chars_total",
    "duplicate_constraint_total",
    "optional_removed_chars_total",
    "avg_prompt_build_calls",
    "avg_contract_stage_attempts",
    "contract_stage1_removals",
    "contract_stage2_removals",
    "contract_stage3_removals",
    "contract_stage4_removals",
)


def main() -> None:
    rows: dict[str, dict[str, object]] = {}
    stage_counts: dict[str, dict[str, object]] = {}
    for path in sorted(LOG_DIR.glob("game_task057_*.jsonl")):
        payload = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if '"game_result"' in line
        ]
        if not payload:
            continue
        record = payload[-1]
        metrics = record.get("ledger_metrics") or {}
        policy = str(record.get("prompt_contract_policy"))
        rows[policy] = {key: metrics.get(key) for key in KEYS}
        stage_counts[policy] = {
            model: (stats or {}).get("contract_removal_stage_counts")
            for model, stats in (record.get("model_stats") or {}).items()
        }
    header = f"{'metric':34s}" + "".join(f"{policy:>14s}" for policy in rows)
    print(header)
    for key in KEYS:
        line = f"{key:34s}" + "".join(f"{str(rows[p].get(key)):>14s}" for p in rows)
        print(line)
    print("\n-- contract_removal_stage_counts（モデル別） --")
    for policy, per_model in stage_counts.items():
        print(policy)
        for model, counts in per_model.items():
            print(f"   {model}: {counts}")


if __name__ == "__main__":
    main()
