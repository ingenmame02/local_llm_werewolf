"""TASK-060: モックゲームのログに配置指標が乗ったかを確認する（LL不使用）。"""
import json
import sys
from pathlib import Path


def main(log_dir: str) -> None:
    logs = sorted(Path(log_dir).glob("game_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not logs:
        print("[FAIL] no game log in", log_dir)
        return
    path = logs[-1]
    result = None
    layouts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("type") == "game_init":
            print("prompt_layout_policy(init):", record.get("prompt_layout_policy"))
        if record.get("type") == "discussion_context_selected":
            label = record.get("prompt_layout")
            layouts[str(label)] = layouts.get(str(label), 0) + 1
        if record.get("type") == "game_result":
            result = record
    if result is None:
        print("[FAIL] no game_result in", path.name)
        return
    metrics = result.get("ledger_metrics", {})
    keys = (
        "avg_prompt_user_chars",
        "avg_prompt_zone_primacy_chars",
        "avg_prompt_zone_middle_chars",
        "avg_prompt_zone_recency_chars",
        "avg_prompt_instructions_to_end_chars",
        "avg_prompt_weak_middle_must_use_chars",
        "avg_prompt_theory_card_chars",
        "avg_gap_question_to_evidence",
        "avg_gap_timeline_to_evidence",
        "avg_gap_own_speeches_to_question",
        "duplicate_constraint_total",
        "prompt_layout_counts",
        "prompt_layout_violations",
        "dedup_saved_chars_total",
    )
    print("log:", path.name, "| diagnostics layout counts:", layouts)
    for key in keys:
        print(f"   {key} = {metrics.get(key)!r}")
    model_stats = result.get("model_stats", {})
    for model, stats in model_stats.items():
        print(
            f"   [{model}] avg_user={stats.get('avg_prompt_user_chars')} "
            f"recency={stats.get('avg_prompt_zone_recency_chars')} "
            f"dilution={stats.get('avg_prompt_instructions_to_end_chars')} "
            f"own_q_gap={stats.get('avg_gap_own_speeches_to_question')} "
            f"layouts={stats.get('prompt_layout_counts')}"
        )
    missing = [key for key in keys if key not in metrics]
    print("[OK]" if not missing else f"[MISSING] {missing}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_tmp_mock_logs")
