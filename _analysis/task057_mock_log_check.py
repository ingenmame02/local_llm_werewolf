"""TASK-057: CLIのモックログ（実機不使用）へ契約指標が出ているかを確認する。"""
import glob
import json
from pathlib import Path

REPO = Path(r"c:\Users\yukic\local_llm_werewolf")
LOG_GLOB = str(REPO / "_analysis" / "mock_task057" / "game_*.jsonl")
FORBIDDEN_KEYS = ("attack_target", "divination_result", "true_role")


def main() -> None:
    paths = sorted(glob.glob(LOG_GLOB))
    assert paths, "モックログが見つからない"
    path = Path(paths[-1])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    selections = [r for r in rows if r.get("type") == "discussion_context_selected"]
    result = next(r for r in rows if r.get("type") == "game_result")
    metrics = result.get("ledger_metrics") or {}
    print("log:", path.name)
    print("prompt_contract_policy:", result.get("prompt_contract_policy"))
    print("selections:", len(selections))
    with_core = [r for r in selections if isinstance(r.get("prompt_core_chars"), int)]
    print("selections_with_contract_diagnostics:", len(with_core), "/", len(selections))
    stages: dict[str, int] = {}
    section_total: dict[str, int] = {}
    for row in selections:
        key = str(row.get("contract_removal_stage"))
        stages[key] = stages.get(key, 0) + 1
        for name, chars in (row.get("prompt_section_chars") or {}).items():
            section_total[name] = int(section_total.get(name, 0)) + int(chars)
    print("stages:", stages)
    for key in (
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
    ):
        print(f"  {key}: {metrics.get(key)}")
    top = sorted(section_total.items(), key=lambda kv: -kv[1])[:6]
    print("  largest sections (chars across requests):", top)
    leaked = [k for k in FORBIDDEN_KEYS if any(k in json.dumps(r, ensure_ascii=False) for r in rows)]
    print("  非公開キーの混入:", leaked or "none")


if __name__ == "__main__":
    main()
