"""TASK-055 事前分析: logs/game_15b1bdf65773 のCO・占い結果行と役職由来情報の対応を見る。"""
import glob
import json
import sys

PATTERNS = sys.argv[1:] or [r"logs\game_15b1bdf65773*"]
CLAIM_TYPES = {
    "claim_role_published",
    "claim_role_corrected",
    "seer_claim",
    "seer_result_claim",
    "seer_result_corrected",
    "claim_role_conflict",
    "claim_role_rejected",
    "duplicate_claim_suppressed",
    "seer_result_claim_missing",
    "discussion_integrity_repaired",
    "ledger_repair",
    "discussion_claim_deterministically_repaired",
    "secret_role_leak_detected",
    "secret_leak_note",
}

for path in sorted(glob.glob(PATTERNS[0])):
    print("==", path)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            kind = event.get("type")
            if kind == "game_init":
                print(
                    "init",
                    {
                        k: event.get(k)
                        for k in ("mode", "seed", "leak_guard_policy", "novelty_policy")
                    },
                )
            elif kind == "game_result":
                seer_results = event.get("seer_results")
                print("result keys:", sorted(event.keys())[:40])
                if seer_results is not None:
                    print("  seer_results(private):", seer_results)
                roles = event.get("roles") or event.get("player_roles")
                if roles:
                    print("  roles(private):", roles)
                claim_ledger = event.get("claim_ledger")
                if claim_ledger:
                    print("  claim_ledger:", json.dumps(claim_ledger, ensure_ascii=False)[:600])
            elif kind == "discussion_message":
                content = str(event.get("content", ""))
                if any(token in content for token in ("CO", "占い", "結果", "RESULT")):
                    print(
                        f"speech d{event.get('day')} t{event.get('discussion_turn')}",
                        event.get("speaker"),
                        content[:260].replace("\n", " / "),
                    )
            elif kind in CLAIM_TYPES:
                print(
                    kind,
                    "d" + str(event.get("day")),
                    event.get("claimant") or event.get("player"),
                    event.get("claimed_role"),
                    event.get("target"),
                    event.get("claimed_result"),
                    event.get("reason"),
                    "src=" + str(event.get("source")),
                )
