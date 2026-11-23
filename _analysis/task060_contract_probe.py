"""TASK-060の事前確認（LL不使用）: 理論カードのマーカー、契約の正準行、除去段階。"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.prompt_contract import (  # noqa: E402
    behavior_contract_lines,
    optional_names_dropped_up_to,
    render_behavior_contract,
    tier_of,
)
from werewolf.prompts import RULE_CARD, WEREWOLF_THEORY_CARD  # noqa: E402

print("theory head:", repr(WEREWOLF_THEORY_CARD[:80]))
print("theory len:", len(WEREWOLF_THEORY_CARD))
print("rule head:", repr(RULE_CARD[:80]))
print("tier:", tier_of("werewolf_theory_card"))
for stage in range(1, 5):
    print("stage", stage, optional_names_dropped_up_to(stage))
print("--- contract lines (default) ---")
for line in behavior_contract_lines():
    print(f"[{line.dimension}] {line.text}")
print("--- rendered in Japanese ---")
print(render_behavior_contract(language="日本語", hint_mode="strict", include_claim=True))

