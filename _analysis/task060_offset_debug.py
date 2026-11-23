"""TASK-060の配置デバッグ（LL不使用）: v3で宣言節順どおりに出ているかを実測する。"""
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task060_layout_effect import _fixture  # noqa: E402

from werewolf.prompt_contract import find_sections  # noqa: E402
from werewolf.prompts import PromptBuilder  # noqa: E402


def main() -> None:
    state, actor, asker = _fixture()
    ctx = PromptBuilder.build_discussion_prompt(
        actor,
        tuple(state.alive_players),
        tuple(state.conversation_log),
        day=state.day,
        state=state,
        include_claim=True,
        context_strategy="hybrid_retrieval",
        question_hint_mode="strict",
        prompt_contract_policy="simplified",
        core_compression_policy="layered",
        prompt_layout_policy="v3",
    )
    prompt = ctx.user_prompt
    print("prompt chars:", len(prompt))
    for marker in ("[CONTINUITY_MEMO", "[CLAIM_LEDGER]", "[HYPOTHESIS_LEDGER]", "[FACT_LEDGER]"):
        hits = [match.start() for match in re.finditer(re.escape(marker), prompt)]
        print(marker, "hits:", hits)
        for hit in hits:
            print("   ", prompt[hit : hit + 70].replace("\n", " | ")[:70])
    spans = find_sections(prompt)
    ordered = sorted(spans.items(), key=lambda item: item[1][0])
    for name, (start, end) in ordered:
        print(f"{start:>6} .. {end:>6}  {name}")


if __name__ == "__main__":
    main()
