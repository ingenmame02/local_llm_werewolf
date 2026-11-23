"""TASK-060の回帰前に確認したい実測（LL不使用）: 言い換え行数・情報欠落・契約の同一性。"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task060_layout_effect import _fixture  # noqa: E402

from werewolf.prompt_contract import find_sections  # noqa: E402
from werewolf.prompts import PromptBuilder  # noqa: E402

CUES = (
    "- Self-Reference:",
    "- Response Language:",
    "ask dead players",
    "recorded in the ledger",
    "True Roles of Dead Players",
    "[OUTPUT_LANGUAGE]",
    "Free:",
    "[BEHAVIOR_CONTRACT",
    "- Separate engine-confirmed facts",
)


def build(layout: str, contract: str, core: str):
    state, actor, asker = _fixture()
    return PromptBuilder.build_discussion_prompt(
        actor,
        tuple(state.alive_players),
        tuple(state.conversation_log),
        day=state.day,
        state=state,
        include_claim=True,
        context_strategy="hybrid_retrieval",
        question_hint_mode="strict",
        prompt_contract_policy=contract,
        core_compression_policy=core,
        prompt_layout_policy=layout,
    )


def main() -> None:
    v3 = build("v3", "simplified", "layered")
    v2 = build("v2", "simplified", "layered")
    legacy = build("legacy", "legacy", "off")
    for label, ctx in (("v3", v3), ("v2", v2), ("legacy", legacy)):
        diag = ctx.diagnostics or {}
        print("-" * 60, label)
        for cue in CUES:
            print(f"   count[{cue}] = {ctx.user_prompt.count(cue)}")
        print("   dup_constraints:", diag.get("prompt_duplicate_constraint_count"))
        print("   core_anchor_chars:", diag.get("core_anchor_chars"),
              "recent:", diag.get("core_recent_chars"),
              "historical:", diag.get("core_historical_chars"))
        print("   dedup_refs:", diag.get("dedup_reference_count"))
    print("system_prompt identical v2/v3:", v2.system_prompt == v3.system_prompt)
    # 発言本文・生存者名・質問文が欠けていないか
    state, actor, asker = _fixture()
    bodies = [m.content for m in state.conversation_log if m.speaker != "system"]
    names = tuple(p.name for p in state.players)
    for label, ctx in (("v3", v3), ("v2", v2)):
        missing_names = [n for n in names if n not in state.alive_players and n not in ctx.user_prompt]
        print(label, "dead names present:", tuple(missing_names))
        print(label, "body counts:", tuple(ctx.user_prompt.count(b) for b in bodies))
        spans = find_sections(ctx.user_prompt)
        guide = spans.get("discussion_guidelines")
        print(label, "guidelines span:", guide, "evidence chars:", spans["conversation_evidence"][1] - spans["conversation_evidence"][0])


if __name__ == "__main__":
    main()
