"""TASK-058 legacy互換チェック（実機LLM不使用）。

`_analysis/prompts_before_task058.py.bak` を一時的に src/werewolf/_prompts_v1ref.py として
コピーし、legacy ポリシーで組み上げた user_prompt が改修前後でバイト一致することを
確認する（simplified は layout v2 を意図的に変えるため対象外）。
"""
import shutil
import sys
from pathlib import Path

REPO = Path(r"c:\Users\yukic\local_llm_werewolf")
sys.path.insert(0, str(REPO / "src"))

SRC_BAK = REPO / "_analysis" / "prompts_before_task058.py.bak"
REF_PATH = REPO / "src" / "werewolf" / "_prompts_v1ref.py"

ok = True
try:
    shutil.copyfile(SRC_BAK, REF_PATH)
    from werewolf.prompts import PromptBuilder as NewBuilder
    from werewolf._prompts_v1ref import PromptBuilder as OldBuilder
    from werewolf.models import Message, Phase
    from werewolf.setup import create_game

    for hint in ("strict", "soft"):
        state = create_game(seed=20261123)
        names = [p.name for p in state.players]
        state.day = 1
        state.discussion_turn = 2
        state.phase = Phase.DAY_DISCUSSION
        state.public_timeline.record_day_start(1)
        state.conversation_log.append(Message("system", "Day 1 begins."))
        for i, name in enumerate(names[:5]):
            msg = Message(name, f"{name}の発言{i}です。投票先は検討中です。")
            state.append_discussion_message(msg)
        actor = next(p for p in state.players if p.name == names[4])
        common = dict(
            alive_players=tuple(state.alive_players),
            conversation_history=tuple(state.conversation_log),
            day=1,
            state=state,
            include_claim=True,
            context_strategy="hybrid_retrieval",
            evidence_char_budget=700,
            question_hint_mode=hint,
        )
        old_ctx = OldBuilder.build_discussion_prompt(
            actor, prompt_contract_policy="legacy", **common
        )
        new_ctx = NewBuilder.build_discussion_prompt(
            actor, prompt_contract_policy="legacy", core_compression_policy="off", **common
        )
        same = old_ctx.user_prompt == new_ctx.user_prompt
        print(f"hint={hint}: legacy parity -> {'OK' if same else 'MISMATCH'}")
        if not same:
            ok = False
            old_t, new_t = old_ctx.user_prompt, new_ctx.user_prompt
            print(" lens:", len(old_t), len(new_t))
            for i in range(min(len(old_t), len(new_t))):
                if old_t[i] != new_t[i]:
                    print(" first diff at", i)
                    print(" old:", repr(old_t[max(0, i - 60): i + 90]))
                    print(" new:", repr(new_t[max(0, i - 60): i + 90]))
                    break
finally:
    REF_PATH.unlink(missing_ok=True)

print("[DONE]", "all-legacy-parity-ok" if ok else "PARITY-FAILED")
sys.exit(0 if ok else 1)
