"""TASK-066切り分け用プローブ: TASK-057の overflow段階除去が変わったかを見る。

実機LLは使わず、テストと同じモック状態で `_build_discussion_prompt_with_budget` を呼び、
診断キー（contract_removal_stage / prompt_char_count / 発言履歴予算）を印字するだけ。
方策 tiered / legacy_early_compression の両方で同じになることも確認する（再生成前の
初回構築なので本来まったく同じであるべき）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from test_task057_prompt_contract import FakeContractClient, _day1_with_speeches, _names, _player  # noqa: E402
from werewolf.llm_actions import OllamaActions  # noqa: E402
from werewolf.prompt_contract import PROMPT_CONTRACT_POLICY_SIMPLIFIED  # noqa: E402

INTERESTING_KEYS = (
    "contract_removal_stage",
    "contract_stage_attempts",
    "optional_removed_chars",
    "prompt_build_calls",
    "user_prompt_char_count",
    "speech_history_char_count",
    "speech_history_budget_chars",
    "needed_tokens",
    "effective_context_limit",
    "effective_requested_num_ctx",
    "conversation_message_count",
)


def _report(label: str, diagnostics: dict[str, object]) -> None:
    print(f"--- {label}")
    for key in INTERESTING_KEYS:
        if key in diagnostics:
            print(f"  {key}={diagnostics[key]!r}")
    extra = {k: v for k, v in sorted(diagnostics.items()) if "budget" in k or "ctx" in k}
    print(f"  budget_keys={extra}")


def main() -> None:
    state = _day1_with_speeches()
    state.prompt_contract_policy = PROMPT_CONTRACT_POLICY_SIMPLIFIED
    actor = _player(state, _names(state)[4])
    actor.num_ctx = 3072
    actions = OllamaActions(state, FakeContractClient(), seed=1001)
    ctx = actions._build_discussion_prompt_with_budget(
        actor, tuple(state.alive_players), tuple(state.conversation_log), 1
    )
    _report("tight num_ctx=3072", dict(ctx.diagnostics or {}))
    print(f"policy={getattr(state, 'retry_context_policy', None)!r}")
    print(f"user_prompt_chars={len(ctx.user_prompt)}")

    # legacy方策でも同じ（再生成前なので差が出てはいけない）
    state.retry_context_policy = "legacy_early_compression"
    legacy = actions._build_discussion_prompt_with_budget(
        actor, tuple(state.alive_players), tuple(state.conversation_log), 1
    )
    print(f"legacy_user_prompt_chars={len(legacy.user_prompt)}")
    _report("legacy policy tight", dict(legacy.diagnostics or {}))


if __name__ == "__main__":
    main()
