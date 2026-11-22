"""TASK-056の表示例（実機LLM不使用）: 発言時点欄と短縮ヘッダの実物を印字する."""
import sys
from pathlib import Path

REPO = Path(r"c:\Users\yukic\local_llm_werewolf")
sys.path.insert(0, str(REPO / "src"))

from werewolf.day_timeline import (
    render_timeline_section,
    timeline_evidence_header,
)
from werewolf.models import Message, Phase
from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game

state = create_game(seed=20261121, config_path=str(REPO / "config" / "players.json"),
                    response_language="日本語")
names = [player.name for player in state.players]

# Day 1 / Turn 2 の状況を再現する（T1で全員発言、Player 8は襲撃死済み、まだ投票前）
state.public_timeline.record_day_start(1)
state.conversation_log.append(Message("system", "Day 1 begins."))
state.phase = Phase.DAY_DISCUSSION
state.discussion_turn = 1
for name in names[:6]:
    state.append_discussion_message(
        Message(name, f"{name}です。まずは状況を確認したいです。")
    )
state.discussion_turn = 2

speaker = next(p for p in state.players if p.is_alive and p.name not in names[:6])
snapshot = state.timeline_snapshot(speaker.name)
block, diag = render_timeline_section(snapshot, language="日本語")
print("=== [DAY_TIMELINE] 実出力例（", speaker.name, "）===")
print(block.rstrip())
print()
print("diagnostics:", {k: diag[k] for k in sorted(diag)})
print()

print("=== 発言ヘッダ（時点つき / 旧形式フォールバック） ===")
stamped = state.conversation_log[1]
print(" stamped :", timeline_evidence_header(
    stamped.day, stamped.turn, stamped.speech_id, stamped.speaker))
print("   chars :", len(timeline_evidence_header(
    stamped.day, stamped.turn, stamped.speech_id, stamped.speaker)))
legacy = Message("Player 3", "従来形の履歴")
print(" legacy  :", timeline_evidence_header(
    1, None, None, legacy.speaker, legacy_index=3))
print("   chars :", len(timeline_evidence_header(1, None, None, legacy.speaker, 3)))
print()

ctx = PromptBuilder.build_discussion_prompt(
    speaker,
    tuple(p.name for p in state.players if p.is_alive),
    tuple(state.conversation_log),
    day=1,
    state=state,
)
text = ctx.user_prompt
start = text.find("[DAY_TIMELINE]")
end = text.find("\n\n", start)
print("=== プロンプト内の位置（節の数）:", text.count("[DAY_TIMELINE]"), "===")
print(text[start:end])
print()
print("diagnostics day_timeline_chars:", ctx.diagnostics.get("day_timeline_chars"),
      "| lines:", ctx.diagnostics.get("day_timeline_lines"),
      "| header_style:", ctx.diagnostics.get("evidence_header_style"),
      "| evidence_chars:", ctx.diagnostics.get("evidence_char_count"))
