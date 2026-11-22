"""TASK-056の実行確認（モッククライアント / 実機LLMは使わない）。

1. explicit（既定）: 発言ヘッダが時点つきになり、`[DAY_TIMELINE]` が1回だけ出る。
2. off: 時系列欄だけ消える（A/B比較の基準側）。ヘッダ短縮と台帳は残る。
3. 未来参照の検出は観測専用（本文は一字一句そのまま＝`body_modified=false`）。
4. 新しい欄が増えても、ヘッダ縮約とあわせて実効プロンプト量は増えないことを計測する。
"""
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(r"c:\Users\yukic\local_llm_werewolf")
sys.path.insert(0, str(REPO / "src"))

from werewolf.cli import GameLogger
from werewolf.engine import GameEngine
from werewolf.llm_actions import OllamaActions
from werewolf.ollama import ChatResponse
from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game

CONFIG = str(REPO / "config" / "players.json")
LOG_DIR = REPO / "_analysis" / "logs_task056"

FORBIDDEN_PUBLIC_KEYS = ("attack_target", "divination_result", "true_role")


class SimClient:
    """COはしないが「まだ起きていない出来事」を根拠にしかねない発言を返す最悪ケース。"""

    timeout = 60.0
    max_retries = 1
    timeout_retries = 1

    def __init__(self, state) -> None:
        self.state = state
        self.names = [player.name for player in state.players]
        self.requests = 0

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {"ollama_ps_context": None}

    def _speaker(self, system: str) -> str:
        for name in self.names:
            if f'You are "{name}"' in system:
                return name
        return ""

    def _other(self, speaker: str) -> str:
        others = [n for n in self.names if n != speaker and n in self.state.alive_players]
        return others[0] if others else speaker

    def chat(self, model, messages, options=None) -> str:
        return self.chat_with_metadata(model, messages, options).content

    def chat_with_metadata(self, model, messages, options=None, tools=None) -> ChatResponse:
        self.requests += 1
        system = messages[0]["content"] if messages else ""
        user = messages[1]["content"] if len(messages) > 1 else ""
        speaker = self._speaker(system)
        other = self._other(speaker)
        third = self._other(other)
        if "decide on your public role claim" in user:
            return ChatResponse(content="[CO] None", attempts=1)
        if "[CO] None, Hold, or your public role claim" in user:
            # 未来参照を断定する文（婉曲なし）と通常の自由な発言を交互に出す
            # （検出しても本文が変わらないこと／誤検出で口調を縛らないことの確認）
            body = (
                f"{other}さんの発言を確認したい。{third}さんの視点も聞いてから決めたい。"
                if self.requests % 2
                else "今日の投票結果は決まった。議論を続ける必要はない。"
            )
            return ChatResponse(content=f"[CO] None\n[BODY] {body}", attempts=1)
        if "Divine 1 undivined living player" in user or "占い対象" in user:
            return ChatResponse(content=f"Seer target: {other}", attempts=1)
        if "Living wolves choose 1 non-wolf" in user:
            return ChatResponse(content=f"Attack target: {other}", attempts=1)
        if "Consult with your partner" in user:
            return ChatResponse(
                content=f"今夜は{other}を攻撃。\nAttack target: {other}", attempts=1
            )
        if "Vote for 1 other living player" in user:
            return ChatResponse(content=f"Vote: {other}", attempts=1)
        return ChatResponse(
            content=f"[CO] None\n[BODY] {other}さんの主張の整合性を確認したい。", attempts=1
        )


def run_case(policy: str) -> dict[str, object]:
    game_id = f"task056_{policy}"
    state = create_game(seed=20261121, config_path=CONFIG, response_language="日本語")
    state.timeline_policy = policy
    state.public_timeline.policy = policy

    client = SimClient(state)
    events: list[dict[str, object]] = []
    logger = GameLogger(
        game_id, LOG_DIR, {"seed": 20261121, "task": "TASK-056"}, private_enabled=True
    )

    def on_event(event: dict[str, object]) -> None:
        events.append(event)
        public_event = {
            key: value
            for key, value in event.items()
            if key not in {"content", "selected_target", "parsed_target", "error"}
        }
        logger.append_record(public_event, event)

    actions = OllamaActions(
        state,
        client,
        seed=20261121,
        options={"question_hint_mode": "none"},
        discussion_claim_mode="legacy",
        event_callback=on_event,
    )
    engine = GameEngine(state, actions, seed=20261121, event_callback=on_event)
    result = engine.run()
    logger.finalize(state, traces=None, error=None)

    public_text = (LOG_DIR / f"game_{game_id}.jsonl").read_text(encoding="utf-8")
    results = [
        json.loads(line)
        for line in public_text.splitlines()
        if '"game_result"' in line
    ]
    metrics = (results[-1].get("ledger_metrics") or {}) if results else {}
    timeline_payload = (results[-1].get("public_timeline") or {}) if results else {}
    speech_events = [event for event in events if event.get("type") == "discussion_message"]
    modified = [e for e in speech_events if e.get("body_modified") is True]
    observed = [e for e in speech_events if e.get("timeline_observed")]
    findings = sum(int(e.get("future_event_reference_count") or 0) for e in speech_events)
    leaks = [key for key in FORBIDDEN_PUBLIC_KEYS if key in json.dumps(timeline_payload)]
    ledger = state.public_timeline
    print(
        f"=== {game_id} === winner={getattr(result, 'winner', None)} "
        f"requests={client.requests}"
    )
    print(
        "  speech:", len(speech_events),
        "| observed:", len(observed),
        "| future_refs:", findings,
        "| body_modified!=false:", len(modified),
        "| speech_id付き:", sum(1 for e in speech_events if "speech_id" in e),
    )
    print(
        "  avg_day_timeline_chars:", metrics.get("avg_day_timeline_chars"),
        "| avg_evidence_char_count:", metrics.get("avg_evidence_char_count"),
    )
    print(
        "  ref_checks:", metrics.get("timeline_reference_checks"),
        "| future_rate:", metrics.get("future_event_reference_rate"),
        "| grounding_rate:", metrics.get("evidence_grounding_rate_at_speech_time"),
    )
    print("  violation_counts:", metrics.get("timeline_violation_counts"))
    print(
        "  台帳イベント:", len(timeline_payload.get("events") or []),
        "| policy:", timeline_payload.get("policy"),
        "| 非公開キー混入:", leaks or "none",
    )
    print(
        "  bodies_checked:", ledger.bodies_checked,
        "| kept_unmodified:", ledger.bodies_kept_unmodified,
        "| withheld_future:", ledger.future_events_withheld,
        "| section_avg_lines:",
        round(ledger.section_lines_total / max(1, ledger.section_requests), 2),
    )
    return {
        "policy": policy,
        "state": state,
        "metrics": metrics,
        "modified_bodies": len(modified),
        "findings": findings,
        "leaks": leaks,
    }


def build(actor, state, history, policy):
    previous = state.timeline_policy
    state.timeline_policy = policy
    try:
        return PromptBuilder.build_discussion_prompt(
            actor,
            tuple(state.alive_players),
            history,
            day=state.day,
            state=state,
            include_claim=True,
            context_strategy="hybrid_retrieval",
            question_hint_mode="none",
        )
    finally:
        state.timeline_policy = previous


def prompt_size_ab(state) -> dict[str, int]:
    """同じ履歴で新旧プロンプトを並べ、新しい欄の実効コストを測る。"""
    actor = next(player for player in state.players if player.is_alive)
    history = tuple(
        message
        for message in state.conversation_log
        if message.speaker != "system" or "begins" in (message.content or "")
    )
    explicit_ctx = build(actor, state, history, "explicit")
    off_ctx = build(actor, state, history, "off")
    # TASK-056以前の挙動（時点スタンプなし＝長いヘッダ・時系列欄なし）を再現する
    legacy_history = tuple(
        replace(message, day=0, turn=0, speech_id=None, timeline_index=None)
        for message in history
    )
    legacy_ctx = build(actor, state, legacy_history, "off")
    lengths = {
        "new": len(explicit_ctx.user_prompt),
        "off": len(off_ctx.user_prompt),
        "old": len(legacy_ctx.user_prompt),
    }
    print("  プロンプト長（同一履歴・同一時点）")
    print(
        "    old(TASK-056前相当):", lengths["old"],
        "| off:", lengths["off"],
        "| explicit:", lengths["new"],
    )
    print(
        "    時系列欄の実費:", lengths["new"] - lengths["off"],
        "| ヘッダ縮約の削減:", lengths["old"] - lengths["off"],
        "| 旧対比の差分:", lengths["new"] - lengths["old"],
    )
    print(
        "    evidence_header_style:",
        explicit_ctx.diagnostics.get("evidence_header_style"),
        "/",
        legacy_ctx.diagnostics.get("evidence_header_style"),
    )
    print(
        "    evidence_char_count: stamped=",
        explicit_ctx.diagnostics.get("evidence_char_count"),
        "legacy=",
        legacy_ctx.diagnostics.get("evidence_char_count"),
        "| 削減=", (
            int(legacy_ctx.diagnostics.get("evidence_char_count") or 0)
            - int(explicit_ctx.diagnostics.get("evidence_char_count") or 0)
        ),
    )
    assert "[DAY_TIMELINE]" in explicit_ctx.user_prompt
    assert "[DAY_TIMELINE]" not in off_ctx.user_prompt
    assert int(explicit_ctx.diagnostics["day_timeline_chars"]) <= 470
    return lengths


if LOG_DIR.exists():
    shutil.rmtree(LOG_DIR)
explicit = run_case("explicit")
off = run_case("off")

print("\n=== プロンプト量のA/B（explicitゲームの状態） ===")
sizes = prompt_size_ab(explicit["state"])

print("\n=== 検証 ===")
explicit_ledger = explicit["state"].public_timeline
off_ledger = off["state"].public_timeline
checks = {
    "explicitで時系列欄が出た": (explicit["metrics"].get("avg_day_timeline_chars") or 0) > 0,
    "offでは時系列欄が出ない": (off["metrics"].get("avg_day_timeline_chars")) in (0, None),
    "future参照を明示側だけ計測": (
        (explicit["metrics"].get("timeline_reference_checks") or 0) > 0
        and (off["metrics"].get("timeline_reference_checks") or 0) == 0
    ),
    "未来参照を検出した（模擬の最悪ケース）": explicit["findings"] > 0,
    "公開本文を修正していない（explicit）": (
        explicit["modified_bodies"] == 0
        and explicit_ledger.bodies_kept_unmodified == explicit_ledger.bodies_checked
        and explicit_ledger.bodies_checked > 0
    ),
    "offは検査そのものを行わない": (
        off_ledger.bodies_checked == 0 and off["modified_bodies"] == 0
    ),
    "公開台帳へ非公開情報なし": not explicit["leaks"] and not off["leaks"],
    "新しい欄の実効コストが480字以内": sizes["new"] - sizes["old"] <= 480,
}
for name, passed in checks.items():
    print(f"  [{'OK' if passed else 'NG'}] {name}")
print("ALL PASS" if all(checks.values()) else "SOME CHECK FAILED")


