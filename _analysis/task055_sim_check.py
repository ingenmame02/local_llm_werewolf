"""TASK-055の実行確認（モッククライアント / 実機LLMは使わない）。

1. allow（既定）: 非保有者の捏造占いCOも公開され、出所は本人申告として分離される。
2. deny: CO・結果行だけが公開前に消え、本文はそのまま残る。
3. 公開ログへエンジン照合の真偽キーが混ざらない（private ログ専用であること）。
"""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO = Path(r"c:\Users\yukic\local_llm_werewolf")
sys.path.insert(0, str(REPO / "src"))

from werewolf.cli import GameLogger
from werewolf.engine import GameEngine
from werewolf.llm_actions import OllamaActions
from werewolf.models import Role
from werewolf.ollama import ChatResponse
from werewolf.setup import create_game

CONFIG = str(REPO / "config" / "players.json")
LOG_DIR = REPO / "_analysis" / "logs_task055"

FORBIDDEN_PUBLIC_KEYS = (
    "engine_supported",
    "unsupported_role_evidence",
    "engine_seer",
    '"category"',
)


class SimClient:
    """COでは常に占い師を名乗り、本文は自由に書くという最悪ケースを再現する。"""

    timeout = 60.0
    max_retries = 1
    timeout_retries = 1

    def __init__(self, state, bluff_names: tuple[str, ...]) -> None:
        self.state = state
        self.names = [p.name for p in state.players]
        self.bluff_names = bluff_names
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
            if speaker in self.bluff_names:
                return ChatResponse(
                    content=f"[CO] Seer CO\n[RESULT] Target={other} Result=Werewolf",
                    attempts=1,
                )
            return ChatResponse(content="[CO] None", attempts=1)
        if "[CO] None, Hold, or your public role claim" in user:
            if speaker in self.bluff_names:
                return ChatResponse(
                    content=(
                        f"[CO] Seer CO\n[RESULT] Target={other} Result=Werewolf\n"
                        f"[BODY] {other}さんは昨夜から言動が不自然です。{third}さんの投票理由も聞きたい。"
                    ),
                    attempts=1,
                )
            return ChatResponse(
                content=(
                    f"[CO] None\n[BODY] {other}さんのCOの根拠を確認したい。"
                    f"{third}さんの視点も聞いてから投票先を決めたい。"
                ),
                attempts=1,
            )
        if "Divine 1 undivined living player" in user or "占い対象" in user:
            return ChatResponse(content=f"Seer target: {other}", attempts=1)
        if "Living wolves choose 1 non-wolf" in user:
            return ChatResponse(content=f"Attack target: {other}", attempts=1)
        if "Consult with your partner" in user:
            return ChatResponse(content=f"今夜は{other}を攻撃。\nAttack target: {other}", attempts=1)
        if "Vote for 1 other living player" in user:
            return ChatResponse(content=f"Vote: {other}", attempts=1)
        return ChatResponse(
            content=f"[CO] None\n[BODY] {other}さんの主張の整合性を確認したい。", attempts=1
        )


def run_case(allow_bluff: bool) -> dict[str, object]:
    game_id = f"task055_{'allow' if allow_bluff else 'deny'}"
    state = create_game(seed=20261120, config_path=CONFIG, response_language="日本語")
    bluff_names = tuple(
        player.name for player in state.players if player.role is not Role.SEER
    )[:2]
    state.allow_bluff_role_claim = allow_bluff

    client = SimClient(state, bluff_names)
    events: list[dict[str, object]] = []
    logger = GameLogger(
        game_id, LOG_DIR, {"seed": 20261120, "task": "TASK-055"}, private_enabled=True
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
        seed=20261120,
        options={"question_hint_mode": "none"},
        discussion_claim_mode="legacy",
        event_callback=on_event,
    )
    engine = GameEngine(state, actions, seed=20261120, event_callback=on_event)
    result = engine.run()
    logger.finalize(state, traces=None, error=None)

    counts = Counter(str(event.get("type")) for event in events)
    public_text = (LOG_DIR / f"game_{game_id}.jsonl").read_text(encoding="utf-8")
    private_text = (LOG_DIR / f"game_{game_id}.private.jsonl").read_text(encoding="utf-8")
    leaks = [key for key in FORBIDDEN_PUBLIC_KEYS if key in public_text]

    seer_name = next(p.name for p in state.players if p.role is Role.SEER)
    # エンジン発行の占い結果（RoleEvidence）は真占い師だけが保有しているべき
    wrong_owners = {item.owner for item in state.role_provenance.evidence} - {seer_name}
    bluff_claimed = [
        item
        for item in state.role_provenance.interpretations
        if item.claimant in bluff_names and item.result is not None
    ]
    bluff_speech = [
        message.content
        for message in state.conversation_log
        if message.speaker in bluff_names and message.speaker != "system"
    ]
    body_chars = sum(len(content) for content in bluff_speech)
    kept_reasoning = sum(1 for content in bluff_speech if "不自然" in content or "聞きたい" in content)
    result_records = [json.loads(line) for line in public_text.splitlines() if '"game_result"' in line]

    print(f"=== {game_id} === winner={getattr(result, 'winner', None)} requests={client.requests}")
    print(
        "  claim_role_published:", counts.get("claim_role_published"),
        "| role_claim_disallowed:", counts.get("role_claim_disallowed"),
        "| fatal_fallbacks:", sum(1 for t in actions.traces if getattr(t, "fatal_safety_fallback", False)),
    )
    print("  公開ログの真偽キー混入:", leaks or "none")
    print(
        "  privateにengine照合:", "engine_supported" in private_text,
        "| privateにRoleEvidence:", "engine_seer" in private_text,
    )
    if result_records:
        provenance = result_records[-1].get("role_provenance", {})
        print("  public role_provenance keys:", sorted(provenance.keys()))
        print("  public metrics:", provenance.get("metrics"))
        claims = provenance.get("claims") or []
        print("  全CO行のラベル:", {c.get("interpretation") for c in claims},
              "| source:", {c.get("source") for c in claims})
    print(
        "  engine evidence owners:", sorted({i.owner for i in state.role_provenance.evidence}),
        "| seer:", seer_name, "| 不正保有者:", sorted(wrong_owners) or "none",
    )
    print(
        "  ブラフ側の申告行:", len(bluff_claimed),
        "| category:", {i.category for i in bluff_claimed},
    )
    print("  state.seer_results:", state.seer_results)
    public_claims = (
        (result_records[-1].get("role_provenance", {}) or {}).get("claims") or []
        if result_records
        else []
    )
    bluff_public_claims = [c for c in public_claims if c.get("claimant") in bluff_names]
    print(
        "  ブラフ発言数:", len(bluff_speech), "| 合計文字数:", body_chars, "| 論点を含む発言数:", kept_reasoning
    )
    print(
        "  公開台帳のブラフ側CO行:", len(bluff_public_claims),
        "| 全CO行ラベル:", {str(c.get("interpretation")) for c in public_claims},
    )
    return {
        "leaks": leaks,
        "counts": counts,
        "allow": allow_bluff,
        "bodies": body_chars,
        "kept_reasoning": kept_reasoning,
        "bluff_public_claims": len(bluff_public_claims),
    }


if LOG_DIR.exists():
    shutil.rmtree(LOG_DIR)
allow = run_case(True)
deny = run_case(False)

print("\n=== 検証 ===")
print("公開ログへ真偽キーが出ない:", not allow["leaks"] and not deny["leaks"])
print("denyで受理不可が発生:", (deny["counts"].get("role_claim_disallowed") or 0) > 0)
print("allowで受理不可ゼロ:", (allow["counts"].get("role_claim_disallowed") or 0) == 0)
print("allowでブラフCOが公開台帳へ載る:", allow["bluff_public_claims"] > 0)
print(
    "denyでブラフCOが公開台帳から消える:",
    deny["bluff_public_claims"] == 0 and (deny["counts"].get("role_claim_disallowed") or 0) > 0,
)
print("両設定で本文の論点は保持:", allow["kept_reasoning"] > 0 and deny["kept_reasoning"] > 0)

