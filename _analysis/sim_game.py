import sys
from pathlib import Path

SRC = Path(r'c:\Users\yukic\local_llm_werewolf\src')
sys.path.insert(0, str(SRC))

from collections import Counter

from werewolf.engine import GameEngine
from werewolf.llm_actions import OllamaActions
from werewolf.models import Role
from werewolf.ollama import ChatResponse
from werewolf.setup import create_game

CONFIG = r'c:\Users\yukic\local_llm_werewolf\config\players.json'


class SimulatedClient:
    """実機風の応答を返す模擬クライアント（形式は正しいが時折幻覚・別称を出す）。"""

    timeout = 60.0
    max_retries = 1
    timeout_retries = 1

    def __init__(self, state) -> None:
        self.state = state
        self.names = [p.name for p in state.players]
        self.requests = 0

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {"ollama_ps_context": None}

    def _others(self, speaker: str) -> list[str]:
        return [n for n in self.names if n != speaker and n in self.state.alive_players]

    def chat(self, model, messages, options=None) -> str:
        return self.chat_with_metadata(model, messages, options).content

    def chat_with_metadata(self, model, messages, options=None, tools=None) -> ChatResponse:
        self.requests += 1
        system = messages[0]["content"] if messages else ""
        user = messages[1]["content"] if len(messages) > 1 else ""
        speaker = ""
        for name in self.names:
            if f'You are "{name}"' in system:
                speaker = name
                break
        others = self._others(speaker)
        other = others[0] if others else speaker
        if "Divine 1 undivined living player" in user or "占い対象" in user:
            content = f"Seer target: {other}"
        elif "Vote for 1 other living player" in user or "投票" in user.split("Eligible")[0]:
            content = f"Vote: プレイヤー{other.split()[-1]}"  # 別称で答える（TASK-043の解析経路）
        elif "Living wolves choose 1 non-wolf" in user:
            content = f"Attack target: {other}"
        elif "Consult with your partner" in user:
            content = f"今夜は{other}を攻撃したい。{speaker}はCOの根拠として残す。\nAttack target: {other}"
        elif "[CO] None, Hold, or your public role claim" in user:
            content = (
                "[CO] None\n"
                f"[BODY] {other}さんの投票理由とCOの整合性を確認したい。"
                "私は公開情報から疑いを整理し、次の質問先を決めたい。"
            )
        elif "decide on your public role claim" in user:
            content = "[CO] None"
        else:
            content = f"[CO] None\n[BODY] {other}さんのCOの整合性を確認したい。"
        return ChatResponse(content=content, attempts=1)


state = create_game(seed=20261115, config_path=CONFIG, response_language="日本語")
client = SimulatedClient(state)
actions = OllamaActions(
    state,
    client,
    seed=20261115,
    options={"question_hint_mode": "none", "temperature": 0.7},
    discussion_claim_mode="legacy",
)
events: list[dict[str, object]] = []
engine = GameEngine(state, actions, seed=20261115, event_callback=events.append)
result = engine.run()

counts = Counter(str(e.get("type")) for e in events)
print('requests:', client.requests)
print('winner:', result.winner if hasattr(result, 'winner') else result)
print('traces:', len(actions.traces), 'fallback traces:',
      sum(1 for t in actions.traces if t.fallback_used))
print('fatal fallbacks:', sum(1 for t in actions.traces if getattr(t, 'fatal_safety_fallback', False)))
for key in sorted(counts):
    if 'fail' in key or 'fallback' in key or 'repair' in key or 'leak' in key or 'dropped' in key:
        print(key, counts[key])
bodies = [m.content for m in state.conversation_log if m.speaker != 'system']
print('speeches:', len(bodies), 'placeholder leaks:',
      sum(1 for b in bodies if 'Player X' in b or 'プレイヤーX' in b))
