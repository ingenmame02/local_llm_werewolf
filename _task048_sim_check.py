"""TASK-048 の A/B 比較（モックのみ・実機Ollamaへは一切送信しない）。

同じ seed で `legacy_repair`（旧挙動: 重複文の除去→意味的再要求）と `publish`（既定:
そのまま公開・再生成なし）を走らせ、修復回数・再生成リクエスト・本文保持率・新規性指標・
既出論点の要約量・リクエスト数を比較する。

`SimulatedClient`（`_analysis/sim_game.py`）はプロンプトの文字列で応答種別を判別するため、
日本語プロンプトでは誤認する。ここでは議論本文を Deterministic な台本で返す
`NoveltySimClient` を使い、連呼・正当な繰り返し・反証・立場変更の計測経路だけを比較する。
"""
import sys
from collections import Counter
from pathlib import Path

SRC = Path(r'c:\Users\yukic\local_llm_werewolf\src')
sys.path.insert(0, str(SRC))
sys.path.insert(0, r'c:\Users\yukic\local_llm_werewolf\_analysis')

from werewolf.engine import GameEngine
from werewolf.llm_actions import OllamaActions
from werewolf.ollama import ChatResponse
from werewolf.setup import create_game

CONFIG = r'c:\Users\yukic\local_llm_werewolf\config\players.json'
_spec_src = Path(r'c:\Users\yukic\local_llm_werewolf\_analysis\sim_game.py').read_text(encoding='utf-8')
# SimulatedClient だけ再定義（実機Ollamaへは一切送信しない・末尾実行は避ける）
_ns = {'__file__': str(_spec_src)}
exec(_spec_src.split('state = create_game')[0], _ns)
SimulatedClient = _ns['SimulatedClient']

# 議論本文の台本（連呼 / 新根拠を添えた再確認 / 反証 / 新論点 / Pass を順に混ぜる）
_SCRIPTS = [
    "{other}さんの今日の投票の理由を確認したい。",
    "私は{other}さんへの投票を提案する。理由は直近の発言の整合性です。",
    "{other}さんのCOは公開判定と矛盾している。別の可能性を確認したい。",
    "{other}さんは信頼できる。私は前の疑いを変更する。",
    "まだ情報不足だ。",
    "Player 3さんの今日の投票の理由を確認したい。",
]


class NoveltySimClient(SimulatedClient):
    """議論本文だけ台本で返す模擬クライアント（夜・投票はSimulatedClientの挙動を維持）。"""

    def __init__(self, state) -> None:
        super().__init__(state)
        self.script_index = 0

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
        markers = [
            "Divine 1 undivined living player",
            "占い対象",
            "Vote for 1 other living player",
            "Living wolves choose 1 non-wolf",
            "Consult with your partner",
        ]
        if any(marker in user for marker in markers):
            return super().chat_with_metadata(model, messages, options, tools)
        script = _SCRIPTS[self.script_index % len(_SCRIPTS)]
        self.script_index += 1
        body = script.format(other=other)
        if body == "まだ情報不足だ。":
            return ChatResponse(content="[CO] None\n[BODY] Pass", attempts=1)
        return ChatResponse(content=f"[CO] None\n[BODY] {body}", attempts=1)


def run(policy: str) -> dict[str, object]:
    state = create_game(seed=20261115, config_path=CONFIG, response_language='日本語')
    client = NoveltySimClient(state)
    events: list[dict[str, object]] = []
    actions = OllamaActions(
        state,
        client,
        seed=20261115,
        options={'question_hint_mode': 'none', 'temperature': 0.7, 'novelty_policy': policy},
        discussion_claim_mode='legacy',
        event_callback=events.append,
    )
    engine = GameEngine(state, actions, seed=20261115, event_callback=events.append)
    result = engine.run()

    bodies = [e for e in events if e.get('action') in {'discussion', 'discussion_body'} and e.get('type') == 'llm_action_completed']
    repairs = [e for e in events if e.get('type') == 'ledger_repair']
    repair_reasons = Counter(str(e.get('reason')) for e in repairs)
    regenerations = [
        e for e in events
        if e.get('type') == 'llm_action_parse_failed'
        and e.get('will_retry') is True
        and e.get('action') in {'discussion', 'discussion_body'}
    ]
    kept = [e for e in events if e.get('type') == 'llm_discussion_low_novelty_kept']
    appended = [e for e in events if e.get('type') == 'discussion_novelty_appended']
    measured = [e for e in events if e.get('type') == 'discussion_novelty_measured']
    novelty_classes = Counter(str(e.get('novelty_primary')) for e in measured)
    sel = [e for e in events if e.get('type') == 'discussion_context_selected']
    return {
        'policy': policy,
        'winner': result.winner,
        'day': result.day,
        'requests': client.requests,
        'bodies': len(bodies),
        'ledger_repairs': len(repairs),
        'repair_reasons': dict(repair_reasons),
        'regenerations': len(regenerations),
        'body_preserved_rate': round(
            sum(1 for e in bodies if e.get('content_preserved')) / max(1, len(bodies)), 3
        ),
        'fallbacks': sum(1 for e in bodies if e.get('fallback_used')),
        'novelty_measurements': len(measured),
        'novelty_present_rate': round(
            sum(1 for e in measured if e.get('has_novelty')) / max(1, len(measured)), 3
        ),
        'novelty_classes': dict(novelty_classes),
        'restated_topics_total': sum(int(e.get('novelty_restated_topics') or 0) for e in measured),
        'legitimate_repetitions_total': sum(
            int(e.get('novelty_legitimate_repetitions') or 0) for e in measured
        ),
        'low_novelty_kept': len(kept),
        'novelty_appends': len(appended),
        'topic_summary_chars_avg': round(
            sum(int(e.get('topic_summary_chars') or 0) for e in sel) / max(1, len(sel)), 1
        ),
        'topic_ledger': state.topic_ledger.metrics(),
        # 公開された本文のサンプル（低新規性计測の妥当性を手元で確認するため）
        'speech_samples': [
            m.content[:160] for m in state.conversation_log if m.speaker != "system"
        ][:6],
    }


lines: list[str] = []
for policy in ('legacy_repair', 'publish', 'append'):
    lines.append(f'--- novelty_policy={policy} ---')
    summary = run(policy)
    for key, value in summary.items():
        lines.append(f'  {key}: {value}')

text = "\n".join(lines) + "\n"
Path(r'c:\Users\yukic\local_llm_werewolf\_task048_sim.txt').write_text(text, encoding='utf-8')
print(text)
