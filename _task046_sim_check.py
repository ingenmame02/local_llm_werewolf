import sys
from pathlib import Path

SRC = Path(r'c:\Users\yukic\local_llm_werewolf\src')
sys.path.insert(0, str(SRC))

from collections import Counter

from werewolf.engine import GameEngine
from werewolf.llm_actions import OllamaActions
from werewolf.setup import create_game

CONFIG = r'c:\Users\yukic\local_llm_werewolf\config\players.json'

sys.path.insert(0, r'c:\Users\yukic\local_llm_werewolf\_analysis')
_spec_src = Path(r'c:\Users\yukic\local_llm_werewolf\_analysis\sim_game.py').read_text(encoding='utf-8')
# SimulatedClient だけ再定義（sim_game.py の末尾実行は避ける）
_class_src = _spec_src.split('state = create_game')[0]
_ns = {'__file__': str(_spec_src)}
exec(_class_src, _ns)
SimulatedClient = _ns['SimulatedClient']

state = create_game(seed=20261115, config_path=CONFIG, response_language='日本語')
client = SimulatedClient(state)
events: list[dict[str, object]] = []
actions = OllamaActions(
    state,
    client,
    seed=20261115,
    options={'question_hint_mode': 'none', 'temperature': 0.7},
    discussion_claim_mode='legacy',
    event_callback=events.append,
)
engine = GameEngine(state, actions, seed=20261115, event_callback=events.append)
result = engine.run()

counts = Counter(str(e.get('type')) for e in events)
print('requests:', client.requests)
print('resolution events:', counts.get('effective_context_resolution'))
print('compression events:', counts.get('context_compression_stage'))
cfg = [e for e in events if e.get('type') == 'llm_request_configuration']
print('config events:', len(cfg), '| budget fields:',
      sum(1 for e in cfg if isinstance(e.get('context_budget_breakdown'), dict)))
print('conflict flagged:', sum(1 for e in cfg if e.get('context_limit_conflict')))
print('configured vs observed:', Counter(
    (str(e.get('configured_context_limit')), str(e.get('server_observed_context_limit')))
    for e in cfg
))
print('requested==effective:', sum(
    1 for e in cfg if e.get('effective_requested_num_ctx') == e.get('requested_num_ctx')
))
sel = [e for e in events if e.get('type') == 'discussion_context_selected']
print('context_selected events:', len(sel))
print('retention all 1.0:', sel and all(e.get('required_evidence_retention') == 1.0 for e in sel))
print('stages:', Counter(str(e.get('compression_stage')) for e in sel))
print('dropped detail samples:', sum(len(e.get('dropped_events_detail') or []) for e in sel))
