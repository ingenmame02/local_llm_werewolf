import json, math

LOG = r'c:\Users\yukic\local_llm_werewolf\logs\game_22754212a062.private.jsonl'
rows = [json.loads(l) for l in open(LOG, encoding='utf-8')]
cfg = [r for r in rows if r.get('type') == 'llm_request_configuration']
sel = [r for r in rows if r.get('type') == 'discussion_context_selected']
gr = [r for r in rows if r.get('type') == 'game_result'][0]
speeches = [m for m in gr['conversation_log'] if m['speaker'] != 'system']

MARGIN = 512
FLOOR = 600
MAX_CHARS = 12000
WRAPPER = len('[Day 2 / Speech 3 / Player 2 / In-game speech] ')


def full_transcript_chars(n_events: int) -> int:
    return sum(len(json.dumps(m['content'], ensure_ascii=False)) + WRAPPER for m in speeches[:n_events])


def budget(raw_chars, ev_chars, num_ctx, num_predict, coef):
    fixed = math.ceil(max(0, raw_chars - ev_chars) * coef)
    available = int(num_ctx) - int(num_predict) - MARGIN
    fitted = int((available - fixed) / coef)
    return min(MAX_CHARS, max(FLOOR, fitted))


for coef in (0.8, 0.5, 0.35):
    floored = trimmed = untouched = 0
    kept = 0
    for c in cfg:
        if c.get('action') not in ('discussion', 'vote', 'wolf_discussion'):
            continue
        prev = [
            r for r in sel
            if r.get('player') == c.get('player')
            and r.get('day') == c.get('day')
            and r.get('sequence', 0) < c.get('sequence', 0)
        ]
        if not prev:
            continue
        ev = prev[-1].get('evidence_char_count') or 0
        full = full_transcript_chars(prev[-1]['transcript_event_count'])
        raw = c.get('raw_prompt_char_count') or 0
        fitted = budget(raw, ev, c.get('requested_num_ctx') or 12288, c.get('num_predict') or 2560, coef)
        if fitted <= FLOOR:
            floored += 1
        if fitted >= full:
            untouched += 1
        else:
            trimmed += 1
            kept += min(fitted, full)
    print(f'coef={coef}: untouched(full transcript)={untouched} trimmed={trimmed} floored_at_600={floored} | kept_chars_when_trimmed={kept}')

print('\nlast turns: full transcript vs applied evidence')
for r in sel[-6:]:
    print(f"  d{r['day']} {r.get('player')} events={r['transcript_event_count']} full={full_transcript_chars(r['transcript_event_count'])} applied={r.get('evidence_char_count')}")
