"""TASK-057 のローカル検証（実機Ollama不使用・モッククライアントのみ）。

目的は「コンテキスト節約」と「一貫性維持」が両立しているかを同じ履歴で測ること。
legacy（TASK-057前の形）と simplified（既定）を並べて、

1. 同じ履歴でのプロンプト字数・推定トークン・core/optional内訳・重複制約数・
   発言本文の二重掲載回数を実測する。
2. モックでのフルゲームを両ポリシーで走らせ、送信字数・参照化による節約・削除段階とあわせて
   パース失敗・再要求・議論時間が変わらないことを確認する（発言の自由は縛らない）。
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(r"c:\Users\yukic\local_llm_werewolf")
sys.path.insert(0, str(REPO / "src"))

from werewolf.cli import GameLogger
from werewolf.engine import GameEngine
from werewolf.llm_actions import OllamaActions, request_token_estimate
from werewolf.ollama import ChatResponse
from werewolf.prompt_contract import (
    PROMPT_CONTRACT_POLICY_LEGACY,
    PROMPT_CONTRACT_POLICY_SIMPLIFIED,
    duplicated_constraints,
    find_sections,
    scan_prompt_constraints,
)
from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game

CONFIG = str(REPO / "config" / "players.json")
LOG_DIR = REPO / "_analysis" / "logs_task057"
SEED = 20261121
FORBIDDEN_PUBLIC_KEYS = ("attack_target", "divination_result", "true_role")

TOPICS = (
    "昨夜の襲撃先",
    "投票の流れ",
    "COを出した理由",
    "占いの対象",
    "議論の順番",
    "発言の分量",
    "保留の意図",
    "狂人の可能性",
    "対立の構図",
    "確認したい候補",
    "夜の占い設定",
    "味方の数",
    "多数派の形成",
    "棄権の動き",
    "初日のCO競争",
    "投票先の言い訳",
    "襲撃先の不自然さ",
    "沉默の理由",
    "質問への回答順",
    "占い結果の非公開",
    "投票の根拠",
    "二重COの可能性",
    "討論の停滞",
    "夜投票の読み",
)
PHRASINGS = (
    "{other}さんに確認したい。{topic}について{third}さんとは食い違うので、今日はそこを詰めておきたい。",
    "{topic}が引っかかっている。{third}さんの見解を先に聞かせてほしい。{other}さんと同じ結論にはまだいけない。",
    "{topic}を整理したい。{third}さんが先に述べた根拠は弱いと思うので、{other}さんの視点も知りたい。",
)


class SimClient:
    """自由な口調の本文を返すだけの実機不使用クライアント（COはしない）。"""

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

    def _others(self, speaker: str) -> tuple[str, str]:
        others = [n for n in self.names if n != speaker and n in self.state.alive_players]
        first = others[0] if others else speaker
        second = others[1] if len(others) > 1 else first
        return first, second

    def chat(self, model, messages, options=None) -> str:
        return self.chat_with_metadata(model, messages, options).content

    def chat_with_metadata(self, model, messages, options=None, tools=None) -> ChatResponse:
        self.requests += 1
        system = messages[0]["content"] if messages else ""
        user = messages[1]["content"] if len(messages) > 1 else ""
        speaker = self._speaker(system)
        other, third = self._others(speaker)
        speaker_index = self.names.index(speaker) if speaker in self.names else 0
        turn = int(getattr(self.state, "discussion_turn", 0) or 0)
        # 論点をずらして発言文をできる限り固有にする（実ログは各自違う文を話すため）
        topic = TOPICS[
            (self.state.day + speaker_index * 3 + turn * 5 + self.requests) % len(TOPICS)
        ]
        phrasing = PHRASINGS[(speaker_index + turn + self.requests) % len(PHRASINGS)]
        body = phrasing.format(other=other, third=third, topic=topic)
        if "decide on your public role claim" in user:
            return ChatResponse(content="[CO] None", attempts=1)
        if "[CO] None, Hold, or your public role claim" in user:
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
        return ChatResponse(content=body, attempts=1)



def build(actor, state, history, policy: str):
    previous = getattr(state, "prompt_contract_policy", None)
    state.prompt_contract_policy = policy
    try:
        return PromptBuilder.build_discussion_prompt(
            actor,
            tuple(state.alive_players),
            history,
            day=state.day,
            state=state,
            include_claim=True,
            context_strategy="hybrid_retrieval",
            question_hint_mode="strict",
            prompt_contract_policy=policy,
        )
    finally:
        state.prompt_contract_policy = previous


def unique_bodies(state) -> dict[str, int]:
    """状態中で1回しか喋られていない本文だけを選ぶ（モックの canned 文の混入を防ぐ）。

    同じ文を複数人が発言していると「別IDなのに同じ文字列」になって重複判定ができないため、
    実ログ相当（各自が別の文を話す）のケースだけを測る。
    """
    counts: dict[str, int] = {}
    for message in state.conversation_log:
        if message.speaker == "system":
            continue
        body = message.content
        if len(body) < 24:
            continue
        counts[body] = counts.get(body, 0) + 1
    return {body: 1 for body, used in counts.items() if used == 1}


def restatement_report(text: str, bodies: dict[str, int]) -> tuple[int, int, dict[str, int]]:
    """証拠ブロック以外で本文が再掲されている字数と、その箇所数を出す。"""
    start = text.find("[CONVERSATION_EVIDENCE")
    if start < 0:
        start = len(text)
    outside = text[:start] + text[_evidence_end(text) :]
    hits: dict[str, int] = {}
    total_chars = 0
    total_count = 0
    for body in bodies:
        occurrences = outside.count(body)
        if occurrences:
            hits[body[:24]] = occurrences
            total_chars += len(body) * occurrences
            total_count += occurrences
    return total_count, total_chars, hits


def _evidence_end(text: str) -> int:
    """[CONVERSATION_EVIDENCE...] の後ろ（CURRENT_TASK 以降）を分ける位置。"""
    end = text.find("[CURRENT_TASK]", text.find("[CONVERSATION_EVIDENCE"))
    return end if end > 0 else len(text)


def prompt_size_ab(state) -> None:
    """同じ履歴で simplified / legacy の送信量・重複・内訳を比べる。"""
    history = tuple(
        message
        for message in state.conversation_log
        if message.speaker != "system" or "begins" in (message.content or "")
    )
    actors = [player for player in state.players if player.is_alive]
    bodies = unique_bodies(state)
    print("== same-history A/B ==")
    totals: dict[str, list[dict[str, object]]] = {}
    for policy in (PROMPT_CONTRACT_POLICY_LEGACY, PROMPT_CONTRACT_POLICY_SIMPLIFIED):
        rows = []
        for actor in actors:
            ctx = build(actor, state, history, policy)
            diag = ctx.diagnostics or {}
            counts = scan_prompt_constraints(ctx.user_prompt)
            dup = duplicated_constraints(counts)
            messages = [
                {"role": "system", "content": ctx.system_prompt},
                {"role": "user", "content": ctx.user_prompt},
            ]
            # 送信見積もり（推定トークン）。両ポリシーで同じ:num_predict は同一値を使う。
            estimate = request_token_estimate(
                messages, getattr(actor, "num_predict", None) or 512
            )
            restate_count, restate_chars, _hits = restatement_report(ctx.user_prompt, bodies)
            rows.append(
                {
                    "chars": len(ctx.user_prompt),
                    "tokens": int(estimate),
                    "core": int(diag.get("prompt_core_chars") or 0),
                    "optional": int(diag.get("prompt_optional_chars") or 0),
                    "dup_kinds": len(dup),
                    "dup_extra": sum(value - 1 for value in dup.values()),
                    "refs": int(diag.get("dedup_reference_count") or 0),
                    "saved": int(diag.get("dedup_saved_chars") or 0),
                    "bodies_restate": restate_count,
                    "bodies_restate_chars": restate_chars,
                    "conflicts": int(diag.get("behavior_contract_conflict_count") or 0),
                    "removed": int(diag.get("optional_removed_chars") or 0),
                }
            )
        totals[policy] = rows
        n = max(1, len(rows))
        print(f"  {policy:10s} n={len(rows)}")
        print(
            f"    chars   avg={sum(r['chars'] for r in rows)/n:8.0f}"
            f"  max={max(r['chars'] for r in rows):6d}"
            f" | tokens avg={sum(r['tokens'] for r in rows)/n:8.0f}"
            f"  max={max(r['tokens'] for r in rows):6d}"
        )
        print(
            f"    core    avg={sum(r['core'] for r in rows)/n:8.0f}"
            f" | optional avg={sum(r['optional'] for r in rows)/n:7.0f}"
            f" removed_optional={sum(r['removed'] for r in rows):6d}"
        )
        print(
            f"    dup     rule_kinds={sum(r['dup_kinds'] for r in rows)/n:5.2f}"
            f"  extra_lines={sum(r['dup_extra'] for r in rows)/n:5.2f}"
            f" | id_refs={sum(r['refs'] for r in rows)/n:5.2f}"
            f"  saved={sum(r['saved'] for r in rows)/n:7.1f}"
        )
        print(
            f"    restate outside evidence: places="
            f"{sum(r['bodies_restate'] for r in rows)/n:5.2f}"
            f" chars={sum(r['bodies_restate_chars'] for r in rows)/n:8.1f}"
            f" | contract_conflicts={sum(r['conflicts'] for r in rows)/n:4.2f}"
        )
    legacy_rows = totals[PROMPT_CONTRACT_POLICY_LEGACY]
    simple_rows = totals[PROMPT_CONTRACT_POLICY_SIMPLIFIED]
    legacy_chars = sum(r["chars"] for r in legacy_rows)
    simple_chars = sum(r["chars"] for r in simple_rows)
    legacy_tokens = sum(r["tokens"] for r in legacy_rows)
    simple_tokens = sum(r["tokens"] for r in simple_rows)
    print(
        f"  正味差: chars {legacy_chars - simple_chars:+d}"
        f" ({(legacy_chars - simple_chars) / max(1, legacy_chars) * 100:.1f}%減)"
        f" | tokens {legacy_tokens - simple_tokens:+d}"
        f" | core最小値 legacy={min(r['core'] for r in legacy_rows)}"
        f" simplified={min(r['core'] for r in simple_rows)}"
    )


def example_reference_lines(state) -> None:
    """参照化された行の実物を出す（空参照・本文の再掲が無いことの確認）。"""
    actor = next(player for player in state.players if player.is_alive)
    history = tuple(
        message
        for message in state.conversation_log
        if message.speaker != "system" or "begins" in (message.content or "")
    )
    ctx = build(actor, state, history, PROMPT_CONTRACT_POLICY_SIMPLIFIED)
    print("== [ref:sN] を使う行（simplified） ==")
    shown = 0
    for line in ctx.user_prompt.splitlines():
        if "[ref:s" in line and shown < 6:
            print("   ", line[:170])
            shown += 1
    legacy_ctx = build(actor, state, history, PROMPT_CONTRACT_POLICY_LEGACY)
    print("   legacy の同一箇所に本文が残っている例:", end=" ")
    samples = [
        message.content
        for message in state.conversation_log
        if message.speaker != "system" and len(message.content) > 24
    ]
    # 履歴に実際に載っている本文だけを調べる（予算で削られた本文は比較対象にしない）
    listed = [body for body in samples if body in ctx.user_prompt]
    print("   同じ本文の掲載箇所（実在する発言のみ）:")
    for body in listed[:3]:
        print(
            f"     {body[:34]}... legacy="
            f"{legacy_ctx.user_prompt.count(body)}箇所"
            f" simplified={ctx.user_prompt.count(body)}箇所"
        )


def duplicate_section_map(state) -> None:
    """どの節が同じ発言本文をまだ載せているか（見出し単位の内訳）を出す。"""
    actor = next(
        player for player in state.players if player.is_alive and player.name != "Player 5"
    )
    history = tuple(
        message
        for message in state.conversation_log
        if message.speaker != "system" or "begins" in (message.content or "")
    )
    bodies = unique_bodies(state)
    for policy in (PROMPT_CONTRACT_POLICY_LEGACY, PROMPT_CONTRACT_POLICY_SIMPLIFIED):
        ctx = build(actor, state, history, policy)
        text = ctx.user_prompt
        # セクション台帳（find_sections）で位置を割り当て、どの節が本文を再掲しているか数える
        spans = find_sections(text)
        ordered = sorted(spans.items(), key=lambda kv: kv[1][0])

        def head_of(pos: int) -> str:
            current = "(登録外の先頭部分)"
            for name, (start, end) in ordered:
                if start <= pos:
                    current = name
                if start <= pos < end:
                    current = name
            return current

        tally: dict[str, list[int]] = {}
        repeated = {body: text.count(body) for body in bodies if text.count(body) > 1}
        for body in repeated:
            for match in re.finditer(re.escape(body), text):
                tally.setdefault(head_of(match.start()), []).append(len(body))
        print(f"-- 同じ本文を2度以上載せている節 ({policy}) actor={actor.name} --")
        ranked = sorted(tally.items(), key=lambda kv: -sum(kv[1]))
        for section, sizes in ranked[:8]:
            note = "（証拠＝本来の1箇所）" if section == "conversation_evidence" else ""
            print(f"   {section}: {len(sizes)}箇所 / {sum(sizes)}字 {note}")




def run_case(policy: str) -> dict[str, object]:
    """モッククライアントでフルゲームを進め、ログへ出た契約指標を読み出す。"""
    game_id = f"task057_{policy}"
    state = create_game(seed=SEED, config_path=CONFIG, response_language="日本語")
    state.prompt_contract_policy = policy
    client = SimClient(state)
    events: list[dict[str, object]] = []
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = GameLogger(
        game_id, LOG_DIR, {"seed": SEED, "task": "TASK-057"}, private_enabled=True
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
        seed=SEED,
        options={"question_hint_mode": "strict"},
        discussion_claim_mode="legacy",
        event_callback=on_event,
    )
    engine = GameEngine(state, actions, seed=SEED, event_callback=on_event)
    result = engine.run()
    logger.finalize(state, traces=None, error=None)

    text = (LOG_DIR / f"game_{game_id}.jsonl").read_text(encoding="utf-8")
    payload = [json.loads(line) for line in text.splitlines() if '"game_result"' in line]
    metrics = (payload[-1].get("ledger_metrics") or {}) if payload else {}
    model_stats = (payload[-1].get("model_stats") or {}) if payload else {}
    selections = [
        event for event in events if event.get("type") == "discussion_context_selected"
    ]
    stages: dict[str, int] = {}
    dropped_names: dict[str, int] = {}
    for event in selections:
        key = str(event.get("contract_removal_stage"))
        stages[key] = stages.get(key, 0) + 1
        for name in event.get("dropped_optional_sections") or []:
            dropped_names[name] = dropped_names.get(name, 0) + 1
    speeches = [event for event in events if event.get("type") == "discussion_message"]
    print(
        f"=== {game_id} === winner={getattr(result, 'winner', None)}"
        f" requests={client.requests} speeches={len(speeches)}"
    )
    print(
        "  policy:", payload[-1].get("prompt_contract_policy"),
        "| selections:", len(selections),
    )
    print(
        "  avg_prompt_user_chars:", metrics.get("avg_prompt_user_chars"),
        "| max:", metrics.get("max_prompt_user_chars"),
        "| avg_core:", metrics.get("avg_prompt_core_chars"),
        "| avg_optional:", metrics.get("avg_prompt_optional_chars"),
    )
    print(
        "  dedup_refs:", metrics.get("dedup_references_total"),
        "| dedup_saved:", metrics.get("dedup_saved_chars_total"),
        "| dup_constraints:", metrics.get("duplicate_constraint_total"),
        "| optional_removed:", metrics.get("optional_removed_chars_total"),
    )
    print("  removal_stages:", stages, "| dropped_sections:", dropped_names or "none")
    print(
        "  parse_failures:",
        sum(int(stats.get("parse_failure_count") or 0) for stats in model_stats.values()),
        "| retries:", metrics.get("retry_requests_sent"),
        "| discussion_seconds:", metrics.get("discussion_elapsed_seconds"),
    )
    leaks = [key for key in FORBIDDEN_PUBLIC_KEYS if key in text]
    print("  非公開キーのログ混入:", leaks or "none")
    return {
        "policy": policy,
        "state": state,
        "metrics": metrics,
        "selections": len(selections),
        "stages": stages,
    }


def main() -> None:
    cases = {}
    for policy in (PROMPT_CONTRACT_POLICY_LEGACY, PROMPT_CONTRACT_POLICY_SIMPLIFIED):
        cases[policy] = run_case(policy)
    # 同じ進行履歴（legacy側の状態）をそのまま使って送信量を比べる
    reference_state = cases[PROMPT_CONTRACT_POLICY_LEGACY]["state"]
    prompt_size_ab(reference_state)
    duplicate_section_map(reference_state)
    example_reference_lines(reference_state)


if __name__ == "__main__":
    main()

