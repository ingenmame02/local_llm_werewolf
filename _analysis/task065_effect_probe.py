"""TASK-065効果プローブ（LL不使用・読み取り専用）。基準ログを新検出器で流す。

1. プレイヤー名表記ゆれ: 旧判定（`Player\\s+\\d+` の完全一致）が未知名前と誤った件数を、
   新しい内部解決（normalized / literal）でどれだけ吸収できるか。公開本文は変更しない。
2. 時系列意図分離: 実機ログの全議論本文を発言順のsnapshotで `separated` と `literal`
   両方へ通し、future_event_reference（実質的未来参照）の件数差と質問行為・行動推測へ
   移った内訳を数える（game_425755ce9a83 Day2 T1 Player 1 の誤検出が焦点）。
3. 存在しない番号は吸収しない（正規化しすぎない）ことの確認。

出力はUTF-8の数値・カテゴリのみ。モデルは呼ばない。
"""
from __future__ import annotations

import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from werewolf.day_timeline import (  # noqa: E402
    FUTURE_SPEECH_FROM_UNSPOKEN,
    TIMELINE_INTENT_POLICY_LITERAL,
    TIMELINE_INTENT_POLICY_SEPARATED,
    TurnSnapshot,
    inspect_public_body,
)
from werewolf.name_resolution import (  # noqa: E402
    NAME_RESOLUTION_POLICY_LITERAL,
    NAME_RESOLUTION_POLICY_NORMALIZED,
    has_unresolved_reference,
    mention_labels,
    scan_references,
)
from werewolf.discussion_format import answers_previous_question  # noqa: E402

LOG = ROOT / "logs" / "game_425755ce9a83.private.jsonl"
OUT = ROOT / "_analysis" / "task065_effect_probe_out.txt"
OLD_UNKNOWN_RE = re.compile(r"Player\s+\d+")


def old_unknown(text: str, known: set[str]) -> list[str]:
    """llm_actions の旧判定を再現する（`Player\\s+\\d+` を実在名と文字列一致で照合）。"""
    return [token for token in OLD_UNKNOWN_RE.findall(text or "") if token not in known]


def load_events() -> tuple[list[dict[str, object]], tuple[str, ...]]:
    events: list[dict[str, object]] = []
    with io.open(LOG, encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                events.append(json.loads(raw))
    names: list[str] = []
    for event in events:
        for name in event.get("players") or []:
            if isinstance(name, str) and name not in names:
                names.append(name)
        speaker = event.get("player") or event.get("speaker")
        if isinstance(speaker, str) and speaker.startswith("Player") and speaker not in names:
            names.append(speaker)
    if not names:
        raise SystemExit("players not found in log")

    def _key(name: str) -> int:
        digits = re.findall(r"\d+", name)
        return int(digits[0]) if digits else 0

    return events, tuple(sorted(names, key=_key))


def walk_snapshots(events, player_names):
    """発言順に状態（既発言・死亡）を積み上げ、各発言の直前snapshotを yield する。"""
    spoken: list[str] = []
    dead: dict[str, int] = {}
    day = 1
    sequence = 0
    next_speech_id = 1
    all_points: list[tuple[str, int, int, int]] = []
    vote_done = False
    for event in events:
        kind = event.get("type")
        if isinstance(event.get("day"), int):
            day = int(event["day"])
        if kind in {"execution", "attack_death", "player_died", "morning_casualty"}:
            name = event.get("player") or event.get("victim") or event.get("name")
            if isinstance(name, str) and name in player_names:
                dead.setdefault(name, int(event.get("day") or day))
        if kind in {"voting_result", "vote_completed", "voting_phase_ended"}:
            vote_done = True
        if kind != "discussion_message":
            continue
        speaker = str(event.get("speaker") or event.get("player") or "?")
        body = str(event.get("content") or "")
        if not body.strip():
            continue
        turn_value = int(event.get("discussion_turn") or 0)
        snapshot = TurnSnapshot(
            day=day,
            turn=turn_value,
            actor=speaker,
            sequence=sequence,
            prior_this_turn=tuple(
                (name, 1) for name in spoken if name != speaker and name not in dead
            ),
            own_speeches_today=tuple((1, 1) for name in spoken if name == speaker),
            prior_today_events=len(spoken),
            speakers_pending=tuple(
                name for name in player_names if name not in spoken and name not in dead
            ),
            spoken_players=tuple(dict.fromkeys(spoken)),
            known_dead=tuple(dead.items()),
            alive_players=tuple(name for name in player_names if name not in dead),
            today_vote_done=vote_done,
            night_pending=not vote_done,
            next_speech_id=next_speech_id,
            all_speech_points=tuple(all_points),
        )
        yield speaker, day, turn_value, body, snapshot
        sequence += 1
        next_speech_id += 1
        all_points.append((speaker, day, turn_value, next_speech_id))
        spoken.append(speaker)


def inspect_all(events, player_names, policy: str) -> dict[str, object]:
    """全議論本文を policy で再検査し、findings行・内訳・質問行為数を集める。"""
    rows: list[str] = []
    totals: Counter[str] = Counter()
    intents: Counter[str] = Counter()
    bodies = 0
    for speaker, day, _turn, body, snapshot in walk_snapshots(events, player_names):
        report = inspect_public_body(body, snapshot, player_names, intent_policy=policy)
        bodies += 1
        fields = report.to_log_fields()
        totals["findings"] += int(fields["future_event_reference_count"])
        totals["unresolvable"] += int(fields["unresolvable_references"])
        category_counts = fields["future_event_reference_categories"]
        if isinstance(category_counts, dict):
            for key, value in category_counts.items():
                totals[f"cat:{key}"] += int(value or 0)
        totals["question_acts"] += int(report.future_question_acts)
        totals["action_speculations"] += int(report.future_action_speculations)
        totals["past_attributions"] += int(report.past_attributions_observed)
        for key, value in dict(report.intent_counts).items():
            intents[key] += int(value or 0)
        for finding in report.findings:
            excerpt = (getattr(finding, "sentence_excerpt", "") or finding.sentence)[:60]
            suffix = f" intent={finding.intent}" if getattr(finding, "intent", "") else ""
            rows.append(f"{speaker} D{day}: [{finding.category}] {excerpt}{suffix}")
    return {"bodies": bodies, "rows": rows, "totals": totals, "intents": intents}



def name_block(events, player_names) -> list[str]:
    """名前表記ゆれ: 旧判定の誤検出を新内部解決がどれだけ吸収するか数える。"""
    known = {str(name) for name in player_names}
    bodies: list[tuple[str, str]] = []
    for event in events:
        who = str(event.get("player") or event.get("speaker") or "?")
        kind = event.get("type")
        if kind == "discussion_message":
            text = str(event.get("content") or "")
        elif kind == "llm_action_completed":
            raw = event.get("raw_response") or event.get("response")
            text = raw if isinstance(raw, str) else ""
        else:
            continue
        if text.strip():
            bodies.append((who, text))

    old_hits: Counter[str] = Counter()
    norm_hits: Counter[str] = Counter()
    lit_hits: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    absorbed = 0
    for who, text in bodies:
        olds = old_unknown(text, known)
        unresolved_norm = [
            ref.token
            for ref in scan_references(text, player_names, policy=NAME_RESOLUTION_POLICY_NORMALIZED)
            if not ref.resolved
        ]
        unresolved_lit = [
            ref.token
            for ref in scan_references(
                text, player_names, policy=NAME_RESOLUTION_POLICY_LITERAL
            )
            if not ref.resolved
        ]
        old_hits[who] += len(olds)
        norm_hits[who] += len(unresolved_norm)
        lit_hits[who] += len(unresolved_lit)
        if olds and not unresolved_norm:
            absorbed += 1
        for token in olds:
            labels[f"old_mismatch:{token!r}"] += 1
        for key, value in mention_labels(
            text, player_names, policy=NAME_RESOLUTION_POLICY_NORMALIZED
        ).items():
            labels[f"label:{key}"] += int(value)

    lines = ["== 1. プレイヤー名表記ゆれ（公開本文は変更しない内部照合のみ） =="]
    lines.append(f"bodies_checked={len(bodies)}")
    lines.append(
        "unknown_tokens: "
        f"old_string_match={sum(old_hits.values())} "
        f"literal_policy={sum(lit_hits.values())} "
        f"normalized_policy={sum(norm_hits.values())}"
    )
    lines.append(f"bodies_where_old_flagged_and_new_absorbed={absorbed}")
    lines.append("by_model_action_owner_of_old_flag:")
    for who, count in sorted(old_hits.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"    {who}: old={count} normalized={norm_hits.get(who, 0)}")
    for label, count in sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))[:18]:
        lines.append(f"    {count:>3}x {label}")
    lines.append("guard（存在しない番号は吸収しない＝未解決として観測）:")
    for probe in ("Player 9", "P 12", "プレイヤー１２", "player 100"):
        refs = scan_references(probe, player_names)
        lines.append(
            f"    {probe!r}: unresolved={bool(has_unresolved_reference(probe, player_names))}"
            f" resolved_names={[r.canonical for r in refs if r.resolved]}"
        )
    return lines


def main() -> None:
    events, player_names = load_events()
    lines: list[str] = []
    lines.append(f"baseline_log={LOG.name}")
    lines.append(f"players={len(player_names)} -> {', '.join(player_names)}")
    lines.append("")
    lines.extend(name_block(events, player_names))

    separated = inspect_all(events, player_names, TIMELINE_INTENT_POLICY_SEPARATED)
    literal = inspect_all(events, player_names, TIMELINE_INTENT_POLICY_LITERAL)
    dropped = [row for row in literal["rows"] if row not in separated["rows"]]
    kept = separated["rows"]

    lines.append("")
    lines.append("== 2. 時系列検出の意図分離（同一本文・発言順snapshot・LL不使用） ==")
    for label, data in (("separated(既定)", separated), ("literal(TASK-059まで)", literal)):
        totals = data["totals"]
        lines.append(
            f"  [{label}] bodies={data['bodies']} findings={totals['findings']} "
            f"speech_from_unspoken={totals[f'cat:{FUTURE_SPEECH_FROM_UNSPOKEN}']} "
            f"unresolvable={totals['unresolvable']}"
        )
        lines.append(
            f"      past_attribution_observed={totals['past_attributions']} "
            f"question_acts={totals['question_acts']} "
            f"action_speculations={totals['action_speculations']}"
        )
        lines.append(f"      intent_counts={dict(sorted(data['intents'].items()))}")
    lines.append(f"  separatedで消えたfindings={len(dropped)} / 残ったfindings={len(kept)}")
    for row in dropped:
        lines.append(f"    - {row}")
    for row in kept:
        lines.append(f"    + {row}")

    # 3. 質問回答判定（旧: 文字列一致 / 新: 内部解決した正式名で照合）
    ordered = [
        (str(e.get("speaker")), str(e.get("content") or ""))
        for e in events
        if e.get("type") == "discussion_message" and str(e.get("content") or "").strip()
    ]
    checks = 0
    variant_only = 0
    legacy_answered = 0
    for index in range(1, len(ordered)):
        previous = ordered[index - 1][1]
        current = ordered[index][1]
        checks += 1
        canonicals = [
            ref.canonical for ref in scan_references(previous, player_names) if ref.canonical
        ]
        if legacy := answers_previous_question(current, previous):
            legacy_answered += 1
        if not canonicals:
            continue
        if not any(name in current for name in canonicals) and any(
            ref.resolved and not ref.exact for ref in scan_references(current, player_names)
        ):
            variant_only += 1
    lines.append("")
    lines.append("== 3. 質問回答判定（旧: 文字列一致 / 新: 内部解決で吸収） ==")
    lines.append(
        f"adjacent_pairs={checks} legacy_answer_flags={legacy_answered} "
        f"canonical_miss_but_variant_resolved={variant_only}"
    )

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
