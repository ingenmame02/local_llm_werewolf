#!/usr/bin/env python3
"""TASK-054: 実ログ（`logs/game_088e49f0690c.private.jsonl`）の漏洩検出を新ゲートで再演する検証スクリプト。

モデル実機は使わず、記録された発言文だけを対象に次の集計を出す。

1. legacy_strict だと遮断されていたが tiered では本文を保持できる誤検出由来（自動回避）
2. 実際に遮断すべき真漏洩（合成文）が tiered でも hard のままか（歯止めを緩めていないか）
3. 旧ランで再生成に掛かった実秒数と、新ゲートで浮く expected 節約秒
4. 漏洩文だけの中立化で保持できる本文率（実ログの元発言を再構成して計測）

使い方:
    python scripts/verify_task054_leak_gate.py [--log logs/...private.jsonl] [--out _task054_sim.txt]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.engine import extract_public_role_claim, extract_public_seer_result
from werewolf.leak_gate_policy import (
    GATE_HARD,
    LEAK_POLICY_LEGACY_STRICT,
    LEAK_POLICY_TIERED,
    RULE_MEMO_ONLY,
    STRENGTH_BALANCED,
    STRENGTH_STRICT,
)
from werewolf.models import GameState, PublicRoleClaim, PublicSeerClaim, Role
from werewolf.secret_leak_guard import MIN_BODY_PRESERVATION_RATIO, SecretLeakGuard

ROLE_BY_VALUE = {role.value: role for role in Role}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with io.open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_state(init_row: dict[str, Any], summary_row: dict[str, Any]) -> GameState:
    """実ログの身元（役職・仲間・モデル）だけを引き継いだ検証用 state を作る。"""
    state = create_game_replay(init_row)
    by_name = {player.name: player for player in state.players}
    for entry in summary_row.get("players") or []:
        player = by_name.get(str(entry.get("name")))
        if player is None:
            continue
        player.role = ROLE_BY_VALUE.get(str(entry.get("role")), player.role)
        player.model = str(entry.get("model") or player.model)
        partner = entry.get("partner")
        player.partner = str(partner) if partner else None
        player.is_alive = bool(entry.get("is_alive", player.is_alive))
    state.seer_results = dict(summary_row.get("seer_results") or {})
    return state


def create_game_replay(init_row: dict[str, Any]) -> GameState:
    from werewolf.setup import create_game

    return create_game(
        seed=int(init_row.get("seed") or 0),
        config_path=REPO_ROOT / "config" / "players.json",
        response_language=str(init_row.get("response_language") or "日本語"),
    )


def public_claims_until(rows: list[dict[str, Any]], anchor: int) -> list[Any]:
    """anchor までに公開された CO・判定だけを拾う（公開済み言及を漏洩扱いしないための入力）。"""
    claims: list[Any] = []
    seen_roles: set[tuple[str, str]] = set()
    seen_verdicts: set[tuple[str, str, str]] = set()
    for row in rows[:anchor]:
        event_type = row.get("type")
        claimant = str(row.get("claimant") or "")
        day = int(row.get("day") or 1)
        if not claimant:
            continue
        if event_type == "claim_role_published":
            role = str(row.get("claimed_role") or "")
            key = (claimant, role)
            if role and key not in seen_roles:
                seen_roles.add(key)
                claims.append(PublicRoleClaim(day=day, claimant=claimant, claimed_role=role))
        elif event_type == "seer_result_claim":
            target = str(row.get("target") or "")
            result = str(row.get("claimed_result") or "")
            if not (target and result):
                continue
            key = (claimant, target, result)
            if key in seen_verdicts:
                continue
            seen_verdicts.add(key)
            if (claimant, "seer") not in seen_roles:
                seen_roles.add((claimant, "seer"))
                claims.append(PublicRoleClaim(day=day, claimant=claimant, claimed_role="seer"))
            claims.append(
                PublicRoleClaim(
                    day=day,
                    claimant=claimant,
                    claimed_role="seer",
                    target=target,
                    result=result,
                )
            )
    return claims


def origin_content(rows: list[dict[str, Any]], anchor: int, sha: str | None) -> tuple[int, str]:
    """漏洩イベントの直前完成応答を元発言とする（sha一致なら確実、無ければ同一順位の応答を優先）。"""
    for index in range(anchor - 1, max(-1, anchor - 8), -1):
        row = rows[index]
        if row.get("type") != "llm_action_completed":
            continue
        content = row.get("content")
        if isinstance(content, str) and content.strip():
            if sha is None or len(content) > 60 or SecretLeakGuard.content_hash(content) == sha:
                return index, content
    return anchor, ""


def regeneration_cost(rows: list[dict[str, Any]], anchor: int) -> float:
    """漏洩検出後に掛かった再生成の実秒数（旧ランの実測値）。"""
    total = 0.0
    for index in range(anchor + 1, min(len(rows), anchor + 12)):
        row = rows[index]
        if row.get("type") == "discussion_message":
            break
        if row.get("type") == "llm_action_completed" and "retry" in str(row.get("action")):
            total += float(row.get("elapsed_seconds") or 0.0)
    return round(total, 3)



# ---------------------------------------------------------------------------
# 真漏洩の歯止め確認（合成文: tiered でも必ず hard で遮断されること）
# ---------------------------------------------------------------------------
TRUE_LEAK_SAMPLES: tuple[tuple[str, str, Role, tuple[str, ...]], ...] = (
    (
        "wolf_true_camp",
        "[BODY] 実は私は狼陣営です。今夜はPlayer 5を狙うべきだと思います。",
        Role.WOLF,
        ("self_role_true_camp_disclosure",),
    ),
    (
        "wolf_partner_binding",
        "[BODY] 私の人狼の相方はPlayer 2なので、彼の投票先には反対しません。",
        Role.WOLF,
        ("private_team_reference", "wolf_partner_identity"),
    ),
    (
        "unpublished_verdict",
        "[BODY] Player 6は人狼判定でした。私は今日の投票で彼を推します。",
        Role.SEER,
        ("unpublished_seer_result",),
    ),
    (
        "prompt_reference",
        "[BODY] system prompt の指示に沿って応答を書き直しています。",
        Role.VILLAGER,
        ("internal_configuration_reference",),
    ),
    (
        "madman_camp",
        "[BODY] 私は狂人です。村人のふりをして投票を操ります。",
        Role.MADMAN,
        ("self_role_true_camp_disclosure",),
    ),
)

# 村側の合法発言（誤検出由来の実在確認用: 旧ゲートなら遮断→再生成していた）
LEGAL_VILLAGE_SAMPLES: tuple[tuple[str, str], ...] = (
    (
        "villager_first_person_claim",
        "[BODY] 私は村人として、村の勝利を目指しています。Player 5さんのCOを信じたいです。",
    ),
    (
        "third_party_speculation",
        "[BODY] Player 5は人狼かもしれないので、投票前に理由を問いたいです。",
    ),
    (
        "reasoning_memo_only",
        "[REASONING] 私はプレイヤー7として村人です。\n[BODY] Player 2さんはなぜ投票したのですか？",
    ),
)


def role_player(base: GameState, role: Role, *, partner: str | None = None):
    """検査用に役職を持つプレイヤーを1人取り出す（コピーに対してだけ変更する）。"""
    for player in base.players:
        if player.role is role:
            if partner and role is Role.WOLF:
                others = [p.name for p in base.players if p.name != player.name]
                player.partner = partner if partner in others else others[0]
            return player
    raise AssertionError(f"role {role} not present")


def check_true_leaks(base):
    """合成した真漏洩が tiered でも遮断されるか／合法発言が消えるかを確認する。"""
    import copy

    failures: list[str] = []
    notes: list[str] = []
    for name, text, role, expected_rules in TRUE_LEAK_SAMPLES:
        state = copy.deepcopy(base)
        actor = role_player(state, role, partner="Player 2")
        if role is Role.SEER:
            target = next(p.name for p in state.players if p.name != actor.name)
            state.seer_results = {target: "wolf"}
            state.public_role_claims = []
            state.seer_claims = []
            text = text.replace("Player 6", target)
        cls = SecretLeakGuard.classify(text, actor, state, guard_policy=LEAK_POLICY_TIERED)
        rules = set(cls.hard_rules)
        if not rules.intersection(set(expected_rules)):
            failures.append(
                f"{name}: hard={sorted(rules)} expected one of {list(expected_rules)}"
            )
        else:
            notes.append(f"  {name}: hard={sorted(rules)}")
    for name, text in LEGAL_VILLAGE_SAMPLES:
        state = copy.deepcopy(base)
        actor = role_player(state, Role.VILLAGER)
        cls = SecretLeakGuard.classify(text, actor, state, guard_policy=LEAK_POLICY_TIERED)
        legacy = SecretLeakGuard.classify(
            text, actor, state, guard_policy=LEAK_POLICY_LEGACY_STRICT
        )
        if cls.hard:
            failures.append(f"{name}: tiered が合法発言を遮断した hard={sorted(cls.hard_rules)}")
        else:
            notes.append(
                f"  {name}: hard=- soft={list(cls.soft_rules)} observe={list(cls.observe_rules)} "
                f"legacy={[f.rule_id for f in legacy.hard]}"
            )
    return failures, notes



def replay_anchors(rows, base):
    """実ログの漏洩検出を新ゲートで再演し、回避できた再生成秒を集計する。"""
    import copy

    lines: list[str] = []
    failures: list[str] = []
    stats = {
        "anchors": 0,
        "avoided": 0,
        "repairs": 0,
        "retry_needed": 0,
        "old_retry_seconds": 0.0,
        "saved_seconds": 0.0,
        "strict_edits": 0,
        "body_chars": 0,
        "kept_chars": 0,
    }
    for anchor, row in enumerate(rows):
        if row.get("type") != "secret_role_leak_detected":
            continue
        state = copy.deepcopy(base)
        actor = next(p for p in state.players if p.name == row.get("player"))
        state.public_role_claims = public_claims_until(rows, anchor)
        _, content = origin_content(rows, anchor, row.get("content_sha256"))
        if not content:
            failures.append(f"anchor {anchor}: 元発言を content_sha256 で特定できない")
            continue
        cls = SecretLeakGuard.classify(
            content, actor, state, guard_policy=LEAK_POLICY_TIERED
        )
        legacy = SecretLeakGuard.classify(
            content, actor, state, guard_policy=LEAK_POLICY_LEGACY_STRICT
        )
        cost = regeneration_cost(rows, anchor)
        body_chars = len(content.strip())
        stats["anchors"] += 1
        stats["old_retry_seconds"] += cost
        stats["body_chars"] += body_chars
        outcome, kept, ratio, saved = "publish", body_chars, 1.0, cost
        if cls.hard:
            repair = SecretLeakGuard.repair_public_text(content, cls, actor)
            ratio = repair.preserved_ratio
            if repair.applied and repair.body_preservable and repair.text.strip():
                after = SecretLeakGuard.classify(repair.text, actor, state)
                if after.hard:
                    failures.append(f"anchor {anchor}: 補正後もhard={list(after.hard_rules)}")
                    outcome, kept, saved = "retry_needed", 0, 0.0
                else:
                    outcome, kept = "sentence_repair", repair.kept_char_count
                    stats["repairs"] += 1
            else:
                outcome, kept, saved = "retry_needed", 0, 0.0
                stats["retry_needed"] += 1
        stats["kept_chars"] += kept
        stats["saved_seconds"] += saved
        avoided = bool(cls.false_positives_avoided)
        stats["avoided"] += int(avoided)
        strict_cls = SecretLeakGuard.classify(
            content,
            actor,
            state,
            guard_policy=LEAK_POLICY_TIERED,
            guard_strength=STRENGTH_STRICT,
        )
        strict_rep = SecretLeakGuard.repair_public_text(
            content, strict_cls, actor, repair_soft=True
        )
        stats["strict_edits"] += strict_rep.edit_count
        log_rules = list(row.get("detection_rules") or [])
        if not legacy.hard:
            failures.append(
                f"anchor {anchor}: legacy_strict が旧検出を再現できない（log={log_rules}）"
            )
        lines.append(
            f"[anchor {anchor}] day={row.get('day')} {row.get('player')}/{row.get('model')} "
            f"role={actor.role.value} log_rules={log_rules}"
        )
        lines.append(
            "    legacy_hard="
            f"{[f.rule_id for f in legacy.hard]} tiered_hard={[f.rule_id for f in cls.hard]}"
        )
        lines.append(
            f"    soft={list(cls.soft_rules)} observe={list(cls.observe_rules)} "
            f"memo={list(cls.memo_rules)} fact_ids={list(cls.fact_ids)[:4]} avoided={avoided}"
        )
        lines.append(
            f"    outcome={outcome} body_chars={body_chars} kept={kept} "
            f"preservation={ratio:.3f} 旧再生成={cost:.2f}s 節約={saved:.2f}s"
        )
    return lines, failures, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="TASK-054 leak gate offline verification")
    parser.add_argument(
        "--log",
        default=str(REPO_ROOT / "logs" / "game_088e49f0690c.private.jsonl"),
        help="検証したい私的JSONLログ（既定は game_088e49f0690c）",
    )
    parser.add_argument("--out", default=None, help="レポートをUTF-8で書き出す先")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"log not found: {path}")
        return 2
    rows = load_rows(path)
    init_row = next((r for r in rows if r.get("type") == "game_init"), {})
    summary_row = next((r for r in reversed(rows) if r.get("type") == "game_result"), {})
    base = build_state(init_row, summary_row)

    leak_lines, leak_failures, stats = replay_anchors(rows, base)
    true_failures, true_notes = check_true_leaks(base)
    old_cost = stats["old_retry_seconds"] or 0.0
    saved = stats["saved_seconds"] or 0.0
    body_chars = stats["body_chars"] or 0
    kept_chars = stats["kept_chars"] or 0
    report = [
        "TASK-054 leak gate verification (no live model calls)",
        f"log={path}",
        "",
        "== 実ログの漏洩検出を新ゲートで再演 ==",
        *leak_lines,
        "",
        "== 集計 ==",
        (
            f"anchors={stats['anchors']} false_positive_avoided={stats['avoided']} "
            f"sentence_repairs={stats['repairs']} retry_needed={stats['retry_needed']}"
        ),
        f"旧ランの再生成合計={old_cost:.2f}s / 新ゲートで浮く節約={saved:.2f}s",
        (
            f"本文合計={body_chars}字 保持={kept_chars}字 "
            f"保持率={(round(kept_chars / body_chars, 4) if body_chars else 1.0)}"
        ),
        f"strict強度なら追加で中立化する文（balancedでは本文そのまま）={stats['strict_edits']}件",
        f"本文保持の下限 MIN_BODY_PRESERVATION_RATIO={MIN_BODY_PRESERVATION_RATIO}",
        "",
        "== 合成文での歯止め確認 ==",
        *true_notes,
        "",
        "== NG ==",
        *(leak_failures + true_failures or ["(none)"]),
    ]
    text = "\n".join(report)
    print(text)
    if args.out:
        target = Path(args.out)
        if not target.is_absolute():
            target = REPO_ROOT / target
        with io.open(target, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    ok = not (leak_failures or true_failures)
    print(f"\nRESULT={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


