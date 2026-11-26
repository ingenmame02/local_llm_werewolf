#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人狼ゲームJSONLログ -> 公開用Markdown変換ツール

logs/game_<ID>.jsonl を読み込み、公開済み情報
(発言 / 公開CO / 投票結果 / 朝の死亡 / 勝敗) のみを抽出し、
プレイヤー名 (Player N) をモデル名に置換した Markdown を出力する。

プライベート情報 (真役職・夜行動の詳細・未公開の占い結果等) は
いかなる場合も出力しない (イベントタイプのホワイトリスト方式)。

使い方:
  python "Public conversion tool/log_to_public_md.py" <ゲームIDまたはjsonlパス> [-o 出力.md]

例:
  python "Public conversion tool/log_to_public_md.py" 425755ce9a83
  # -> Public conversion tool/public_425755ce9a83.md が生成される
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# 公開Markdownに載せるイベントタイプのホワイトリスト (これ以外は完全に無視)
PUBLIC_TYPES = {
    "game_init",
    "first_night_no_attack",
    "morning_casualty",
    "discussion_message",
    "claim_role_published",
    "seer_result_claim",
    "voting_result",
    "winner_decided",
    "game_result",
}

ROLE_LABELS = {
    "seer": "占い師",
    "werewolf": "人狼",
    "madman": "狂人",
    "neutral": "中立",
    "guardian": "守護霊",
}
CAMP_LABELS = {"village": "村人", "werewolf": "人狼", "madman": "狂人", "neutral": "中立"}
RESULT_LABELS = {"human": "人間", "werewolf": "人狼"}
WIN_REASON_LABELS = {
    "no_living_wolves": "生存する人狼がいなくなったため",
    "wolves_equal_village": "村人と人狼の勢力が並んだため",
    "no_living_villagers": "生存する村人がいなくなったため",
}

# Player N / PlayerN / プレイヤーN (全角数字含む) を捕まえる
NAME_RE = re.compile(r"(?:Player|プレイヤー)\s*([0-9０-９]+)")
_FULLWIDTH = str.maketrans("０１２３４５６７８９", "0123456789")


def resolve_log_path(target: str, logs_dir: Path) -> Path:
    """ゲームIDまたはパスから対象ログファイルを見つける。"""
    p = Path(target)
    if p.is_file():
        return p
    stem = p.stem  # "game_xxx.jsonl" -> "xxx" でも指定できるようにする
    if stem.startswith("game_"):
        stem = stem[len("game_"):]
    for candidate in (logs_dir / f"game_{stem}.jsonl", logs_dir / f"game_{target}.jsonl", p):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"[ERROR] ログが見つかりません: {target} (検索先: {logs_dir})")


def load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            print(f"[WARN] {lineno}行目のJSON解析に失敗 - スキップ", file=sys.stderr)
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def player_index(name: str | None) -> int | None:
    m = NAME_RE.search(str(name or ""))
    return int(m.group(1).translate(_FULLWIDTH)) if m else None


def build_player_model_map(events: list[dict]) -> dict[int, str]:
    """Player番号 -> モデル名 の対応表を作る。game_result が最優先、
    無ければ個々のアクションイベントの player/model ペアから補完する。"""
    mapping: dict[int, str] = {}
    for ev in events:
        if ev.get("type") == "game_result":
            for pl in ev.get("players") or []:
                idx = player_index(pl.get("name"))
                model = pl.get("model")
                if idx is not None and model:
                    mapping[idx] = str(model)
    for ev in events:
        idx = player_index(ev.get("player") or ev.get("speaker") or ev.get("claimant") or ev.get("voter"))
        model = ev.get("model")
        if idx is not None and model and idx not in mapping:
            mapping[idx] = str(model)
    return mapping


def convert_names(text: str, mapping: dict[int, str], leftover: set[str]) -> str:
    """文中の Player N / PlayerN をモデル名に置換する。
    対応が存在しない場合は元の表記を残し leftover に記録する。"""

    def repl(m: re.Match) -> str:
        idx = int(m.group(1).translate(_FULLWIDTH))
        model = mapping.get(idx)
        if model is None:
            leftover.add(m.group(0))
            return m.group(0)
        return model

    return NAME_RE.sub(repl, text)


# === 集計 ===


def new_day() -> dict:
    return {"note": None, "casualty_seen": False, "casualty_dead": set(),
            "turns": {}, "claims": [], "voting": None}


def aggregate(events: list[dict]) -> dict:
    days: dict[int, dict] = {}
    meta = {"timestamp": None, "mode": None, "personalities": {}}
    result: dict = {"winner": None, "reason": None}
    final_alive: set[int] | None = None
    causes: dict[int, str] = {}
    prev_alive: set[int] | None = None

    def bucket(d: int) -> dict:
        b = days.get(d)
        if b is None:
            b = days[d] = new_day()
        return b

    def alive_set(ev: dict) -> set[int]:
        return {i for i in map(player_index, ev.get("alive") or []) if i is not None}

    for ev in events:
        etype = ev.get("type")
        if etype not in PUBLIC_TYPES or ev.get("private"):
            continue
        if etype == "game_init":
            meta["timestamp"] = ev.get("timestamp")
            meta["mode"] = ev.get("mode")
            for name, prof in (ev.get("personality_profiles") or {}).items():
                idx = player_index(name)
                if idx is not None:
                    meta["personalities"][idx] = str(prof)
            continue
        if etype == "first_night_no_attack":
            bucket(int(ev.get("day") or 1))["note"] = "初夜の襲撃はルールにより発生なし"
            continue
        if etype == "winner_decided":
            result["winner"] = ev.get("winner")
            result["reason"] = ev.get("reason")
            continue
        if etype == "morning_casualty":
            d = int(ev.get("day") or 1)
            b = bucket(d)
            alive = alive_set(ev)
            dead = {i for i in map(player_index, ev.get("dead") or []) if i is not None}
            if prev_alive and not dead:
                dead = prev_alive - alive
            for idx in dead:
                causes[idx] = f"Day {d - 1} 夜の襲撃(朝{d}に判明)" if d > 1 else f"Day {d} 夜の襲撃"
            b["casualty_seen"] = True
            b["casualty_dead"] = dead
            if alive:
                prev_alive = alive
                final_alive = alive
            continue
        if etype == "discussion_message":
            d = int(ev.get("day") or 1)
            turn = int(ev.get("discussion_turn") or 1)
            idx = player_index(ev.get("speaker") or ev.get("player"))
            if idx is None:
                continue
            bucket(d)["turns"].setdefault(turn, []).append((idx, str(ev.get("content") or "")))
            continue
        if etype in ("claim_role_published", "seer_result_claim"):
            bucket(int(ev.get("day") or 1))["claims"].append(ev)
            continue
        if etype == "voting_result":
            d = int(ev.get("day") or 1)
            b = bucket(d)
            b["voting"] = ev
            for name in ev.get("executed") or []:
                idx = player_index(name)
                if idx is not None:
                    causes[idx] = f"Day {d} 投票による処刑"
            alive = alive_set(ev)
            if alive:
                prev_alive = alive
                final_alive = alive
            continue
        if etype == "game_result":
            aw = {i for i in map(player_index, ev.get("alive_players") or []) if i is not None}
            if aw:
                final_alive = aw
            result["winner"] = result["winner"] or ev.get("winner")
            result["reason"] = result["reason"] or ev.get("winner_reason")
    return {"days": days, "meta": meta, "result": result,
            "final_alive": final_alive, "causes": causes}


# === 描画ヘルパー ===


def quote_body(converted: str) -> str:
    t = converted.strip()
    if not t or t in {"Pass", "パス"}:
        return "> *(スキップ)*"
    lines = [ln.rstrip() for ln in t.splitlines() if ln.strip()]
    return "\n".join("> " + ln for ln in lines) or "> *(スキップ)*"


def render_claims(claims: list[dict], M) -> list[str]:
    """公開COイベントを重複排除して見やすく整形する。"""
    role_entries: dict[tuple[int, str], dict] = {}
    div_entries: dict[tuple[int, int], dict] = {}
    for c in claims:
        who = player_index(c.get("player") or c.get("claimant"))
        if who is None:
            continue
        if c.get("claimed_role"):
            key = (who, str(c["claimed_role"]))
            e = role_entries.setdefault(key, {"reason": None})
            rsn = c.get("reason")
            if rsn and not e["reason"]:
                e["reason"] = str(rsn)
        tgt = player_index(c.get("target"))
        if tgt is not None and c.get("claimed_result"):
            de = div_entries.setdefault((who, tgt), {"result": None})
            if de["result"] is None:
                de["result"] = str(c["claimed_result"])
    out: list[str] = []
    claimants = sorted({k[0] for k in role_entries} | {k[0] for k in div_entries})
    for who in claimants:
        own_roles = [(role, e) for (w, role), e in role_entries.items() if w == who]
        own_divs = [(tgt, de) for (w, tgt), de in div_entries.items() if w == who]
        if own_roles:
            for role, e in own_roles:
                label = ROLE_LABELS.get(role, f"{role}役")
                line = f"- **{M(who)}**: {label} CO"
                if e["reason"]:
                    line += f"(理由: {e['reason']})"
                out.append(line)
            for tgt, de in own_divs:
                rl = RESULT_LABELS.get(de["result"], str(de["result"]))
                out.append(f"  - 占い結果の申告: **{M(tgt)}** -> {rl}")
        else:
            for tgt, de in own_divs:
                rl = RESULT_LABELS.get(de["result"], str(de["result"]))
                out.append(f"- **{M(who)}**(役職名なし): 占い結果の申告: **{M(tgt)}** -> {rl}")
    return out


def render_voting(v: dict, M) -> list[str]:
    out = ["#### 投票", "", "| 投票者 | 投票先 |", "| --- | --- |"]
    votes = v.get("votes") or {}
    for voter_name, target_name in votes.items():
        vi, ti = player_index(voter_name), player_index(target_name)
        left = M(vi) if vi is not None else "?"
        right = M(ti) if ti is not None else "*(記入なし)*"
        out.append(f"| `{left}` | `{right}` |")
    executed = [M(i) for i in (x for x in map(player_index, v.get("executed") or []) if x is not None)]
    out.append("")
    out.append("**処刑:** " + ("、".join(executed) if executed else "なし(投票が割れて決定者なし)"))
    return out


# === メイン描画 ===


def render_markdown(agg: dict, mapping: dict[int, str], leftover: set[str], game_id: str) -> str:
    def M(idx: int) -> str:
        return convert_names(f"Player {idx}", mapping, leftover)

    days, meta, res = agg["days"], agg["meta"], agg["result"]
    final_alive, causes = agg["final_alive"], agg["causes"]
    L: list[str] = []

    date = (meta.get("timestamp") or "")[:10] or "(不明)"
    mode = meta.get("mode") or "(不明)"
    winner_txt = ""
    if res.get("winner"):
        camp = CAMP_LABELS.get(str(res["winner"]), str(res["winner"]))
        winner_txt = f"{camp}陣営の勝利"
        reason = WIN_REASON_LABELS.get(str(res.get("reason") or ""), res.get("reason"))
        if reason:
            winner_txt += f"({reason})"

    L.append(f"# 人狼ゲーム公開記録: {game_id}")
    L.append("")
    L.append(f"- 対局日: {date} (UTC)")
    L.append(f"- 実行モード: {mode}")
    L.append(f"- 参加モデル: {len(mapping)}体")
    if winner_txt:
        L.append(f"- 結果: **{winner_txt}**")
    L.append("")
    L.append("## 参加モデル")
    L.append("")
    L.append("| モデル名 | 人物設定 | 最期 |")
    L.append("| --- | --- | --- |")
    for idx in sorted(mapping):
        pers = meta["personalities"].get(idx, "-")
        if final_alive is not None and idx in final_alive:
            fate = "生存で最終ターン終了"
        else:
            fate = causes.get(idx, "(死亡時期不明)")
        L.append(f"| `{mapping[idx]}` | {pers} | {fate} |")

    for d in sorted(days):
        b = days[d]
        L.append("")
        L.append(f"## Day {d}")
        L.append("")
        if b["note"]:
            L.append(f"- {b['note']}")
        if b["casualty_seen"]:
            dead = sorted(b["casualty_dead"])
            txt = "、".join(f"`{M(i)}`" for i in dead) if dead else "なし"
            L.append(f"- 朝の死亡者: {txt}")
        for turn in sorted(b["turns"]):
            L.append("")
            L.append(f"### 議論ターン {turn}")
            for idx, content in b["turns"][turn]:
                conv = convert_names(content, mapping, leftover)
                L.append("")
                L.append(f"**{M(idx)}**:")
                L.append("")
                L.append(quote_body(conv))
        claim_lines = render_claims(b["claims"], M)
        if claim_lines:
            L.append("")
            L.append("#### 公開CO記録")
            L.append("")
            L.extend(claim_lines)
        if b["voting"] is not None:
            L.append("")
            L.extend(render_voting(b["voting"], M))

    L.append("")
    L.append("---")
    L.append("")
    L.append("## 最終結果")
    L.append("")
    if winner_txt:
        L.append(f"- **{winner_txt}**")
    surv = sorted(final_alive or [])
    L.append("- 最終生存者: " + ("、".join(f"`{mapping.get(i, f'Player {i}')}`" for i in surv) or "(不明)"))
    L.append("")
    L.append("> ※ この記録はゲームログから `log_to_public_md.py` で自動生成したもの。")
    L.append("> ※ プレイヤー番号は発言文中の参照を含めモデル名に置換済み。")
    L.append("> ※ 真の役職・未公開の夜行動の詳細は記載していない(公開されたCO・占い申告のみ)。")
    return "\n".join(L) + "\n"


def find_game_id(events: list[dict], path: Path) -> str:
    for ev in events:
        gid = ev.get("game_id")
        if gid:
            return str(gid)
    stem = path.stem
    return stem[len("game_"):] if stem.startswith("game_") else stem


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="プレイヤー名をモデル名に置き換えた公開用Markdownを生成する")
    ap.add_argument("target", help="ゲームID(例: 425755ce9a83) または game_<ID>.jsonl へのパス")
    ap.add_argument("-o", "--output", default=None,
                    help="出力先MD(省略時: このスクリプトと同じフォルダの public_<ゲームID>.md)")
    ap.add_argument("--logs-dir", default=None, help="ログ格納ディレクトリ(省略時: このスクリプトの ../logs)")
    args = ap.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    logs_dir = Path(args.logs_dir) if args.logs_dir else script_dir.parent / "logs"
    log_path = resolve_log_path(args.target, logs_dir)
    events = load_events(log_path)
    mapping = build_player_model_map(events)
    if not mapping:
        raise SystemExit("[ERROR] プレイヤーとモデルの対応が取れませんでした(game_result/アクションイベントなし?)")

    agg = aggregate(events)
    leftover: set[str] = set()
    game_id = find_game_id(events, log_path)
    md = render_markdown(agg, mapping, leftover, game_id)

    residual = sorted({m.group(0) for m in NAME_RE.finditer(md)})
    out_path = Path(args.output) if args.output else script_dir / f"public_{game_id}.md"
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(md)

    n_msgs = sum(len(v) for b in agg["days"].values() for v in b["turns"].values())
    n_votes = sum(1 for b in agg["days"].values() if b["voting"] is not None)
    print(f"[OK] 生成完了: {out_path}")
    print(f"     Days={len(agg['days'])} / 発言={n_msgs}件 / 投票フェーズ={n_votes}回 / モデル数={len(mapping)}")
    bad = sorted(set(leftover) | {m if isinstance(m, str) else str(m) for m in residual})
    if bad:
        print(f"[WARN] 未置換のプレイヤー表記が残存(対応モデル不明): {bad}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
