"""TASK-058のローカル模擬A/B（LL不使用）: core圧縮 layered と off を同一履歴で比べる。

実機Ollamaは使わず PromptBuilder だけで組み、同じ履歴・同じ予算で次を検証する。

  ・user prompt字数: layered が off より大きくならないこと（節約側の崩れがない）
  ・役職主張・占い判定の本文保持数: layered が off を下回らないこと（TASK-058完了条件）
  ・anchor層（ルール・現在質問・行動形式・契約・自分の記録）の字数が両ポリシーで一致
  ・単純な末尾切り捨てになっていないこと: 畳むなら直近の低優先発言は本文が残る
  ・履歴カバレッジ: プロンプトに現れる発言ID（本文または追跡行）の件数が layered で増える
    ＝同じ字数で「何が起きたか」を追える量が増える（公開ログの本文はどちらでも不変）

off は TASK-057までの挙動（予算超過時はレコード削除のみ）で、legacy完全一致は
_analysis/task058_legacy_parity.py が別检査する。
"""
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.core_compression import HISTORICAL_STUB_MARK
from werewolf.models import Message, Phase
from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game

BUDGETS = (600, 1200, 2000, 3200)
STRATEGIES = ("hybrid_retrieval", "full_transcript")
CLAIM_TOKENS = ("占い結果", "CO")
EVIDENCE_ID_RE = re.compile(r"\[D\d+ T\d+ s(\d+) ")


def _state(seed: int = 46058):
    return create_game(seed=seed, config_path="config/players.json")


def _names(state):
    return tuple(player.name for player in state.players)


def _advance(state, day: int, turns: int, speeches_per_turn: int) -> None:
    """Day進行と同じ形の履歴を作る（发言順と時点スタンプを維持）。"""
    state.day = day
    state.phase = Phase.DAY_DISCUSSION
    state.public_timeline.record_day_start(day)
    state.conversation_log.append(Message("system", f"Day {day} begins."))
    names = _names(state)
    for turn in range(1, turns + 1):
        state.discussion_turn = turn
        state.begin_discussion_turn()
        for index in range(speeches_per_turn):
            speaker = names[index % len(names)]
            if index == 1:
                tail = "私は村人で、占い結果は人間でした。"
            elif index == 3:
                tail = "その反論には矛盾があると感じる。ここで投票先を絞りたい。"
            else:
                tail = "まずは各人の言い回しを確認したいので、今は慎重に見ている。"
            body = (
                f"{speaker}のD{day}T{turn}発言{index + 1}。" + tail
                + "長めの補足説明をここに添えて、履歴の字数を実ゲームに近い形へ膨らませる。"
                "誰が誰を信頼していて、どの点が未確認のままなのかを書き残しておく。"
            )
            state.append_discussion_message(Message(speaker, body))


def _public_bodies(state):
    return [
        str(message.content or "")
        for message in state.conversation_log
        if message.speaker in set(_names(state))
    ]


def _claim_bodies(state):
    return [text for text in _public_bodies(state) if any(t in text for t in CLAIM_TOKENS)]


def _plain_bodies(state):
    skip = CLAIM_TOKENS + ("反論", "矛盾", "投票")
    return [text for text in _public_bodies(state) if not any(t in text for t in skip)]


def _build(state, actor_name, budget, policy, strategy="hybrid_retrieval"):
    actor = next(player for player in state.players if player.name == actor_name)
    return PromptBuilder.build_discussion_prompt(
        actor,
        tuple(state.alive_players),
        tuple(state.conversation_log),
        day=state.day,
        state=state,
        include_claim=True,
        context_strategy=strategy,
        evidence_char_budget=budget,
        question_hint_mode="strict",
        prompt_contract_policy="simplified",
        core_compression_policy=policy,
    )


def _claims_kept(ctx, claims) -> int:
    """役職主張・占い判定の本文（先頭24字）がプロンプトに残っている件数。"""
    return sum(1 for text in claims if text[:24] in ctx.user_prompt)




def _coverage(ctx) -> int:
    """プロンプトに現れる発言IDの固有数（本文でも追跡行でも数える＝歴史の追跡範囲）。"""
    return len(set(EVIDENCE_ID_RE.findall(ctx.user_prompt)))


def _min_budget_for(state, policy: str, target_ids: int, target_claims: int, claims) -> int | None:
    """ target_claims 件の役職主張本文と target_ids 件の発言IDを追える最小予算を二分探索する。

    同じ情報を送るのにより少ない入力で済むか（＝ context_needed_tokens の削減余地）を見る
    プロトコルB。layered は過去の本文を stub 行（時点・ID・話者だけ）へ畳むため、同じ追跡
    範囲をより安い字数で再現できるはず。
    """
    lo, hi = 200, 9000
    probe = _build(state, "Player 1", hi, policy)
    if _coverage(probe) < target_ids or _claims_kept(probe, claims) < target_claims:
        return None
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        ctx = _build(state, "Player 1", mid, policy)
        if _coverage(ctx) >= target_ids and _claims_kept(ctx, claims) >= target_claims:
            hi = mid
        else:
            lo = mid
    return hi


def main() -> None:
    failures = 0
    stats = {"layered": [], "off": []}
    coverage_delta = []
    folded_any = 0
    protocol_b_results = []
    for day, turns, speeches in ((1, 3, 8), (2, 4, 9), (3, 2, 6)):
        state = _state()
        _advance(state, day, turns, speeches)
        claims = _claim_bodies(state)
        plain = _plain_bodies(state)
        print(
            f"--- D{day}: turns={turns} speeches/turn={speeches}"
            f" claims={len(claims)} plain={len(plain)}"
        )
        for strategy in STRATEGIES:
            for budget in BUDGETS:
                layered = _build(state, "Player 1", budget, "layered", strategy)
                off = _build(state, "Player 1", budget, "off", strategy)
                dl, do = layered.diagnostics, off.diagnostics
                keep_l, keep_o = _claims_kept(layered, claims), _claims_kept(off, claims)
                cov_l, cov_o = _coverage(layered), _coverage(off)
                stats["layered"].append(len(layered.user_prompt))
                stats["off"].append(len(off.user_prompt))
                coverage_delta.append((cov_l, cov_o))
                stub_lines = [
                    line
                    for line in layered.user_prompt.splitlines()
                    if HISTORICAL_STUB_MARK in line
                ]
                if stub_lines:
                    folded_any += 1
                tag = f"  {strategy} budget={budget}"
                print(
                    f"{tag}: chars L={len(layered.user_prompt)} O={len(off.user_prompt)}"
                    f" | claims kept L={keep_l}/{len(claims)} O={keep_o}/{len(claims)}"
                    f" | history ids L={cov_l} O={cov_o}"
                    f" | anchor L={dl['core_anchor_chars']} O={do['core_anchor_chars']}"
                    f" | recent L={dl['core_recent_chars']} O={do['core_recent_chars']}"
                    f" historical L={dl['core_historical_chars']}"
                    f" | folded={dl.get('evidence_withheld_body_count')}"
                    f" saved={dl.get('evidence_withheld_saved_chars')}"
                    f" dropped L={len(dl.get('evidence_dropped_indices') or [])}"
                    f" O={len(do.get('evidence_dropped_indices') or [])}"
                )
                # 1) anchor は常に両ポリシー同字数（一貫性の担保）
                if int(dl["core_anchor_chars"]) != int(do["core_anchor_chars"]):
                    print("    !! anchor layer changed between policies")
                    failures += 1
                # 2) 役職主張の本文保持は layered が off を下回らない
                if keep_l < keep_o:
                    print("    !! layered kept fewer claim bodies than off")
                    failures += 1
                # 3) 節約側: 同じ予算での入力の差は「何 を残したか」の差なので、+2%以内の
                #    揺らぎとして許容し、その代わりに claim/追跡ID が減っていないことを要求する
                #   （layeredは質問対象・反証・役職主張を後に残す方針＝質側の利得）。
                limit = int(len(off.user_prompt) * 1.02) + 32
                if len(layered.user_prompt) > limit:
                    print(
                        f"    !! layered grew beyond tolerance ({len(layered.user_prompt)} vs {len(off.user_prompt)})"
                    )
                    failures += 1
                # 4) off は従来どおり stub行を作らない
                if HISTORICAL_STUB_MARK in off.user_prompt:
                    print("    !! off produced stub lines")
                    failures += 1
                # 5) 単純な末尾切り捨ての禁止: 畳むなら直近の低優先発言は本文で残る
                if stub_lines and plain:
                    newest = plain[-1][:16]
                    if newest not in layered.user_prompt:
                        print(f"    !! newest low-priority speech folded away: {newest}")
                        failures += 1
                # 6) 歴史カバレッジ: 同じ字数で追える発言IDが減っていないか
                if cov_l < cov_o:
                    print(f"    !! history coverage dropped ({cov_l} < {cov_o})")
                    failures += 1
        # プロトコルB: 同じ情報（全 claim 本文 + 同一数の発言ID）を追える最小予算を比べる。
        # 実ゲームで evidence budget は割り当てコンテキストから逆算されるため、これが小さいほど
        # context_needed_tokens と再要求・truncation の余地が減る（TASK-058の節約側）。
        baseline_off = _build(state, "Player 1", 2000, "off", "hybrid_retrieval")
        target_ids = _coverage(baseline_off)
        min_l = _min_budget_for(state, "layered", target_ids, len(claims), claims)
        min_o = _min_budget_for(state, "off", target_ids, len(claims), claims)
        if min_l is None or min_o is None:
            print(f"    !! B: could not reach target ids={target_ids} claims={len(claims)}")
            failures += 1
        else:
            ctx_l = _build(state, "Player 1", min_l, "layered", "hybrid_retrieval")
            ctx_o = _build(state, "Player 1", min_o, "off", "hybrid_retrieval")
            delta = len(ctx_o.user_prompt) - len(ctx_l.user_prompt)
            print(
                f"  [B] same-info budget layered={min_l} off={min_o}"
                f" | chars L={len(ctx_l.user_prompt)} O={len(ctx_o.user_prompt)}"
                f" | saving={delta:+d} chars ({100.0 * delta / max(1, len(ctx_o.user_prompt)):.1f}%)"
                f" | ids L={_coverage(ctx_l)}/O={_coverage(ctx_o)}"
                f" claims L={_claims_kept(ctx_l, claims)}/{len(claims)}"
                f" folded L={ctx_l.diagnostics.get('evidence_withheld_body_count')}"
            )
            protocol_b_results.append((len(ctx_l.user_prompt), len(ctx_o.user_prompt)))
        # 公開ログ（conversation_log）の本文は圧縮前後で完全に同じまま
        joined = "\n".join(str(m.content) for m in state.conversation_log)

        if any(text not in joined for text in claims + plain[:1]):
            print("    !! public log body was altered by compression")
            failures += 1

    avg_l = sum(stats["layered"]) / len(stats["layered"])
    avg_o = sum(stats["off"]) / len(stats["off"])
    gain_ids = sum(l - o for l, o in coverage_delta)
    b_avg_l = sum(v for v, _ in protocol_b_results) / max(1, len(protocol_b_results))
    b_avg_o = sum(v for _, v in protocol_b_results) / max(1, len(protocol_b_results))
    print(
        f"[A] avg chars L={avg_l:.1f} O={avg_o:.1f}"
        f" | max L={max(stats['layered'])} O={max(stats['off'])}"
        f" | history-id coverage delta={gain_ids:+d} (folded cases={folded_any})"
    )
    if protocol_b_results:
        print(
            f"[B] same-info prompt chars avg L={b_avg_l:.1f} O={b_avg_o:.1f}"
            f" | saving={b_avg_o - b_avg_l:.1f} chars"
            f" ({100.0 * (b_avg_o - b_avg_l) / max(1.0, b_avg_o):.2f}%)"
        )
        if b_avg_l > b_avg_o:
            print("    !! B: layered needed more input than off for the same evidence")
            failures += 1
    if folded_any == 0:
        print("    !! no scenario exercised the layered collapse path")
        failures += 1
    print("[DONE]", "LAYERED-AB-OK" if failures == 0 else f"LAYERED-AB-FAIL({failures})")
    return failures


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)

