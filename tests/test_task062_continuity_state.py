"""TASK-062の回帰テスト（実機LL不使用）: 日をまたぐ論理状態の正規化と記憶減衰対策。

実測観測（game_e6b6c7cc3481）では、Day 3 で core が 11,613字→21,501字へ膨らむ一方
`core_historical_chars` は各日0（＝歴史欄が畳まれず、発言本文が積み上がる）で、過去の
仮説が現在の事実として読まれたり、未回答の問いが日を超えて垂れ流れたりしていた。
本テストは `continuity_state.ContinuityState` が「エンジン発行の短い状態差分」だけで
論理を継続し、次のことを固定する（すべて決定的・LL不使用）:

1. 権威レベルの分離（エンジン確定／公開申告・未検証／自分の未検証仮説／反証済み）。
   古い仮説が新しい事実として読まれないこと、反証は履歴として出ること。
2. 差分提示: 初回は持ち越し状態を1度だけ、以降は変わった項目だけ。変わっていない項目は
   個数とIDだけ示して本文を再掲しない（プロンプト量を日に比例して増やさない）。
3. 同じ時点での組み直し（予算段階・再要求）では同じ文言を返し、差分基準が進まないこと。
4. 行数上限。出し切れなかった項目はチェックポイント側へ記録されたままになる。
5. プロンプト統合（layout v4）: 状態節は発言証拠の直後・現在質問の直前に置かれ、
   v3 / off では出ない。口調・論点・結論を決める新しい命令文は増えない。
6. チェックポイントが正規化した日の平常発言本文はID行へ畳まれ、公開ログの本文は不変。
"""
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.continuity_state import (
    AUTHORITY_DERIVED,
    AUTHORITY_ENGINE,
    AUTHORITY_PUBLIC_STATEMENT,
    AUTHORITY_REFUTED,
    CONTINUITY_POLICY_LEGACY,
    CONTINUITY_POLICY_NORMALIZED,
    RENDER_CHANGED_MAX,
    RENDER_CURRENT_MAX,
    RENDER_OVERTURNED_MAX,
    RENDER_QUESTIONS_MAX,
    ContinuityState,
    continuity_policy_label,
    normalize_continuity_policy,
    point_label,
)
from werewolf.engine import GameEngine
from werewolf.hypothesis_ledger import (
    REFUTATION_CONTESTED,
    REFUTATION_HISTORICAL,
)
from werewolf.models import GameState, Message, Phase
from werewolf.prompt_contract import find_sections
from werewolf.prompt_layout import (
    CONTINUITY_STATE_SECTION,
    PROMPT_LAYOUT_V3,
    PROMPT_LAYOUT_V4,
)
from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game

LONG_SPEECH = (
    "{name}の{index}番目の発言です。{rival}の占い師COと投票の動きを比べると、昨夜の襲撃先と"
    "の整合が怪しく見えるので、今日の投票ではその点を確認したいと考えています。"
    "加えて、{rival}が先に述べた候補については根拠が弱いと感じています。"
)

SEED = 2062061


def _state(seed: int = SEED) -> GameState:
    return create_game(
        seed=seed, config_path="config/players.json", response_language="日本語"
    )


def _names(state: GameState) -> tuple[str, ...]:
    return tuple(player.name for player in state.players)


def _player(state: GameState, name: str):
    return next(player for player in state.players if player.name == name)


def _start_day(state: GameState, day: int = 1, turn: int = 1) -> None:
    """日をまたいだ議論開始（日の開始メッセージ＋時系列台帳＋ターン初期化）。"""
    state.day = day
    state.conversation_log.append(Message("system", f"Day {day} begins."))
    state.public_timeline.record_day_start(day)
    state.phase = Phase.DAY_DISCUSSION
    state.discussion_turn = turn
    state.begin_discussion_turn()


def _say(state: GameState, name: str, content: str):
    return state.append_discussion_message(Message(name, content))


def _speeches(state: GameState, count: int, start_index: int = 1) -> None:
    """平文の通常発言（低優先）を `count` 本積む（recent層が日に比例して増える状況を作る）。"""
    names = _names(state)
    for offset in range(count):
        index = start_index + offset
        speaker = names[offset % len(names)]
        rival = names[(offset + 3) % len(names)]
        _say(state, speaker, LONG_SPEECH.format(name=speaker, index=index, rival=rival))


def _ctx(
    state: GameState,
    actor_name: str,
    layout: str = "v4",
    contract: str = "simplified",
    compression: str = "layered",
):
    actor = _player(state, actor_name)
    return PromptBuilder.build_discussion_prompt(
        actor,
        tuple(state.alive_players),
        tuple(state.conversation_log),
        day=state.day,
        state=state,
        include_claim=True,
        context_strategy="hybrid_retrieval",
        question_hint_mode="strict",
        prompt_contract_policy=contract,
        core_compression_policy=compression,
        prompt_layout_policy=layout,
    )


def _section(prompt: str, name: str) -> str:
    spans = find_sections(prompt)
    start, end = spans[name]
    return prompt[start:end]


# ---------------------------------------------------------------------------
# 1. ポリシーとstate連携
# ---------------------------------------------------------------------------
def test_normalizes_aliases_and_labels() -> None:
    assert normalize_continuity_policy(None) == CONTINUITY_POLICY_NORMALIZED
    assert normalize_continuity_policy("normalized") == CONTINUITY_POLICY_NORMALIZED
    assert normalize_continuity_policy("OFF ") == CONTINUITY_POLICY_LEGACY
    assert normalize_continuity_policy(True) == CONTINUITY_POLICY_NORMALIZED
    assert normalize_continuity_policy(False) == CONTINUITY_POLICY_LEGACY
    assert normalize_continuity_policy("legacy") == CONTINUITY_POLICY_LEGACY
    assert normalize_continuity_policy("未知値") == CONTINUITY_POLICY_NORMALIZED
    assert continuity_policy_label(CONTINUITY_POLICY_LEGACY) == CONTINUITY_POLICY_LEGACY
    assert point_label(2, 3) == "D2T3"


def test_game_state_default_syncs_ledger_policy() -> None:
    state = _state()
    assert state.continuity_policy == CONTINUITY_POLICY_NORMALIZED
    assert state.continuity_state.policy == CONTINUITY_POLICY_NORMALIZED
    assert state.continuity_state.enabled is True
    # off へ変えて __post_init__ を回すと台帳側のポリシーも揃う（A/B比較の基準側）
    state.continuity_policy = "off"
    GameState.__post_init__(state)
    assert state.continuity_policy == CONTINUITY_POLICY_LEGACY
    assert state.continuity_state.policy == CONTINUITY_POLICY_LEGACY
    assert state.continuity_state.enabled is False


def test_off_policy_emits_no_section_and_keeps_old_behaviour() -> None:
    """off はチェックポイントを作らない（旧挙動＝本文積み増しのまま、比較基準側）。"""
    state = _state()
    state.continuity_policy = CONTINUITY_POLICY_LEGACY
    GameState.__post_init__(state)
    _start_day(state)
    _speeches(state, 4)
    actor_name = _names(state)[5]
    block, diag = state.render_continuity_for(actor_name)
    assert block == ""
    assert diag.get("continuity_policy") == CONTINUITY_POLICY_LEGACY
    assert state.issue_continuity_checkpoint("day-1-pre_vote") is None
    assert state.continuity_state.checkpoints == []
    assert state.continuity_fold_days() == ()


# ---------------------------------------------------------------------------
# 2. エンジン発行のチェックポイント（公開情報のID付き短行だけ）
# ---------------------------------------------------------------------------
def _claim(claimant: str, role: str, day: int):
    from werewolf.models import PublicRoleClaim

    return PublicRoleClaim(day=day, claimant=claimant, claimed_role=role)


def _seer_claim(claimant: str, target: str, result: str, day: int):
    from werewolf.models import PublicSeerClaim

    return PublicSeerClaim(day=day, claimant=claimant, target=target, result=result)


def _day_with_events(state: GameState) -> GameState:
    """CO・占い申告・投票・処刑・襲撃死が揃った Day 2 の実形を作る。"""
    names = _names(state)
    _start_day(state, day=1, turn=1)
    _speeches(state, 3)
    seer = next(p for p in state.players if p.role.value == "seer")
    scan_target = next(
        n for n in names if n != seer.name and _player(state, n).role.value != "wolf"
    )
    state.public_role_claims.append(_claim(seer.name, "Seer", 1))
    state.conversation_log.append(
        Message(seer.name, f"[CO] {seer.name} claims the Seer role on Day 1.")
    )
    state.seer_claims.append(_seer_claim(seer.name, scan_target, "human", 1))
    # 8人構成（names[0..7]）。投票先は Player 7、襲撃死は Player 6 にする（死亡は
    # mark_dead を呼ばないので生存扱いのまま、状態としての記録だけを作る）
    state.vote_history.append((1, False, dict(zip(names[:6], [names[6]] * 6))))
    state.executions.append((1, names[6]))
    _start_day(state, day=2, turn=1)
    state.attack_deaths.append((2, names[5]))
    return state


# ---------------------------------------------------------------------------
# 2. エンジン発行のチェックポイント（公開情報のID付き短行だけ）
# ---------------------------------------------------------------------------
def test_checkpoint_holds_public_state_in_short_id_lines() -> None:
    state = _day_with_events(_state())
    names = _names(state)
    record = state.issue_continuity_checkpoint("day-2-morning")
    assert record is not None
    text = "\n".join(record["lines"])
    # 確定事実（エンジン記録）と、真偽未検証の公開申告を権威レベルで分ける
    assert "CONFIRMED" in text
    assert f"[death:execution:1:{names[6]}]" in text
    assert "[vote:1:" in text
    assert "truth unverified" in text
    assert "[claim:1]" in text or "[seer_claim:1]" in text
    # 発言全文の写しを載せない（行は短い状態行のみ）
    assert max(len(line) for line in record["lines"]) <= 300
    assert record["char_count"] == len(text)
    # 正規化が効くのは checkpoint より前の日だけ（当日の本文はまだ生かす）
    assert state.continuity_fold_days() == ()
    state.day = 3
    assert state.continuity_fold_days() == (1, 2)


def test_engine_issues_checkpoints_at_boundaries_without_secret_leak() -> None:
    state = _state()
    events: list[dict[str, object]] = []
    engine = GameEngine(state, None, seed=state.seed, event_callback=events.append)
    _start_day(state, day=1, turn=1)
    _speeches(state, 2)
    wolves = [player.name for player in state.players if player.role.value == "wolf"]
    secret_line = f"{wolves[0]}と相談して今夜狙う先を決めた（秘匿）"
    state.wolf_chat_log.append(Message(wolves[0], secret_line))
    record = engine._issue_continuity_checkpoint("day-1-pre_vote")
    assert record is not None and record["checkpoint_id"] == "C1"
    emitted = [event for event in events if event.get("type") == "continuity_checkpoint"]
    assert len(emitted) == 1
    text = "\n".join(str(line) for line in emitted[0]["lines"])
    assert secret_line not in text
    assert "night consult" not in text
    public_blob = str(state.continuity_state.serialize_public())
    assert secret_line not in public_blob
    assert "seer_result" not in public_blob


def test_metrics_and_public_serialization_shape() -> None:
    state = _day_with_events(_state())
    state.issue_continuity_checkpoint("day-2-morning")
    metrics = state.continuity_metrics()
    assert metrics["continuity_policy"] == CONTINUITY_POLICY_NORMALIZED
    assert metrics["continuity_checkpoints"] == 1
    assert int(metrics["continuity_checkpoint_char_total"]) > 0
    assert set(metrics["continuity_chars_by_day"]) <= {"D2"}
    public = state.continuity_state.serialize_public()
    assert set(public) == {"metrics", "checkpoints"}
    entry = public["checkpoints"][0]
    assert {
        "checkpoint_id",
        "boundary",
        "day",
        "turn",
        "issued_at",
        "char_count",
        "evidence_ids",
    } <= set(entry)
    # 本人限定の未検証仮説（推理メモ由来）は公開ログへ出さない
    private_memo = "私が仮に置いた容疑者の私的な読み"
    state.store_model_reasoning_memo(_names(state)[0], private_memo)
    assert private_memo not in str(state.continuity_state.serialize_public())


# ---------------------------------------------------------------------------
# 3. 差分提示（初回は持ち越しを1度だけ、以降は変わった項目だけ・不変は個数だけ）
# ---------------------------------------------------------------------------
def test_first_presentation_carries_state_once_with_authority_levels() -> None:
    state = _day_with_events(_state())
    names = _names(state)
    actor_name = names[1]
    block, diag = state.render_continuity_for(actor_name)
    assert block.startswith("[CONTINUITY_CHECKPOINT")
    assert "Authority levels differ per line" in block
    assert "[CONFIRMED]" in block
    assert "executed by vote on Day 1" in block
    assert f"{len(state.alive_players)} alive: " in block
    assert diag["continuity_first_presentation"] is True
    levels = list(diag["continuity_authority_levels"])
    assert AUTHORITY_ENGINE in levels
    # 公の申告（CO・占い結果）は台帳側が本文を持つので、状態節ではID参照だけを持つ。
    # 権威レベルは分かれて記録される（同じ本文を2度送らない）
    assert AUTHORITY_PUBLIC_STATEMENT in levels
    assert "Confirmed state carried over" in block
    assert "Unchanged since your previous turn" not in block


def test_repeat_build_at_the_same_point_returns_identical_text() -> None:
    """予算段階や再要求で組み直しても、モデルへ見える状態文言はぶれない。"""
    state = _day_with_events(_state())
    actor_name = _names(state)[1]
    first, first_diag = state.render_continuity_for(actor_name)
    second, second_diag = state.render_continuity_for(actor_name)
    assert first == second
    assert dict(first_diag) == dict(second_diag)
    unchanged_before = int(state.continuity_state.unchanged_presented)
    updated_before = int(state.continuity_state.items_updated)
    state.render_continuity_for(actor_name)
    assert int(state.continuity_state.unchanged_presented) == unchanged_before
    assert int(state.continuity_state.items_updated) == updated_before


def test_next_turn_shows_only_changes_and_does_not_restate_unchanged() -> None:
    state = _day_with_events(_state())
    names = _names(state)
    actor_name = names[1]
    first, _ = state.render_continuity_for(actor_name)
    alive_line = f"{len(state.alive_players)} alive: " + ", ".join(state.alive_players)
    assert alive_line in first
    # ターンを進めて新しい処刑と投票だけを「変わった項目」として確認する
    state.discussion_turn = 2
    state.executions.append((2, names[6]))
    state.vote_history.append((2, False, dict(zip(names[:5], [names[6]] * 5))))
    second, diag = state.render_continuity_for(actor_name)
    assert "- Changed since you last saw this state:" in second
    assert "executed by vote on Day 2" in second
    assert f"[vote:2:initial:{names[0]}]" in second
    # 変わっていない生存者一覧は再掲せず、個数とIDだけの行へ畳む
    assert alive_line not in second
    assert "Unchanged since your previous turn" in second
    assert int(diag["continuity_items_unchanged"]) >= 1
    assert int(diag["continuity_state_chars"]) < len(first) + 900


def test_changed_lines_are_capped_with_an_explicit_note() -> None:
    """行数上限（RENDER_CHANGED_MAX）を超えても、出し切れなかった数を注記で示す。"""
    state = _day_with_events(_state())
    names = _names(state)
    actor_name = names[1]
    state.render_continuity_for(actor_name)
    state.discussion_turn = 3
    # 8人構成なので、処刑5件＋投票先5件（合計10項目）を変化させて上限超過を作る
    for executed in (names[6], names[7], names[0], names[3], names[5]):
        if executed == actor_name:
            continue
        state.executions.append((3, executed))
        state.vote_history.append((3, False, {names[1]: executed}))
    assert len(state.executions) >= 6  # Day1処刑 + Day3で複数（上限超過の確認）
    block, diag = state.render_continuity_for(actor_name)
    shown_rows = [line for line in block.splitlines() if "executed by vote on Day 3" in line]
    assert len(shown_rows) <= RENDER_CHANGED_MAX
    assert int(diag["continuity_items_changed"]) > len(shown_rows)
    assert "more changes are recorded in checkpoint" in block


# ---------------------------------------------------------------------------
# 4. 自分の読み（未検証）・反証の扱い
# ---------------------------------------------------------------------------
def _own_reading(state: GameState, actor_name: str, subject: str) -> str:
    """本人限定の私的仮説を1件登録して hypothesis_id を返す（他プレイヤーには渡らない）。"""
    memo = f"{subject}を暫定的に疑っている根拠を確認したい。"
    state.store_model_reasoning_memo(actor_name, memo)
    state.record_hypotheses_from_reasoning_memo(
        actor_name, memo, state.day, state.discussion_turn
    )
    record = next(
        row
        for row in state.hypothesis_ledger.records.values()
        if row.generated_by == actor_name
    )
    return record.hypothesis_id


def test_own_readings_are_marked_unverified_and_free_to_drop() -> None:
    state = _day_with_events(_state())
    names = _names(state)
    actor_name = names[2]
    _own_reading(state, actor_name, names[4])
    block, diag = state.render_continuity_for(actor_name)
    assert "- Your standing unverified readings" in block
    assert "[YOUR-UNVERIFIED]" in block
    assert names[4] in block
    assert "update or drop them freely" in block  # 維持義務ではなく自由として出す
    assert AUTHORITY_DERIVED in list(diag["continuity_authority_levels"])
    assert diag["continuity_first_presentation"] is True
    # 自分の仮説は他人のプロンプトへ出ない（可視規則は HypothesisLedger のまま）
    other_block, _ = state.render_continuity_for(names[5])
    assert "[YOUR-UNVERIFIED]" not in other_block


def test_refuted_reading_shows_as_history_and_its_old_line_disappears() -> None:
    state = _day_with_events(_state())
    names = _names(state)
    actor_name = names[2]
    hypothesis_id = _own_reading(state, actor_name, names[4])
    first, _ = state.render_continuity_for(actor_name)
    assert "[YOUR-UNVERIFIED]" in first
    # エンジン照合で反証された状態へ変える（古い行が新事実として残らないようにする）
    record = state.hypothesis_ledger.records[hypothesis_id]
    record.refutation_status = REFUTATION_CONTESTED
    record.counter_evidence_ids = ["[seer_claim:2]"]
    state.discussion_turn = 2
    block, diag = state.render_continuity_for(actor_name)
    assert "- Overturned or refuted (history, not current facts):" in block
    assert "[NO-LONGER-STANDS]" in block
    assert "[seer_claim:2]" in block
    assert AUTHORITY_REFUTED in list(diag["continuity_authority_levels"])
    assert int(state.continuity_state.items_overturned) >= 1
    assert int(state.continuity_state.hypothesis_refutations) >= 1
    # 自分の未検証だった行は差し替えられ、二重に「いま有効」としては出ない
    assert [line for line in block.splitlines() if line.startswith("- [YOUR-UNVERIFIED]")] == []
    refuted_row = next(line for line in block.splitlines() if "[NO-LONGER-STANDS]" in line)
    assert names[4] in refuted_row and "contested" in refuted_row


def test_historical_refutation_is_also_history_not_a_current_fact() -> None:
    state = _day_with_events(_state())
    names = _names(state)
    actor_name = names[2]
    hypothesis_id = _own_reading(state, actor_name, names[3])
    state.render_continuity_for(actor_name)
    record = state.hypothesis_ledger.records[hypothesis_id]
    record.refutation_status = REFUTATION_HISTORICAL
    record.actionable = False
    state.discussion_turn = 2
    block, diag = state.render_continuity_for(actor_name)
    assert "[NO-LONGER-STANDS]" in block
    assert "historical" in block
    assert int(diag["continuity_items_overturned"]) >= 1


def test_open_questions_are_capped_and_dropped_once_resolved() -> None:
    state = _day_with_events(_state())
    names = _names(state)
    actor_name = names[2]
    state.refresh_reasoning_memos()
    memo = state.reasoning_memos[actor_name]
    total = RENDER_QUESTIONS_MAX + 3
    memo.unresolved_questions = [
        f"{actor_name}さんへ。この投票案はどう考えますか？ [speech:{index}]"
        for index in range(1, total + 1)
    ]
    block, diag = state.render_continuity_for(actor_name)
    question_rows = [line for line in block.splitlines() if "open question:" in line]
    assert len(question_rows) == RENDER_QUESTIONS_MAX
    assert "older open points stay unresolved too" in block
    assert int(diag["continuity_open_questions"]) == total
    # 問いが解消されたら状態から落ちる（日をまたいで未解決を垂れ流さない）
    state.discussion_turn = 2
    memo.unresolved_questions = []
    second, second_diag = state.render_continuity_for(actor_name)
    assert "Still unanswered toward you" not in second
    assert int(second_diag["continuity_open_questions"]) == 0
    assert int(state.continuity_state.questions_resolved) >= total


def test_state_section_never_orders_reasoning_or_presses_a_conclusion() -> None:
    """状態節は口調・論点・結論・推理順を決めない（自由行を縛る命令文を増やさない）。"""
    state = _day_with_events(_state())
    names = _names(state)
    actor_name = names[2]
    _own_reading(state, actor_name, names[4])
    state.refresh_reasoning_memos()
    state.reasoning_memos[actor_name].unresolved_questions = [
        f"{actor_name}への質問です。[speech:2] の根拠をどう見ますか？"
    ]
    first, _ = state.render_continuity_for(actor_name)
    state.discussion_turn = 2
    record = next(
        row
        for row in state.hypothesis_ledger.records.values()
        if row.generated_by == actor_name
    )
    record.refutation_status = REFUTATION_CONTESTED
    record.counter_evidence_ids = ["[speech:5]"]
    second, _ = state.render_continuity_for(actor_name)
    banned = re.compile(
        r"^\s*-?\s*(Must|Always|Never|Do not|Don't|You should|Avoid|Repeat|Conclude|Argue|"
        r"Start by|First |Prioritize|Defend|Agree|Disagree|禁止|必须)",
        re.IGNORECASE,
    )
    for text in (first, second):
        offenders = [line for line in text.splitlines() if banned.match(line)]
        assert offenders == [], offenders
    # 状態は情報であり、結論の固定・推理順の指定・口調の設定ではないと明記されている
    assert "you may change your mind freely" in second
    assert "does not order your reasoning" in second


# ---------------------------------------------------------------------------
# 5. プロンプト統合（layout v4）: 状態節は発言証拠の直後・現在質問の直前にある
# ---------------------------------------------------------------------------
# 台帳側で plain（低優先）と判定される文章にする。`投票` `反証` `占い` `判定` `結果` などの
# 語は priority が上がり、チェックポイント起因の stub 化対象外になるため使わない。
PLAIN_SPEECH = (
    "{name}の{index}番目の発言です。{rival}の話していた順番と昨夜の出来事を並べてみると、"
    "もう少し詳しく確認したくなってきました。私としては、{rival}が最初に触れた件と、"
    "その後に続けた説明の続きを、この場で改めて聞きたい気持ちです。"
    "{name}として別の角度から見ると、まだ判断しかねる部分が残っています。"
)


def _multi_day_state(days: int = 3, per_turn: int = 3) -> tuple[GameState, str]:
    """checkpoint を日ごとに発行しつつ、平常発言を積み上げた Day `days` の盤面を作る。"""
    state = _state()
    names = _names(state)
    engine = GameEngine(state, None, seed=state.seed)
    for day in range(1, days + 1):
        state.day = day
        state.conversation_log.append(Message("system", f"Day {day} begins."))
        state.public_timeline.record_day_start(day)
        state.phase = Phase.DAY_DISCUSSION
        for turn in range(1, 4):
            state.discussion_turn = turn
            state.begin_discussion_turn()
            for offset in range(per_turn):
                speaker = names[(day * 4 + turn + offset) % len(names)]
                rival = names[(day * 4 + turn + offset + 3) % len(names)]
                body = PLAIN_SPEECH.format(name=speaker, index=offset, rival=rival)
                _say(state, speaker, body)
                state.record_topics_from_speech(speaker, body, day, turn)
        engine._issue_continuity_checkpoint(f"day-{day}-post_execution")
    return state, names[2]


def test_v4_places_the_state_section_between_evidence_and_question() -> None:
    state, actor_name = _multi_day_state()
    ctx = _ctx(state, actor_name, "v4")
    diag = dict(ctx.diagnostics or {})
    assert diag["prompt_layout"] == PROMPT_LAYOUT_V4
    assert diag["prompt_layout_violations"] == []
    prompt = ctx.user_prompt
    assert "[CONTINUITY_CHECKPOINT" in prompt
    spans = find_sections(prompt)
    checkpoint_start = prompt.index("[CONTINUITY_CHECKPOINT")
    # 発言証拠の直後・現在質問の直前（状態→質問の順で読める配置）
    assert spans["conversation_evidence"][0] < checkpoint_start
    assert checkpoint_start < spans["question_block"][1]
    assert int(diag["continuity_section_chars"]) > 0
    # v3（同じ連鎖から状態節だけを抜いた配置）では状態節が出ない＝A/Bの基準側は無傷
    v3_ctx = _ctx(state, actor_name, "v3")
    assert "[CONTINUITY_CHECKPOINT" not in v3_ctx.user_prompt
    assert int(dict(v3_ctx.diagnostics or {}).get("continuity_state_chars") or 0) == 0
    assert dict(v3_ctx.diagnostics or {})["prompt_layout"] == PROMPT_LAYOUT_V3
    # off + v4 では節そのものを作らない（旧挙動との切り替え）
    state.continuity_policy = CONTINUITY_POLICY_LEGACY
    GameState.__post_init__(state)
    off_ctx = _ctx(state, actor_name, "v4")
    assert "[CONTINUITY_CHECKPOINT" not in off_ctx.user_prompt


def test_checkpoint_folds_earlier_plain_bodies_without_touching_the_public_log() -> None:
    """正規化した日の平常発言は ID 行へ畳み、公開ログの本文は一字一句そのまま。"""
    state, actor_name = _multi_day_state(days=4)
    bodies_before = [str(message.content) for message in state.conversation_log]
    normalized = _ctx(state, actor_name, "v4")
    norm_diag = dict(normalized.diagnostics or {})
    assert sorted(int(day) for day in norm_diag["continuity_fold_days"]) == [1, 2, 3]
    folded = int(norm_diag["evidence_continuity_folded_count"])
    saved = int(norm_diag["evidence_continuity_fold_saved_chars"])
    assert folded > 0 and saved > 0
    # 畳んだ行には時点・話者・[speech:N] と「本文は載せていない」注記が残る（空参照にしない）
    stub_rows = [line for line in normalized.user_prompt.splitlines() if "withheld" in line]
    assert stub_rows
    assert any("[speech:" in line for line in stub_rows)
    # off（旧挙動）では同じ発言が本文のまま積まれ、recent層が減らない
    state.continuity_policy = CONTINUITY_POLICY_LEGACY
    GameState.__post_init__(state)
    legacy = _ctx(state, actor_name, "v4")
    legacy_diag = dict(legacy.diagnostics or {})
    assert int(legacy_diag["evidence_continuity_folded_count"]) == 0
    assert int(legacy_diag.get("continuity_state_chars") or 0) == 0
    assert int(legacy_diag["core_recent_chars"]) > int(norm_diag["core_recent_chars"])
    # 状態節ぶんだけ増えても、本文の再掲ぶんの方が大きい（節約が正味で効く）
    assert int(norm_diag["core_recent_chars"]) + int(norm_diag["continuity_state_chars"]) < int(
        legacy_diag["core_recent_chars"]
    )
    # 公開ログ側の本文は無傷（プロンプトで省略しても履歴は消えない）
    assert [str(message.content) for message in state.conversation_log] == bodies_before


def test_state_metrics_show_no_day_over_day_growth_of_the_state_section() -> None:
    """状態節の字数は日をまたいで線形増加しない（本文積み増しの代替であることの証明）。"""
    state, actor_name = _multi_day_state(days=4)
    per_day: dict[int, int] = {}
    for day in (1, 2, 3, 4):
        state.day = day
        block, diag = state.render_continuity_for(actor_name, day=day, turn=1)
        if block:
            per_day[day] = int(diag["continuity_state_chars"])
    assert per_day
    deltas = [
        per_day[day] - per_day[previous]
        for previous, day in zip(sorted(per_day), sorted(per_day)[1:])
    ]
    # 最大でも1日あたりの増加は小さな上限内に収まる（発言全文の再掲とは違う）
    assert max((abs(delta) for delta in deltas), default=0) < 1200
    metrics = state.continuity_metrics()
    assert metrics["continuity_max_day_delta"] < 1200
