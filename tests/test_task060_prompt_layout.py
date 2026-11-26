"""TASK-060の回帰テスト（LL不使用）: 議論プロンプトの節順（layout v3）と配置評価モデル。

実機ログの観測（TASK-057/058/059）で、命令文の重複・理論カードの先頭常置・現在質問と
発言証拠の離間・出力形式仕様の後ろへのめり込みが、節約・一貫性・回答率を削っていた。
だからといって口調・論点・結論や推理の順番を決める指示は足さない（発言の自由は契約の
Free行どおりモデル側に残す）。本テストは以下を固定する（実機LLMは使わない）。

1. ポリシー: 既定は v5（TASK-064の能力境界節込み）。`prompt_contract_policy=legacy` では旧字节順
   が勝つ（A/B基準を壊さない）。v4（TASK-062の論理状態）と v3（TASK-060）も明示指定で参照できる。
2. 凍結した節順（PROMPT_LAYOUT_V3_ORDER）はハード制約を満たし、コストモデルの貪欲探索が
   同じ順を最良として再現する（決定的・LL不使用）。
3. v3 では理論カードが独立節（optional／中部帯）になり、段階除去で落とせる。
4. ヘッダの言い換え行（一人称・応答言語・死者への質問）は他所に正準が1箇所あるため
   ヘッダからは落ちる。落とした制約は契約・ルール・案内に残る。
5. 終端は「議論の案内→秘匿境界→強い契約→出力形式→最終チェックリスト」。
6. 自分の発言と推理メモは現在質問の隣、時系列と主張索引は発言証拠の隣に寄る。
7. 新しい指示文は増やしていない（案内の各行はv2と同じ文言）。情報欠落も起きていない。
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.core_compression import (
    PROMPT_LAYOUT_LEGACY,
    PROMPT_LAYOUT_V2,
    PROMPT_LAYOUT_V2_ORDER,
)
from werewolf.models import GameState, Message, Phase
from werewolf.prompt_contract import (
    drop_sections,
    find_sections,
    behavior_contract_lines,
    inspect_behavior_contract,
    optional_names_dropped_up_to,
    render_behavior_contract,
    section_meta,
    tier_of,
)
from werewolf.prompt_layout import (
    ATTENTION_MIDDLE_BASE,
    ATTENTION_PRIMACY_EDGE,
    CONTINUITY_STATE_SECTION,
    NO_PRIMACY_SECTIONS,
    PROMPT_LAYOUT_POLICY_AUTO,
    PROMPT_LAYOUT_POLICY_DEFAULT,
    PROMPT_LAYOUT_POLICY_V2,
    PROMPT_LAYOUT_POLICY_V3,
    PROMPT_LAYOUT_POLICY_V4,
    PROMPT_LAYOUT_POLICY_V5,
    PROMPT_LAYOUT_V3,
    PROMPT_LAYOUT_V3_ORDER,
    PROMPT_LAYOUT_V4,
    PROMPT_LAYOUT_V4_ORDER,
    PROMPT_LAYOUT_V5,
    PROMPT_LAYOUT_V5_ORDER,
    REPRESENTATIVE_SECTION_CHARS,
    ROLE_BOUNDARY_SECTION,
    adopted_layout_score,
    adjacency_gap,
    attention_prior,
    attention_prior_span,
    measure_zones,
    normalize_prompt_layout_policy,
    post_format_dilution,
    prompt_layout_policy_label,
    resolve_prompt_layout,
    score_order,
    search_prompt_orders,
    search_prompt_orders_v5,
    v3_layout_score,
    v4_layout_score,
    validate_declared_order,
    weak_middle_must_use_chars,
)
from werewolf.prompts import PromptBuilder, WEREWOLF_THEORY_CARD
from werewolf.setup import create_game


LONG_SPEECH = (
    "{name}の{index}番目の発言です。{rival}の占い師COと投票の動きを比べると、"
    "昨夜の襲撃先との整合が怪しく見えるので、今日の投票ではその点を確認したいと考えています。"
)


def _state(seed: int = 1060) -> GameState:
    return create_game(seed=seed, config_path="config/players.json", response_language="日本語")


def _names(state: GameState) -> tuple[str, ...]:
    return tuple(player.name for player in state.players)


def _player(state: GameState, name: str):
    return next(player for player in state.players if player.name == name)


def _start_day1(state: GameState, turn: int = 3) -> None:
    state.public_timeline.record_day_start(state.day)
    state.conversation_log.append(Message("system", f"Day {state.day} begins."))
    state.phase = Phase.DAY_DISCUSSION
    state.discussion_turn = turn
    state.begin_discussion_turn()


def _say(state: GameState, name: str, content: str):
    return state.append_discussion_message(Message(name, content))


def _day1_with_speeches(seed: int = 1060) -> GameState:
    """4発言＋直前の質問（本人は未発言）という Day1 の実形（TASK-057と同じ作り方）。"""
    state = _state(seed=seed)
    _start_day1(state)
    names = _names(state)
    for index, name in enumerate(names[:4]):
        rival = names[(index + 1) % len(names)]
        _say(state, name, LONG_SPEECH.format(name=name, index=index + 1, rival=rival))
    actor_name = names[4]
    asker = names[3]
    _say(
        state,
        asker,
        f"{actor_name}さんに聞きます。{asker}が先に述べた投票案と比べて、"
        f"{actor_name}はどう考えますか？根拠を具体的に教えてください。",
    )
    return state


def _ctx(state: GameState, actor_name: str, layout: str, contract: str = "simplified"):
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
        core_compression_policy="layered",
        prompt_layout_policy=layout,
    )


# ---------------------------------------------------------------------------
# 1. ポリシー解決（既定 v3 / legacy契約では旧字节順が勝つ）
# ---------------------------------------------------------------------------
def test_layout_policy_defaults_to_v5_and_normalizes_aliases() -> None:
    # TASK-064: 既定は v5（能力境界節を現在質問の直後・自分の発言の直前へ置く配置）。
    # v4（論理状態のみ）と v3 は A/B比較の基準側として明示指定で残る。
    assert PROMPT_LAYOUT_POLICY_DEFAULT == PROMPT_LAYOUT_POLICY_V5
    assert normalize_prompt_layout_policy(None) == "v5"
    assert normalize_prompt_layout_policy("V2 ") == "v2"
    assert normalize_prompt_layout_policy("v3") == PROMPT_LAYOUT_POLICY_V3
    assert normalize_prompt_layout_policy("continuity") == "v4"
    assert normalize_prompt_layout_policy("boundary") == "v5"
    assert normalize_prompt_layout_policy(PROMPT_LAYOUT_V5) == "v5"
    assert normalize_prompt_layout_policy("auto") == PROMPT_LAYOUT_POLICY_AUTO
    assert normalize_prompt_layout_policy(True) is PROMPT_LAYOUT_POLICY_V5
    assert normalize_prompt_layout_policy(False) is PROMPT_LAYOUT_POLICY_V2
    # 旧ラベル（legacy/off）は「契約に追随」を意味する auto として扱う
    assert normalize_prompt_layout_policy(PROMPT_LAYOUT_LEGACY) == PROMPT_LAYOUT_POLICY_AUTO
    assert normalize_prompt_layout_policy("未知値") == "v5"
    assert prompt_layout_policy_label("v2_attention") == "v2"
    # レイアウトラベル（ログに出る値）も設定値へ写る
    assert prompt_layout_policy_label(PROMPT_LAYOUT_V5) == "v5"
    assert prompt_layout_policy_label(PROMPT_LAYOUT_V4) == "v4"
    assert prompt_layout_policy_label(PROMPT_LAYOUT_V3) == "v3"


def test_legacy_contract_keeps_the_pre_task060_byte_order() -> None:
    # A/B比較の基準（TASK-057までの字节順）を layout 指定で壊さない
    assert resolve_prompt_layout("v3", "legacy") == PROMPT_LAYOUT_LEGACY
    assert resolve_prompt_layout("v2", "legacy") == PROMPT_LAYOUT_LEGACY
    assert resolve_prompt_layout("v4", "legacy") == PROMPT_LAYOUT_LEGACY
    assert resolve_prompt_layout("v5", "legacy") == PROMPT_LAYOUT_LEGACY
    assert resolve_prompt_layout(None, "legacy_v1") == PROMPT_LAYOUT_LEGACY
    # simplified では auto / v5 が v5、明示した v4 / v3 / v2 はそれぞれの配置のまま
    assert resolve_prompt_layout("auto", "simplified") == PROMPT_LAYOUT_V5
    assert resolve_prompt_layout("v5", "simplified") == PROMPT_LAYOUT_V5
    assert resolve_prompt_layout("v4", "simplified") == PROMPT_LAYOUT_V4
    assert resolve_prompt_layout("v3", "simplified") == PROMPT_LAYOUT_V3
    assert resolve_prompt_layout("v2", "simplified") == PROMPT_LAYOUT_V2


def test_game_state_normalizes_layout_policy() -> None:
    state = _state()
    assert state.prompt_layout_policy == PROMPT_LAYOUT_POLICY_DEFAULT
    state.prompt_layout_policy = "v2_attention"
    GameState.__post_init__(state)
    assert state.prompt_layout_policy == "v2"
    state.prompt_layout_policy = "v4_continuity"
    GameState.__post_init__(state)
    assert state.prompt_layout_policy == "v4"
    state.prompt_layout_policy = "bogus"
    GameState.__post_init__(state)
    assert state.prompt_layout_policy == "v5"


# ---------------------------------------------------------------------------
# 2. attention prior とコストモデル（U字カーブ・長い節の実効値）
# ---------------------------------------------------------------------------
def test_attention_prior_is_a_u_curve_with_middle_as_the_floor() -> None:
    assert attention_prior(0.0) > ATTENTION_MIDDLE_BASE
    assert attention_prior(ATTENTION_PRIMACY_EDGE / 2) > ATTENTION_MIDDLE_BASE
    # 中部（lost-in-the-middle の底）が最も見えにくい
    assert attention_prior(0.5) == ATTENTION_MIDDLE_BASE
    assert attention_prior(1.0) > attention_prior(0.5)
    # 先頭を過ぎた primacy の利得は消える（先頭に長く積んでも守られない）
    assert attention_prior(ATTENTION_PRIMACY_EDGE + 0.05) <= ATTENTION_MIDDLE_BASE + 1e-9


def test_long_sections_are_scored_by_their_whole_span() -> None:
    total = 20000
    head_point = attention_prior(0.01)
    # 先頭から始まっても中部まで伸びる節は、先頭の点評価より実効値が下がる
    long_from_head = attention_prior_span(0, int(total * 0.45), total)
    assert long_from_head < head_point
    # 同じ長さの節でも終端へ寄せる方が実効値が高い（配置改善がスコアへ出る）
    early = attention_prior_span(int(total * 0.30), int(total * 0.30) + 2000, total)
    late = attention_prior_span(int(total * 0.80), int(total * 0.80) + 2000, total)
    assert late > early


def test_score_order_penalizes_moving_the_question_away_from_evidence() -> None:
    chars = dict(REPRESENTATIVE_SECTION_CHARS)
    good = score_order(PROMPT_LAYOUT_V3_ORDER, chars)
    bad_order = list(PROMPT_LAYOUT_V3_ORDER)
    bad_order.remove("question_block")
    bad_order.insert(bad_order.index("day_timeline"), "question_block")
    bad = score_order(tuple(bad_order), chars)
    assert bad.score < good.score
    assert bad.adjacency_penalty > good.adjacency_penalty


# ---------------------------------------------------------------------------
# 3. 凍結した v3 節順（ハード制約と探索再現）
# ---------------------------------------------------------------------------
def test_frozen_v5_order_is_legal_and_reproduced_by_the_search() -> None:
    """凍結した v5 節順がハード制約を満たし、境界込みの探索が同じ順を最良として再現する。"""
    assert validate_declared_order(PROMPT_LAYOUT_V5_ORDER) == []
    assert len(set(PROMPT_LAYOUT_V5_ORDER)) == len(PROMPT_LAYOUT_V5_ORDER)
    chars = dict(REPRESENTATIVE_SECTION_CHARS)
    frozen = adopted_layout_score()
    assert frozen.violations == ()
    assert frozen.order == PROMPT_LAYOUT_V5_ORDER
    # 代表字数での貪欲探索（単一ブロック移動＋2ブロック入替）が凍結順をそのまま出す
    best = search_prompt_orders_v5(chars, top_k=1)[0]
    assert tuple(best["layout_order"]) == PROMPT_LAYOUT_V5_ORDER
    assert best["hard_violations"] == []
    # v2 の宣言順はハード制約で弾かれる（案内が形式の後／自分の記録が分断）
    v2_like = score_order(PROMPT_LAYOUT_V2_ORDER, chars)
    assert {item["kind"] for item in v2_like.violations} == {
        "anchor_split",
        "guidelines_after_format",
    }
    assert frozen.score > v2_like.score
    # 境界節を末尾側（議論案内の直前）へ足すだけの配置より、v5 は上（順番が挙動を決める）
    appended = list(PROMPT_LAYOUT_V4_ORDER)
    appended.insert(appended.index("discussion_guidelines"), ROLE_BOUNDARY_SECTION)
    assert score_order(tuple(appended), chars).score < frozen.score


def test_v4_baseline_stays_legal_and_is_kept_for_ab_reference() -> None:
    """A/B基準（TASK-062の v4）は節順そのままで無傷のまま参照できる（境界節なし）。"""
    assert validate_declared_order(PROMPT_LAYOUT_V4_ORDER) == []
    assert len(set(PROMPT_LAYOUT_V4_ORDER)) == len(PROMPT_LAYOUT_V4_ORDER)
    baseline = v4_layout_score()
    assert baseline.order == PROMPT_LAYOUT_V4_ORDER
    assert baseline.violations == ()
    assert ROLE_BOUNDARY_SECTION not in PROMPT_LAYOUT_V4_ORDER
    assert CONTINUITY_STATE_SECTION in PROMPT_LAYOUT_V4_ORDER
    # v3（状態節も境界節なし）も参照可能
    v3_chars = {
        key: value
        for key, value in dict(REPRESENTATIVE_SECTION_CHARS).items()
        if key not in {CONTINUITY_STATE_SECTION, ROLE_BOUNDARY_SECTION}
    }
    v3_baseline = v3_layout_score(v3_chars)
    assert v3_baseline.order == PROMPT_LAYOUT_V3_ORDER
    assert v3_baseline.violations == ()


def test_state_section_must_follow_its_id_index() -> None:
    """論理状態節は主張索引（CLAIM_LEDGER）より前に出せない（存在しないIDを辿らせる）。"""
    too_early = list(PROMPT_LAYOUT_V4_ORDER)
    too_early.remove(CONTINUITY_STATE_SECTION)
    too_early.insert(too_early.index("werewolf_theory_card"), CONTINUITY_STATE_SECTION)
    kinds = {item["kind"] for item in validate_declared_order(tuple(too_early))}
    assert "state_before_index" in kinds
    # v4 の実位置は索引の後・発言証拠の後・現在質問の前（読み順が状態→質問になる）
    order = PROMPT_LAYOUT_V4_ORDER
    assert order.index("claim_ledger") < order.index(CONTINUITY_STATE_SECTION)
    assert order.index("conversation_evidence") < order.index(CONTINUITY_STATE_SECTION)
    assert order.index(CONTINUITY_STATE_SECTION) < order.index("question_block")
    assert CONTINUITY_STATE_SECTION in NO_PRIMACY_SECTIONS



def test_hard_constraints_reject_inducing_or_reordered_placement() -> None:
    # 誘導の強い理論カードを事実台帳より前（＝先頭帯）へ置く案は弾く
    theory_first = list(PROMPT_LAYOUT_V3_ORDER)
    theory_first.remove("werewolf_theory_card")
    theory_first.insert(1, "werewolf_theory_card")
    kinds = {item["kind"] for item in validate_declared_order(tuple(theory_first))}
    assert "heuristics_before_facts" in kinds
    assert "werewolf_theory_card" in NO_PRIMACY_SECTIONS
    # 自分の発言と推理メモを離す配置も弾く（立場の更新が途切れるため）
    split = list(PROMPT_LAYOUT_V3_ORDER)
    split.remove("reasoning_memo_own")
    split.insert(split.index("fact_ledger") + 1, "reasoning_memo_own")
    assert any(
        item["kind"] == "anchor_split" for item in validate_declared_order(tuple(split))
    )
    # 最終チェックリストを末尾から外す配置も弾く
    tail_broken = list(PROMPT_LAYOUT_V3_ORDER)
    tail_broken.remove("output_checklist")
    tail_broken.insert(0, "output_checklist")
    assert validate_declared_order(tuple(tail_broken))


# ---------------------------------------------------------------------------
# 4. v3で実際に組んだプロンプト（同じ盤面・同じ発言者での配置比較）
# ---------------------------------------------------------------------------
def _state_with_own_history(seed: int = 1060):
    """本人が発言済みで、直前の発言が本人への質問になっている Day1 の実形。"""
    state = _state(seed=seed)
    _start_day1(state, turn=4)
    names = _names(state)
    actor_name = names[2]
    speakers = (actor_name, names[0], names[1])
    for index, name in enumerate(speakers):
        rival = names[(index + 3) % len(names)]
        _say(state, name, LONG_SPEECH.format(name=name, index=index + 1, rival=rival))
    asker = names[3] if names[3] != actor_name else names[5]
    _say(
        state,
        asker,
        f"{actor_name}さんに聞きます。{asker}が先に述べた投票案と比べて、"
        f"{actor_name}はどう考えますか？根拠を具体的に教えてください。",
    )
    state.store_model_reasoning_memo(
        actor_name,
        "占い師COを出した2名が昨夜の襲撃先と矛盾していないか、今日の発言順を確認したい。",
    )
    return state, actor_name


def _section_text(prompt: str, name: str) -> str:
    spans = find_sections(prompt)
    start, end = spans[name]
    return prompt[start:end]


def test_v3_chains_evidence_into_the_question_and_the_actors_own_records() -> None:
    state, actor_name = _state_with_own_history()
    v3 = _ctx(state, actor_name, "v3")
    diag = v3.diagnostics or {}
    assert diag["prompt_layout"] == PROMPT_LAYOUT_V3
    assert diag["prompt_layout_violations"] == []
    spans = find_sections(v3.user_prompt)
    chain = (
        "claim_ledger",
        "day_timeline",
        "conversation_evidence",
        "question_block",
        "own_speeches",
        "reasoning_memo_own",
        "discussion_guidelines",
        "leak_boundary",
        "behavior_contract",
        "instructions",
    )
    assert all(name in spans for name in chain)
    assert [spans[name][0] for name in chain] == sorted(spans[name][0] for name in chain)
    # 証拠の直後に質問、その直後に自分の発言と推理メモ（距離はプロンプト長の2%以内）
    assert adjacency_gap(v3.user_prompt, "conversation_evidence", "question_block") < 0.02
    assert adjacency_gap(v3.user_prompt, "question_block", "own_speeches") < 0.02
    assert adjacency_gap(v3.user_prompt, "day_timeline", "conversation_evidence") < 0.02


def test_v2_leaves_the_actors_own_records_far_from_the_question() -> None:
    """比較基準（v2）。自分の記録が質問から離れた位置にあることを実測で確認する。"""
    state, actor_name = _state_with_own_history()
    v3 = _ctx(state, actor_name, "v3")
    v2 = _ctx(state, actor_name, "v2")
    own_gap_v3 = adjacency_gap(v3.user_prompt, "question_block", "own_speeches")
    own_gap_v2 = adjacency_gap(v2.user_prompt, "question_block", "own_speeches")
    assert own_gap_v3 < own_gap_v2 / 10
    timeline_gap_v3 = adjacency_gap(v3.user_prompt, "day_timeline", "conversation_evidence")
    timeline_gap_v2 = adjacency_gap(v2.user_prompt, "day_timeline", "conversation_evidence")
    assert timeline_gap_v3 < timeline_gap_v2 / 5


def test_v3_keeps_the_output_format_spec_at_the_end_of_the_prompt() -> None:
    state, actor_name = _state_with_own_history()
    v3 = _ctx(state, actor_name, "v3")
    v2 = _ctx(state, actor_name, "v2")
    # 出力形式仕様の後ろに、形式以外の節が載らない（案内に押されて遠ざからない）
    assert post_format_dilution(v3.user_prompt) == 0
    assert post_format_dilution(v2.user_prompt) > 1000
    spans = find_sections(v3.user_prompt)
    assert spans["output_checklist"][1] == len(v3.user_prompt)
    assert weak_middle_must_use_chars(v3.user_prompt) < len(v3.user_prompt)


def test_theory_card_leaves_the_head_and_becomes_droppable() -> None:
    state, actor_name = _state_with_own_history()
    v3 = _ctx(state, actor_name, "v3")
    assert tier_of("werewolf_theory_card") != "core"
    assert "werewolf_theory_card" in optional_names_dropped_up_to(2)
    meta = section_meta("werewolf_theory_card")
    assert meta is not None and "heuristic" in meta.purpose
    prompt = v3.user_prompt
    assert WEREWOLF_THEORY_CARD in prompt
    pruned, removed, saved = drop_sections(prompt, ("werewolf_theory_card",))
    assert removed == ("werewolf_theory_card",)
    assert saved == len(WEREWOLF_THEORY_CARD)
    assert WEREWOLF_THEORY_CARD not in pruned
    # anchor（証拠・現在質問・契約）は削れたまま残る
    for marker in ("[CONVERSATION_EVIDENCE", "[Current Question]", "[BEHAVIOR_CONTRACT"):
        assert marker in pruned
    assert len(pruned) == len(prompt) - saved


def test_restatements_fall_back_to_their_single_canonical_source() -> None:
    state, actor_name = _state_with_own_history()
    v3 = _ctx(state, actor_name, "v3")
    v2 = _ctx(state, actor_name, "v2")
    legacy = _ctx(state, actor_name, "legacy", contract="legacy")
    prompt = v3.user_prompt
    # ヘッダの言い換え（一人称・応答言語・投票台帳の参照限定）はヘッダから消える
    assert "- Self-Reference:" not in prompt
    assert "- Response Language:" not in prompt
    assert "recorded in the ledger" not in prompt
    # 正準側は残る（応答言語＝[OUTPUT_LANGUAGE]、一人称・表現の自由＝契約の各行）
    assert "[OUTPUT_LANGUAGE]" in prompt
    contract_lines = behavior_contract_lines(
        language="日本語", hint_mode="strict", include_claim=True
    )
    self_reference = next(
        line.text for line in contract_lines if line.dimension == "self_reference"
    )
    expression = next(line.text for line in contract_lines if line.dimension == "expression")
    assert self_reference in prompt
    assert expression in prompt  # 口調・論点・結論が自由である旨は契約に1箇所残る
    assert prompt.count("ask dead players") == 1
    assert prompt.count("True Roles of Dead Players") == 1
    # 同じ制約の再掲回数は v2 より減り、legacy はTASK-057のまま
    diag3, diag2, diagL = (v3.diagnostics or {}, v2.diagnostics or {}, legacy.diagnostics or {})
    assert int(diag3["prompt_duplicate_constraint_count"]) < int(
        diag2["prompt_duplicate_constraint_count"]
    )
    assert "- Self-Reference:" in legacy.user_prompt
    assert "- Response Language:" in legacy.user_prompt


def test_the_layout_change_adds_no_new_instructions() -> None:
    """配置変えで「言い方・論点・結論・推理順」を決める文言を足していないことの保証。"""
    state, actor_name = _state_with_own_history()
    v3 = _ctx(state, actor_name, "v3")
    v2 = _ctx(state, actor_name, "v2")
    assert v3.system_prompt == v2.system_prompt
    assert _section_text(v3.user_prompt, "behavior_contract") == _section_text(
        v2.user_prompt, "behavior_contract"
    )
    guide_v3 = [line.strip() for line in _section_text(v3.user_prompt, "discussion_guidelines").splitlines()]
    prompt_v2 = v2.user_prompt
    added = [line for line in guide_v3 if line and line not in prompt_v2]
    # 案内の各行はv2に存在する文言だけ（新規指示ゼロ）。減った行は他所に正準がある言い換え。
    assert added == []


def test_reordering_keeps_every_speech_and_name_present() -> None:
    state, actor_name = _state_with_own_history()
    v3 = _ctx(state, actor_name, "v3")
    v2 = _ctx(state, actor_name, "v2")
    bodies = [message.content for message in state.conversation_log if message.speaker != "system"]
    for name in state.alive_players:
        assert name in v3.user_prompt
    for dead in [name for name in _names(state) if name not in state.alive_players]:
        assert dead in v3.user_prompt
    # 発言本文は各1回（証拠欄のみ）。配置が変わっても二重送信・欠落が起きていない
    assert all(v3.user_prompt.count(body) == 1 for body in bodies)
    assert all(v2.user_prompt.count(body) == 1 for body in bodies)
    spans = find_sections(v3.user_prompt)
    evidence_chars = spans["conversation_evidence"][1] - spans["conversation_evidence"][0]
    v2_spans = find_sections(v2.user_prompt)
    v2_evidence_chars = v2_spans["conversation_evidence"][1] - v2_spans["conversation_evidence"][0]
    assert abs(evidence_chars - v2_evidence_chars) <= max(40, int(v2_evidence_chars * 0.05))
    # 節約は効いている（同じ盤面・同じ発言で総字数が減る）
    assert len(v3.user_prompt) < len(v2.user_prompt)


def test_legacy_contract_still_emits_the_task057_byte_order() -> None:
    state, actor_name = _state_with_own_history()
    legacy = _ctx(state, actor_name, "v3", contract="legacy")  # layout指定より契約が勝つ
    diag = legacy.diagnostics or {}
    assert diag["prompt_layout"] == PROMPT_LAYOUT_LEGACY
    prompt = legacy.user_prompt
    assert "[BEHAVIOR_CONTRACT" not in prompt
    ledger_start = find_sections(prompt)["fact_ledger"][0]
    assert prompt.index("[Werewolf Theory Card") > ledger_start
    assert legacy.user_prompt.count("ask dead players") >= 2


def test_diagnostics_report_placement_quality_for_ab_runs() -> None:
    state, actor_name = _state_with_own_history()
    v3 = _ctx(state, actor_name, "v3")
    diag = v3.diagnostics or {}
    for key in (
        "prompt_layout",
        "prompt_layout_policy",
        "prompt_layout_violations",
        "prompt_zone_primacy_chars",
        "prompt_zone_middle_chars",
        "prompt_zone_recency_chars",
        "prompt_instructions_to_end_chars",
        "prompt_weak_middle_must_use_chars",
        "prompt_gap_question_to_evidence",
        "prompt_gap_timeline_to_evidence",
        "prompt_gap_own_speeches_to_question",
        "prompt_theory_card_chars",
    ):
        assert key in diag, key
    assert diag["prompt_layout_policy"] == "v3"
    assert diag["prompt_theory_card_chars"] == len(WEREWOLF_THEORY_CARD)
    assert int(diag["prompt_zone_recency_chars"]) > 0
    zones = measure_zones(v3.user_prompt)
    assert zones["unregistered"] == 0  # 全台帳が拾える（節順の測定が信用できる）


def test_logger_aggregates_placement_metrics_for_ab(tmp_path) -> None:
    """A/B比較に必要な配置指標が、ログの台帳とモデル別 stats に集約されること。"""
    import json as _json

    from werewolf.cli import GameLogger

    state, actor_name = _state_with_own_history()
    model = _player(state, actor_name).model
    base = {
        "day": 1,
        "phase": "day_discussion",
        "phase_label": "昼・議論",
        "player": actor_name,
        "model": model,
        "action": "discussion",
        "private": False,
    }
    logger = GameLogger("task060", tmp_path, {"seed": 1060}, private_enabled=False)
    first = {
        **base,
        "type": "discussion_context_selected",
        "prompt_layout": PROMPT_LAYOUT_V3,
        "prompt_user_char_count": 12000,
        "prompt_core_chars": 5700,
        "prompt_duplicate_constraint_count": 2,
        "prompt_layout_violations": [],
        "prompt_zone_primacy_chars": 1300,
        "prompt_zone_middle_chars": 6000,
        "prompt_zone_recency_chars": 4700,
        "prompt_instructions_to_end_chars": 0,
        "prompt_weak_middle_must_use_chars": 1800,
        "prompt_theory_card_chars": len(WEREWOLF_THEORY_CARD),
        "prompt_gap_question_to_evidence": 0.005,
        "prompt_gap_timeline_to_evidence": 0.02,
        "prompt_gap_own_speeches_to_question": 0.008,
    }
    second = {
        **first,
        "prompt_user_char_count": 13000,
        "prompt_zone_middle_chars": 4000,
        "prompt_gap_own_speeches_to_question": 0.012,
        "prompt_layout_violations": [
            {"kind": "out_of_order", "name": "own_speeches", "after": "conversation_evidence"}
        ],
    }
    logger.append_record(first)
    logger.append_record(second)
    logger.finalize(state)

    rows = [
        _json.loads(line)
        for line in (tmp_path / "game_task060.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = next(row for row in rows if row.get("type") == "game_result")
    metrics = result["ledger_metrics"]
    assert metrics["prompt_layout_counts"][PROMPT_LAYOUT_V3] == 2
    assert metrics["avg_prompt_zone_middle_chars"] == 5000.0
    assert metrics["avg_prompt_instructions_to_end_chars"] == 0.0
    assert metrics["avg_gap_own_speeches_to_question"] == 0.01
    assert metrics["avg_gap_question_to_evidence"] == 0.005
    assert metrics["avg_prompt_theory_card_chars"] == float(len(WEREWOLF_THEORY_CARD))
    assert metrics["prompt_layout_violations"] == 1
    model_stats = result["model_stats"][model]
    assert model_stats["prompt_contracted_requests"] == 2
    assert model_stats["avg_prompt_user_chars"] == 12500.0
    assert model_stats["avg_gap_own_speeches_to_question"] == 0.01
    assert model_stats["avg_prompt_zone_recency_chars"] == 4700.0
    assert model_stats["avg_prompt_instructions_to_end_chars"] == 0.0
    # JSONLとしてそのまま書ける（実機ログでA/Bできるように）
    _json.dumps(metrics)




