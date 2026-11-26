"""TASK-064の回帰テスト（LL不使用）: 役職騙りと能力行使の境界整理。

実ログ game_bcd7c97e7e48 では、村人（Player 4）が Day 1 T3 で実行記録のない占い結果を
口にし、後で `seer_result_without_claim` の補正が「その結果を出しなさい」と再強化した。
当時は「エンジンが実際に実行した能力行動」と「本人が公に語った主張」が同じ形で見え、
未検証のブラフが事実のように扱われていた。本テストは以下を固定する（実機LLMは使わない）。

1. 分類は4出所（engine_action / public_role_claim / unsupported_role_action / inference）。
   検出は観測専用で、本文の言い換え・削除・再生成は一切行わない。
2. 疑問形・仮定形・引用・否定は断定的な能力行使として数えない（誤検出を増やさない）。
3. 公開面（プロンプト・公開台帳・イベント）には実行記録との一致可否を出さない
   （TASK-055と同じ線引き。真偽は管理者ビュー専用）。
4. `[ROLE_ACTION_BOUNDARY]` は現在質問の直後・自分の発言の直前（v5で凍結）。見出し＋1行で、
   エンジン結果・CO本文・推測を再掲しない（コンテキスト節約）。新しい命令文も増やさない。
5. CO補正の再要求は能力実行記録・主張状態・現在質問・出力形式だけを含み、通常プロンプトより
   明確に短い。保有していない能力を「エンジンが実行した」とは書かない。
6. ログにはカテゴリ合計・境界字数・隣接距離・CO初出→能力表現のターン距離・補正の短縮倍率が
   集約される（口調・論点には無関係な数値だけ）。
"""
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.engine import GameEngine
from werewolf.llm_actions import OllamaActions
from werewolf.models import GameState, Message, Phase, PublicRoleClaim, Role
from werewolf.prompt_contract import find_sections
from werewolf.prompt_layout import (
    PROMPT_LAYOUT_V4_ORDER,
    PROMPT_LAYOUT_V5_ORDER,
    REPRESENTATIVE_SECTION_CHARS,
    ROLE_BOUNDARY_SECTION,
    score_order,
    validate_declared_order,
)
from werewolf.prompts import PromptBuilder
from werewolf.role_action_provenance import (
    CATEGORY_ENGINE_ACTION,
    CATEGORY_INFERENCE,
    CATEGORY_PUBLIC_ROLE_CLAIM,
    CATEGORY_UNSUPPORTED_ROLE_ACTION,
    KIND_ATTACK,
    KIND_DIVINATION,
    RoleActionProvenance,
    detect_ability_expressions,
    extract_spoken_verdict,
    turn_delta,
    turn_position,
)
from werewolf.setup import create_game


def _state(seed: int = 2064) -> GameState:
    return create_game(
        seed=seed, config_path="config/players.json", response_language="日本語"
    )


def _names(state: GameState) -> tuple[str, ...]:
    return tuple(player.name for player in state.players)


def _player(state: GameState, name: str):
    return next(player for player in state.players if player.name == name)


def _cast(state: GameState, name: str, role: Role):
    player = _player(state, name)
    player.role = role
    player.partner = None
    return player


def _start_day(state: GameState, turn: int = 3) -> None:
    state.public_timeline.record_day_start(state.day)
    state.conversation_log.append(Message("system", f"Day {state.day} begins."))
    state.phase = Phase.DAY_DISCUSSION
    state.discussion_turn = turn
    state.begin_discussion_turn()


LONG_SPEECH = (
    "{name}の{index}番目の発言です。{rival}の占い師COと投票の動きを比べると、"
    "昨夜の襲撃先との整合が怪しく見えるので、今日の投票ではその点を確認したいと考えています。"
)


def _day1_with_speeches(seed: int = 2064) -> GameState:
    """4発言＋本人宛ての質問（本人は未発言）という Day 1 の実形（TASK-060と同じ作り方）。"""
    state = _state(seed=seed)
    _start_day(state)
    names = _names(state)
    for index, name in enumerate(names[:4]):
        rival = names[(index + 1) % len(names)]
        state.append_discussion_message(
            Message(name, LONG_SPEECH.format(name=name, index=index + 1, rival=rival))
        )
    actor_name = names[4]
    asker = names[3]
    state.append_discussion_message(
        Message(
            asker,
            f"{actor_name}さんに聞きます。{asker}が先に述べた投票案と比べて、"
            f"{actor_name}はどう考えますか？根拠を具体的に教えてください。",
        )
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
# 1. 能力行使表現の検出（観測専用・口調は問わない）
# ---------------------------------------------------------------------------
def test_assertion_is_detected_and_hedged_forms_are_not_counted_as_use() -> None:
    names = ("Player 1", "Player 2", "Player 4")
    plain = detect_ability_expressions(
        "私が占った結果、Player 1は狼だった。投票を確認したい。",
        actor_name="Player 4",
        known_player_names=names,
    )
    assert len(plain) == 1
    assert plain[0]["kind"] == KIND_DIVINATION
    assert plain[0]["first_person"] is True and plain[0]["hedged"] is False

    hedged_lines = (
        "Player 2が先に「私は占った」と言っていましたね。",  # 引用
        "私は占っていません。COもしていません。",  # 否定
        "あなたが占った結果は何ですか？",  # 疑問（相手宛て）
    )
    for line in hedged_lines:
        found = detect_ability_expressions(
            line, actor_name="Player 4", known_player_names=names
        )
        assert all(item["hedged"] for item in found), line


def test_third_person_ability_reference_is_inference_not_self_use() -> None:
    names = ("Player 1", "Player 2", "Player 4")
    found = detect_ability_expressions(
        "Player 2が占ったのは夜の設定と合わないはずです。",
        actor_name="Player 4",
        known_player_names=names,
    )
    assert len(found) == 1
    assert found[0]["first_person"] is False
    assert found[0]["attributed_to"] == "Player 2"


def test_spoken_verdict_extraction_works_for_both_forms() -> None:
    assert extract_spoken_verdict("私が占った結果、Player 1は狼だった。") == (
        "Player 1",
        "wolf",
    )
    assert extract_spoken_verdict("対象=Player 8 判定=人間") == ("Player 8", "human")
    assert extract_spoken_verdict("今日は投票を見合わせたい。") is None


def test_turn_position_and_delta_are_monotonic_across_days() -> None:
    assert turn_delta((1, 3), (2, 1)) > 0
    assert turn_delta((2, 1), (1, 3)) == 0  # 過去へは遡らない（単調な距離軸）
    assert turn_position(2, 2) - turn_position(1, 2) == 3  # 1日=3ターン


# ---------------------------------------------------------------------------
# 2. 台帳（RoleActionProvenance）: 4出所への分類と観測専用であること
# ---------------------------------------------------------------------------
def test_bluff_assertion_is_unsupported_and_true_seer_matches_engine_record() -> None:
    tracker = RoleActionProvenance()
    bluff = tracker.observe_public_speech(
        player="Player 4",
        day=1,
        discussion_turn=3,
        text="私が占った結果、Player 1は狼だった。だから今日投票したい。",
        known_player_names=("Player 1", "Player 4"),
    )
    assert [item.category for item in bluff] == [CATEGORY_UNSUPPORTED_ROLE_ACTION]
    assert tracker.unsupported_role_action_count == 1
    # ブラフでも発言そのものは受理される（禁止しない）。カテゴリは観測値。
    assert tracker.allow_bluff_role_claim is True

    seer_tracker = RoleActionProvenance()
    seer_tracker.record_engine_action("Player 5", KIND_DIVINATION, 1)
    true_read = seer_tracker.observe_public_speech(
        player="Player 5",
        day=2,
        discussion_turn=1,
        text="私が占った結果、Player 8は狼でした。理由はその振る舞いです。",
        own_engine_results={"Player 8": "wolf"},
        known_player_names=("Player 5", "Player 8"),
    )
    assert [item.category for item in true_read] == [CATEGORY_ENGINE_ACTION]
    assert true_read[0].supported is True


def test_conflicting_verdict_is_recorded_against_the_players_own_result() -> None:
    tracker = RoleActionProvenance()
    tracker.record_engine_action("Player 5", KIND_DIVINATION, 1)
    matching = tracker.observe_public_speech(
        player="Player 5",
        day=2,
        discussion_turn=2,
        text="私が占った結果、Player 9は人間でした。",
        own_engine_results={"Player 9": "human"},
        known_player_names=("Player 5", "Player 9"),
    )
    assert matching and matching[0].conflict is False
    wrong = tracker.observe_public_speech(
        player="Player 5",
        day=3,
        discussion_turn=1,
        text="私が占った結果、Player 2は狼でした。",
        own_engine_results={"Player 2": "human"},
        known_player_names=("Player 2", "Player 5"),
    )
    assert any(item.conflict for item in wrong)
    assert tracker.claim_vs_engine_conflict_count == 1


def test_inference_and_hedged_mentions_do_not_count_as_unsupported_use() -> None:
    tracker = RoleActionProvenance()
    tracker.observe_public_speech(
        player="Player 4",
        day=2,
        discussion_turn=1,
        text="Player 2が占ったはずです。私は占っていません。もし私なら占います。",
        known_player_names=("Player 2", "Player 4"),
    )
    assert tracker.category_counts[CATEGORY_INFERENCE] >= 1
    assert tracker.unsupported_role_action_count == 0


def test_attack_assertion_needs_the_players_own_attack_record() -> None:
    tracker = RoleActionProvenance()
    madman = tracker.observe_public_speech(
        player="Player 3",
        day=2,
        discussion_turn=1,
        text="私たちは昨夜 Player 7 を襲った。",
        known_player_names=("Player 3", "Player 7"),
    )
    assert [item.category for item in madman] == [CATEGORY_UNSUPPORTED_ROLE_ACTION]
    wolf_tracker = RoleActionProvenance()
    wolf_tracker.record_engine_action("Player 3", KIND_ATTACK, 2)
    wolf_items = wolf_tracker.observe_public_speech(
        player="Player 3",
        day=2,
        discussion_turn=2,
        text="私たちは昨夜 Player 7 を襲った。理由は投票の並びだ。",
        known_player_names=("Player 3", "Player 7"),
    )
    assert [item.category for item in wolf_items] == [CATEGORY_ENGINE_ACTION]


def test_claim_then_ability_distance_and_public_view_stays_neutral() -> None:
    tracker = RoleActionProvenance()
    tracker.mark_public_claim("Player 4", day=1, discussion_turn=2)
    delta = tracker.mark_ability_assertion("Player 4", day=2, discussion_turn=3)
    assert delta is not None and delta > 0
    admin = tracker.admin_metrics()
    assert admin["per_player"]["Player 4"]["claim_to_ability_turn_delta"] == delta
    public_blob = json.dumps(tracker.serialize_public(), ensure_ascii=False)
    # 公開面はカテゴリ合計と中立ラベルのみ（本人のCOが本物かを示す語を出さない）
    for forbidden in ('"supported"', '"category"', '"conflict"', '"engine_actions"'):
        assert forbidden not in public_blob
    assert (
        tracker.metrics()["role_action_provenance_counts"][CATEGORY_PUBLIC_ROLE_CLAIM]
        == 1
    )


# ---------------------------------------------------------------------------
# 3. `[ROLE_ACTION_BOUNDARY]` の配置とコスト（v5 / A/B基準の無傷さ）
# ---------------------------------------------------------------------------
def _state_with_engine_record_and_claim(seed: int = 2064):
    """エンジンが占った記録と本人の公開COがある Day 2 相当の実形を作る。"""
    state = _day1_with_speeches(seed=seed)
    actor_name = _names(state)[4]
    seer = _cast(state, actor_name, Role.SEER)
    tracker = state.role_action_provenance
    tracker.record_engine_action(actor_name, KIND_DIVINATION, 1)
    tracker.record_engine_action(_names(state)[0], KIND_ATTACK, 2)
    state.public_role_claims.append(
        PublicRoleClaim(
            day=1,
            claimant=actor_name,
            claimed_role="seer",
            target="Player 1",
            result="wolf",
            reason="初日の言動が不自然だから占い師として申告した",
            vote_proposal="Player 1へ投票したい",
        )
    )
    tracker.mark_public_claim(actor_name, day=1, discussion_turn=2)
    return state, seer.name


def test_boundary_section_sits_after_the_question_and_before_own_speeches() -> None:
    state, actor_name = _state_with_engine_record_and_claim()
    ctx = _ctx(state, actor_name, "v5")
    spans = find_sections(ctx.user_prompt)
    assert ROLE_BOUNDARY_SECTION in spans
    start, end = spans[ROLE_BOUNDARY_SECTION]
    assert spans["question_block"][1] <= start  # 現在質問の直後
    assert end <= spans["own_speeches"][0] + 1  # 自分の発言の直前
    # v5 の凍結節順（claim_ledger → ... → question → boundary → own_speeches）と読む順序が一致
    order = PROMPT_LAYOUT_V5_ORDER
    assert (
        order.index("claim_ledger")
        < order.index(ROLE_BOUNDARY_SECTION)
        < order.index("own_speeches")
    )
    diag = ctx.diagnostics or {}
    assert diag["prompt_layout"] == "v5_role_boundary"
    assert diag["prompt_layout_violations"] == []
    assert int(diag["role_boundary_section_chars"]) > 0
    assert float(diag["prompt_gap_role_boundary_to_own_speeches"]) < 0.02


def test_boundary_is_one_short_line_pair_and_restates_no_bodies() -> None:
    state, actor_name = _state_with_engine_record_and_claim()
    ctx = _ctx(state, actor_name, "v5")
    start, end = find_sections(ctx.user_prompt)[ROLE_BOUNDARY_SECTION]
    section = ctx.user_prompt[start:end]
    body = [message.content for message in state.conversation_log if message.speaker != "system"]
    # 発言本文・COの理由文・投票提案をここへ再掲しない（CLAIM_LEDGERと私人欄が正準）
    for text in body:
        assert text not in section
    assert "初日の言動が不自然だから" not in section
    assert "Player 1へ投票したい" not in section
    # 実行回数は出すが、占い判定の語（狼/人間）をこの節に載せない
    assert "Wolf" not in section and "狼" not in section
    # コンテキスト節約: 見出し＋定義1行＋状態1行（2行で完結）
    assert len(section) <= 320
    assert section.count("[ROLE_ACTION_BOUNDARY]") == 1


def test_boundary_adds_no_new_instructions_or_freedom_loss() -> None:
    """この節から命令・禁止・口調指定を増やさない（設計思想4: 議論の自由を縛らない）。"""
    state, actor_name = _state_with_engine_record_and_claim()
    ctx = _ctx(state, actor_name, "v5")
    start, end = find_sections(ctx.user_prompt)[ROLE_BOUNDARY_SECTION]
    section = ctx.user_prompt[start:end]
    for forbidden in (
        "You must",
        "Do not",
        "Never ",
        "Stop ",
        "Always ",
        "してください",
        "しなさい",
        "禁止",
        "禁じる",
    ):
        assert forbidden not in section, forbidden
    # 同じ盤面・同じ発言で、発言本文の欠落も二重送信も起きていない
    for message in state.conversation_log:
        if message.speaker == "system":
            continue
        assert ctx.user_prompt.count(message.content) <= 1


def test_v4_and_v3_keep_the_byte_order_without_the_boundary_section() -> None:
    state, actor_name = _state_with_engine_record_and_claim()
    v5 = _ctx(state, actor_name, "v5")
    v4 = _ctx(state, actor_name, "v4")
    v3 = _ctx(state, actor_name, "v3")
    assert ROLE_BOUNDARY_SECTION not in find_sections(v4.user_prompt)
    assert ROLE_BOUNDARY_SECTION not in find_sections(v3.user_prompt)
    assert "[ROLE_ACTION_BOUNDARY]" in v5.user_prompt
    # 節約: v5の追加は1節ぶん（代表200字）に収まり、v4より増え過ぎない
    assert len(v5.user_prompt) - len(v4.user_prompt) <= 300
    assert len(v4.user_prompt) - len(v3.user_prompt) >= 0  # 基準側は壊れていない


def test_hard_constraint_rejects_the_boundary_before_its_claim_index() -> None:
    too_early = list(PROMPT_LAYOUT_V5_ORDER)
    too_early.remove(ROLE_BOUNDARY_SECTION)
    too_early.insert(too_early.index("werewolf_theory_card"), ROLE_BOUNDARY_SECTION)
    kinds = {item["kind"] for item in validate_declared_order(tuple(too_early))}
    assert "boundary_before_index" in kinds
    # 索引より前に置いた案はコストモデルでも弾かれる（違反ぶんが減点される）
    penalised = score_order(tuple(too_early), dict(REPRESENTATIVE_SECTION_CHARS))
    assert penalised.violations
    # v4（境界なし）と v5（境界あり）はどちらも無違反で宣言できる
    assert validate_declared_order(PROMPT_LAYOUT_V4_ORDER) == []
    assert validate_declared_order(PROMPT_LAYOUT_V5_ORDER) == []



# ---------------------------------------------------------------------------
# 4. 公開発言の観測: 本文は一字一句そのまま、分類だけ管理者ビューへ
# ---------------------------------------------------------------------------
def _engine_with_events(state: GameState, events: list[dict[str, object]]) -> GameEngine:
    class _NoActions:
        def wolf_discuss(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("このテストでは夜行動を実行しない")

    return GameEngine(state, _NoActions(), seed=state.seed, event_callback=events.append)


BLUFF_SENTENCE = "私が占った結果、Player 1は狼だった。今日の投票で確認したい。"


def test_public_speech_body_is_unchanged_and_observation_is_admin_side() -> None:
    state = _day1_with_speeches(seed=4101)
    speaker = _names(state)[5]
    _cast(state, speaker, Role.VILLAGER)
    events: list[dict[str, object]] = []
    engine = _engine_with_events(state, events)
    engine._append_discussion_message(Message(speaker, BLUFF_SENTENCE))

    event = next(e for e in events if e["type"] == "discussion_message")
    # 公開本文は一字一句そのまま（言い換え・削除・メタ文の挿入なし）
    assert event["content"] == BLUFF_SENTENCE
    assert state.conversation_log[-1].content == BLUFF_SENTENCE
    observations = event["role_ability_observations"]
    assert len(observations) == 1
    assert observations[0]["category"] == CATEGORY_UNSUPPORTED_ROLE_ACTION
    # 台帳は観測するだけで、発言を消したり書き換えたりしていない
    assert state.role_action_provenance.unsupported_role_action_count == 1


def test_question_or_quoted_ability_wording_never_counts_as_self_use_in_engine() -> None:
    state = _day1_with_speeches(seed=4102)
    speaker = _names(state)[5]
    _cast(state, speaker, Role.VILLAGER)
    events: list[dict[str, object]] = []
    engine = _engine_with_events(state, events)
    hedged = "Player 3が「私が占った」と言っていましたね。私は占っていません。"
    engine._append_discussion_message(Message(speaker, hedged))
    assert state.role_action_provenance.unsupported_role_action_count == 0
    assert state.conversation_log[-1].content == hedged


def test_claim_first_then_ability_reports_a_turn_distance_only() -> None:
    state = _day1_with_speeches(seed=4103)
    speaker = _names(state)[5]
    _cast(state, speaker, Role.SEER)
    state.discussion_turn = 2
    tracker = state.role_action_provenance
    tracker.mark_public_claim(speaker, day=1, discussion_turn=2)
    state.day = 2
    events: list[dict[str, object]] = []
    engine = _engine_with_events(state, events)
    engine._append_discussion_message(Message(speaker, BLUFF_SENTENCE))
    event = next(e for e in events if e["type"] == "discussion_message")
    delta = event.get("claim_to_ability_turn_delta")
    assert isinstance(delta, int) and delta > 0
    assert event.get("claim_to_ability_order") == "claim_then_ability"
    # 距離は数値だけ（どのCOが本物かを示す語はこのイベントに無い）
    blob = json.dumps(event, ensure_ascii=False)
    assert '"engine_supported"' not in blob


# ---------------------------------------------------------------------------
# 5. CO補正の再要求（能力実行記録・主張状態・現在質問・出力形式だけ）
# ---------------------------------------------------------------------------
class _NoClient:  # pragma: no cover - このテストでは通信しない
    def chat(self, *args, **kwargs):
        raise AssertionError("補正プロンプトの検証でLLMを呼んではならない")


def _actions(state: GameState, events: list[dict[str, object]]) -> OllamaActions:
    return OllamaActions(
        state, _NoClient(), seed=state.seed, event_callback=events.append
    )


def test_correction_prompt_separates_engine_record_from_own_claim() -> None:
    state = _day1_with_speeches(seed=4201)
    events: list[dict[str, object]] = []
    actions = _actions(state, events)
    bluffing = _cast(state, _names(state)[4], Role.MADMAN)

    text = actions._seer_co_correction_prompt(bluffing, None)
    assert "[DONE]" in text and "[CLAIM]" in text
    assert "never executed a divination action" in text
    # 未保有の能力を「エンジンが実行した」とは書かない／出していない結果を促さない
    assert "your engine-issued result" not in text
    assert "[RESULT]" not in text
    # 長い旧プロンプトは再送しない（証拠欄・理論カード・ルール本文を再び載せない）
    for heavy in ("[CONVERSATION_EVIDENCE", "Werewolf Theory", "[SYSTEM_RULES]"):
        assert heavy not in text


def test_correction_prompt_shows_result_only_to_the_true_holder() -> None:
    state = _day1_with_speeches(seed=4202)
    seer_name = _names(state)[4]
    _cast(state, seer_name, Role.SEER)
    state.seer_results["Player 6"] = "wolf"
    state.seer_result_history.append((1, "Player 6", "wolf"))
    state.role_action_provenance.record_engine_action(seer_name, KIND_DIVINATION, 1)
    events: list[dict[str, object]] = []
    actions = _actions(state, events)
    text = actions._seer_co_correction_prompt(_player(state, seer_name), None)
    assert "Engine executed divination for you" in text
    assert "your engine-issued result" in text
    assert "Player 6" in text  # 本人分の結果だけ（他者分は載せない）


def test_correction_prompt_stays_far_shorter_than_the_discussion_prompt() -> None:
    state = _day1_with_speeches(seed=4203)
    actor_name = _names(state)[4]
    _cast(state, actor_name, Role.MADMAN)
    events: list[dict[str, object]] = []
    actions = _actions(state, events)
    correction = actions._seer_co_correction_prompt(_player(state, actor_name), None)
    normal = _ctx(state, actor_name, "v5")
    assert len(correction) < len(normal.user_prompt) * 0.4
    # 現在の質問は拾う（長文プロンプトを再送せずに質問だけ渡す）
    assert "Current question to answer" in correction


def test_correction_prompt_keeps_options_within_the_bluff_policy() -> None:
    state = _day1_with_speeches(seed=4204)
    actor_name = _names(state)[4]
    _cast(state, actor_name, Role.MADMAN)
    events: list[dict[str, object]] = []
    actions = _actions(state, events)
    spoken = ("Player 1", "wolf")

    allowed = actions._seer_co_correction_prompt(_player(state, actor_name), spoken)
    assert "unverified" in allowed and "[CO] Seer" in allowed
    assert "Option B" in allowed  # 結果行を外す/答える選択肢も残す（結論は強制しない）

    state.allow_bluff_role_claim = False
    denied = actions._seer_co_correction_prompt(_player(state, actor_name), spoken)
    assert "disallows publishing a role you do not hold" in denied


# ---------------------------------------------------------------------------
# 6. ログ集約: 公開ログは中立な数値だけ、分類の真偽は管理者レコード専用
# ---------------------------------------------------------------------------
def test_logger_aggregates_boundary_metrics_and_keeps_truth_admin_only(
    tmp_path,
) -> None:
    import json as _json

    from werewolf.cli import GameLogger

    state = _day1_with_speeches(seed=4301)
    actor_name = _names(state)[4]
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
    logger = GameLogger("task064", tmp_path, {"seed": 4301}, private_enabled=True)

    # 1) v5配置の実測（境界節の字数・現在質問と自分の発言との距離）
    logger.append_record(
        {
            **base,
            "type": "discussion_context_selected",
            "prompt_layout": "v5_role_boundary",
            "prompt_user_char_count": 13900,
            "role_boundary_section_chars": 180,
            "prompt_gap_role_boundary_to_question": 0.0099,
            "prompt_gap_role_boundary_to_own_speeches": 0.0,
        }
    )

    # 2) 公開発言の観測（cliの on_engine_event と同じ形で公開側は真偽キーを落とす）
    bluff_event = {
        **base,
        "type": "discussion_message",
        "speaker": actor_name,
        "content": BLUFF_SENTENCE,
        "role_ability_observations": [
            {
                "player": actor_name,
                "day": 1,
                "discussion_turn": 3,
                "kind": KIND_DIVINATION,
                "assertion": True,
                "supported": False,
                "category": CATEGORY_UNSUPPORTED_ROLE_ACTION,
                "conflict": False,
                "interpretation": "public_claim_line",
            }
        ],
        "role_action_model": model,
        "claim_to_ability_turn_delta": 4,
        "claim_to_ability_order": "claim_then_ability",
    }
    public_view = {
        key: value
        for key, value in bluff_event.items()
        if key
        not in {
            "role_ability_observations",
            "claim_to_ability_turn_delta",
            "claim_to_ability_order",
        }
    }
    logger.append_record(public_view, bluff_event)

    # 3) CO初出→能力表現のターン距離（中立な数値だけのイベント）
    logger.append_record(
        {
            **base,
            "type": "claim_to_ability_distance",
            "turn_delta": 6,
            "order": "ability_then_claim",
        }
    )

    # 4) CO補正プロンプトの実測（短縮ぶんと、追加された能力表現ゼロの記録）
    logger.append_record(
        {
            **base,
            "type": "co_correction_boundary",
            "reask_prompt_chars": 900,
            "normal_prompt_chars": 13000,
            "ability_assertions_before": 1,
            "ability_assertions_after": 1,
            "ability_assertions_added": 0,
            "engine_backed": False,
        }
    )

    # 台帳側にも本人の実行記録と婉曲観測を積んで finalize する
    tracker = state.role_action_provenance
    tracker.record_engine_action(actor_name, KIND_DIVINATION, 1)
    tracker.observe_public_speech(
        player=actor_name,
        day=1,
        discussion_turn=2,
        text="もし私が占ったのは Player 3 かもしれません。私は実際に占ってはいません。",
        known_player_names=_names(state),
    )
    logger.finalize(state)

    rows = [
        _json.loads(line)
        for line in (tmp_path / "game_task064.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    result = next(row for row in rows if row.get("type") == "game_result")
    metrics = result["ledger_metrics"]
    assert metrics["role_boundary_sections"] == 1
    assert metrics["avg_claim_boundary_chars"] == 180.0
    assert metrics["avg_gap_role_boundary_to_question"] == 0.0099
    assert metrics["avg_gap_role_boundary_to_own_speeches"] == 0.0
    assert metrics["unsupported_role_action_count"] == 1
    assert metrics["role_ability_assertions_observed"] == 1
    assert metrics["co_correction_events"] == 1
    assert metrics["avg_co_correction_reask_prompt_chars"] == 900.0
    assert metrics["co_correction_shortening_ratio"] == 0.0692
    assert metrics["retry_added_ability_expressions"] == 0
    # CO初出→能力表現の距離は 4（観測経由）と 6（距離イベント）の平均
    assert metrics["avg_claim_to_ability_turn_delta"] == 5.0
    assert metrics["hedged_ability_mentions_observed"] >= 1

    stats = result["model_stats"][model]
    assert stats["unsupported_role_action_count"] == 1
    assert stats["avg_claim_boundary_chars"] == 180.0
    assert stats["avg_gap_role_boundary_to_question"] == 0.0099
    assert stats["avg_co_correction_reask_prompt_chars"] == 900.0
    assert stats["co_correction_shortening_ratio"] == 0.0692

    # 公開ログには分類の真偽キーが一切残らない（管理者レコード専用の線引き）
    public_text = (tmp_path / "game_task064.jsonl").read_text(encoding="utf-8")
    # （中立な合計値 engine_actions_recorded は出してよい。保有者別の内訳を出さない）
    for forbidden in (
        "role_ability_observations",
        '"supported"',
        '"category"',
        '"conflict"',
        '"per_player"',
        '"engine_actions":',
    ):
        assert forbidden not in public_text
    # 公開側の game_result に載る境界台帳は中立合計のみ
    public_blob = _json.dumps(result["role_action_provenance"], ensure_ascii=False)
    assert "per_player" not in public_blob
    assert result["role_action_provenance"]["metrics"]["engine_actions_recorded"] == 1

    private_text = (tmp_path / "game_task064.private.jsonl").read_text(encoding="utf-8")
    assert "engine_actions" in private_text and "per_player" in private_text



