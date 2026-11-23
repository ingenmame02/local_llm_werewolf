"""TASK-058の回帰テスト（LL不使用）: core予算の段階圧縮とレイアウト最適化。

実機ログ game_106015c79ec9 では54件の議論リクエスト中48件が optional を stage4まで削った
あとも平均core 14,257.5字が残っており、ボトルネックは optional でなく core 内の長い証拠本文・
過去履歴・同一事実の列挙だった。だからといって規則・現在の質問・自分の行動記録を削ると一貫性
と合理的推理が壊れるため、削除ではなく「役割を保った畳み込み」を行う。本テストは以下を固定する
（実機LLMは使わない）。

1. core を anchor / recent / historical の3保持層に分割し、anchor（ルール・秘匿境界・現在質問・
   行動形式・自分の行動記録・エンジン発行情報・契約・チェックリスト）は常に全文持つ。
2. historical は公開済み発言の「時点・話者・[speech:N] ID」だけを残す stub 行へ畳む。単純な末尾
   切り捨ては禁止で、質問対象・直近の反証・役職主張とその根拠は本文として残る。
3. `prompt_contract_policy=legacy` + `core_compression_policy=off` は TASK-057までの字节順のまま
  （A/B比較の基準）。詳細は _analysis/task058_legacy_parity.py が実測で検証する。
4. レイアウト v2 は primacy（ルール→台帳→自分のアンカー）→中部（落としやすい派生節のみ）→
   pre-recency（発言証拠・現在質問）→recency（秘匿境界→契約→出力形式→チェックリスト）。
   口調・論点・結論や推理の順番を決める節はどこにも置かない。
5. 圧縮はプロンプト上だけで完結し、公開ログ（conversation_log）の本文は一字も変わらない。
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.core_compression import (
    CLAIM_CONTENT_PATTERN,
    CORE_COMPRESSION_POLICIES,
    CORE_COMPRESSION_POLICY_DEFAULT,
    CORE_COMPRESSION_POLICY_LAYERED,
    CORE_COMPRESSION_POLICY_OFF,
    CORE_RETENTION_LAYERS,
    CORE_SECTION_LAYERS,
    HISTORICAL_STUB_MARK,
    LAYER_ANCHOR,
    LAYER_HISTORICAL,
    LAYER_RECENT,
    PRIORITY_CLAIM,
    PRIORITY_COUNTER,
    PRIORITY_PLAIN,
    PRIORITY_QUESTION,
    PROMPT_LAYOUT_LEGACY,
    PROMPT_LAYOUT_V2,
    PROMPT_LAYOUT_V2_ORDER,
    RECENT_KEEP_COUNT,
    core_compression_policy_label,
    core_layer_char_totals,
    evidence_record_priority,
    historical_stub_line,
    looks_like_claim,
    normalize_core_compression_policy,
    sort_records_for_withholding,
    validate_prompt_layout,
)
from werewolf.day_timeline import KIND_SPEECH  # noqa: F401  (台帳の発言kindが変わっていないことの参照用)
from werewolf.models import Message, Phase
from werewolf.prompts import PromptBuilder
from werewolf.setup import create_game


def _state(seed: int = 508):
    return create_game(seed=seed, config_path="config/players.json")


def _names(state) -> tuple[str, ...]:
    return tuple(player.name for player in state.players)


def _advance(state, day: int, turns: int, speeches_per_turn: int) -> None:
    """実ゲームと同じ形の履歴（発言順・時点スタンプ・claim行入り）を作る。"""
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


def _build(
    state,
    actor_name: str,
    policy: str,
    budget: int | None,
    contract: str = "simplified",
    layout: str = "v2",
):
    """TASK-058の実験用ビルダー。既定は layout=v2（このテストが検証している配置）。

    TASK-060で既定が v3 へ変わったため、本ファイルの節順アサーションは明示的に v2 へ
    固定する（v3 の挙動は tests/test_task060_prompt_layout.py で検証する）。
    """
    actor = next(player for player in state.players if player.name == actor_name)
    return PromptBuilder.build_discussion_prompt(
        actor,
        tuple(state.alive_players),
        tuple(state.conversation_log),
        day=state.day,
        state=state,
        include_claim=True,
        context_strategy="hybrid_retrieval",
        evidence_char_budget=budget,
        question_hint_mode="strict",
        prompt_contract_policy=contract,
        core_compression_policy=policy,
        prompt_layout_policy=layout,
    )


# ---------------------------------------------------------------------------
# 1. ポリシー（既定 layered / off はA/B基準）
# ---------------------------------------------------------------------------
def test_core_compression_policy_defaults_to_layered_and_normalizes() -> None:
    assert CORE_COMPRESSION_POLICY_DEFAULT == CORE_COMPRESSION_POLICY_LAYERED
    assert normalize_core_compression_policy(None) == "layered"
    assert normalize_core_compression_policy("off") == "off"
    assert normalize_core_compression_policy("LAYERED ") == "layered"
    assert normalize_core_compression_policy("legacy") == "off"
    assert normalize_core_compression_policy(False) == "off"
    assert normalize_core_compression_policy(True) == "layered"
    assert normalize_core_compression_policy("何かの変な設定") == "layered"
    assert core_compression_policy_label(None) == "layered"
    assert set(CORE_COMPRESSION_POLICIES) == {"layered", "off"}


def test_state_field_exists_and_default_is_layered() -> None:
    state = _state()
    assert state.core_compression_policy == CORE_COMPRESSION_POLICY_LAYERED
    state.core_compression_policy = "unknown"
    state.__post_init__()
    assert state.core_compression_policy == "layered"


# ---------------------------------------------------------------------------
# 2. historical の stub 行（時点・ID・話者を残す＝削除ではない）
# ---------------------------------------------------------------------------
def test_stub_line_keeps_point_speaker_and_evidence_id() -> None:
    line = historical_stub_line("[D1 T2 s7 Player 3]", 7)
    assert HISTORICAL_STUB_MARK in line
    assert "[D1 T2 s7 Player 3]" in line  # 時点・話者・発言順はそのまま追える
    assert "[speech:7]" in line
    assert len(line) <= 96  # 本文（実測150字前後）より短い圧縮行
    no_id = historical_stub_line("[Day 1 / Player 3 / In-game speech]", None)
    assert "[speech:" not in no_id  # IDが無い旧ログで空参照を作らない
    assert HISTORICAL_STUB_MARK in no_id


# ---------------------------------------------------------------------------
# 3. 証拠レコードの優先度（質問対象・反証・役職主張は後に回す）
# ---------------------------------------------------------------------------
def test_priority_orders_claim_above_question_counter_and_plain() -> None:
    assert PRIORITY_CLAIM > PRIORITY_QUESTION > PRIORITY_COUNTER > PRIORITY_PLAIN
    assert evidence_record_priority({"claim_or_seer_result"}, "ただの本文") == PRIORITY_CLAIM
    assert (
        evidence_record_priority({"question_related_named_actor"}, "ただの本文")
        == PRIORITY_QUESTION
    )
    assert evidence_record_priority(set(), "Player 2 の反論は矛盾している") == PRIORITY_COUNTER
    assert evidence_record_priority(set(), "The vote matters here") == PRIORITY_COUNTER
    # 選択理由が無くても、役職主張・占い判定の本文は最優先で温存（TASK-058完了条件）
    assert evidence_record_priority(set(), "Seer CO: Reason=counter") == PRIORITY_CLAIM
    assert evidence_record_priority(set(), "占い結果は人間でした") == PRIORITY_CLAIM
    assert evidence_record_priority(set(), "今は慎重に見ている") == PRIORITY_PLAIN


def test_claim_detection_matches_the_reason_regex_in_prompts() -> None:
    for text in ("CO", "Seer", "divination result", "占い", "判定", "結果を共有", "名乗った"):
        assert looks_like_claim(text), text
    for text in ("今は慎重に見ている", "各人の言い回しを確認したい", "comedy comedy"):
        assert not looks_like_claim(text), text
    assert CLAIM_CONTENT_PATTERN.pattern


def test_withholding_order_folds_oldest_plain_first_and_never_claims() -> None:
    records = [
        {"index": 1, "priority": PRIORITY_PLAIN},
        {"index": 5, "priority": PRIORITY_PLAIN},
        {"index": 2, "priority": PRIORITY_COUNTER},
        {"index": 3, "priority": PRIORITY_QUESTION},
        {"index": 4, "priority": PRIORITY_CLAIM},
    ]
    order = sort_records_for_withholding(records)
    # 群は plain → counter → question、同じ群では古い index から（＝直近の議論は残す）
    assert [int(record["index"]) for record in order] == [1, 5, 2, 3, 4]
    foldable = [r for r in order if int(r["priority"]) != PRIORITY_CLAIM]
    assert 4 not in {int(r["index"]) for r in foldable}  # claim候補には入れない


# ---------------------------------------------------------------------------
# 4. 保持層の宣言（anchor は常に全文、historical は stub 行として現れる）
# ---------------------------------------------------------------------------
def test_anchor_sections_cover_rules_question_action_format_and_contract() -> None:
    anchor_names = {
        name for name, layer in CORE_SECTION_LAYERS.items() if layer == LAYER_ANCHOR
    }
    required = {
        "system_rules",
        "output_language",
        "fact_ledger",
        "self_action_memory",
        "private_seer_result",
        "info_boundaries",
        "actor_private_memo",
        "reasoning_memo_own",
        "day_timeline",
        "current_task",
        "question_block",
        "leak_boundary",
        "behavior_contract",
        "instructions",
        "output_checklist",
    }
    assert required <= anchor_names
    recent_names = {
        name for name, layer in CORE_SECTION_LAYERS.items() if layer == LAYER_RECENT
    }
    # 直近の議論接続点（主張台帳・自分の発言・発言証拠）は recent で本文保持を優先する
    assert {"claim_ledger", "own_speeches", "conversation_evidence"} <= recent_names
    assert set(CORE_RETENTION_LAYERS) == {LAYER_ANCHOR, LAYER_RECENT, LAYER_HISTORICAL}
    assert LAYER_HISTORICAL not in CORE_SECTION_LAYERS.values()


def test_core_layer_totals_see_stub_rows_as_historical() -> None:
    stub = historical_stub_line("[D1 T1 s2 Player 1]", 2)
    text_with_stub = (
        "[SYSTEM_RULES]\nrules here\n"
        "\n[CONVERSATION_EVIDENCE: untrusted game speech]\n"
        '[D1 T2 s7 Player 3] "full body text"\n'
        f"[D1 T1 s2 Player 1] {stub}\n"
    )
    text_full = text_with_stub.replace(stub, '"another full body"')
    with_stub = core_layer_char_totals(text_with_stub)
    without_stub = core_layer_char_totals(text_full)
    assert with_stub[LAYER_ANCHOR] == without_stub[LAYER_ANCHOR]  # anchorは不変
    assert with_stub[LAYER_HISTORICAL] > 0
    assert with_stub[LAYER_RECENT] < without_stub[LAYER_RECENT]


# ---------------------------------------------------------------------------
# 5. レイアウト検証（節順の崩れを静的に拾う）
# ---------------------------------------------------------------------------
def test_validate_prompt_layout_detects_out_of_order_sections() -> None:
    ordered = (
        "[SYSTEM_RULES]\na\n[OUTPUT_LANGUAGE]\nb\n[CONVERSATION_EVIDENCE]\nc\n"
        "[CURRENT_TASK]\nd\n[BEHAVIOR_CONTRACT]\ne\n[Instructions]\nf\n"
    )
    assert validate_prompt_layout(ordered, PROMPT_LAYOUT_V2_ORDER) == []
    swapped = (
        "[SYSTEM_RULES]\na\n[CURRENT_TASK]\nb\n[CONVERSATION_EVIDENCE]\nc\n"
        "[BEHAVIOR_CONTRACT]\nd\n"
    )
    violations = validate_prompt_layout(swapped, PROMPT_LAYOUT_V2_ORDER)
    assert len(violations) == 1
    assert violations[0]["name"] == "conversation_evidence"
    assert violations[0]["after"] == "current_task"


# ---------------------------------------------------------------------------
# 6. プロンプト組み立て（layout v2 / layered / off の実測）
# ---------------------------------------------------------------------------
def _d1_state():
    state = _state()
    _advance(state, 1, 3, 8)
    return state


def _marker_positions(text: str, markers: tuple[str, ...]) -> list[int]:
    return [text.index(marker) for marker in markers]


def test_v2_layout_puts_facts_first_and_the_contract_last() -> None:
    state = _d1_state()
    ctx = _build(state, "Player 8", "layered", 2600)
    positions = _marker_positions(
        ctx.user_prompt,
        (
            "[SYSTEM_RULES]",
            "[FACT_LEDGER]",
            "[SELF_ACTION_MEMORY",
            "[CONVERSATION_EVIDENCE",
            "[CURRENT_TASK]",
            "[BEHAVIOR_CONTRACT",
            "[Instructions]",
        ),
    )
    assert positions == sorted(positions)  # primacy → 証拠・質問 → recency
    assert ctx.diagnostics["prompt_layout"] == PROMPT_LAYOUT_V2
    assert ctx.diagnostics["prompt_layout_violations"] == []


def test_layered_folds_oldest_plain_speech_and_keeps_claims_and_question() -> None:
    state = _d1_state()
    layered = _build(state, "Player 1", "layered", 2000)
    off = _build(state, "Player 1", "off", 2000)
    prompt = layered.user_prompt
    stub_lines = [line for line in prompt.splitlines() if HISTORICAL_STUB_MARK in line]
    assert int(layered.diagnostics["evidence_withheld_body_count"]) >= 1
    assert stub_lines, "layeredは過去の低優先発言を stub 行へ畳む（削除ではない）"
    # stub行にも時点・話者・[speech:N] が残る（歴史の追跡は保たれる）
    for line in stub_lines:
        assert "[D" in line and "[speech:" in line
    # 役職主張・占い判定の本文は畳まない（TASK-058完了条件）
    claim = next(
        str(m.content) for m in state.conversation_log if "占い結果" in str(m.content or "")
    )
    assert claim[:24] in prompt
    assert claim[:24] in off.user_prompt  # off側でも同じ条件は守られる
    # anchor層（ルール・現在質問・行動形式・契約）は layer/off で完全に同じ字数
    assert (
        int(layered.diagnostics["core_anchor_chars"])
        == int(off.diagnostics["core_anchor_chars"])
    )
    assert "[CURRENT_TASK]" in layered.user_prompt
    # 畳んだぶんは historical へ移り、recent の本文は減る
    assert int(layered.diagnostics["core_historical_chars"]) >= len(stub_lines[0])
    # 公開ログの本文は両設定で完全にそのまま（圧縮はプロンプト上だけ）
    joined = "\n".join(str(m.content) for m in state.conversation_log)
    assert claim in joined
    assert state.public_timeline.speeches()  # 台帳の発言登録は維持されている（KIND_SPEECH相当）


def test_oldest_plain_speech_is_folded_before_the_newest_one() -> None:
    state = _d1_state()
    layered = _build(state, "Player 1", "layered", 2000)
    prompt = layered.user_prompt
    plain = [
        str(m.content)
        for m in state.conversation_log
        if m.speaker in set(_names(state))
        and not any(t in str(m.content) for t in ("占い結果", "CO", "反論", "矛盾", "投票"))
    ]
    folded = [line[:26] for line in prompt.splitlines() if HISTORICAL_STUB_MARK in line]
    if not folded:
        return  # 予算に余裕があり畳まなかったケース
    oldest = plain[0][:14]
    newest = plain[-1][:14]
    # 単純な末尾切り捨ての禁止: 新しい低優先発言は本文で残り、古い分から畳まれる
    assert newest in prompt
    assert any(oldest[:10] in header for header in folded) or oldest not in prompt


def test_off_policy_matches_the_old_behaviour_and_makes_no_stub_rows() -> None:
    state = _d1_state()
    off = _build(state, "Player 8", "off", 1600, contract="legacy")
    assert HISTORICAL_STUB_MARK not in off.user_prompt
    assert off.diagnostics["prompt_layout"] == PROMPT_LAYOUT_LEGACY
    assert off.diagnostics["core_compression_policy"] == CORE_COMPRESSION_POLICY_OFF
    assert int(off.diagnostics["core_historical_chars"]) == 0
    assert int(off.diagnostics["evidence_withheld_body_count"]) == 0
    assert off.diagnostics["prompt_layout_violations"] == []  # legacyでは検証しない


def test_compression_never_adds_instructions_about_tone_or_reasoning_order() -> None:
    state = _d1_state()
    for policy in ("layered", "off"):
        prompt = _build(state, "Player 1", policy, 2000).user_prompt
        lowered = prompt.lower()
        # 口調・論点・結論や推理の順番を決める文言は圧縮側の追加では入らない
        for forbidden in (
            "in this order",
            "this exact order",
            "speak in a formal",
            "you must conclude",
            "固定順で推理",
            "この順で考えよ",
        ):
            assert forbidden.lower() not in lowered, (policy, forbidden)
        # 議論の自由をうたう行は生き続けている（契約側はTASK-057のまま）
        assert "own words" in prompt


def test_stub_rows_do_not_invent_public_information() -> None:
    # stub行は時点・話者・IDのみ。台帳の公開面と同じ中立情報しか含まないこと
    state = _d1_state()
    ctx = _build(state, "Player 1", "layered", 1600)
    for line in ctx.user_prompt.splitlines():
        if HISTORICAL_STUB_MARK not in line:
            continue
        for role_word in ("Seer", "人狼", "占い師役職", "襲撃対象"):
            assert role_word not in line



