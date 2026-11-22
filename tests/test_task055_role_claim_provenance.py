"""TASK-055の回帰テスト（LL不使用）: 役職主張と役職由来情報のプロヴェナンス分離。

実ログ game_15b1bdf65773 の再現が動機。真占い師のエンジン発行判定は1件だけなのに、
Day 1 で3人が占い師COし、そのうち2人が「Player 1 = 狼」という捏造判定を公開した。
当時の台帳では「実際に占い師が持っていた情報」と「発言で申告されただけの行」が同じ形
（`claim_ledger.public_seer_results`）で並び、他プレイヤーの推理は捏造判定を準確定の
根拠として扱ってしまった。本テストは以下を固定する（実機LLMは使わない）。

1. エンジンが発行した情報は `RoleEvidence`（保有者名つきの `private_action`）だけに載る。
2. 公開されたCO行・結果行は出所つき（`public_claim` / `compatibility_text`）で分離される。
3. `allow_bluff_role_claim` は既定 True（ブラフ自由）。False のときだけ CO・結果行を
   公開前に除去し、本文は一字一句そのまま残す（再生成も口調の強制もしない）。
4. 公開面（プロンプト・公開ログ）には全CO行を例外なく中立ラベル `public_claim_line` で
   出し、エンジン照合の真偽は管理者ログ専用にする（真偽が漏れると推論ゲームが壊れる）。
5. 捏造判定は `state.seer_results`（本物の占い結果）・確定FactLedger・他人の非公開情報へ
   書き込まれない。
6. 本人向けの一貫性欄は実際に知っている情報と自分の公開主張を分けて数行で示し、
   口調・論点・結論を指示しない。
"""
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from werewolf.engine import GameEngine, extract_public_role_claim
from werewolf.llm_actions import OllamaActions
from werewolf.models import GameState, Message, Role
from werewolf.ollama import ToolCall
from werewolf.prompts import PromptBuilder
from werewolf.role_provenance import (
    ALLOW_BLUFF_ROLE_CLAIM_DEFAULT,
    CATEGORY_DISALLOWED,
    CATEGORY_ENGINE_ISSUED,
    CATEGORY_SELF_REPORTED,
    CATEGORY_UNSUPPORTED,
    PUBLIC_LINE_LABEL,
    SOURCE_COMPATIBILITY_TEXT,
    SOURCE_PRIVATE_ACTION,
    SOURCE_PUBLIC_CLAIM,
    RoleProvenanceLedger,
    bluff_role_claim_policy_label,
    normalize_allow_bluff_role_claim,
    normalize_provenance_source,
)
from werewolf.setup import create_game

# game_15b1bdf65773 Day 1 で実際に公開された形（村人の捏造占いCO）
BLUFF_SEER_PUBLICATION = (
    "[CO] Seer CO\n"
    "[RESULT] Target=Player 1 Result=Werewolf\n"
    "Player 1さんの言動は初日から不自然で、夜の行動の説明が毎回ずれています。"
)


def _state() -> GameState:
    return create_game(
        seed=1001, config_path="config/players.json", response_language="日本語"
    )


def _player(state: GameState, name: str):
    return next(player for player in state.players if player.name == name)


def _cast(state: GameState, name: str, role: Role) -> object:
    """検証したい役職へ振り替える（CO台帳の挙動確認に使う）。"""
    player = _player(state, name)
    player.role = role
    player.partner = None
    return player


def _engine(state: GameState, events: list[dict[str, object]]) -> GameEngine:
    class _NoActions:
        def wolf_discuss(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("このテストでは夜行動を実行しない")

    return GameEngine(state, _NoActions(), seed=state.seed, event_callback=events.append)


def _seer_evidence(state: GameState, owner: str, target: str, result: str) -> str:
    """エンジンが発行した占い結果（夜行動相当）を台帳へ登録する。"""
    return state.role_provenance.record_engine_divination(
        owner=owner, day=state.day, target=target, result=result
    )


# ---------------------------------------------------------------------------
# 1. 台帳そのもの: エンジン発行情報と公開申告の出所を分ける
# ---------------------------------------------------------------------------
def test_engine_issued_evidence_carries_owner_and_private_source() -> None:
    ledger = RoleProvenanceLedger()
    evidence_id = ledger.record_engine_divination(
        owner="Player 5", day=2, target="Player 1", result="human"
    )
    record = ledger.evidence[-1]
    assert record.evidence_id == evidence_id
    assert record.owner == "Player 5"
    assert record.source == SOURCE_PRIVATE_ACTION
    # 再実行（同じ夜の対象）は統合し、判定が変わった場合だけ別行として残す
    again = ledger.record_engine_divination(
        owner="Player 5", day=2, target="Player 1", result="human"
    )
    assert again == evidence_id and len(ledger.evidence) == 1
    changed = ledger.record_engine_divination(
        owner="Player 5", day=2, target="Player 1", result="wolf"
    )
    assert changed == evidence_id
    assert {item.result for item in ledger.evidence} == {"human", "wolf"}


def test_bluff_verdict_is_unsupported_but_true_seer_verdict_matches() -> None:
    ledger = RoleProvenanceLedger()
    ledger.record_engine_divination(
        owner="Player 5", day=1, target="Player 1", result="human"
    )
    # 村人が「Player 1 は狼」と申告してもエンジン発行の判定とは一致しない
    bluff = ledger.record_claim(
        claim_ref="[claim:1]",
        day=1,
        claimant="Player 4",
        claimed_role="seer",
        source=SOURCE_COMPATIBILITY_TEXT,
        actor_role=Role.VILLAGER.value,
        target="Player 1",
        result="wolf",
    )
    assert bluff.category == CATEGORY_UNSUPPORTED
    assert bluff.engine_supported is False and bluff.engine_evidence_ids == ()
    # 真占い師本人の申告だけが engine_supported（この情報は管理者専用）
    real = ledger.record_claim(
        claim_ref="[claim:2]",
        day=2,
        claimant="Player 5",
        claimed_role="seer",
        source=SOURCE_PUBLIC_CLAIM,
        actor_role=Role.SEER.value,
        target="Player 1",
        result="human",
    )
    assert real.category == CATEGORY_ENGINE_ISSUED
    assert real.engine_supported is True
    # 他人の結果を引用しただけでは一致扱いにしない（推測は engine_supported にならない）
    other = ledger.record_claim(
        claim_ref="[claim:3]",
        day=2,
        claimant="Player 6",
        claimed_role="seer",
        source=SOURCE_PUBLIC_CLAIM,
        actor_role=Role.WOLF.value,
        target="Player 1",
        result="human",
    )
    assert other.category == CATEGORY_UNSUPPORTED
    assert ledger.admin_metrics()["unsupported_role_evidence_lines"] == 2
    assert ledger.admin_metrics()["engine_supported_verdict_lines"] == 1


def test_public_view_never_exposes_engine_match_and_labels_every_line() -> None:
    ledger = RoleProvenanceLedger()
    ledger.record_engine_divination(
        owner="Player 5", day=1, target="Player 1", result="human"
    )
    for index, (claimant, role_name) in enumerate(
        (("Player 4", "seer"), ("Player 5", "seer"), ("Player 2", "villager")), 1
    ):
        ledger.record_claim(
            claim_ref=f"[claim:{index}]",
            day=1,
            claimant=claimant,
            claimed_role=role_name,
            source=SOURCE_COMPATIBILITY_TEXT,
            actor_role="villager",
            target="Player 1" if role_name == "seer" else None,
            result="wolf" if role_name == "seer" else None,
        )
    public = json.dumps(ledger.serialize_public(), ensure_ascii=False)
    # 公開ログには真偽を示す語を一切出さない（中立ラベルだけ）
    for forbidden in (
        "engine_supported",
        "unsupported_role_evidence",
        '"category"',
        "private_action",
    ):
        assert forbidden not in public
    assert PUBLIC_LINE_LABEL in public
    # 全CO行が同じ読み方（本人申告）に揃っている
    views = [item.public_view() for item in ledger.interpretations]
    assert {view["interpretation"] for view in views} == {PUBLIC_LINE_LABEL}
    assert {view["source"] for view in views} == {SOURCE_COMPATIBILITY_TEXT}
    # 管理者ビューでは初めて真偽とエンジンIDが読める
    admin_rows = {row["claim_ref"]: row for row in ledger.serialize_admin()["claims"]}
    assert admin_rows["[claim:1]"]["category"] == CATEGORY_UNSUPPORTED
    assert ledger.evidence[0].evidence_id not in admin_rows["[claim:1]"]["engine_evidence_ids"]
    assert ledger.admin_metrics()["verdict_lines_self_reported"] == 2


def test_disallowed_rows_stay_out_of_the_public_serialization() -> None:
    """不受理行を役職名つきで公開すると「実保有者でないこと」が漏れるため管理者専用にする。"""
    ledger = RoleProvenanceLedger()
    ledger.record_claim(
        claim_ref="[claim:1]",
        day=1,
        claimant="Player 5",
        claimed_role="seer",
        source=SOURCE_PUBLIC_CLAIM,
        actor_role="seer",
    )
    ledger.record_disallowed_claim(
        day=1,
        claimant="Player 4",
        claimed_role="seer",
        source=SOURCE_PUBLIC_CLAIM,
    )
    public = ledger.serialize_public()
    assert [row["claimant"] for row in public["claims"]] == ["Player 5"]
    assert json.dumps(public, ensure_ascii=False).count("Player 4") == 0
    # 中立カウンタだけ立つ（誰が試みたかは判らない）
    assert public["metrics"]["role_claims_disallowed"] == 1
    admin = ledger.serialize_admin()
    assert "Player 4" in json.dumps(admin["claims"], ensure_ascii=False)


def test_source_and_policy_normalization() -> None:
    assert ALLOW_BLUFF_ROLE_CLAIM_DEFAULT is True
    assert normalize_allow_bluff_role_claim("allow") is True
    assert normalize_allow_bluff_role_claim("deny") is False
    assert normalize_allow_bluff_role_claim(False) is False
    assert normalize_allow_bluff_role_claim(None) is True  # 不明値は既定の許可
    assert bluff_role_claim_policy_label(False) == "deny"
    # ツール由来のCOも「本人の公開主張」に寄せる（エンジン発行の情報にはしない）
    assert normalize_provenance_source("tool") == SOURCE_PUBLIC_CLAIM
    assert normalize_provenance_source("private_action") == SOURCE_PRIVATE_ACTION
    assert normalize_provenance_source("何かの変な語") == SOURCE_COMPATIBILITY_TEXT
    assert CATEGORY_DISALLOWED != CATEGORY_SELF_REPORTED


# ---------------------------------------------------------------------------
# 2. エンジン受理: ブラフCOは自由（既定）で、本物の占い結果には混ざらない
# ---------------------------------------------------------------------------
def test_default_policy_publishes_bluff_co_as_own_statement() -> None:
    state = _state()
    events: list[dict[str, object]] = []
    engine = _engine(state, events)
    _cast(state, "Player 4", Role.VILLAGER)
    # 村人が捏造占いCOをしても、CO自体はそのまま公開される（発言の自由を保つ）
    publication = engine._record_public_role_claim("Player 4", BLUFF_SEER_PUBLICATION)
    assert publication.role_published is True
    assert publication.result_published is True
    assert publication.role_claim_disallowed is False

    published = state.public_role_claims[-1]
    assert published.source == SOURCE_COMPATIBILITY_TEXT
    assert published.evidence_ids  # 台帳ID（公開用ID）を持つ
    # CO行と結果行が別IDで台帳に残る（TASK-047のID参照方式を踏襲し本文の再掲を増やさない）
    assert state.public_role_claims[0].evidence_ids == ("[claim:1]",)
    assert published.evidence_ids[0] == "[claim:2]"
    # 本物の占い結果（エンジン発行）は書き換わらない＝他人の推理の根拠にならない
    assert "Player 1" not in state.seer_results
    assert not any(
        fact.kind == "seer_result" and fact.description.startswith("Player 1")
        for fact in state.fact_ledger.confirmed_facts
    )

    provenance = state.role_provenance.interpretations[-1]
    assert provenance.claimant == "Player 4"
    assert provenance.target == "Player 1"
    assert provenance.category == CATEGORY_UNSUPPORTED
    assert provenance.engine_supported is False
    # 公開イベントには中立ラベルだけで、真偽を示す語を載せない
    claim_event = next(e for e in events if e["type"] == "claim_role_published")
    assert claim_event["interpretation"] == PUBLIC_LINE_LABEL
    assert claim_event["claim_ref"] == state.public_role_claims[0].evidence_ids[0]
    result_event = next(e for e in events if e["type"] == "seer_result_claim")
    assert result_event["interpretation"] == PUBLIC_LINE_LABEL
    assert result_event["claim_ref"] == published.evidence_ids[0]
    public_event_text = json.dumps(events, ensure_ascii=False)
    for forbidden in ("engine_supported", "unsupported_role_evidence"):
        assert forbidden not in public_event_text


def test_deny_policy_drops_only_the_claim_lines_and_keeps_body() -> None:
    state = _state()
    state.allow_bluff_role_claim = False
    events: list[dict[str, object]] = []
    engine = _engine(state, events)
    _cast(state, "Player 4", Role.VILLAGER)

    publication = engine._record_public_role_claim("Player 4", BLUFF_SEER_PUBLICATION)
    assert publication.role_claim_disallowed is True
    assert publication.role_published is False
    # 台帳（CO・占い結果申告）へ登録されない。本物の占い結果にも確定Factにもならない
    assert state.public_role_claims == []
    assert state.seer_claims == []
    assert state.claim_ledger.public_seer_results == {}
    assert state.fact_ledger.public_claims == []

    disallowed = state.role_provenance.interpretations[-1]
    assert disallowed.category == CATEGORY_DISALLOWED
    assert state.role_provenance.metrics()["role_claims_disallowed"] == 1
    # 不受理行を役職名つきで公開ログへ出さない（実保有者でないことが外から判ってしまう）
    public_log = state.role_provenance.serialize_public()
    assert public_log["claims"] == []
    assert public_log["metrics"]["role_claims_disallowed"] == 1
    assert state.role_provenance.serialize_admin()["claims"][-1]["category"] == CATEGORY_DISALLOWED

    event = next(e for e in events if e["type"] == "role_claim_disallowed")
    assert event["claimant"] == "Player 4"
    assert event["body_preserved"] is True
    # 試みた役職を公開イベントへ書かない（真偽が他者へ漏れないようにする）
    assert "claimed_role" not in event

    # 除去するのは CO行・結果行だけ。それ以外の本文は一字一句そのまま残る
    stripped = engine._strip_role_claim_text(BLUFF_SEER_PUBLICATION, "Player 4")
    assert "[CO]" not in stripped and "[RESULT]" not in stripped
    assert "Player 1さんの言動は初日から不自然で、夜の行動の説明が毎回ずれています。" in stripped


def test_deny_policy_still_publishes_true_holder_claim() -> None:
    state = _state()
    state.allow_bluff_role_claim = False
    events: list[dict[str, object]] = []
    engine = _engine(state, events)
    seer = _cast(state, "Player 1", Role.SEER)
    _seer_evidence(state, "Player 1", "Player 2", "human")

    publication = engine._record_public_role_claim(
        "Player 1",
        "[CO] Seer CO\n[RESULT] Target=Player 2 Result=Human\n占ったPlayer 2さんは人間でした。",
    )
    assert publication.role_published is True
    assert any(e["type"] == "claim_role_published" for e in events)
    verdict_rows = [
        item
        for item in state.role_provenance.interpretations
        if item.target == "Player 2" and item.result == "human"
    ]
    assert verdict_rows and verdict_rows[-1].category == CATEGORY_ENGINE_ISSUED
    # エンジン発行情報（RoleEvidence）の保有者は管理者ビューで初めて判る
    admin = state.role_provenance.serialize_admin()
    assert admin["evidence"][0]["owner"] == seer.name
    assert state.role_provenance.serialize_public()["metrics"]["role_claims_recorded"] >= 1


def test_free_text_body_keeps_everything_when_policy_allows() -> None:
    state = _state()
    events: list[dict[str, object]] = []
    engine = _engine(state, events)
    _cast(state, "Player 3", Role.WOLF)
    body = (
        "[CO] Villager CO\n"
        "夜も静かに動いていました。投票はPlayer 5さんに賛成です。"
    )
    publication = engine._record_public_role_claim("Player 3", body)
    # 狼の村人COも既定では自由（遮断しない）。出所は公開主張として記録される
    assert publication.role_published is True
    assert state.public_role_claims[-1].claimed_role == Role.VILLAGER.value
    provenance = state.role_provenance.interpretations[-1]
    assert provenance.category == CATEGORY_SELF_REPORTED
    assert provenance.source == SOURCE_COMPATIBILITY_TEXT
    # 許可設定では本文を触らない（除去・中性化を新たに行わない）
    assert engine._strip_role_claim_text(body, "Player 3") != ""

    # 不許可設定でも「主張句」だけ落ち、同じ文の後半（投票提案）は残る
    state.allow_bluff_role_claim = False
    english = "I am a Villager, and I propose voting for Player 5."
    assert extract_public_role_claim(english, "Player 3") == Role.VILLAGER.value
    stripped = engine._strip_role_claim_text(english, "Player 3")
    assert "Villager" not in stripped
    assert "I propose voting for Player 5." in stripped
    # COを含まない本文は一字も消えない
    plain = "Player 5さんに投票しましょう。理由を聞きたいです。"
    assert engine._strip_role_claim_text(plain, "Player 3") == plain


# ---------------------------------------------------------------------------
# 3. プロンプト: 全CO行を中立ラベルに揃え、本人向けの一貫性欄は短く保つ
# ---------------------------------------------------------------------------
def _alive(state: GameState) -> tuple[str, ...]:
    return tuple(player.name for player in state.players if player.is_alive)


def _publish_both_seer_claims(state: GameState) -> None:
    """捏造占いCO（村人）と本物の占いCO（占い師）を同じ公開形で登録する。"""
    events: list[dict[str, object]] = []
    engine = _engine(state, events)
    _seer_evidence(state, "Player 1", "Player 2", "human")
    _cast(state, "Player 4", Role.VILLAGER)
    _cast(state, "Player 1", Role.SEER)
    engine._record_public_role_claim("Player 4", BLUFF_SEER_PUBLICATION)
    engine._record_public_role_claim(
        "Player 1",
        "[CO] Seer CO\n[RESULT] Target=Player 2 Result=Human\n昨夜占ったPlayer 2さんは人間でした。",
    )


def test_claim_ledger_labels_every_line_neutrally() -> None:
    state = _state()
    _publish_both_seer_claims(state)
    text = PromptBuilder._discussion_claim_ledger(state)
    assert "Self-reported Verdict" in text
    assert "Provenance:" in text
    # 捏造COと本物のCOで表記を変えない（真偽が推測できてしまうため）
    assert text.count("(claimed by this player only)") == 2
    for forbidden in ("engine_supported", "unsupported_role_evidence", "private_action"):
        assert forbidden not in text
    # ID参照は保たれる（本文の再掲を増やさない＝コンテキスト節約）
    assert "[seer_claim:" in text and "[claim:" in text


def test_info_boundary_section_separates_known_from_claimed() -> None:
    state = _state()
    _publish_both_seer_claims(state)
    claimant = _cast(state, "Player 4", Role.VILLAGER)
    text = PromptBuilder._info_boundary_section(claimant, state)
    assert "[情報の境界" in text
    # 実際に知っている非公開情報と、自分が公開済みの主張が別欄で分かる
    assert "エンジンが発行したもの" in text
    assert "Player 1=Werewolf" in text
    # 結論・口調・論点を決める指示にはしない（議論の自由を保つ）
    assert "口調・論点・結論を決めない" in text
    # コンテキスト節約: 5行前後に収め、発言本文を再掲しない
    assert len(text) < 600, f"info boundary too long: {len(text)}"
    assert "言動は初日から不自然" not in text
    # 他人のエンジン発行情報（保有者名）をここから読み取れない
    assert "engine_seer" not in text
    # 設定（ブラフ可/不可）を本人へ伝える
    assert "どの役職のCOも自由" in text
    state.allow_bluff_role_claim = False
    deny_text = PromptBuilder._info_boundary_section(claimant, state)
    assert "実保有者だけがCOと判定を公開できる" in deny_text
    # 英語話者でも同じ枠（真偽キーは出さない）
    english_player = _cast(state, "Player 7", Role.VILLAGER)
    english_player.response_language = "English"
    english_text = PromptBuilder._info_boundary_section(english_player, state)
    assert "[INFO_BOUNDARIES" in english_text and "engine_supported" not in english_text
    assert PromptBuilder._info_boundary_section(claimant, None) == ""


def test_discussion_prompt_carries_info_boundary_without_truth_leak() -> None:
    state = _state()
    state.conversation_log.append(Message(speaker="system", content="Day 1 begins."))
    _publish_both_seer_claims(state)
    outsider = _cast(state, "Player 6", Role.VILLAGER)
    ctx = PromptBuilder.build_discussion_prompt(
        outsider,
        _alive(state),
        tuple(state.conversation_log),
        day=1,
        state=state,
        include_claim=True,
        context_strategy="hybrid_retrieval",
        question_hint_mode="soft",
    )
    assert int(ctx.diagnostics["info_boundary_chars"]) > 0
    assert "[情報の境界" in ctx.user_prompt
    # 他プレイヤーのプロンプトへ「どちらが本物の占い師か」を示す情報を出さない
    for forbidden in ("engine_supported", "unsupported_role_evidence", "engine_seer"):
        assert forbidden not in ctx.user_prompt
    # クレーム台帳には中立の出所説明が1回だけ（セクションを増やして節約を損なわない）
    assert ctx.user_prompt.count("Provenance:") == 1


def test_co_decision_prompt_follows_bluff_policy() -> None:
    state = _state()
    actor = _cast(state, "Player 4", Role.VILLAGER)
    alive = _alive(state)
    allow_ctx = PromptBuilder.build_discussion_claim_prompt(actor, alive, 1, state)
    assert "Anyone may claim or counter-claim in Werewolf." in allow_ctx.user_prompt
    state.allow_bluff_role_claim = False
    deny_ctx = PromptBuilder.build_discussion_claim_prompt(actor, alive, 1, state)
    assert "only the true holder of a role may publish" in deny_ctx.user_prompt
    assert "Anyone may claim or counter-claim" not in deny_ctx.user_prompt
    # 出力指定行を増やしていない（コンテキスト節約: 差し替えは1文分のみ）
    assert len(deny_ctx.user_prompt) - len(allow_ctx.user_prompt) < 200


# ---------------------------------------------------------------------------
# 4. ツールCO経路（claim_role）でも同じ出所・ポリシーを適用する
# ---------------------------------------------------------------------------
class _NoClient:  # pragma: no cover - このテストでは通信しない
    def chat(self, *args, **kwargs):
        raise AssertionError("ツール受理の検証でLLMを呼んではならない")


def _actions(state: GameState, events: list[dict[str, object]]) -> OllamaActions:
    return OllamaActions(
        state, _NoClient(), seed=state.seed, event_callback=events.append
    )


def test_tool_claim_publishes_bluff_and_keeps_provenance() -> None:
    state = _state()
    events: list[dict[str, object]] = []
    actions = _actions(state, events)
    actor = _cast(state, "Player 2", Role.MADMAN)
    call = ToolCall(
        name="claim_role",
        arguments={"role": "seer", "target": "Player 3", "result": "wolf"},
    )
    result = actions._handle_public_tool_call(actor, call, "discussion", time.time())
    assert result["status"] == "published"
    published = state.public_role_claims[-1]
    assert published.source == SOURCE_PUBLIC_CLAIM
    assert published.evidence_ids[0].startswith("[claim:")
    # CO行と結果行の両方が公開用IDで追える（エンジン発行IDはここに現れない）
    seer_refs = [ref for ref in published.evidence_ids if ref.startswith("[seer_claim:")]
    assert len(seer_refs) == 2
    assert not any("engine_seer" in ref for ref in published.evidence_ids)
    provenance = state.role_provenance.interpretations[-1]
    assert provenance.category == CATEGORY_UNSUPPORTED
    assert provenance.engine_supported is False
    event = next(e for e in events if e["type"] == "claim_role_published")
    assert event["interpretation"] == PUBLIC_LINE_LABEL
    # 本物の占い結果へ捏造が混ざらない
    assert state.seer_results.get("Player 3") is None


def test_tool_claim_is_rejected_only_when_policy_denies() -> None:
    state = _state()
    events: list[dict[str, object]] = []
    actions = _actions(state, events)
    actor = _cast(state, "Player 2", Role.MADMAN)
    call = ToolCall(name="claim_role", arguments={"role": "seer"})

    allowed = actions._handle_public_tool_call(actor, call, "discussion", time.time())
    assert allowed["status"] == "published"
    before_claims = len(state.public_role_claims)

    state.allow_bluff_role_claim = False
    denied = actions._handle_public_tool_call(actor, call, "discussion", time.time())
    assert denied["status"] == "rejected"
    assert len(state.public_role_claims) == before_claims
    disallow_event = next(e for e in events if e["type"] == "role_claim_disallowed")
    assert disallow_event["player"] == "Player 2"
    assert "claimed_role" not in disallow_event  # 試みた役職を公開イベントに残さない
    # 同じ役職の実保有者なら不許可設定でも通る
    holder = _cast(state, "Player 5", Role.SEER)
    holder_result = actions._handle_public_tool_call(
        holder,
        ToolCall(name="claim_role", arguments={"role": "seer"}),
        "discussion",
        time.time(),
    )
    assert holder_result["status"] == "published"




