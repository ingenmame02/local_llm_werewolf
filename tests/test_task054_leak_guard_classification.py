"""TASK-054の回帰テスト（LL不使用）: 漏洩ガードのゲート分類と本文保持。

設計思想の検証:
1. コンテキストを節約して性能を上げる:
   - 検査は実際に公開される面（CO・結果ブロック・本文）だけを対象にし、公開されない
     【推理メモ】/`[REASONING]` の検出は観測（memo-only）へ逃がす。
   - 村側の役職宣言など誤検出由来では全文再生成せず、警告・観測だけ残す。
2. モデルの一貫性を保つ:
   - FactLedger / ClaimLedger で裏付く「既に公開済みのCO・占い判定」の再言及は漏洩にしない
     （``[claim:N]`` / ``[seer_claim:N]`` の事実IDを検査結果へ残す）。
   - 補正は漏洩した文だけを選び、他の文は一字一句そのまま保つ（本文保持率で検証）。
3. 合理的な議論をさせる:
   - 他者の役職推測は常に合法（gate=observe）。未公開の真の情報だけを hard にする。
4. モデルの自由を縛らない:
   - hardゲートは「非公開の真情報の開示」に限定し、口調・論点・結論は指定しない。
   - ガード強度はモデル別実績から選び（balanced / strict）、遮断すべき情報の歯止めは緩めない。
"""
from __future__ import annotations

import io
import json

import pytest

from werewolf.engine import GameEngine
from werewolf.leak_gate_policy import (
    GATE_HARD,
    GATE_OBSERVE,
    GATE_SOFT,
    LEAK_POLICY_LEGACY_STRICT,
    LEAK_POLICY_TIERED,
    RULE_INTERNAL_CONFIG,
    RULE_MEMO_ONLY,
    RULE_PARTNER_CO_MENTION,
    RULE_PRIVATE_TEAM_REFERENCE,
    RULE_PUBLIC_CLAIM_RESTATEMENT,
    RULE_PUBLISHED_VERDICT_RESTATEMENT,
    RULE_SELF_ROLE_META,
    RULE_SELF_ROLE_PUBLIC_CLAIM,
    RULE_SELF_ROLE_TRUE_CAMP,
    RULE_THIRD_PARTY_SPECULATION,
    RULE_UNPUBLISHED_SEER_RESULT,
    STRENGTH_BALANCED,
    STRENGTH_STRICT,
    LeakGuardProfiles,
    normalize_leak_guard_policy,
)
from werewolf.models import GameState, Player, PublicRoleClaim, Role
from werewolf.prompts import leak_boundary_block
from werewolf.secret_leak_guard import (
    MIN_BODY_PRESERVATION_RATIO,
    SecretLeakGuard,
)
from werewolf.setup import create_game

FP_VILLAGER_BODY = (
    "[BODY] こんにちは、皆さん。私は村人として、村の勝利を目指しています。"
    "Player 1さんのSeer COは初日で信頼に値するものですが、まだ確信は持てません。"
)


def _state() -> GameState:
    return create_game(
        seed=1001, config_path="config/players.json", response_language="日本語"
    )


def _player(state: GameState, name: str) -> Player:
    return next(player for player in state.players if player.name == name)


def _cast(
    state: GameState,
    name: str,
    role: Role,
    *,
    partner: str | None = None,
    model: str = "test-model",
) -> Player:
    """検証したい役職へ振り替える（ガードの検査対象は自分自身だけなので安全）。"""
    player = _player(state, name)
    player.role = role
    player.partner = partner
    player.model = model
    return player


class _NeverRetryActions:
    """soft / observe 指摘でモデルへ再生成要求してはいけないことを検証するためのモック。"""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def retry_after_secret_leak(self, *args, **kwargs):  # pragma: no cover - 呼ばれたら失敗
        self.calls.append((args, kwargs))
        raise AssertionError("soft/observe指摘で安全再生成をしてはならない")


class _StubRetryActions(_NeverRetryActions):
    def __init__(self, candidate: str) -> None:
        super().__init__()
        self.candidate = candidate

    def retry_after_secret_leak(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.candidate


class _LegacySignatureActions:
    """旧シグネチャ（カテゴリのみ）の actions を使った場合の後方互換確認用。"""

    def __init__(self, candidate: str) -> None:
        self.candidate = candidate
        self.categories: tuple[str, ...] | None = None

    def retry_after_secret_leak(self, player, day, alive, history, categories):
        self.categories = tuple(categories)
        return self.candidate


def _engine(state: GameState, actions: object, events: list[dict[str, object]]) -> GameEngine:
    return GameEngine(state, actions, seed=state.seed, event_callback=events.append)


def test_villager_first_person_role_claim_is_not_a_leak() -> None:
    state = _state()
    actor = _cast(state, "Player 3", Role.MADMAN, model="mistral")
    cls = SecretLeakGuard.classify(FP_VILLAGER_BODY, actor, state)
    # 村人COはブラフ含め合法なゲーム内主張。旧ゲートはここで全文再生成していた。
    assert cls.hard == ()
    assert cls.soft_rules == (RULE_SELF_ROLE_PUBLIC_CLAIM,)
    assert cls.soft[0].gate == GATE_SOFT
    assert cls.false_positives_avoided is True
    assert "self_role_assertion" in cls.legacy_hard_rules
    # 後方互換API（inspect）は hard のみ返し、公開を止めない
    assert SecretLeakGuard.inspect(FP_VILLAGER_BODY, actor, state) == ()


def test_reasoning_memo_never_gates_public_speech() -> None:
    state = _state()
    actor = _cast(state, "Player 7", Role.VILLAGER, model="gpt-oss")
    text = (
        "[REASONING] 私はプレイヤー7として村人です。\n"
        "[BODY] Player 2さん、なぜPlayer 4さんに投票したのですか？"
    )
    cls = SecretLeakGuard.classify(text, actor, state)
    assert cls.hard == ()
    assert RULE_MEMO_ONLY in cls.memo_rules
    assert cls.public_regions == 1
    assert cls.false_positives_avoided is True
    legacy = SecretLeakGuard.classify(text, actor, state, guard_policy=LEAK_POLICY_LEGACY_STRICT)
    assert "self_role_assertion" in {finding.rule_id for finding in legacy.hard}


def test_published_claim_and_verdict_restatement_keeps_body_with_fact_ids() -> None:
    """既に公開済みのCO・判定の再言及は漏洩ではなく、台帳ID付きの観測として残す。"""
    state = _state()
    seer = _cast(state, "Player 1", Role.SEER, model="calm3")
    state.seer_results = {"Player 3": "human"}
    state.public_role_claims = [
        PublicRoleClaim(
            day=1, claimant="Player 1", claimed_role="seer", target="Player 3", result="human"
        )
    ]
    state.seer_claims = []
    text = (
        "[BODY] Player 3さんの人間という占い結果は既に公開済みです。"
        "私はその判定を踏まえて投票先を考えたいです。"
    )
    cls = SecretLeakGuard.classify(text, seer, state)
    assert cls.hard == ()
    assert RULE_PUBLISHED_VERDICT_RESTATEMENT in cls.observe_rules
    verdict_note = next(f for f in cls.observe if f.rule_id == RULE_PUBLISHED_VERDICT_RESTATEMENT)
    assert any(fid.startswith("[seer_claim") for fid in verdict_note.fact_ids)
    assert verdict_note.gate == GATE_OBSERVE

    # 自分の公開済みCOの再宣言も observe（[claim:N]）であり、遮断・再生成の対象にしない
    claim_text = "[BODY] 私はSeerです。引き続き公表情報だけで判断します。"
    claim_cls = SecretLeakGuard.classify(claim_text, seer, state)
    assert claim_cls.hard == ()
    assert RULE_PUBLIC_CLAIM_RESTATEMENT in claim_cls.observe_rules


def test_third_party_role_speculation_is_legitimate_discussion() -> None:
    state = _state()
    actor = _cast(state, "Player 2", Role.VILLAGER, model="test-model")
    text = "[BODY] Player 5は人狼かもしれないので、投票前に理由を問いたいです。"
    cls = SecretLeakGuard.classify(text, actor, state)
    assert cls.hard == ()
    assert RULE_THIRD_PARTY_SPECULATION in cls.observe_rules
    assert SecretLeakGuard.inspect(text, actor, state) == ()


def test_wolf_camp_disclosure_and_partner_binding_stay_hard() -> None:
    """歯止めは緩めていない: 真陣営の開示と仲間関係の明示は必ず遮断する。"""
    state = _state()
    wolf = _cast(state, "Player 6", Role.WOLF, partner="Player 2", model="deepseek")
    camp = "[BODY] 実は私は狼陣営の一員です。今夜はPlayer 5を狙うべきだと思います。"
    camp_cls = SecretLeakGuard.classify(camp, wolf, state)
    assert RULE_SELF_ROLE_TRUE_CAMP in camp_cls.hard_rules
    assert all(finding.gate == GATE_HARD for finding in camp_cls.hard)

    binding = "[BODY] 私の人狼の相方はPlayer 2なので、彼の投票先には反対しません。"
    binding_cls = SecretLeakGuard.classify(binding, wolf, state)
    assert set(binding_cls.hard_rules).issubset({RULE_PRIVATE_TEAM_REFERENCE})
    assert binding_cls.hard_rules == (RULE_PRIVATE_TEAM_REFERENCE,)

    # 名前と一人称が別々に現れただけの共起（実ログP8型）は soft。本文は消さない。
    co_mention = "[BODY] Player 2さんや私を疑う理由を、はっきり明確にしてほしいです。"
    co_cls = SecretLeakGuard.classify(co_mention, wolf, state)
    assert co_cls.hard == ()
    assert RULE_PARTNER_CO_MENTION in co_cls.soft_rules


def test_madman_true_role_disclosure_is_hard() -> None:
    state = _state()
    madman = _cast(state, "Player 4", Role.MADMAN, model="test-model")
    text = "[BODY] 私は狂人です。村人のふりをして投票を操ります。"
    cls = SecretLeakGuard.classify(text, madman, state)
    assert cls.hard_rules == (RULE_SELF_ROLE_TRUE_CAMP,)


def test_unpublished_verdict_and_config_reference_are_hard() -> None:
    state = _state()
    seer = _cast(state, "Player 1", Role.SEER, model="calm3")
    state.seer_results = {"Player 6": "wolf"}
    state.public_role_claims = []
    state.seer_claims = []
    text = (
        "[BODY] Player 6は人狼判定でした。私は今日の投票で彼を推します。"
        "他の二人はまだ公表情報で判断できます。"
    )
    cls = SecretLeakGuard.classify(text, seer, state)
    assert RULE_UNPUBLISHED_SEER_RESULT in cls.hard_rules
    verdict = next(f for f in cls.hard if f.rule_id == RULE_UNPUBLISHED_SEER_RESULT)
    assert verdict.region in {"body", "plain"}
    assert verdict.char_start >= 0 and verdict.char_end > verdict.char_start

    villager = _cast(state, "Player 2", Role.VILLAGER, model="llama-3")
    config_text = (
        "[BODY] num_ctxの設定が短いので応答を端折りました。llama-3の出力だと判断しにくいです。"
    )
    config_cls = SecretLeakGuard.classify(config_text, villager, state)
    assert RULE_INTERNAL_CONFIG in config_cls.hard_rules
    assert SecretLeakGuard.inspect(config_text, villager, state) == config_cls.hard


def test_legacy_strict_reproduces_pre_task054_blocking() -> None:
    """A/B比較用の legacy_strict は旧来どおり誤検出でも遮断する（tiered との差分が見える）。"""
    state = _state()
    actor = _cast(state, "Player 3", Role.MADMAN, model="mistral")
    tiered = SecretLeakGuard.classify(FP_VILLAGER_BODY, actor, state)
    legacy = SecretLeakGuard.classify(
        FP_VILLAGER_BODY, actor, state, guard_policy=LEAK_POLICY_LEGACY_STRICT
    )
    assert tiered.hard == ()
    assert [f.rule_id for f in legacy.hard] == ["self_role_assertion"]
    assert all(f.region == "legacy" and f.char_start < 0 for f in legacy.hard)
    assert SecretLeakGuard.inspect(
        FP_VILLAGER_BODY, actor, state, guard_policy="legacy_strict"
    ) == legacy.hard
    assert normalize_leak_guard_policy("unknown-value") == LEAK_POLICY_TIERED
    assert normalize_leak_guard_policy("legacy") == LEAK_POLICY_LEGACY_STRICT


def test_sentence_repair_neutralizes_only_the_leaking_sentence() -> None:
    """本文全体を書き直さず、漏洩した文だけを選んで残りの文は一字一句そのまま保つ。"""
    state = _state()
    seer = _cast(state, "Player 1", Role.SEER, model="calm3")
    state.seer_results = {"Player 6": "wolf"}
    state.public_role_claims = []
    state.seer_claims = []
    keep_a = "初日のCO矛盾はまだ解明されていないので、説明を聞きたいです。"
    keep_b = "私は公表情報だけを見て投票先を決めたいです。"
    text = (
        "[BODY]\nPlayer 6は人狼判定でした。\n"
        f"{keep_a}"
        f"{keep_b}"
    )
    cls = SecretLeakGuard.classify(text, seer, state)
    repair = SecretLeakGuard.repair_public_text(text, cls, seer)
    assert repair.applied is True
    assert repair.edit_count == 1
    assert RULE_UNPUBLISHED_SEER_RESULT in repair.replaced_rules
    assert repair.preserved_ratio >= MIN_BODY_PRESERVATION_RATIO
    assert repair.body_preservable is True
    assert keep_a in repair.text and keep_b in repair.text
    assert "[BODY]" in repair.text

    assert "人狼判定" not in repair.text
    assert SecretLeakGuard.classify(repair.text, seer, state).hard == ()


def test_meta_sentence_is_dropped_instead_of_neutralized() -> None:
    state = _state()
    actor = _cast(state, "Player 2", Role.VILLAGER, model="test-model")
    text = (
        "[BODY] 私は設定上は村人ですが、騙って勝つつもりです。"
        "皆さんはPlayer 6さんの投票先をどう見ていますか？"
    )
    cls = SecretLeakGuard.classify(text, actor, state)
    assert RULE_SELF_ROLE_META in cls.hard_rules
    repair = SecretLeakGuard.repair_public_text(text, cls, actor)
    assert RULE_SELF_ROLE_META in repair.dropped_rules
    assert repair.replaced_rules == ()
    assert "設定上" not in repair.text
    assert "Player 6" in repair.text


def test_legacy_classification_escalates_because_spans_are_unlocalizable() -> None:
    """legacy_strict の結果では文位置が不明のため決定的補正できず、再要求側へ回る。"""
    state = _state()
    actor = _cast(state, "Player 3", Role.MADMAN, model="mistral")
    legacy = SecretLeakGuard.classify(
        FP_VILLAGER_BODY, actor, state, guard_policy=LEAK_POLICY_LEGACY_STRICT
    )
    repair = SecretLeakGuard.repair_public_text(FP_VILLAGER_BODY, legacy, actor)
    assert repair.applied is False
    assert repair.body_preservable is False


def test_guard_strength_varies_by_model_and_does_not_rewrite_balanced_bodies() -> None:
    """強度は一律にしない: 誤検出由来は本文そのまま、真漏洩モデルだけ曖昧文も中立化する。"""
    state = _state()
    actor = _cast(state, "Player 3", Role.MADMAN, model="mistral")
    cls = SecretLeakGuard.classify(FP_VILLAGER_BODY, actor, state)
    balanced = SecretLeakGuard.repair_public_text(FP_VILLAGER_BODY, cls, actor)
    assert balanced.applied is False and balanced.text == FP_VILLAGER_BODY

    strict = SecretLeakGuard.repair_public_text(
        FP_VILLAGER_BODY, cls, actor, repair_soft=True
    )
    assert strict.applied is True
    assert RULE_SELF_ROLE_PUBLIC_CLAIM in strict.replaced_rules

    profiles = LeakGuardProfiles()
    for _ in range(2):
        profiles.observe(
            "mistral",
            gate=GATE_SOFT,
            rule_ids=(RULE_SELF_ROLE_PUBLIC_CLAIM,),
            body_char_count=100,
            kept_char_count=100,
        )
    profiles.observe("deepseek", gate=GATE_HARD, rule_ids=(RULE_SELF_ROLE_TRUE_CAMP,))
    assert profiles.strength_for("mistral") == STRENGTH_BALANCED
    assert profiles.strength_for("deepseek") == STRENGTH_STRICT
    assert profiles.summary()["deepseek"]["guard_strength"] == STRENGTH_STRICT


def test_engine_publishes_soft_and_observe_notes_without_regenerating() -> None:
    """警告・観測のみの発言は本文をそのまま公開する（旧来はここで再生成していた）。"""
    state = _state()
    events: list[dict[str, object]] = []
    actions = _NeverRetryActions()
    engine = _engine(state, actions, events)
    published = engine._guard_public_text(
        "Player 3", "discussion", FP_VILLAGER_BODY, ()
    )
    assert published == FP_VILLAGER_BODY
    assert actions.calls == []
    notes = [event for event in events if event.get("type") == "secret_leak_note"]
    assert len(notes) == 1
    assert notes[0]["false_positive_avoided"] is True
    assert RULE_SELF_ROLE_PUBLIC_CLAIM in list(notes[0]["note_rules"])
    assert [event for event in events if event.get("type") == "secret_role_leak_detected"] == []
    profile = state.leak_guard_profiles.profiles[_player(state, "Player 3").model]
    assert profile.soft_notes >= 1


def test_engine_repairs_the_leaking_sentence_and_skips_regeneration() -> None:
    state = _state()
    wolf_name = "Player 6"
    _cast(state, wolf_name, Role.WOLF, partner="Player 2", model="deepseek")
    body = (
        "[BODY]\n私の人狼の相方はPlayer 2です。\n初日のCO矛盾はまだ解明されていないので、投票前に説明を聞きたいです。"
        "初日のCO矛盾はまだ解明されていないので、投票前に説明を聞きたいです。"
        "私は公表情報だけで判断します。"
    )
    events: list[dict[str, object]] = []
    actions = _NeverRetryActions()
    engine = _engine(state, actions, events)
    published = engine._guard_public_text(wolf_name, "discussion", body, ())
    assert "[BODY]" in published
    assert actions.calls == []
    assert "相方" not in published
    assert "初日のCO矛盾" in published and "公表情報だけ" in published
    repairs = [e for e in events if e.get("type") == "secret_leak_sentence_repair"]
    assert len(repairs) == 1
    assert repairs[0]["regeneration_avoided"] is True
    assert float(repairs[0]["retry_seconds"]) == 0.0
    detections = [e for e in events if e.get("type") == "secret_role_leak_detected"]
    assert len(detections) == 1
    assert RULE_PRIVATE_TEAM_REFERENCE in list(detections[0]["detection_rules"])


def test_engine_uses_safety_retry_only_when_body_is_not_preservable() -> None:
    """漏洩が本文の大半を占めるときだけモデルへ再要求する（ルール別の短い案内付きで）。"""
    state = _state()
    wolf_name = "Player 6"
    _cast(state, wolf_name, Role.WOLF, partner="Player 2", model="deepseek")
    body = "[BODY]\n実は私は狼陣営です。\n"
    candidate = "[BODY]\n私は公表情報だけで判断します。Player 5さんの説明を聞きたいです。\n"
    events: list[dict[str, object]] = []
    actions = _StubRetryActions(candidate)
    engine = _engine(state, actions, events)
    published = engine._guard_public_text(wolf_name, "discussion", body, ())
    assert len(actions.calls) == 1
    _, kwargs = actions.calls[0]
    assert kwargs.get("rule_ids") and RULE_SELF_ROLE_TRUE_CAMP in tuple(kwargs["rule_ids"])
    assert isinstance(kwargs.get("retained_body"), str)
    assert published == candidate.strip()
    regenerations = [e for e in events if e.get("type") == "secret_leak_regeneration"]
    assert len(regenerations) == 1
    assert regenerations[0]["outcome"] == "regenerated"
    assert regenerations[0]["regeneration_avoided"] is False
    assert float(regenerations[0]["retry_seconds"]) >= 0.0
    assert RULE_SELF_ROLE_TRUE_CAMP in list(regenerations[0]["hard_rules"])


def test_engine_accepts_actions_without_the_new_keyword_arguments() -> None:
    """旧シグネチャの actions（カテゴリのみ）でも再要求が回る（後方互換）。"""
    state = _state()
    wolf_name = "Player 6"
    _cast(state, wolf_name, Role.WOLF, partner="Player 2", model="deepseek")
    actions = _LegacySignatureActions("[BODY]\n私は公表情報だけで判断します。\n")
    engine = _engine(state, actions, [])
    published = engine._guard_public_text(wolf_name, "discussion", "[BODY]\n実は私は狼陣営です。\n", ())
    assert actions.categories and "secret_role_leak" in actions.categories
    assert published.startswith("[BODY]") or "公表情報だけ" in published


def test_prompt_leak_boundary_does_not_prescribe_the_village_side() -> None:
    """プロンプト側で村側に「as a villager」等を書かせない（量産された誤検出由来の断ち切り）。"""
    state = _state()
    seer = _cast(state, "Player 1", Role.SEER)
    villager_ja = leak_boundary_block(seer, "日本語")
    assert "自分の役職をゲーム内主張" in villager_ja
    assert "村人の立場で話す" not in villager_ja

    english = leak_boundary_block(seer, "English")
    assert "Declaring your own role as an in-game claim is legitimate" in english
    assert "as a villager" not in english.lower()

    wolf = _cast(state, "Player 6", Role.WOLF, partner="Player 2")
    wolf_ja = leak_boundary_block(wolf, "日本語")
    assert "仲間・相方" in wolf_ja
    assert "自分の役職をゲーム内主張" not in wolf_ja


def test_cli_logger_counts_gate_metrics(tmp_path) -> None:
    """ログで誤検出自動回避数・本文保持率・再要求秒を比較できる（TASK-054完了条件）。"""
    from werewolf.cli import GameLogger

    state = _state()
    model = _player(state, "Player 3").model
    logger = GameLogger("task054", tmp_path, {"mode": "mock"}, private_enabled=False)
    base = {
        "day": 1,
        "phase": "day_discussion",
        "phase_label": "昼・議論",
        "player": "Player 3",
        "model": model,
        "action": "discussion",
        "private": False,
        "elapsed_seconds": 0.0,
    }
    logger.append_record(
        {
            **base,
            "type": "secret_leak_note",
            "note_rules": [RULE_SELF_ROLE_PUBLIC_CLAIM],
            "false_positive_avoided": True,
            "body_char_count": 200,
            "kept_char_count": 200,
        }
    )
    logger.append_record(
        {
            **base,
            "type": "secret_leak_sentence_repair",
            "replaced_rules": [RULE_UNPUBLISHED_SEER_RESULT],
            "dropped_rules": [],
            "edit_count": 1,
            "body_char_count": 100,
            "kept_char_count": 85,
            "regeneration_avoided": True,
            "retry_seconds": 0.0,
        }
    )
    logger.append_record(
        {
            **base,
            "type": "secret_leak_regeneration",
            "hard_rules": [RULE_PARTNER_CO_MENTION],
            "outcome": "redacted_after_retry",
            "retry_seconds": 12.5,
            "body_char_count": 90,
            "kept_char_count": 40,
            "regeneration_avoided": False,
        }
    )
    logger.append_record(
        {
            **base,
            "type": "secret_role_leak_detected",
            "detection_rules": [RULE_SELF_ROLE_TRUE_CAMP],
            "leak_categories": ["secret_role_leak"],
            "guard_strength": STRENGTH_STRICT,
            "content_sha256": "0" * 64,
            "safety_retry": 0,
            "memo_region_rules": [RULE_MEMO_ONLY],
        }
    )
    logger.finalize(state)

    rows = [
        json.loads(line)
        for line in io.open(tmp_path / "game_task054.jsonl", encoding="utf-8")
        if line.strip()
    ]
    summary = rows[-1]
    metrics = summary["ledger_metrics"]
    assert metrics["secret_leak_hard"] == 1
    assert metrics["secret_leak_notes"] == 1
    assert metrics["secret_leak_sentence_repairs"] == 1
    assert metrics["secret_leak_regenerations"] == 1
    assert metrics["secret_leak_redactions"] == 1
    assert metrics["secret_leak_false_positives_avoided"] == 1
    assert float(metrics["secret_leak_retry_seconds"]) == 12.5
    assert int(metrics["secret_leak_body_char_total"]) == 390
    assert int(metrics["secret_leak_body_char_kept"]) == 325
    assert metrics["leak_body_preservation_rate"] == round(325 / 390, 4)
    assert metrics["leak_guard_policy"] == LEAK_POLICY_TIERED
    stats = summary["model_stats"][model]
    assert stats["leak_rule_counts"][RULE_SELF_ROLE_PUBLIC_CLAIM] == 1
    assert stats["leak_rule_counts"][RULE_SELF_ROLE_TRUE_CAMP] == 1
    assert stats["leak_memo_only_count"] == 1
    assert stats["leak_guard_strengths"][STRENGTH_STRICT] == 1



