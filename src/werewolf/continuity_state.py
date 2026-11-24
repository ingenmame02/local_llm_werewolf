"""TASK-062: 日をまたぐ論理状態の正規化（ContinuityState）と記憶減衰対策。

過去の発言全文をプロンプトに積み増すと、Day 3 で core が膨張して「いま何が確定し、
どの仮説が反証され、どれが有効か」を見失う（game_e6b6c7cc3481: 平均core が
Day1 11,613字 → Day3 21,501字、`core_historical_chars` は各日0）。本モジュールは
**生の会話を長く持つ代わりに、エンジンが発行する短い状態差分**で論理を継続する。

4つの設計思想との対応:
1. コンテキスト節約: 同じ状態を繰り返し送らない。変わった項目だけを1行で出し、
   変わっていない項目は `unchanged` として数えるだけ（本文もIDも再掲しない）。
   checkpoint が正規化した日の平常発言は、prompts側が本文をID行へ畳める。
2. モデルの一貫性: 各項目に `observed_at`（いつ出てきたか）と `evidence_id`
   （どの証拠か）と権威レベルを付与し、古い仮説が新しい確定事実として扱われるのを防ぐ。
3. 合理的な議論: `current / changed / overturned / unresolved` の状態だけを示す。
   「前回の結論を維持せよ」とは書かない。矛盾の説明はモデルの自由に残す。
4. 議論の自由: 新しい禁止・誘導命令は追加しない。表示するのは状態であり、
   口調・論点・結論・推理順には干渉しない。自分の立場を変えることも自由である。

既存台帳（FactLedger / ClaimLedger / HypothesisLedger）を再掲するのではなく、
そこから「次の発言に必要な差分」だけを生成する。LLMを呼ばない決定的な計算のみ。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 循環import回避。実行時は duck typing で GameState を扱う。
    from .models import GameState

# ---------------------------------------------------------------------------
# ポリシー（既定 normalized。off はA/B比較用の旧挙動＝checkpointなし）
# ---------------------------------------------------------------------------

CONTINUITY_POLICY_NORMALIZED = "normalized"
CONTINUITY_POLICY_LEGACY = "off"
CONTINUITY_POLICIES: tuple[str, ...] = (
    CONTINUITY_POLICY_NORMALIZED,
    CONTINUITY_POLICY_LEGACY,
)
CONTINUITY_POLICY_DEFAULT = CONTINUITY_POLICY_NORMALIZED


def normalize_continuity_policy(value: object) -> str:
    """設定値を \"normalized\" / \"off\" へ正規化する（不明値は既定の正規化）。"""
    if isinstance(value, bool):
        return CONTINUITY_POLICY_NORMALIZED if value else CONTINUITY_POLICY_LEGACY
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in CONTINUITY_POLICIES:
            return candidate
        if candidate in {"on", "true", "state", "diff", "checkpoint"}:
            return CONTINUITY_POLICY_NORMALIZED
        if candidate in {"none", "legacy", "false", "old", "before"}:
            return CONTINUITY_POLICY_LEGACY
    return CONTINUITY_POLICY_DEFAULT


def continuity_policy_label(policy: object) -> str:
    """ログ・CLI表示用のラベル（正規化済みの文字列を返す）。"""
    return normalize_continuity_policy(policy)


# ---------------------------------------------------------------------------
# 権威レベル（確定事実と未検証の派生を混ぜないための表示記号）
# ---------------------------------------------------------------------------

# エンジンが記録した事実のみ（死亡・投票・生存者数）。モデルの推理はここに入れない。
AUTHORITY_ENGINE = "engine_confirmed"
# 公開された発言・CO・占い申告。「その人がそう言った」までは事実、内容の真偽は未検証。
AUTHORITY_PUBLIC_STATEMENT = "public_statement_unverified"
# モデルの発言から導いた未検証の解釈（自分の仮説・立場など）
AUTHORITY_DERIVED = "derived_unverified"
# 反証・立場変更・死亡者参照で現在成り立たなくなった項目（古い行を新事実として扱わない）
AUTHORITY_REFUTED = "refuted_or_superseded"

CONTINUITY_AUTHORITIES: tuple[str, ...] = (
    AUTHORITY_ENGINE,
    AUTHORITY_PUBLIC_STATEMENT,
    AUTHORITY_DERIVED,
    AUTHORITY_REFUTED,
)

# 表示タグ（短く権威が分かる形。本文の言い回しは変えない）
AUTHORITY_TAGS: dict[str, str] = {
    AUTHORITY_ENGINE: "[CONFIRMED]",
    AUTHORITY_PUBLIC_STATEMENT: "[SAID-unverified]",
    AUTHORITY_DERIVED: "[YOUR-UNVERIFIED]",
    AUTHORITY_REFUTED: "[NO-LONGER-STANDS]",
}

# 項目の種類（状態の区別。口調や論点ではなく「どの情報か」だけの分類）
KIND_FACT = "confirmed_fact"
KIND_CLAIM = "public_claim"
KIND_VERDICT = "public_verdict"
KIND_STANCE = "own_stance"
KIND_SUPPORT = "own_support"
KIND_COUNTER = "own_counter_evidence"
KIND_QUESTION = "open_question"
KIND_REFUTATION = "refuted_hypothesis"

CONTINUITY_KINDS: tuple[str, ...] = (
    KIND_FACT,
    KIND_CLAIM,
    KIND_VERDICT,
    KIND_STANCE,
    KIND_SUPPORT,
    KIND_COUNTER,
    KIND_QUESTION,
    KIND_REFUTATION,
)

# 提示量の上限（状態の総量ではなく「いま必要な行数」で抑える＝日をまたいでも増やさない）
RENDER_CHANGED_MAX = 4
RENDER_OVERTURNED_MAX = 3
RENDER_QUESTIONS_MAX = 3
RENDER_CURRENT_MAX = 5
# 変化なし項目は行を出さず、この1行で個数とIDの一部だけ示す（同じ状態の再掲を省く）
UNCHANGED_SUMMARY_IDS = 8


def point_label(day: int, turn: int) -> str:
    """観測時点ラベル（`D2T1` 形。夜・投票境界は turn=0）。"""
    return f"D{int(day)}T{int(turn or 0)}"


def _clamp_text(text: object, limit: int = 120) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(1, limit - 1)].rstrip() + "\u2026"


# チェックポイントの1行上限。状態行は短く保ち、本文の写しを混ぜないための担保になる。
CHECKPOINT_LINE_LIMIT = 280


def _clip(line: str, limit: int = CHECKPOINT_LINE_LIMIT) -> str:
    text = " ".join(str(line or "").split())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "\u2026"


@dataclass(frozen=True)
class ContinuityItem:
    """正規化された論理状態1項目。発言全文の写しではなく、時点・証拠ID付きの短い行だけ持つ。"""

    item_id: str
    kind: str
    subject: str
    statement: str
    authority: str
    observed_at: str = ""
    evidence_ids: tuple[str, ...] = ()
    first_day: int = 1
    last_day: int = 1

    def signature(self) -> tuple[object, ...]:
        """前回提示からの差分判定用。権威が変われば「変更」として出す（旧と新を区別する）。"""
        return (
            self.statement,
            self.authority,
            tuple(sorted(self.evidence_ids)),
            self.observed_at,
        )

    def render_line(self) -> str:
        tag = AUTHORITY_TAGS.get(self.authority, "[STATE]")
        parts = [f"{tag} {self.subject}: {self.statement}"]
        if self.evidence_ids:
            shown = ", ".join(self.evidence_ids[:3])
            omitted = len(self.evidence_ids) - min(3, len(self.evidence_ids))
            parts.append(f"evidence {shown}" + (f" (+{omitted} more)" if omitted > 0 else ""))
        if self.observed_at:
            parts.append(f"observed {self.observed_at}")
        return "- " + "; ".join(parts)


@dataclass
class PlayerContinuity:
    """1プレイヤー分の論理状態と、最後に提示した状態（差分の基準）。"""

    name: str
    items: dict[str, ContinuityItem] = field(default_factory=dict)
    presented: dict[str, tuple[object, ...]] = field(default_factory=dict)
    # 最後に提示した時点（同一時点での重複構築で差分基準が進まないためのガード）
    last_point: tuple[int, int] | None = None
    # 同一時点の再構築（予算段階のやり直し・再要求）では同じ文言を返す。
    # プロンプトが組み直されるたびに差分基準が進むと、モデルへ見える状態がぶれる。
    last_block: str = ""
    last_diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass
class ContinuityCheckpoint:
    """エンジン発行の状態チェックポイント（日終了・投票前に作成。ID付き短行のみ）。"""

    checkpoint_id: str
    boundary: str
    day: int
    turn: int
    issued_at: str
    lines: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    char_count: int = 0
    # この時点で正規化済みとして数えた項目（以後、本文の再掲対象から外れる）
    normalized_item_ids: tuple[str, ...] = ()


def _hypothesis_authority(record) -> str:
    """HypothesisRecord の反証状態を権威レベルへ写す（古い仮説を新事実として出さない）。"""
    status = str(getattr(record, "refutation_status", "") or "")
    if status in {"contested", "historical", "superseded"}:
        return AUTHORITY_REFUTED
    return AUTHORITY_DERIVED


@dataclass
class ContinuityState:
    """日をまたぐ論理状態の正規化台帳。次の発言に必要な差分だけを生成する。

    - 項目は内部で更新し続けるが、プロンプトには「変わった項目」「反証された項目」
      「未解決の問い」だけを出す（変わっていない項目は個数だけ）。
    - checkpoint は公開情報だけで作る（人狼相談・占い結果などの秘匿情報は載せない）。
      自分の未検証仮説は本人向け欄でのみ提示し、可視規則は HypothesisLedger に従う。
    """

    policy: str = CONTINUITY_POLICY_DEFAULT
    players: dict[str, PlayerContinuity] = field(default_factory=dict)
    checkpoints: list[ContinuityCheckpoint] = field(default_factory=list)
    # 診断カウンター（実機A/Bで「状態が増えたか・再掲が減ったか」を見る）
    items_updated: int = 0
    items_overturned: int = 0
    hypothesis_refutations: int = 0
    hypothesis_abandonments: int = 0
    topic_reintroductions: int = 0
    unchanged_presented: int = 0
    questions_resolved: int = 0
    evidence_ids_referenced: int = 0
    state_chars_by_day: dict[int, int] = field(default_factory=dict)
    _seen_topics_by_day: dict[int, set[str]] = field(default_factory=dict, repr=False)
    _checkpoint_seq: int = 0

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return normalize_continuity_policy(self.policy) == CONTINUITY_POLICY_NORMALIZED

    def player_state(self, actor_name: str) -> PlayerContinuity:
        player = self.players.get(actor_name)
        if player is None:
            player = PlayerContinuity(name=actor_name)
            self.players[actor_name] = player
        return player

    def normalized_days(self, before_day: int) -> tuple[int, ...]:
        """チェックポイントが既に正規化した日（当日より前）。

        prompts 側はこれらの日の「通常の発言本文」を ID 行へ畳める。状態はここに残って
        いるので、会話全文を再送しなくても追跡できる（recent層の単調増加を止める狙い）。
        当日issued のチェックポイントは当日自身を正規化しない（Day 2の朝に出したC1は
        Day 2の本文を生かしたまま、Day 3以降で1日分の本文を畳める）。
        """
        target = int(before_day)
        if not any(int(cp.day) < target for cp in self.checkpoints):
            return ()
        return tuple(range(1, target))

    def latest_checkpoint(self) -> ContinuityCheckpoint | None:
        return self.checkpoints[-1] if self.checkpoints else None

    def checkpoint_text(self) -> str:
        """直近チェックポイントの内容（再要求の短縮プロンプトで優先して渡すための公開テキスト）."""
        checkpoint = self.latest_checkpoint()
        if checkpoint is None:
            return ""
        return "\n".join(checkpoint.lines) + "\n"

    # ------------------------------------------------------------------
    # 現在の論理状態の収集（既存台帳の再掲ではなく、正規化された差分の元）
    # ------------------------------------------------------------------

    def _public_fact_items(self, state: "GameState", day: int, turn: int) -> list[ContinuityItem]:
        """確定事実（エンジン記録だけ）。CO・占い申告は未検証側へ回す権威分離をする。"""
        items: list[ContinuityItem] = []
        for death_day, name in getattr(state, "attack_deaths", ()) or ():
            items.append(
                ContinuityItem(
                    item_id=f"F-attack-{death_day}-{name}",
                    kind=KIND_FACT,
                    subject=str(name),
                    statement=f"attacked during the night entering Day {death_day}",
                    authority=AUTHORITY_ENGINE,
                    observed_at=point_label(death_day, 0),
                    evidence_ids=(f"[death:attack:{death_day}:{name}]",),
                    first_day=int(death_day),
                    last_day=day,
                )
            )
        for death_day, name in getattr(state, "executions", ()) or ():
            items.append(
                ContinuityItem(
                    item_id=f"F-execution-{death_day}-{name}",
                    kind=KIND_FACT,
                    subject=str(name),
                    statement=f"executed by vote on Day {death_day}",
                    authority=AUTHORITY_ENGINE,
                    observed_at=point_label(death_day, 3),
                    evidence_ids=(f"[death:execution:{death_day}:{name}]",),
                    first_day=int(death_day),
                    last_day=day,
                )
            )
        for vote_day, runoff, votes in getattr(state, "vote_history", ()) or ():
            ballot = "runoff" if runoff else "initial"
            for target in sorted({target for target in votes.values()}):
                voters = sorted(voter for voter, chosen in votes.items() if chosen == target)
                items.append(
                    ContinuityItem(
                        item_id=f"F-vote-{vote_day}-{ballot}-{target}",
                        kind=KIND_FACT,
                        subject=str(target),
                        statement=(
                            f"received {len(voters)} vote(s) in the Day {vote_day} "
                            f"{'runoff' if runoff else 'initial'} ballot"
                        ),
                        authority=AUTHORITY_ENGINE,
                        observed_at=point_label(vote_day, 3),
                        evidence_ids=tuple(
                            f"[vote:{vote_day}:{ballot}:{voter}]" for voter in voters[:3]
                        ),
                        first_day=int(vote_day),
                        last_day=day,
                    )
                )
        alive = tuple(getattr(state, "alive_players", ()) or ())
        if alive:
            items.append(
                ContinuityItem(
                    item_id=f"F-alive-{len(alive)}",
                    kind=KIND_FACT,
                    subject="alive players",
                    statement=f"{len(alive)} alive: " + ", ".join(alive),
                    authority=AUTHORITY_ENGINE,
                    # 生存者数は「いまの状態」であり観測イベントではない。刻印をターン毎に
                    # 変えると差分扱いで再掲になってしまうため、IDと本文だけで安定させる。
                    observed_at="",
                    evidence_ids=(),
                    first_day=1,
                    last_day=day,
                )
            )
        return items

    def _public_statement_items(self, state: "GameState", day: int) -> list[ContinuityItem]:
        """公開CO・占い申告（内容の真偽は未検証）。本文は [CLAIM_LEDGER] が持つのでID参照にする。"""
        items: list[ContinuityItem] = []
        for index, claim in enumerate(getattr(state, "public_role_claims", ()) or (), 1):
            items.append(
                ContinuityItem(
                    item_id=f"C-claim-{index}",
                    kind=KIND_CLAIM,
                    subject=str(claim.claimant),
                    statement=f"claimed {claim.claimed_role} on Day {claim.day}",
                    authority=AUTHORITY_PUBLIC_STATEMENT,
                    observed_at=point_label(int(claim.day), 0),
                    evidence_ids=(f"[claim:{index}]",),
                    first_day=int(claim.day),
                    last_day=day,
                )
            )
        for index, claim in enumerate(getattr(state, "seer_claims", ()) or (), 1):
            if claim.target is None or claim.result is None:
                continue
            verdict = "Werewolf" if claim.result == "wolf" else "Human"
            items.append(
                ContinuityItem(
                    item_id=f"C-verdict-{index}",
                    kind=KIND_VERDICT,
                    subject=str(claim.target),
                    statement=(
                        f"{claim.claimant} publicly reported a {verdict} verdict on Day {claim.day}"
                    ),
                    authority=AUTHORITY_PUBLIC_STATEMENT,
                    observed_at=point_label(int(claim.day), 0),
                    evidence_ids=(f"[seer_claim:{index}]",),
                    first_day=int(claim.day),
                    last_day=day,
                )
            )
        return items

    def _own_reasoning_items(
        self, state: "GameState", actor_name: str, day: int
    ) -> tuple[list[ContinuityItem], list[ContinuityItem]]:
        """本人の未検証仮説と、反証・立場変更で成り立たなくなった項目に二分する。

        可視規則は HypothesisLedger.visible_for に従う（他人の私的仮説を渡さない）。
        """
        ledger = getattr(state, "hypothesis_ledger", None)
        if ledger is None:
            return [], []
        try:
            visible = list(ledger.visible_for(actor_name))
        except Exception:  # pragma: no cover - 台帳外の入力に対する保守
            return [], []
        own: list[ContinuityItem] = []
        overturned: list[ContinuityItem] = []
        for record in visible:
            subject = str(getattr(record, "subject", "") or "")
            stance = str(getattr(record, "stance", "") or "")
            statement = _clamp_text(getattr(record, "statement", ""), 90)
            status = str(getattr(record, "refutation_status", "") or "")
            observed = str(
                getattr(record, "observed_at", "")
                or point_label(
                    int(getattr(record, "day", 1) or 1),
                    int(getattr(record, "turn", 0) or 0),
                )
            )
            evidence_ids = tuple(
                str(eid) for eid in (getattr(record, "evidence_ids", ()) or ())[:3]
            )
            counter_ids = tuple(
                str(eid) for eid in (getattr(record, "counter_evidence_ids", ()) or ())[:3]
            )
            self_made = str(getattr(record, "generated_by", "")) == actor_name
            item_id = f"H-{getattr(record, 'hypothesis_id', f'{actor_name}-{subject}-{stance}')}"
            who = "your" if self_made else f"{getattr(record, 'generated_by', '?')}'s"
            detail = f"{who} {stance} reading of {subject}"
            if statement:
                detail = f"{detail}: {statement}"
            common = dict(
                observed_at=observed,
                first_day=int(getattr(record, "first_day", 1) or 1),
                last_day=int(getattr(record, "last_updated_day", day) or day),
            )
            if status in {"contested", "historical", "superseded"}:
                overturned.append(
                    ContinuityItem(
                        item_id=f"{item_id}-refuted",
                        kind=KIND_REFUTATION,
                        subject=subject,
                        statement=f"{detail} is {status}"
                        + (f" (counter {', '.join(counter_ids)})" if counter_ids else ""),
                        authority=AUTHORITY_REFUTED,
                        evidence_ids=(counter_ids or evidence_ids),
                        **common,
                    )
                )
                continue
            if self_made:
                own.append(
                    ContinuityItem(
                        item_id=item_id,
                        kind=KIND_STANCE,
                        subject=subject,
                        statement=detail,
                        authority=AUTHORITY_DERIVED,
                        evidence_ids=evidence_ids,
                        **common,
                    )
                )
        return own, overturned

    def _open_question_items(self, state: "GameState", actor_name: str, day: int) -> list[ContinuityItem]:
        """本人宛の未回答質問（全件収集）。表示上限は render 側で注記付き守る。"""
        memos = getattr(state, "reasoning_memos", {}) or {}
        memo = memos.get(actor_name)
        if memo is None:
            return []
        items: list[ContinuityItem] = []
        for entry in list(getattr(memo, "unresolved_questions", ()) or ()):
            text = _clamp_text(entry, 140)
            token_ids = tuple(word for word in text.split() if word.startswith("[speech:"))[:2]
            items.append(
                ContinuityItem(
                    item_id=f"Q-{token_ids[0].rstrip(']') if token_ids else abs(hash(text)) % (10 ** 12)}",
                    kind=KIND_QUESTION,
                    subject="open question",
                    statement=text,
                    authority=AUTHORITY_PUBLIC_STATEMENT,
                    observed_at=point_label(day, 0),
                    evidence_ids=token_ids,
                    first_day=day,
                    last_day=day,
                )
            )
        return items

    def sync(self, state: "GameState", actor_name: str, day: int, turn: int) -> PlayerContinuity:
        """現在の状態へ項目を統合する。変わった項目だけを更新として数える（再掲はしない）。"""
        player = self.player_state(actor_name)
        incoming: list[ContinuityItem] = []
        incoming.extend(self._public_fact_items(state, day, turn))
        incoming.extend(self._public_statement_items(state, day))
        own, overturned = self._own_reasoning_items(state, actor_name, day)
        incoming.extend(own)
        incoming.extend(overturned)
        incoming.extend(self._open_question_items(state, actor_name, day))
        for item in incoming:
            existing = player.items.get(item.item_id)
            if existing is None:
                player.items[item.item_id] = item
                continue
            if existing.signature() != item.signature():
                merged = ContinuityItem(
                    item_id=item.item_id,
                    kind=item.kind,
                    subject=item.subject,
                    statement=item.statement,
                    authority=item.authority,
                    observed_at=item.observed_at or existing.observed_at,
                    evidence_ids=tuple(
                        dict.fromkeys((*existing.evidence_ids, *item.evidence_ids))
                    )[:4],
                    first_day=min(existing.first_day, item.first_day),
                    last_day=day,
                )
                player.items[item.item_id] = merged
                self.items_updated += 1
                if (
                    existing.authority != AUTHORITY_REFUTED
                    and merged.authority == AUTHORITY_REFUTED
                ):
                    self.items_overturned += 1
                    if merged.kind in {KIND_REFUTATION, KIND_STANCE}:
                        self.hypothesis_refutations += 1
        # 解消した未回答質問は状態から落とす（Day 3 まで古い問いを持ち回らない）
        live_questions = {item.item_id for item in incoming if item.kind == KIND_QUESTION}
        stale = [
            key
            for key, item in player.items.items()
            if item.kind == KIND_QUESTION and key not in live_questions
        ]
        for key in stale:
            player.items.pop(key, None)
            self.questions_resolved += 1
        # 反証・差し替えられた仮説の古い行を状態から落とす。unchanged へ残ると「いまも有効」に
        # 読まれうるため、成り立たない読みは反証行だけを残して消す（自由な立場変更は妨げない）。
        superseded_keys = {
            item.item_id[: -len("-refuted")]
            for item in incoming
            if item.kind == KIND_REFUTATION and item.item_id.endswith("-refuted")
        }
        for key in superseded_keys:
            removed = player.items.pop(key, None)
            if removed is not None:
                self.items_overturned += 1
                if removed.kind == KIND_STANCE:
                    # 自分の読みが反証で落ちた回数は、立場の更新量として別カウンターに数える
                    self.hypothesis_abandonments += 1
                    self.hypothesis_refutations += 1
        return player

    def render_for(
        self,
        state: "GameState",
        actor_name: str,
        day: int,
        turn: int,
    ) -> tuple[str, dict[str, object]]:
        """本人向けの `[CONTINUITY_CHECKPOINT]`。変わった項目だけを出し、不変は個数だけ示す。"""
        if not self.enabled:
            return "", {"continuity_policy": continuity_policy_label(self.policy)}
        player = self.player_state(actor_name)
        point = (int(day), int(turn or 0))
        # 同じ時点での組み直し（予算段階・再要求）では前回と同じ文言を返す。差分基準が進んで
        # モデルへ見える状態がぶれるのを防ぎ、カウンタの二重計上も抑える。
        if player.last_point == point and player.last_block:
            return player.last_block, dict(player.last_diagnostics)
        player = self.sync(state, actor_name, day, turn)
        baseline = player.presented
        first_time = player.last_point is None and not baseline
        changed: list[ContinuityItem] = []
        overturned: list[ContinuityItem] = []
        questions: list[ContinuityItem] = []
        own_current: list[ContinuityItem] = []
        confirmed_carry: list[ContinuityItem] = []
        unchanged: list[ContinuityItem] = []
        for item in player.items.values():
            previous = baseline.get(item.item_id)
            if item.kind == KIND_REFUTATION:
                (overturned if previous != item.signature() else unchanged).append(item)
                continue
            if item.kind == KIND_QUESTION:
                questions.append(item)
                continue
            if previous is None:
                if not first_time:
                    # 前回提示してから現れた項目（新しい襲撃死・投票・処刑・CO・自分の仮説）は
                    # 「変わった項目」としてすぐ出す。unchanged へ埋めて新情報を隠さない。
                    changed.append(item)
                    continue
                # 初回は「自分の未検証の読み」と「持ち越しの確定状態」を1度だけ示す。
                # 以降は差分のみにして、同じ状態の再掲でプロンプトを増やさない。
                if item.authority == AUTHORITY_ENGINE:
                    confirmed_carry.append(item)
                elif item.authority == AUTHORITY_DERIVED:
                    own_current.append(item)
                else:
                    unchanged.append(item)
                continue
            (changed if previous != item.signature() else unchanged).append(item)

        lines: list[str] = [
            "[CONTINUITY_CHECKPOINT: engine-normalized logical state]",
            (
                "Authority levels differ per line: CONFIRMED = recorded by the game engine; "
                "SAID-unverified = a player publicly stated it; YOUR-UNVERIFIED = your own earlier "
                "inference; NO-LONGER-STANDS = refuted or superseded, so read it as history rather "
                "than a current fact. An `observed DdTt` stamp says when the item appeared. "
                "This section carries state only - it does not order your reasoning, fix your "
                "conclusion, or set your tone; you may change your mind freely and explain why."
            ),
        ]
        marker = self.checkpoints[-1].checkpoint_id if self.checkpoints else "C0"
        truncated = 0
        if first_time and (confirmed_carry or own_current):
            if confirmed_carry:
                ordered = sorted(confirmed_carry, key=lambda i: (i.last_day, i.item_id))
                shown = ordered[-RENDER_CURRENT_MAX:]
                truncated += len(ordered) - len(shown)
                lines.append("- Confirmed state carried over:")
                lines.extend(item.render_line() for item in shown)
                if len(ordered) > len(shown):
                    lines.append(
                        f"- (+{len(ordered) - len(shown)} further carried items are recorded in "
                        f"checkpoint {marker}, where the same authority levels apply.)"
                    )
            if own_current:
                ordered = sorted(own_current, key=lambda i: (i.last_day, i.item_id))
                shown = ordered[-RENDER_CURRENT_MAX:]
                truncated += len(ordered) - len(shown)
                lines.append("- Your standing unverified readings (update or drop them freely):")
                lines.extend(item.render_line() for item in shown)
                if len(ordered) > len(shown):
                    lines.append(
                        f"- (+{len(ordered) - len(shown)} more of your own readings are recorded "
                        f"in checkpoint {marker} as unverified.)"
                    )
        if changed:
            ordered = sorted(changed, key=lambda i: (-i.last_day, i.item_id))
            shown = ordered[-RENDER_CHANGED_MAX:]
            truncated += len(ordered) - len(shown)
            lines.append("- Changed since you last saw this state:")
            lines.extend(item.render_line() for item in shown)
            if len(ordered) > len(shown):
                lines.append(
                    f"- (+{len(ordered) - len(shown)} more changes are recorded in checkpoint "
                    f"{marker}, each with its own authority level and observed stamp.)"
                )
        if overturned:
            ordered = sorted(overturned, key=lambda i: (-i.last_day, i.item_id))
            shown = ordered[-RENDER_OVERTURNED_MAX:]
            truncated += len(ordered) - len(shown)
            lines.append("- Overturned or refuted (history, not current facts):")
            lines.extend(item.render_line() for item in shown)
            if len(ordered) > len(shown):
                lines.append(
                    f"- (+{len(ordered) - len(shown)} further readings no longer stand; they are "
                    f"recorded in checkpoint {marker}.)"
                )
        if questions:
            shown = questions[-RENDER_QUESTIONS_MAX:]
            truncated += len(questions) - len(shown)
            lines.append("- Still unanswered toward you:")
            lines.extend(item.render_line() for item in shown)
            if len(questions) > len(shown):
                lines.append(
                    f"- (+{len(questions) - len(shown)} older open points stay unresolved too.)"
                )
        if unchanged and not first_time:
            ids = ", ".join(sorted(item.item_id for item in unchanged)[:UNCHANGED_SUMMARY_IDS])
            extra = max(0, len(unchanged) - UNCHANGED_SUMMARY_IDS)
            self.unchanged_presented += extra
            lines.append(
                f"- Unchanged since your previous turn: {len(unchanged)} items are not restated "
                f"(they sit in checkpoint {marker}"
                + (f"; ids {ids}" if ids else "")
                + (f", +{extra} more" if extra else "")
                + ")."
            )
        block = "\n".join(lines) + "\n\n"
        referenced_ids = sorted(
            {eid for item in player.items.values() for eid in item.evidence_ids}
        )
        diagnostics: dict[str, object] = {
            "continuity_policy": continuity_policy_label(self.policy),
            "continuity_state_chars": len(block),
            "continuity_items_total": len(player.items),
            "continuity_items_changed": len(changed),
            "continuity_items_overturned": len(overturned),
            "continuity_items_unchanged": len(unchanged),
            "continuity_open_questions": len(questions),
            # 行数上限でこの場に出し切れなかった項目数（状態そのものは失わない）
            "continuity_items_truncated": int(truncated),
            "continuity_evidence_ids": len(referenced_ids),
            "continuity_first_presentation": bool(first_time),
            "continuity_checkpoint_count": len(self.checkpoints),
            "continuity_normalized_days": list(self.normalized_days(day)),
            "continuity_hypothesis_refutations": self.hypothesis_refutations,
            "continuity_questions_resolved": self.questions_resolved,
            "continuity_topic_reintroductions": self.topic_reintroductions,
            "continuity_authority_levels": sorted(
                {item.authority for item in player.items.values()}
            ),
        }
        # 同じ時点でのプロンプト再構築（予算段階のやり直し）で差分基準が進まないようにする
        player.last_block = block
        player.last_diagnostics = dict(diagnostics)
        if first_time or player.last_point != point:
            player.presented = {
                item.item_id: item.signature() for item in player.items.values()
            }
            player.last_point = point
            self.state_chars_by_day[int(day)] = max(
                int(self.state_chars_by_day.get(int(day), 0)), len(block)
            )
            self.evidence_ids_referenced = max(
                self.evidence_ids_referenced, len(referenced_ids)
            )
            self._track_topics(state, actor_name, day)
        return block, diagnostics

    def _track_topics(self, state: "GameState", actor_name: str, day: int) -> None:
        """同一論点の再導入回数を観測する（状態提示が連呼を減らしたかをA/Bで読むため）。"""
        ledger = getattr(state, "topic_ledger", None)
        if ledger is None:
            return
        try:
            summary, _diag = ledger.summarize_for(
                actor_name,
                int(day),
                dead_names=frozenset(
                    player.name
                    for player in getattr(state, "players", ())
                    if not player.is_alive
                ),
            )
        except Exception:  # pragma: no cover - 台帳外の入力に対する保守
            return
        folded = " ".join(str(summary or "").lower().split())
        if not folded:
            return
        seen_earlier = {
            key
            for seen_day, keys in self._seen_topics_by_day.items()
            if seen_day < int(day)
            for key in keys
        }
        if folded in seen_earlier:
            self.topic_reintroductions += 1
        self._seen_topics_by_day.setdefault(int(day), set()).add(folded)

    def checkpoint(
        self, state: "GameState", boundary: str, *, turn: int | None = None
    ) -> ContinuityCheckpoint:
        """エンジン発行の状態チェックポイント（日終了・投票前）。短いID付き行だけを作る。

        モデルの過去推理本文を「要約」という名前で確定事実へ昇格させることはしない。
        確定事実と未検証の公開申告を違う権威レベルの行に分けて残す。
        """
        day = int(getattr(state, "day", 1) or 1)
        point_turn = int(0 if turn is None else turn)
        facts = self._public_fact_items(state, day, point_turn)
        statements = self._public_statement_items(state, day)
        lines: list[str] = [f"[CONTINUITY_CHECKPOINT {point_label(day, point_turn)} @{boundary}]"]
        # 1行あたりの上限（CHECKPOINT_LINE_LIMIT）。状態行は短く保ち、ID列だけが長くなる場合は
        # 継続行へ折り返す（発言全文の写しを載せないという設計上の制約を行長で担保する）。
        grouped: dict[int, list[ContinuityItem]] = {}
        for item in facts:
            grouped.setdefault(item.first_day, []).append(item)
        for group_day in sorted(grouped):
            entries = sorted(grouped[group_day], key=lambda i: i.item_id)
            shown_entries = entries[:3]
            extra = len(entries) - len(shown_entries)
            for entry in shown_entries:
                row = f"- Day {group_day} CONFIRMED: {entry.subject}: {entry.statement}"
                if entry.evidence_ids:
                    row += f"; ids {', '.join(entry.evidence_ids)}"
                lines.append(_clip(row, CHECKPOINT_LINE_LIMIT))
            if extra > 0:
                rest_ids = [eid for item in entries[3:] for eid in item.evidence_ids]
                lines.append(
                    _clip(
                        f"- Day {group_day} CONFIRMED (+{extra} more items): "
                        + (", ".join(rest_ids) if rest_ids else "see the engine ledger"),
                        CHECKPOINT_LINE_LIMIT,
                    )
                )
        if statements:
            by_day: dict[int, int] = {}
            ids_in_order: list[str] = []
            for item in statements:
                by_day[item.first_day] = by_day.get(item.first_day, 0) + 1
                ids_in_order.extend(item.evidence_ids)
            counts = " / ".join(
                f"Day {day_no}: {count} statement(s)" for day_no, count in sorted(by_day.items())
            )
            head = f"- Public statements (truth unverified): {counts}"
            # ID列は280字每条で折り返す（1行に全IDを積まない）
            chunk: list[str] = []
            rows: list[str] = []
            for eid in ids_in_order:
                candidate = ", ".join(chunk + [eid])
                if len(f"{head}; ids {candidate}") > CHECKPOINT_LINE_LIMIT and chunk:
                    rows.append(f"{head}; ids {', '.join(chunk)}" if not rows else f"- ids {', '.join(chunk)}")
                    chunk = [eid]
                else:
                    chunk.append(eid)
            if chunk:
                prefix = head if not rows else "- ids"
                rows.append(f"{prefix}; ids {', '.join(chunk)}")
            remaining = len(ids_in_order) - sum(row.count("[") for row in rows)
            lines.extend(rows[:3])
            if remaining > 0 or len(rows) > 3:
                lines.append(
                    f"- (+further statement ids are recorded in the claim ledger by day number.)"
                )
        lines.append(
            "- Authority levels are kept apart on purpose; an older line is not a new fact."
        )
        self._checkpoint_seq += 1
        record = ContinuityCheckpoint(
            checkpoint_id=f"C{self._checkpoint_seq}",
            boundary=str(boundary),
            day=day,
            turn=point_turn,
            issued_at=point_label(day, point_turn),
            lines=tuple(lines),
            evidence_ids=tuple(
                dict.fromkeys(
                    eid for item in (*facts, *statements) for eid in item.evidence_ids
                )
            ),
            char_count=len("\n".join(lines)),
            normalized_item_ids=tuple(item.item_id for item in (*facts, *statements)),
        )
        if self.enabled:
            self.checkpoints.append(record)
        return record

    def day_growth(self) -> dict[str, object]:
        """日別の状態字数と増加量（単調増加していないかをログで確認できるようにする）。"""
        series = [int(self.state_chars_by_day[day]) for day in sorted(self.state_chars_by_day)]
        deltas = [series[i + 1] - series[i] for i in range(len(series) - 1)]
        return {
            "continuity_chars_by_day": {
                f"D{day}": int(value)
                for day, value in sorted(self.state_chars_by_day.items())
            },
            "continuity_day_deltas": [int(delta) for delta in deltas],
            "continuity_max_day_delta": max((int(delta) for delta in deltas), default=0),
            "continuity_days_increased": sum(1 for delta in deltas if delta > 0),
        }

    def metrics(self) -> dict[str, object]:
        return {
            "continuity_policy": continuity_policy_label(self.policy),
            "continuity_checkpoints": len(self.checkpoints),
            "continuity_checkpoint_char_total": sum(
                int(checkpoint.char_count) for checkpoint in self.checkpoints
            ),
            "continuity_players_tracked": len(self.players),
            "continuity_items_per_player": {
                name: len(player.items) for name, player in sorted(self.players.items())
            },
            "continuity_items_updated": self.items_updated,
            "continuity_items_overturned": self.items_overturned,
            "continuity_hypothesis_refutations": self.hypothesis_refutations,
            "continuity_questions_resolved": self.questions_resolved,
            "continuity_unchanged_presented": self.unchanged_presented,
            "continuity_topic_reintroductions": self.topic_reintroductions,
            "continuity_evidence_ids_referenced": self.evidence_ids_referenced,
            **self.day_growth(),
        }

    def serialize_public(self) -> dict[str, object]:
        """公開ログ用（本人限定の状態・秘匿情報は載せない）。"""
        return {
            "metrics": self.metrics(),
            "checkpoints": [
                {
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "boundary": checkpoint.boundary,
                    "day": checkpoint.day,
                    "turn": checkpoint.turn,
                    "issued_at": checkpoint.issued_at,
                    "char_count": checkpoint.char_count,
                    "evidence_ids": list(checkpoint.evidence_ids),
                }
                for checkpoint in self.checkpoints
            ],
        }


__all__ = [
    "AUTHORITY_DERIVED",
    "AUTHORITY_ENGINE",
    "AUTHORITY_PUBLIC_STATEMENT",
    "AUTHORITY_REFUTED",
    "AUTHORITY_TAGS",
    "CONTINUITY_AUTHORITIES",
    "CONTINUITY_KINDS",
    "CONTINUITY_POLICIES",
    "CONTINUITY_POLICY_DEFAULT",
    "CONTINUITY_POLICY_LEGACY",
    "CONTINUITY_POLICY_NORMALIZED",
    "ContinuityCheckpoint",
    "ContinuityItem",
    "ContinuityState",
    "KIND_CLAIM",
    "KIND_COUNTER",
    "KIND_FACT",
    "KIND_QUESTION",
    "KIND_REFUTATION",
    "KIND_STANCE",
    "KIND_SUPPORT",
    "KIND_VERDICT",
    "PlayerContinuity",
    "RENDER_CHANGED_MAX",
    "RENDER_CURRENT_MAX",
    "RENDER_OVERTURNED_MAX",
    "RENDER_QUESTIONS_MAX",
    "UNCHANGED_SUMMARY_IDS",
    "continuity_policy_label",
    "normalize_continuity_policy",
    "point_label",
]
