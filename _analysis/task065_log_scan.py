"""TASK-065: 基準ログ game_425755ce9a83 の残課題を数える（読み取り専用・LL不使用）。

観点:
1. コンテキスト整合: configured / server_observed / loaded(/api/ps) / requested の不一致
2. 出力予算: thinking-only / generation_cap / timeout / transport / output_format を
   モデル×行動別に分ける
3. 時系列検出: speech_from_unspoken_player の内容（質問行為だったか）
4. プレイヤー名表記ゆれ: unknown_player_reference の元になった本文の実態
"""
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "game_425755ce9a83.private.jsonl"


def rows():
    with io.open(LOG, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    data = list(rows())
    out = []
    out.append(f"events={len(data)}")

    out.append("\n== 1. context resolution ==")
    for rec in data:
        if rec.get("type") == "effective_context_resolution":
            out.append(
                f"{rec.get('player')}/{rec.get('model')} configured={rec.get('configured_context_limit')}"
                f" server_observed={rec.get('server_observed_context_limit')}({rec.get('server_observed_context_source')})"
                f" adopted={rec.get('effective_context_limit')} conflict={rec.get('context_limit_conflict')}"
                f" candidates={rec.get('context_budget_candidates')}"
            )
    out.append("\n== loaded (/api/ps) keys on attempts ==")
    ps_keys = Counter()
    for rec in data:
        for key in (
            "ollama_ps_context",
            "ollama_ps_processor",
            "modelfile_num_ctx",
            "loaded_context_limit",
            "effective_num_ctx",
            "requested_num_ctx",
            "num_ctx_source",
        ):
            if key in rec:
                ps_keys[key] += 1
    out.append(str(ps_keys))

    out.append("\n== 2. generation attempts (model x action x category fields) ==")
    per = Counter()
    for rec in data:
        if rec.get("type") != "llm_generation_attempt":
            continue
        meta = rec.get("response_metadata") or {}
        cat = (
            rec.get("failure_reason"),
            rec.get("done_reason"),
            bool(rec.get("timeout")),
            bool(rec.get("transport_error") or rec.get("error")),
            rec.get("output_budget_category") or rec.get("budget_failure_category"),
        )
        per[(rec.get("player"), rec.get("model"), rec.get("action"), cat)] += 1
    for key, count in sorted(per.items(), key=lambda kv: str(kv[0])):
        out.append(f"{count} {key}")

    out.append("\n== retries / timeout events ==")
    for rec in data:
        if rec.get("timeout") or rec.get("timeout_retries") or rec.get("transport_retries"):
            out.append(
                f"type={rec.get('type')} player={rec.get('player')} model={rec.get('model')}"
                f" action={rec.get('action')} timeout={rec.get('timeout')}"
                f" timeout_retries={rec.get('timeout_retries')} transport_retries={rec.get('transport_retries')}"
                f" retry_strategy={rec.get('retry_strategy')} retry_prompt_style={rec.get('retry_prompt_style')}"
                f" elapsed={rec.get('elapsed_seconds')}"
            )

    out.append("\n== 3. timeline findings (future references) ==")
    for rec in data:
        findings = rec.get("future_event_reference_findings")
        if findings:
            out.append(
                f"player={rec.get('player')} day={rec.get('day')} turn? action={rec.get('action')}"
                f" cats={rec.get('future_event_reference_categories')}"
            )
            for f in findings:
                out.append(f"   - {f}")

    out.append("\n== 4. unknown_player_reference raw bodies ==")
    pattern = re.compile(r"(?:Player|player|PLAYER)\s*(\d+)")
    for rec in data:
        reason_blob = json.dumps(rec, ensure_ascii=False)
        if "unknown_player_reference" not in reason_blob:
            continue
        raw_keys = [k for k in rec if "resp" in k or "raw" in k or "text" in k or "body" in k]
        snippet = ""
        for key in ("response", "raw_response", "original_text_before_repair", "candidate_text"):
            value = rec.get(key)
            if isinstance(value, str) and value:
                snippet = value
                break
        names = sorted(set(pattern.findall(snippet))) if snippet else []
        out.append(
            f"type={rec.get('type')} player={rec.get('player')} reason={rec.get('reason')}"
            f" text_keys={raw_keys} names_in_snippet={names}"
        )
        if snippet:
            for sent in re.split(r"[。！？\n]", snippet):
                m = pattern.search(sent)
                if m and (m.group(0) != f"Player {m.group(1)}"):
                    out.append(f"   variant-> {sent[:120]}")

    out.append("\n== question checks / answers ==")
    qc = Counter()
    for rec in data:
        if rec.get("type") == "discussion_question_check":
            qc[(bool(rec.get("question_directed")), bool(rec.get("question_answered")))] += 1
    out.append(str(qc))

    print("\n".join(out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
