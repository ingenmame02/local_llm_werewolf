"""TASK-065配線位置の確認: 名前照合・thinking能力・時系列集計の該当行をUTF-8で書き出す。

意図分離（day_timeline）と名前正規化（unknown_player_reference / 質問回答判定 / 台帳登録）、
モデル×行動別のthinking安定化（能力フラグのスコープ）をどの関数へ繋ぐかを確定するため、
各モジュールの該当シンボル出現行と文脈1行を表示する。読み取り専用・LL不使用。
"""
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "werewolf"
OUT = ROOT / "_analysis" / "task065_wiring.txt"

TARGETS = {
    "llm_actions.py": [
        ("unknown_player_reference", r"unknown_player_reference"),
        ("known-name regex", r"Player\\s\+|プレイヤー\\d|P\\d\+"),
        ("question answer check", r"def _question_answer_check|answers_previous_question"),
        ("thinking capability", r"model_capabilities|thinking_budget|timed_out"),
        ("budget trace api", r"_output_trace\.|record_final_body|annotate_last_attempt|short_retry_prompt"),
        ("context diag", r"context_diagnostics|modelfile_num_ctx|ollama_ps_context|effective_context_resolution"),
        ("transport fail", r"def _transport_failure_reason|reason=."),
    ],
    "prompts.py": [
        ("directed to player", r"def _is_directed_to_player"),
        ("pending speakers line", r"pending|未発言|これから"),
        ("name regex", r"Player\\s\+\|\プレイヤー"),
    ],
    "topic_ledger.py": [("register def", r"def register|player_name|mentions")],
    "hypothesis_ledger.py": [("register def", r"def register|player|mention")],
    "dead_player_resolution.py": [("names", r"def |Player|プレイヤー")],
    "self_consistency.py": [("names", r"def |Player\\d|プレイヤー")],
    "engine.py": [
        ("timeline inspect call", r"inspect_public_body|_timeline_reference_fields"),
        ("ledger registration", r"\.register\(|record_speech|observation_metrics"),
    ],
    "cli.py": [
        ("timeline category", r"_timeline_category_counts|future_event_reference_rate"),
        ("intent counters add", r"citation_status_counts|hedged_pending_citations"),
        ("thinking stats", r"thinking_budget_fallbacks|timeout_count|short_retry"),
        ("flag defs", r"--day-timeline|--continuity|--no-body-reservation"),
        ("result metrics", r"\"avg_|ledger_metrics\["),
    ],
    "output_budget_trace.py": [
        ("public api", r"^def |^class |^[A-Z_]+:"),
        ("scoped?", r"model|budget_key|scope"),
    ],
}


def main() -> None:
    chunks: list[str] = []
    for name, entries in TARGETS.items():
        path = SRC / name
        lines = io.open(path, encoding="utf-8").read().splitlines()
        blocks: list[str] = []
        total_lines = len(lines)
        for label, pattern in entries:
            out: list[str] = []
            for index, line in enumerate(lines):
                try:
                    hit = line.rstrip().find(pattern) >= 0
                except Exception:  # noqa: BLE001 - regex fallback below
                    hit = False
            # pattern is treated as plain substring OR regex (whichever matches more reliably)
            import re

            rx = re.compile(pattern) if any(ch in pattern for ch in "\\[(") else None
            for index, line in enumerate(lines):
                text = line.strip()
                matched = bool(rx.search(text)) if rx else (pattern in text)
                if matched:
                    out.append(f"{index + 1}: {text[:120]}")
            blocks.append(f"--- [{label}] {pattern!r} hits={len(out)}\n" + "\n".join(out[:30]))
        chunks.append(f"===== {name} ({total_lines} lines)\n" + "\n".join(blocks))
    OUT.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
