"""TASK-065レビュー対応の効果測定（モック・LL不使用）。

`game_425755ce9a83` の Player 1 相当（設定 num_ctx=16384 / Modelfile・env=4096、
起動時はモデル未ロードで /api/ps=None → 送信後に実ロード値 4096）を再現し、
次の3点を数値で示す:

1. cap方策では解決結果の採用値が縮退し、実際の送信num_ctxへ反映される
   （修正前は `adopted_num_ctx=4096` でも送信側は 16384 を使い続けていた）
2. warn方策（既定）は送信値を変えない
3. provisional解決は送信後に実ロード値を取り直す（キャッシュが固定されない）

実機Ollamaには一切接続しない。
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "_analysis" / "task065_context_fix_probe_out.txt"

from werewolf.llm_actions import OllamaActions  # noqa: E402
from werewolf.setup import create_game  # noqa: E402


class CaptureClient:
    """ps を「未ロード→送信後」と変化させるモック。送信オプションを記録する。"""

    def __init__(self, *, modelfile: int, ps_after: int | None) -> None:
        self.timeout = 30.0
        self.max_retries = 1
        self.timeout_retries = 0
        self._modelfile = modelfile
        self._ps_after = ps_after
        self.sends = 0
        self.sent_ctx: list[object] = []

    def context_diagnostics(self, model: str) -> dict[str, object]:
        return {
            "ollama_ps_context": (self._ps_after if self.sends else None),
            "modelfile_num_ctx": self._modelfile,
        }

    def chat(self, model, messages, options=None):
        self.sends += 1
        self.sent_ctx.append((options or {}).get("num_ctx"))
        return "[CO] None\n[BODY] 了解しました。情報集めを続けます。"


def run_case(policy: str, *, configured: int, modelfile: int, ps_after, seed: int) -> list[str]:
    os.environ["OLLAMA_NUM_CTX"] = str(modelfile)
    os.environ.pop("OLLAMA_CONTEXT_LENGTH", None)
    state = create_game(seed=seed, config_path="config/players.json", response_language="日本語")
    state.context_integrity_policy = policy
    actor = state.players[0]
    actor.num_ctx = configured
    actor.recommended_num_ctx = None
    actor.max_num_ctx = None
    events: list[dict] = []
    client = CaptureClient(modelfile=modelfile, ps_after=ps_after)
    actions = OllamaActions(state, client, seed=seed, event_callback=events.append)
    lines: list[str] = [f"### policy={policy} configured={configured} env/modelfile={modelfile} loaded_after_send={ps_after}"]

    res1 = actions._resolve_context_budget(actor, "discussion", time.perf_counter())
    lines.append(
        f"  initial   : limit={res1.limit} source={res1.source} provisional={res1.provisional}"
        f" loaded={res1.loaded_context_from_ps}"
    )
    actions.discuss_body(actor.name, 1, tuple(state.alive_players), tuple(state.conversation_log))
    sent1 = client.sent_ctx[-1]
    lines.append(f"  send#1    : num_ctx={sent1}")
    res2 = actions._resolve_context_budget(actor, "discussion", time.perf_counter())
    lines.append(
        f"  refresh   : limit={res2.limit} source={res2.source} provisional={res2.provisional}"
        f" loaded={res2.loaded_context_from_ps}"
    )
    actions.discuss_body(actor.name, 1, tuple(state.alive_players), tuple(state.conversation_log))
    sent2 = client.sent_ctx[-1]
    lines.append(f"  send#2    : num_ctx={sent2}")

    resolution = [e for e in events if e["type"] == "effective_context_resolution"]
    stages = [e.get("context_resolution_stage") for e in resolution]
    adopted = [e.get("adopted_num_ctx") for e in resolution]
    lines.append(f"  stages    : {stages} adopted={adopted}")
    configs = [e for e in events if e["type"] == "llm_request_configuration"]
    lines.append(
        "  cfg log   : "
        + " / ".join(
            f"requested={c.get('requested_num_ctx')} eff_limit={c.get('effective_context_limit')}"
            f" src_after={c.get('context_budget_source_after_integrity')}"
            for c in configs[:2]
        )
    )
    ok = "n/a"
    if policy == "cap":
        ok = "PASS" if isinstance(sent1, int) and isinstance(sent2, int) and sent2 <= 4096 else "FAIL"
    lines.append(f"  verdict   : {ok}")
    lines.append("")
    return lines


def main() -> None:
    out: list[str] = []
    out.extend(run_case("cap", configured=16384, modelfile=4096, ps_after=None, seed=2065101))
    out.extend(run_case("warn", configured=16384, modelfile=4096, ps_after=None, seed=2065102))
    out.extend(run_case("cap", configured=16384, modelfile=4096, ps_after=4096, seed=2065103))
    out.extend(run_case("off", configured=16384, modelfile=4096, ps_after=None, seed=2065104))
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
