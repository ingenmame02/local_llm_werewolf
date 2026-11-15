import json, io, sys

SKIP = {"type", "timestamp", "game_id", "sequence", "day", "phase", "phase_label",
        "private", "discussion_claim_mode", "elapsed_seconds", "player", "model", "action"}


def dump(src, dst, skip_types=()):
    out = io.open(dst, "w", encoding="utf-8")
    for line in io.open(src, encoding="utf-8"):
        e = json.loads(line)
        t = e["type"]
        if t in skip_types:
            continue
        keys = [k for k in e.keys() if k not in SKIP]
        out.write("[{}] d{} {} {} {}\n".format(e["sequence"], e.get("day"), t,
                                               e.get("player", ""), e.get("action", "")))
        for k in keys:
            s = json.dumps(e[k], ensure_ascii=False)
            out.write("   {}: {}\n".format(k, s[:1500]))
        out.write("\n")
    out.close()


dump("logs/game_0055be638e37.jsonl", "_analysis_public.txt",
     ("llm_request_configuration", "llm_action_started", "discussion_context_selected"))
dump("logs/game_0055be638e37.private.jsonl", "_analysis_private.txt",
     ("llm_request_configuration", "llm_action_started", "discussion_context_selected"))
print("ok")
