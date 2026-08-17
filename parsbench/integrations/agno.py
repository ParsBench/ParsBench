"""Agno → Trace. Feed a RunOutput/RunResponse (or its .messages list)."""

from parsbench.appeval.trace import Trace


def to_trace(run) -> Trace:
    msgs = getattr(run, "messages", None) or (run if isinstance(run, list) else [])
    out = []
    for m in msgs:
        if isinstance(m, dict):
            out.append(m)
            continue
        content = getattr(m, "content", None)
        entry = {
            "role": getattr(m, "role", "assistant"),
            "content": content if content is None or isinstance(content, str) else str(content),
        }
        if getattr(m, "tool_calls", None):
            entry["tool_calls"] = m.tool_calls  # already OpenAI-format dicts
        if getattr(m, "tool_call_id", None):
            entry["tool_call_id"] = m.tool_call_id
        out.append(entry)
    return Trace.from_messages(out, final_output=getattr(run, "content", None),
                               raw=run)
