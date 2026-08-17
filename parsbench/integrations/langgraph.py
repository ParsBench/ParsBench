"""LangChain/LangGraph → Trace. Feed a graph result/state with `messages`
(LangChain BaseMessage objects or plain dicts)."""

from parsbench.appeval.trace import Trace

_ROLES = {"human": "user", "ai": "assistant", "tool": "tool", "system": "system"}


def to_trace(state) -> Trace:
    msgs = (state.get("messages") or []) if isinstance(state, dict) else state
    out: list[dict] = []
    for m in msgs:
        if isinstance(m, dict):
            out.append(m)
            continue
        content = getattr(m, "content", "")
        entry: dict = {
            "role": _ROLES.get(getattr(m, "type", ""), "assistant"),
            "content": content if isinstance(content, str) else str(content),
        }
        tool_calls = getattr(m, "tool_calls", None) or []
        if tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.get("id"),
                    "function": {"name": tc.get("name", ""), "arguments": tc.get("args", {})},
                }
                for tc in tool_calls
            ]
        if getattr(m, "tool_call_id", None):
            entry["tool_call_id"] = m.tool_call_id
        out.append(entry)
    return Trace.from_messages(out, raw=state)
