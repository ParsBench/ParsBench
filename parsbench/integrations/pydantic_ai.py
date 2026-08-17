"""Pydantic AI → Trace. Feed an AgentRunResult (or its all_messages())."""

import json

from parsbench.appeval.trace import Trace


def to_trace(result) -> Trace:
    msgs = result.all_messages() if hasattr(result, "all_messages") else result
    out: list[dict] = []
    for m in msgs:
        for part in getattr(m, "parts", []):
            kind = getattr(part, "part_kind", "")
            if kind == "user-prompt":
                out.append({"role": "user", "content": str(part.content)})
            elif kind == "system-prompt":
                out.append({"role": "system", "content": str(part.content)})
            elif kind == "text":
                out.append({"role": "assistant", "content": part.content})
            elif kind == "tool-call":
                args = getattr(part, "args", {})
                if not isinstance(args, (dict, str)):
                    args = json.loads(getattr(part, "args_as_json_str", lambda: "{}")())
                out.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": getattr(part, "tool_call_id", None),
                                "function": {"name": part.tool_name, "arguments": args},
                            }
                        ],
                    }
                )
            elif kind == "tool-return":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": getattr(part, "tool_call_id", ""),
                        "content": str(part.content),
                    }
                )
    return Trace.from_messages(out, final_output=getattr(result, "output", None),
                               raw=result)
