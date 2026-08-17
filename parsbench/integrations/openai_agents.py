"""OpenAI Agents SDK → Trace. Feed a RunResult (or its to_input_list())."""

from parsbench.appeval.trace import Trace


def to_trace(run_result) -> Trace:
    items = (
        run_result.to_input_list()
        if hasattr(run_result, "to_input_list")
        else list(run_result)
    )
    messages: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": item.get("call_id"),
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "{}"),
                            },
                        }
                    ],
                }
            )
        elif kind == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": str(item.get("output", "")),
                }
            )
        elif "role" in item:
            content = item.get("content")
            if isinstance(content, list):  # Responses-format content parts
                content = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            messages.append({"role": item["role"], "content": content})

    return Trace.from_messages(
        messages, final_output=getattr(run_result, "final_output", None),
        raw=run_result,
    )
