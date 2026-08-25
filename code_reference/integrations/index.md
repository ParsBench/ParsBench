# Integrations

Source code in `parsbench/integrations/openai_agents.py`

```
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
```

Source code in `parsbench/integrations/langgraph.py`

```
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
```

Source code in `parsbench/integrations/pydantic_ai.py`

```
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
```

Source code in `parsbench/integrations/agno.py`

```
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
```

Duck-typed OTel SpanProcessor that records finished spans.

Source code in `parsbench/integrations/otel.py`

```
class TraceCollector:
    """Duck-typed OTel SpanProcessor that records finished spans."""

    def __init__(self):
        self.spans = []

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        self.spans.append(span)

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000):
        return True

    def to_trace(self) -> Trace:
        return spans_to_trace(self.spans)
```

Source code in `parsbench/integrations/otel.py`

```
def spans_to_trace(spans) -> Trace:
    messages = []
    final = ""
    for span in spans:
        attrs = _attrs(span)
        op = attrs.get("gen_ai.operation.name")
        kind = (attrs.get("openinference.span.kind") or "").upper()

        if op == "execute_tool" or kind == "TOOL":
            name = attrs.get("gen_ai.tool.name") or attrs.get("tool.name") or _name(span)
            args = _json(
                attrs.get("gen_ai.tool.call.arguments")
                or attrs.get("input.value")
                or {}
            )
            if not isinstance(args, dict):
                args = {"_raw": args}
            result = attrs.get("gen_ai.tool.call.result") or attrs.get("output.value")
            call_id = attrs.get("gen_ai.tool.call.id") or _name(span)
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": call_id, "function": {"name": name, "arguments": args}}
                    ],
                }
            )
            if result is not None:
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": str(result)}
                )
        elif op in ("chat", "invoke_agent") or kind in ("LLM", "AGENT", "CHAIN"):
            text = _last_text(
                attrs.get("gen_ai.output.messages") or attrs.get("output.value")
            )
            if text:
                final = text

    if final:
        messages.append({"role": "assistant", "content": final})
    return Trace.from_messages(messages, raw=list(spans))
```

Source code in `parsbench/integrations/adk.py`

```
def persian_response_match(reference: str, response: str) -> float:
    ref = normalize(reference).split()
    got = normalize(response).split()
    if not ref or not got:
        return float(ref == got)
    common: dict[str, int] = {}
    for token in ref:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in got:
        if common.get(token, 0) > 0:
            common[token] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(got)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)
```

Source code in `parsbench/integrations/langfuse.py`

```
def push(evaluation_result, *, name: str = "parsbench", host: str | None = None,
         public_key: str | None = None, secret_key: str | None = None,
         _post=None):
    host = (host or os.environ["LANGFUSE_HOST"]).rstrip("/")
    public_key = public_key or os.environ["LANGFUSE_PUBLIC_KEY"]
    secret_key = secret_key or os.environ["LANGFUSE_SECRET_KEY"]

    now = datetime.now(timezone.utc).isoformat()
    trace_id = str(uuid.uuid4())
    batch = [
        {
            "id": str(uuid.uuid4()),
            "type": "trace-create",
            "timestamp": now,
            "body": {"id": trace_id, "name": name},
        }
    ]
    for golden_result in evaluation_result.golden_results:
        for check_result in golden_result.check_results:
            if check_result.skipped:
                continue
            batch.append(
                {
                    "id": str(uuid.uuid4()),
                    "type": "score-create",
                    "timestamp": now,
                    "body": {
                        "id": str(uuid.uuid4()),
                        "traceId": trace_id,
                        "name": check_result.check,
                        "value": check_result.score,
                        "comment": f"{golden_result.golden_name}"
                        + (f" — {check_result.reason}" if check_result.reason else ""),
                    },
                }
            )

    if _post is None:  # pragma: no cover - exercised via injection in tests
        import requests

        _post = requests.post
    response = _post(
        f"{host}/api/public/ingestion",
        json={"batch": batch},
        auth=(public_key, secret_key),
        timeout=30,
    )
    response.raise_for_status()
    return trace_id
```
