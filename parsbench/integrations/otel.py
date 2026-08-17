"""OpenTelemetry spans → Trace.

TraceCollector quacks like an OTel SpanProcessor, so it plugs straight into
any OTel-instrumented app (Pydantic AI, LangGraph, ADK, instrumented OpenAI
Agents SDK) without parsbench depending on opentelemetry:

    collector = TraceCollector()
    tracer_provider.add_span_processor(collector)
    ... run the app ...
    trace = collector.to_trace()

spans_to_trace() understands the three attribute dialects in the wild:
OTel gen_ai.* semconv, OpenInference, and OpenLLMetry-ish variants.
ponytail: heuristic mapping over experimental conventions — extend the
attribute lookups as the semconv stabilizes, don't restructure.
"""

import json

from parsbench.appeval.trace import Trace


def _attrs(span) -> dict:
    if isinstance(span, dict):
        return span.get("attributes") or {}
    return dict(getattr(span, "attributes", None) or {})


def _name(span) -> str:
    if isinstance(span, dict):
        return span.get("name", "")
    return getattr(span, "name", "") or ""


def _json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _last_text(value) -> str | None:
    """Pull assistant text out of gen_ai.output.messages-style payloads,
    and out of whole-response-object payloads (OpenInference instrumentors
    often record `output.value` as the serialized response: `{"choices":
    [...]}`, `{"generations": [[...]]}`, `{"content": ...}`)."""
    value = _json(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "output", "message"):
            if value.get(key) is not None:
                found = _last_text(value[key])
                if found:
                    return found
        for key in ("choices", "generations", "messages"):
            items = value.get(key)
            if isinstance(items, list) and items:
                found = _last_text(items[-1])
                if found:
                    return found
        return None
    if isinstance(value, list) and value:
        last = value[-1]
        if isinstance(last, list):  # LangChain generations: [[{"text": ...}]]
            return _last_text(last)
        if isinstance(last, dict):
            content = last.get("content") or last.get("parts")
            if isinstance(content, list):
                return "".join(
                    str(p.get("content") or p.get("text") or "") if isinstance(p, dict)
                    else str(p)
                    for p in content
                )
            if content is not None:
                return (
                    _last_text(content)
                    if isinstance(content, (str, dict))
                    else str(content)
                )
            return _last_text(last)  # e.g. {"message": {"content": ...}}
    return None


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
