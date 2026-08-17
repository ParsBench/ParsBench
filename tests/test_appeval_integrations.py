"""App-eval integrations: pass^k, judge resolution, adapters, OTel, Langfuse."""

from types import SimpleNamespace

from parsbench.appeval import AppEvaluator, Golden, ToolCall
from parsbench.appeval.judge import resolve_judge
from parsbench.integrations import adk, agno, langfuse, langgraph, openai_agents, otel, pydantic_ai


# --- pass^k ------------------------------------------------------------------

def test_pass_hat_k_flaky_app():
    calls = {"n": 0}

    def flaky(_):
        calls["n"] += 1
        return "پاسخ درست" if calls["n"] % 2 == 1 else "غلط"

    golden = Golden(input="سلام", contains=["درست"])
    result = AppEvaluator([golden]).evaluate(flaky, n_runs=4)
    assert result.golden_results[0].run_passes == [True, False, True, False]
    assert result.pass_hat_k(1) == 0.5
    assert result.pass_hat_k(4) == 0.0

    result = AppEvaluator([golden]).evaluate(lambda _: "پاسخ درست", n_runs=3)
    assert result.pass_hat_k() == 1.0


# --- judge resolution --------------------------------------------------------

def test_resolve_judge_passthrough_and_env(monkeypatch):
    fn = lambda p: "نمره: ۵"
    assert resolve_judge(fn) is fn
    monkeypatch.delenv("PARSBENCH_JUDGE", raising=False)
    assert resolve_judge(None) is None
    monkeypatch.setenv("PARSBENCH_JUDGE", "gpt-x")
    monkeypatch.setenv("PARSBENCH_JUDGE_API_KEY", "test")
    assert callable(resolve_judge(None))  # builds an OpenAI-compatible caller


def test_judge_client_gets_production_retry_and_timeout(monkeypatch):
    import openai

    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=None))

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    from parsbench.appeval.judge import resolve_model

    resolve_model("gpt-x")
    assert seen.get("max_retries", 0) >= 3    # batch evals must ride out 429s
    assert seen.get("timeout") is not None    # and never hang a CI job forever


def test_resolve_model_temperature_none_omits_param(monkeypatch):
    import openai

    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content="سلام")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    from parsbench.appeval.judge import resolve_model

    caller = resolve_model("gpt-x", temperature=None)
    caller("پیام")
    assert "temperature" not in calls[0]


def test_judge_drops_temperature_when_model_rejects_it(monkeypatch):
    # reasoning models (gpt-5 family, o-series) 400 on temperature=0
    import httpx
    import openai

    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if "temperature" in kwargs:
                raise openai.BadRequestError(
                    "Unsupported value: 'temperature' does not support 0 with this model.",
                    response=httpx.Response(400, request=httpx.Request("POST", "http://j")),
                    body=None,
                )
            message = SimpleNamespace(content="نمره: ۵")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    judge = resolve_judge("gpt-reasoning")
    assert judge("سلام") == "نمره: ۵"
    assert "temperature" not in calls[-1]
    judge("دوباره")  # the fallback sticks — no wasted failing call per prompt
    assert len(calls) == 3 and "temperature" not in calls[-1]


# --- adapters ----------------------------------------------------------------

def test_openai_agents_adapter():
    run_result = SimpleNamespace(
        to_input_list=lambda: [
            {"role": "user", "content": "بلیط"},
            {"type": "function_call", "call_id": "c1", "name": "search_flights",
             "arguments": '{"date": "1405-07-05"}'},
            {"type": "function_call_output", "call_id": "c1", "output": "پرواز ۸"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "پرواز ساعت ۸"}]},
        ],
        final_output="پرواز ساعت ۸",
    )
    trace = openai_agents.to_trace(run_result)
    assert trace.tool_calls[0].name == "search_flights"
    assert trace.tool_calls[0].arguments == {"date": "1405-07-05"}
    assert trace.final_output == "پرواز ساعت ۸"


def test_langgraph_adapter():
    msgs = [
        SimpleNamespace(type="human", content="بلیط", tool_calls=None, tool_call_id=None),
        SimpleNamespace(type="ai", content="", tool_call_id=None,
                        tool_calls=[{"id": "1", "name": "search_flights",
                                     "args": {"date": "۱۴۰۵/۰۷/۰۵"}}]),
        SimpleNamespace(type="tool", content="پرواز ۸", tool_calls=None, tool_call_id="1"),
        SimpleNamespace(type="ai", content="پرواز ساعت ۸", tool_calls=None, tool_call_id=None),
    ]
    trace = langgraph.to_trace({"messages": msgs})
    assert trace.tool_calls[0].arguments == {"date": "۱۴۰۵/۰۷/۰۵"}
    assert trace.final_output == "پرواز ساعت ۸"
    # end to end with Jalali/Gregorian equivalence through the adapter
    golden = Golden(input="بلیط", tools=[ToolCall("search_flights", date="2026-09-27")])
    result = AppEvaluator([golden]).score_traces([trace])
    assert result.passed


def test_pydantic_ai_adapter():
    msgs = [
        SimpleNamespace(parts=[SimpleNamespace(part_kind="user-prompt", content="بلیط")]),
        SimpleNamespace(parts=[SimpleNamespace(part_kind="tool-call", tool_name="search_flights",
                                               args={"date": "1405-07-05"}, tool_call_id="c1")]),
        SimpleNamespace(parts=[SimpleNamespace(part_kind="tool-return", content="پرواز ۸",
                                               tool_call_id="c1")]),
        SimpleNamespace(parts=[SimpleNamespace(part_kind="text", content="پرواز ساعت ۸")]),
    ]
    trace = pydantic_ai.to_trace(SimpleNamespace(all_messages=lambda: msgs, output="پرواز ساعت ۸"))
    assert trace.tool_calls[0].name == "search_flights"
    assert trace.final_output == "پرواز ساعت ۸"


def test_agno_adapter():
    run = SimpleNamespace(
        content="پرواز ساعت ۸، قیمت ۲٬۵۰۰٬۰۰۰ ریال",
        messages=[
            SimpleNamespace(role="user", content="بلیط", tool_calls=None, tool_call_id=None),
            SimpleNamespace(role="assistant", content=None, tool_call_id=None,
                            tool_calls=[{"id": "1", "function": {
                                "name": "search_flights",
                                "arguments": '{"date": "1405-07-05"}'}}]),
            SimpleNamespace(role="tool", content="پرواز ۸", tool_calls=None, tool_call_id="1"),
            SimpleNamespace(role="assistant", content="پرواز ساعت ۸، قیمت ۲٬۵۰۰٬۰۰۰ ریال",
                            tool_calls=None, tool_call_id=None),
        ],
    )
    trace = agno.to_trace(run)
    assert trace.tool_calls[0].arguments == {"date": "1405-07-05"}
    assert trace.final_output == "پرواز ساعت ۸، قیمت ۲٬۵۰۰٬۰۰۰ ریال"
    golden = Golden(input="بلیط", tools=[ToolCall("search_flights", date="2026-09-27")],
                    contains=["250 هزار تومان"])
    assert AppEvaluator([golden]).score_traces([trace]).passed


def test_otel_spans_both_dialects():
    spans = [
        {"name": "execute_tool search_flights",
         "attributes": {"gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": "search_flights",
                        "gen_ai.tool.call.arguments": '{"date": "1405-07-05"}'}},
        {"name": "chat",
         "attributes": {"gen_ai.operation.name": "chat",
                        "gen_ai.output.messages":
                            '[{"role": "assistant", "content": "پرواز ساعت ۸"}]'}},
    ]
    trace = otel.spans_to_trace(spans)
    assert trace.tool_calls[0].arguments == {"date": "1405-07-05"}
    assert trace.final_output == "پرواز ساعت ۸"

    collector = otel.TraceCollector()
    collector.on_end({"name": "my_tool",
                      "attributes": {"openinference.span.kind": "TOOL",
                                     "tool.name": "search_flights",
                                     "input.value": '{"date": "1405-07-05"}',
                                     "output.value": "پرواز ۸"}})
    collector.on_end({"name": "llm",
                      "attributes": {"openinference.span.kind": "LLM",
                                     "output.value": "پرواز ساعت ۸"}})
    trace = collector.to_trace()
    assert trace.tool_calls[0].name == "search_flights"
    assert trace.final_output == "پرواز ساعت ۸"


# --- adk metric + langfuse export -------------------------------------------

def test_persian_response_match():
    assert adk.persian_response_match("پرواز ساعت ۸ صبح", "پرواز ساعت ۸ صبح") == 1.0
    assert adk.persian_response_match("پرواز ساعت ۸", "پرواز ساعت 8") == 1.0  # digits
    assert adk.persian_response_match("الف ب", "ج د") == 0.0


def test_result_to_langfuse_method():
    result = AppEvaluator([Golden(input="س", contains=["الف"])]).score_traces(
        [[{"role": "assistant", "content": "الف"}]]
    )
    posted = {}

    def fake_post(url, json=None, auth=None, timeout=None):
        posted["url"] = url

        class Response:
            def raise_for_status(self):
                pass

        return Response()

    trace_id = result.to_langfuse(host="http://lf.local", public_key="pk",
                                  secret_key="sk", _post=fake_post)
    assert trace_id and posted["url"] == "http://lf.local/api/public/ingestion"


def test_langfuse_push_payload():
    golden = Golden(input="سلام", contains=["درود"])
    result = AppEvaluator([golden]).evaluate(lambda _: "درود بر تو")

    sent = {}

    def fake_post(url, json=None, auth=None, timeout=None):
        sent.update(url=url, payload=json, auth=auth)
        return SimpleNamespace(raise_for_status=lambda: None)

    trace_id = langfuse.push(result, host="http://lf.local", public_key="pk",
                             secret_key="sk", _post=fake_post)
    assert sent["url"] == "http://lf.local/api/public/ingestion"
    types = [e["type"] for e in sent["payload"]["batch"]]
    assert types[0] == "trace-create" and "score-create" in types
    score = next(e for e in sent["payload"]["batch"] if e["type"] == "score-create")
    assert score["body"]["traceId"] == trace_id
    assert score["body"]["value"] == 1.0
