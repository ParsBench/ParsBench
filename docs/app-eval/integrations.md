# Framework integrations

The evaluator doesn't care what your app is built with. It scores a `Trace`,
and there are three ways to produce one:

1. Return a string or an OpenAI-format message list from your app function.
   `evaluate()` builds the trace for you. No integration needed.
2. Use an adapter to convert your framework's run object into a `Trace`.
3. Collect spans via OpenTelemetry for frameworks without an adapter.

## Adapters

Every adapter is duck-typed. Importing it never requires the framework to be
installed, so `parsbench` stays dependency-light:

```python
from parsbench.integrations import openai_agents, langgraph, pydantic_ai, agno

trace = openai_agents.to_trace(run_result)     # OpenAI Agents SDK RunResult
trace = langgraph.to_trace(state)              # LangGraph graph state
trace = pydantic_ai.to_trace(result)           # Pydantic AI AgentRunResult
trace = agno.to_trace(run)                     # Agno RunOutput
```

Typical use inside `evaluate()`:

```python
from agents import Runner
from parsbench.integrations import openai_agents

def app(message):
    run_result = Runner.run_sync(my_agent, message)
    return openai_agents.to_trace(run_result)

result = evaluator.evaluate(app)
```

Or run the app yourself and score afterwards:

```python
result = evaluator.score_traces([openai_agents.to_trace(r) for r in run_results])
```

## OpenTelemetry: the universal path

Anything OTel-instrumented (CrewAI, LlamaIndex, Google ADK, instrumented
LangChain) feeds through the universal collector. No adapter needed:

```python
from parsbench.integrations.otel import TraceCollector

collector = TraceCollector()
tracer_provider.add_span_processor(collector)
...   # run your app
result = evaluator.score_traces([collector.to_trace()])
```

The collector reads the GenAI semantic-convention attributes that these
frameworks emit (messages, tool calls, model usage) and assembles them into a
`Trace`. See
[`examples/with_crewai_otel.py`](https://github.com/ParsBench/ParsBench/blob/main/examples/with_crewai_otel.py)
for a complete CrewAI setup.

## Building a Trace by hand

When nothing above fits (a bot behind an HTTP API, logs replayed from
production), construct traces directly:

```python
from parsbench.appeval import Message, ToolCall, Trace

trace = Trace(
    messages=[
        Message(role="user", content="..."),
        Message(role="assistant", tool_calls=[ToolCall("search_flights", date="1405-07-05")]),
        Message(role="assistant", content="..."),
    ],
    final_output="...",
    latency=1.9,          # feeds the max_latency budget check
    cost=0.0004,          # feeds the max_cost budget check
)

result = evaluator.score_traces([trace])
```

`Trace.from_messages([...])` builds one from OpenAI-format dicts, which is
also what `evaluate()` does when your app returns a message list.

## Google ADK

For Google ADK agents, collect via OTel as above. The
`parsbench.integrations.adk` module additionally provides
`persian_response_match`, a drop-in replacement for ADK's own
`response_match_score`. ADK's default is ROUGE-1 with an ASCII-oriented
tokenizer, which collapses on Perso-Arabic script; the replacement computes
token-level F1 over Persian-normalized tokens.

## Langfuse

`result.to_langfuse()` pushes per-check scores to a Langfuse instance for
dashboarding alongside your production traces. See
[CI & regression tracking](ci.md#exporting-results).
