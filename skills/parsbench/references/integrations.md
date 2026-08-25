# Framework integrations

The evaluator scores a `Trace`. Three ways to produce one:

1. Return a string or an OpenAI-format message list from the app function;
   `evaluate()` builds the trace. No integration needed.
2. Use an adapter to convert a framework run object into a `Trace`.
3. Collect spans via OpenTelemetry for frameworks without an adapter.

## Adapters

Duck-typed; importing an adapter never requires the framework installed:

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
`evaluator.score_traces([openai_agents.to_trace(r) for r in run_results])`.

## OpenTelemetry: the universal path

Anything OTel-instrumented (CrewAI, LlamaIndex, Google ADK, instrumented
LangChain):

```python
from parsbench.integrations.otel import TraceCollector

collector = TraceCollector()
tracer_provider.add_span_processor(collector)
...   # run the app
result = evaluator.score_traces([collector.to_trace()])
```

The collector reads GenAI semantic-convention attributes (messages, tool
calls, model usage). Complete CrewAI setup: `examples/with_crewai_otel.py`
in the repo.

## Building a Trace by hand

For bots behind an HTTP API or replayed production logs:

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

`Trace.from_messages([...])` builds one from OpenAI-format dicts.

## Google ADK

Collect via OTel as above. `parsbench.integrations.adk` also provides
`persian_response_match`, a drop-in replacement for ADK's
`response_match_score`: ADK's default is ROUGE-1 with an ASCII-oriented
tokenizer that collapses on Perso-Arabic script; the replacement computes
token-level F1 over Persian-normalized tokens.

## Langfuse

`result.to_langfuse()` pushes per-check scores to a Langfuse instance
(reads `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`;
returns the created trace id).
