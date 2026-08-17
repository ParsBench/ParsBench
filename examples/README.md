# ParsBench examples — evaluate your Persian agent, whatever framework you use

Every example runs the **same real case**: «پروازیار», a Persian flight-booking
bot with a `search_flights` tool and a `book_flight` tool it must not call
without confirmation. The golden expects a **Gregorian** date and a price in
**toman with Latin digits**; the bot answers with a **Jalali** date and
**rials in Persian digits** — ParsBench's normalization layer matches them.
Only the framework changes between files, so you can diff the integrations.

| File | Framework | Needs API key |
|---|---|---|
| [`quickstart.py`](quickstart.py) | none — plain function returning OpenAI messages | no |
| [`with_openai_sdk.py`](with_openai_sdk.py) | OpenAI SDK (manual tool loop) | yes |
| [`with_openai_agents.py`](with_openai_agents.py) | OpenAI Agents SDK | yes |
| [`with_langgraph.py`](with_langgraph.py) | LangGraph / LangChain | yes |
| [`with_pydantic_ai.py`](with_pydantic_ai.py) | Pydantic AI | yes |
| [`with_agno.py`](with_agno.py) | Agno | yes |
| [`with_crewai_otel.py`](with_crewai_otel.py) | CrewAI via OpenTelemetry — the universal path for any instrumented framework (LlamaIndex, ADK, ...) | yes |
| [`ci_with_pytest.py`](ci_with_pytest.py) | pytest / CI (`result.assert_passed()`) | no |

## Industry scenarios

[`industry/`](industry/) holds vertical-specific evaluations — banking,
e-commerce, healthcare (all offline), telecom user-simulation and
knowledge-base RAG (live) — each showing the checks that matter for that
product. Start there to see what evaluating *your* product looks like.

## Setup

```bash
pip install parsbench            # plus the framework of the example you run
export OPENAI_API_KEY=...        # any OpenAI-compatible gateway works:
export OPENAI_BASE_URL=...       # AvalAI, OpenRouter, Ollama, ...
export MODEL=gpt-4o-mini         # optional override

python examples/quickstart.py    # works offline, start here
```

Judge-based checks (`output=`, `context=`, `refuses=`) need a judge model;
without one they skip gracefully:

```bash
export PARSBENCH_JUDGE=gpt-4o-mini            # + PARSBENCH_JUDGE_BASE_URL / _API_KEY if not OpenAI
```

## The pattern

Two ways in, one result out:

```python
from parsbench.appeval import AppEvaluator

evaluator = AppEvaluator(goldens=[...])

# 1. run-for-me: hand evaluate() your function
result = evaluator.evaluate(bot)

# 2. traces mode: run the app yourself, convert what happened
trace = openai_agents.to_trace(run_result)    # or langgraph / pydantic_ai / agno / otel
result = evaluator.score_traces([trace])

print(result)                                 # Persian failure reasons
```
