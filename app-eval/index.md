# Evaluating your AI app

ParsBench's app evaluation layer tests **your Persian chatbot or agent**, whatever framework it is built with. It is the part of ParsBench you wire into CI so a prompt tweak that breaks tool calls, leaks a forbidden answer, or quotes the wrong price never reaches production.

English-shaped eval harnesses silently fail on Persian apps: they treat «۲۵۰ هزار تومان» and «۲٬۵۰۰٬۰۰۰ ریال» as different answers, «۱۴۰۵/۰۷/۰۵» and `2026-09-27` as different dates, and «می‌روم» / «می روم» / «میروم» as different words. ParsBench's [normalization layer](https://parsbench.github.io/ParsBench/app-eval/normalization/index.md) makes those equivalences hold in every check, on both text answers and tool-call arguments.

Like the rest of ParsBench, the API is class-based: an `AppEvaluator` holds your goldens the way a `Task` holds its dataset, and `evaluate()` takes the thing under test, your app instead of a model.

## Sixty seconds, no API key

Your app is any function that takes a user message and returns a string, a `Trace`, or an OpenAI-format message list:

```
from parsbench.appeval import AppEvaluator, Golden, ToolCall

def my_bot(message):          # stand-in for your app
    return [
        {"role": "assistant", "tool_calls": [
            {"id": "1", "function": {"name": "search_flights",
                                     "arguments": '{"date": "1405-07-05"}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "پرواز PY-101"},
        {"role": "assistant", "content": "پرواز ساعت ۸ صبح، قیمت ۲٬۵۰۰٬۰۰۰ ریال"},
    ]

evaluator = AppEvaluator(goldens=[
    Golden(
        input="بلیط تهران-مشهد برای ۵ مهر ۱۴۰۵ می‌خوام. قیمتش چنده؟",
        tools=[ToolCall("search_flights", date="2026-09-27")],  # Jalali == Gregorian
        contains=["250 هزار تومان"],                            # rials == tomans
        forbidden_tools=["book_flight"],
    ),
])
result = evaluator.evaluate(my_bot)
print(result)
```

Every filled `Golden` field switches on its check. There is no separate metric configuration and no YAML.

## Two ways in

```
# 1. run-for-me: hand evaluate() your function (sync or async)
result = evaluator.evaluate(bot)

# 2. traces mode: run the app yourself, score what happened
result = evaluator.score_traces([trace_or_message_list])
```

A crash inside your app is reported as a failing `app_error` check on that golden, so one broken case never aborts the suite. Async apps work everywhere, including notebooks and servers with a running event loop.

## Where to go next

- [Goldens & Checks](https://parsbench.github.io/ParsBench/app-eval/goldens/index.md): every `Golden` field, what its check asserts, and the `metrics=` filter.
- [The Judge](https://parsbench.github.io/ParsBench/app-eval/judge/index.md): configuring the judge model for LLM-judged checks, and calibrating it against human labels.
- [Persian Normalization](https://parsbench.github.io/ParsBench/app-eval/normalization/index.md): what the matching layer equates and why, with examples.
- [Framework Integrations](https://parsbench.github.io/ParsBench/app-eval/integrations/index.md): OpenAI Agents SDK, LangGraph, Pydantic AI, Agno, and any OTel-instrumented app.
- [Multi-turn Simulation](https://parsbench.github.io/ParsBench/app-eval/simulation/index.md): an LLM plays an Iranian user against your bot; a judge scores the conversation.
- [Generating Goldens](https://parsbench.github.io/ParsBench/app-eval/generating-goldens/index.md): bootstrap a test suite from your product docs or knowledge base.
- [CI & Regression Tracking](https://parsbench.github.io/ParsBench/app-eval/ci/index.md): pytest integration, concurrency, flaky-agent consistency (`pass^k`), diffs, Langfuse export.
- [The Viewer](https://parsbench.github.io/ParsBench/app-eval/viewer/index.md): `parsbench view`, the local UI over recorded runs.

Runnable end-to-end examples for each framework and several industries live in [`examples/`](https://github.com/ParsBench/ParsBench/tree/main/examples); see the [Examples](https://parsbench.github.io/ParsBench/examples/index.md) page.
