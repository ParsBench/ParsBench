---
name: parsbench
description: >-
  Write evaluations for Persian (Farsi) chatbots and AI agents, or benchmark
  LLMs on Persian tasks, using the parsbench Python library. Use when the user
  wants to test a Persian bot, write goldens/evals, check Persian text
  equivalence (rial/toman amounts, Jalali/Gregorian dates, Persian digits,
  ZWNJ), simulate Iranian users, run Persian evals in CI, or rank models on
  Persian benchmarks like ParsiNLU or Persian MMLU.
---

# ParsBench

ParsBench evaluates AI in Persian. Two separate jobs:

1. **App evaluation** (`parsbench.appeval`): test the user's own chatbot or
   agent, whatever framework it uses. This is what most users want.
2. **Model benchmarking** (`parsbench.benchmarks` / `.tasks` / `.models`):
   score an LLM on 13 ready-made Persian tasks. Read
   `references/benchmarking.md` for this; the rest of this file is app eval.

Install: `pip install parsbench` (Python >= 3.12; on 3.10/3.11 pin
`parsbench==0.1.7`, which has no appeval). Docs:
https://parsbench.github.io/ParsBench/ and, for the full API in one file,
https://parsbench.github.io/ParsBench/llms-full.txt

This skill describes parsbench 0.3.x. If the installed version differs,
verify signatures against llms-full.txt before relying on defaults stated
here.

## Mental model

An `AppEvaluator` holds `Golden` objects. Each filled `Golden` field switches
on one check; empty fields don't run. No YAML, no metric registry:

```python
from parsbench.appeval import AppEvaluator, Golden, ToolCall

evaluator = AppEvaluator(goldens=[
    Golden(
        input="بلیط تهران-مشهد برای ۵ مهر ۱۴۰۵ می‌خوام. قیمتش چنده؟",
        tools=[ToolCall("search_flights", date="2026-09-27")],  # Jalali == Gregorian
        contains=["250 هزار تومان"],                            # rials == tomans
        forbidden_tools=["book_flight"],
    ),
])
result = evaluator.evaluate(my_bot)   # my_bot: fn(message) -> str | Trace | OpenAI message list
result.assert_passed()                # raises with the failing checks (pytest-ready)
```

Two entry points: `evaluator.evaluate(app)` runs the app for you (sync or
async both work), `evaluator.score_traces([...])` scores traces or message
lists you produced yourself. A crash inside the app becomes a failing
`app_error` check on that golden, never an aborted suite.

Every text and tool-argument comparison passes through the Persian
normalization layer automatically: digit scripts, ZWNJ, rial/toman amounts,
and numeric Jalali/Gregorian dates all compare equal. Exact rules in
`references/normalization.md`.

## Workflow: writing an eval suite for a user's bot

1. Wrap their bot as `fn(message)` returning a string, a `Trace`, or an
   OpenAI-format message list. For framework run objects (OpenAI Agents SDK,
   LangGraph, Pydantic AI, Agno, anything OTel-instrumented), use an adapter:
   `references/integrations.md`.
2. Write goldens covering: correct tool calls with argument values, forbidden
   actions (`forbidden_tools=`, `not_contains=`), must-state facts
   (`contains=`), refusal cases (`refuses=True`), and budgets (`max_steps=`).
   Field-by-field reference: `references/goldens.md`.
3. Deterministic checks (tools, contains, format, budgets) need no API key.
   For judged checks (`output=`, `context=`, `refuses=`) configure a judge:
   `references/judge.md`.
4. Wire into pytest/CI and handle flakiness with `n_runs=` and pass^k:
   `references/ci.md`.
5. Tell the user about `parsbench view`, the built-in local UI over recorded
   runs (also covered in `references/ci.md`).

To bootstrap goldens from the user's docs/FAQ instead of writing them by
hand, and for multi-turn simulation with an Iranian-user persona, read
`references/goldens.md` (generation section) and `references/simulation.md`.

## Gotchas (these burn people)

- **Month names don't match.** Date equivalence covers numeric dates only
  («۱۴۰۵/۰۷/۰۵» == `2026-09-27`; a year below 1600 reads as Jalali). «۵ مهر»
  in text is NOT matched by `contains=` or tool-arg checks. Never write a
  golden expecting month-name equivalence; month names are a simulator trap
  instead.
- **Bare numbers are exact rials.** `contains=["250 هزار تومان"]` matches any
  equivalent amount in any unit, but a bare number in the app's answer only
  matches at the exact rial value. When one side has no currency word, both
  rial and toman readings are accepted.
- **Judge checks skip silently without a judge.** A suite with `output=` or
  `refuses=` and no judge configured still passes its deterministic checks
  and marks judge checks skipped. Set `PARSBENCH_JUDGE` (plus
  `OPENAI_API_KEY`/`OPENAI_BASE_URL`, any OpenAI-compatible gateway: AvalAI,
  OpenRouter, local Ollama) or pass `judge=` explicitly.
- **Set `PARSBENCH_NO_RECORD=1` in CI.** Otherwise every run writes into the
  project-local `.parsbench/` viewer store.
- **`prefer_concurrency=True` requires a thread-safe app and judge.** It is
  off by default for a reason; stateful bots usually aren't.
- **Iran network access.** OpenAI is often unreachable from Iran. Point
  `OPENAI_BASE_URL` at AvalAI/OpenRouter/Ollama, and `HF_ENDPOINT` at a
  mirror for benchmark datasets.

## Reference files

- `references/goldens.md`: every Golden field, `metrics=` filtering, dict/JSON
  goldens, reading results, generating goldens from docs.
- `references/normalization.md`: exact equivalence rules and the importable
  normalize functions.
- `references/judge.md`: judge configuration, env var precedence, calibration
  against human labels.
- `references/simulation.md`: multi-turn Iranian-user simulation, traps,
  `ConversationGolden`.
- `references/integrations.md`: framework adapters, OTel collector, building
  a `Trace` by hand, Langfuse export.
- `references/ci.md`: pytest, `parsbench test`, `n_runs`/pass^k, diffing
  runs, the viewer.
- `references/benchmarking.md`: models, the 13 tasks, benchmarks, and
  leaderboards.
