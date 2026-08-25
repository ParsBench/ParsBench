# Examples

The repository's
[`examples/`](https://github.com/ParsBench/ParsBench/tree/main/examples)
directory holds runnable, end-to-end app evaluations. Every framework
example runs the **same real case**: «پروازیار», a Persian flight-booking bot
with a `search_flights` tool and a `book_flight` tool it must not call
without confirmation. The golden expects a Gregorian date and a price in
toman with Latin digits; the bot answers with a Jalali date and rials in
Persian digits. ParsBench's
[normalization layer](app-eval/normalization.md) matches them. Only the
framework changes between files, so you can diff the integrations.

| File | Framework | Needs API key |
|---|---|---|
| [`quickstart.py`](https://github.com/ParsBench/ParsBench/blob/main/examples/quickstart.py) | none, a plain function returning OpenAI messages | no |
| [`with_openai_sdk.py`](https://github.com/ParsBench/ParsBench/blob/main/examples/with_openai_sdk.py) | OpenAI SDK (manual tool loop) | yes |
| [`with_openai_agents.py`](https://github.com/ParsBench/ParsBench/blob/main/examples/with_openai_agents.py) | OpenAI Agents SDK | yes |
| [`with_langgraph.py`](https://github.com/ParsBench/ParsBench/blob/main/examples/with_langgraph.py) | LangGraph / LangChain | yes |
| [`with_pydantic_ai.py`](https://github.com/ParsBench/ParsBench/blob/main/examples/with_pydantic_ai.py) | Pydantic AI | yes |
| [`with_agno.py`](https://github.com/ParsBench/ParsBench/blob/main/examples/with_agno.py) | Agno | yes |
| [`with_crewai_otel.py`](https://github.com/ParsBench/ParsBench/blob/main/examples/with_crewai_otel.py) | CrewAI via OpenTelemetry, the universal path for any instrumented framework | yes |
| [`ci_with_pytest.py`](https://github.com/ParsBench/ParsBench/blob/main/examples/ci_with_pytest.py) | pytest / CI (`result.assert_passed()`) | no |

## Industry scenarios

[`examples/industry/`](https://github.com/ParsBench/ParsBench/tree/main/examples/industry)
holds vertical-specific evaluations, each showing the checks that matter for
that product. Start here to see what evaluating *your* product looks like:

- `banking_support.py` (offline): rial/toman equivalence in answers and
  tool args, OTP-gated `forbidden_tools`, compliance `not_contains`,
  credential refusal.
- `ecommerce_orders.py` (offline): Jalali/Gregorian delivery dates, order
  ids compared as text (a wrong id fails, proven), a `max_steps` budget.
- `medical_triage.py` (offline): booking args across calendars and digit
  scripts, dosage-advice refusal, emergency escalation to «۱۱۵».
- `telecom_sales_simulation.py` (needs an API key): multi-turn
  [Iranian-user simulation](app-eval/simulation.md) with toman/rial
  confusion and Finglish, goal and criteria judging.
- `rag_faq_support.py` (needs an API key): goldens
  [generated](app-eval/generating-goldens.md) from `faq_netyar.md`,
  correctness and faithfulness judged in Persian.

The offline examples use scripted stand-in bots on purpose: swap the
stand-in for the function that calls your real bot and the goldens keep
working.

## Running them

```bash
pip install parsbench            # plus the framework of the example you run
export OPENAI_API_KEY=...        # any OpenAI-compatible gateway works:
export OPENAI_BASE_URL=...       # AvalAI, OpenRouter, Ollama, ...
export MODEL=gpt-4o-mini         # optional override

python examples/quickstart.py    # works offline, start here
```

Judge-based checks (`output=`, `context=`, `refuses=`) need a judge model
and skip gracefully without one; see [The Judge](app-eval/judge.md). The
offline examples run in the project's CI, so they stay working.

## Benchmark notebooks

For the model-benchmarking side, these Colab notebooks benchmark real
Persian-capable models with ParsBench:

- [Aya](https://huggingface.co/CohereForAI): [notebook](https://colab.research.google.com/drive/1aPayB9AaheDxT7zS4A_4SAMH3a7mIDFX?usp=sharing)
- [Ava](https://huggingface.co/MehdiHosseiniMoghadam): [notebook](https://drive.google.com/file/d/1ToJ8gTQz1ifU70EBAM7fZG2LIOY4zAp0/view?usp=sharing)
- [Dorna](https://huggingface.co/PartAI): [notebook](https://drive.google.com/file/d/1f64d0GnmcQIZ-tlN8cg49pPdiwlVlWvi/view?usp=sharing)
- [MaralGPT](https://huggingface.co/MaralGPT): [notebook](https://drive.google.com/file/d/1ZfjxPa4CfAZdQgtPaEt3nnX180A825ZF/view?usp=sharing)
