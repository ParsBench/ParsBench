# ParsBench

## Overview

ParsBench is a toolkit for making AI work well in Persian. It has two pillars:

- **App evaluation.** Test *your own* Persian chatbot or agent, whatever framework it is built with, and wire the checks into CI. The Persian-aware matching layer understands digit scripts, rial/toman amounts, Jalali dates, and ZWNJ spacing, so checks that would silently fail in an English-shaped eval harness hold up on Persian text.
- **Model benchmarking.** Evaluate and rank LLMs on ready-made Persian tasks (ParsiNLU, Persian MMLU, FarsTail, Persian Math, and more), each bundled with its dataset, prompt templates, and scorer.

Both share the same class-based API: create an evaluator, hand it the thing under test, read the result. No YAML, no metric registry, one import.

## Show me

Evaluating an app. The app is any function that takes a message and returns an answer:

```
from parsbench.appeval import AppEvaluator, Golden, ToolCall

evaluator = AppEvaluator(goldens=[
    Golden(
        input="بلیط تهران-مشهد برای ۵ مهر ۱۴۰۵ می‌خوام. قیمتش چنده؟",
        tools=[ToolCall("search_flights", date="2026-09-27")],  # Jalali == Gregorian
        contains=["250 هزار تومان"],                            # rials == tomans
        forbidden_tools=["book_flight"],
    ),
])
evaluator.evaluate(my_bot).assert_passed()
```

Benchmarking models:

```
from parsbench.benchmarks import CustomBenchmark
from parsbench.models import OpenAIModel
from parsbench.tasks import ParsiNLUMultipleChoice, PersianMath

benchmark = CustomBenchmark(
    models=[OpenAIModel(...)],
    tasks=[ParsiNLUMultipleChoice, PersianMath],
)
result = benchmark.run(prompt_lang="fa", prompt_shots=[0, 3])
result.show_radar_plot()
```

Start with [Getting Started](https://parsbench.github.io/ParsBench/getting-started/index.md), then go deeper with [App Evaluation](https://parsbench.github.io/ParsBench/app-eval/index.md) or the [Benchmarking tutorial](https://parsbench.github.io/ParsBench/tutorial/models/index.md).

## Key features

- **App evaluation for Persian products**: goldens with tool-call, content, format, budget, and judge-based checks; multi-turn user simulation with an Iranian-user persona; golden generation from your docs; judge calibration.
- **Persian-aware normalization**: «۲۵۰ هزار تومان» matches «۲٬۵۰۰٬۰۰۰ ریال», «۱۴۰۵/۰۷/۰۵» matches `2026-09-27`, and «می‌روم» matches «می روم», in every check.
- **Framework agnostic**: adapters for the OpenAI Agents SDK, LangGraph, Pydantic AI, and Agno, plus a universal OpenTelemetry collector for everything else (CrewAI, LlamaIndex, Google ADK, ...).
- **A local run viewer**: `parsbench view` shows live runs, a goldens × checks matrix, full traces, RTL simulation replay, and run-vs-run diffs.
- **Ready-made Persian benchmarks**: 13 tasks with datasets and prompt templates, benchmarking tools to compare and rank models, radar/bar plots, and a leaderboard builder.
- **Customizable API**: create custom models, tasks, scores, checks, and benchmarks with plain Python classes.

## Motivation

I was trying to fine-tune an open-source LLM for the Persian language and needed a way to measure whether it was actually any good. That led me to [this paper](https://arxiv.org/abs/2404.02403), great work preparing datasets and evaluation methods for testing ChatGPT on Persian, with the code shared in [this repository](https://github.com/Ipouyall/Benchmarking_ChatGPT_for_Persian).

So I built a handy framework that packages various tasks and datasets for evaluating LLMs on Persian, reusing parts of their work (datasets, metrics, basic prompt templates). ParsBench powered the [Open Persian LLM Leaderboard](https://huggingface.co/spaces/ParsBench/leaderboard), and has since grown a second pillar: helping teams that *build* Persian AI products ship them with confidence, not just ranking base models.

## Example notebooks

- Benchmark [Aya](https://huggingface.co/CohereForAI) models:
- Benchmark [Ava](https://huggingface.co/MehdiHosseiniMoghadam) models:
- Benchmark [Dorna](https://huggingface.co/PartAI) models:
- Benchmark [MaralGPT](https://huggingface.co/MaralGPT) models:

Runnable app-evaluation examples, one per framework plus industry scenarios (banking, e-commerce, healthcare, telecom, RAG), live in [`examples/`](https://github.com/ParsBench/ParsBench/tree/main/examples); see the [Examples](https://parsbench.github.io/ParsBench/examples/index.md) page.

## For LLMs

These docs are also published in LLM-friendly form: [`llms.txt`](https://parsbench.github.io/ParsBench/llms.txt) (index) and [`llms-full.txt`](https://parsbench.github.io/ParsBench/llms-full.txt) (everything inlined). Paste either into your assistant to give it the whole API.

## Sponsors

Here are the companies/people who have helped keep this project maintained. If you want to support the project, see the [donation page](https://parsbench.github.io/ParsBench/donation/index.md).

- [AvalAI](https://avalai.ir/): gave us free OpenAI API credit several times through their "AvalAward" program, which funded R&D and benchmarking GPT models.
- [Basalam](https://basalam.com/): voluntarily helped run the benchmarks on open-weight models and build the [ParsBench Leaderboard](https://huggingface.co/spaces/ParsBench/leaderboard).

## Contributing

Contributions are welcome! Please refer to the [contribution guidelines](https://parsbench.github.io/ParsBench/contribution/index.md) for how to get involved.

## License

ParsBench is distributed under the Apache-2.0 license.

## Contact

For support or questions, contact [shahriarshm81@gmail.com](mailto:shahriarshm81@gmail.com) or open an issue on [GitHub](https://github.com/ParsBench/ParsBench/issues).
