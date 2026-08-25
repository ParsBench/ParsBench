# ParsBench

<div align="center">
    <a href="https://github.com/ParsBench/ParsBench">
        <img src="https://raw.githubusercontent.com/ParsBench/ParsBench/main/docs/imgs/banner-black.png" alt="ParsBench banner" width="480" height="240">
    </a>
    <br>
    <a href="https://parsbench.github.io/ParsBench/">
        <img src="https://shields.io/badge/-docs-blue" alt="docs">
    </a>
    <a href="https://pypi.python.org/pypi/parsbench">
        <img src="https://img.shields.io/pypi/v/parsbench.svg" alt="pypi">
    </a>
    <a href="https://github.com/ParsBench/ParsBench/actions/workflows/ci.yml">
        <img src="https://github.com/ParsBench/ParsBench/actions/workflows/ci.yml/badge.svg" alt="CI">
    </a>
    <a href="https://huggingface.co/ParsBench">
        <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-me-on-HF-md-dark.svg" alt="huggingface">
    </a>
</div>

ParsBench is a toolkit for making AI work well in Persian. It has two pillars:

- **App evaluation.** Test your own Persian chatbot or agent, whatever
  framework it is built with, and wire the checks into CI. The Persian-aware
  matching layer understands digit scripts, rial/toman amounts, Jalali dates,
  and ZWNJ spacing, so checks that would silently fail in an English-shaped
  eval harness hold up on Persian text.
- **Model benchmarking.** Evaluate and rank LLMs on 13 ready-made Persian
  tasks (ParsiNLU, Persian MMLU, FarsTail, Persian Math, and more), each
  bundled with its dataset, prompt templates, and scorer. This is the toolkit
  behind the [ParsBench Leaderboard](https://huggingface.co/spaces/ParsBench/leaderboard).

Both share the same class-based API: create an evaluator, hand it the thing
under test, read the result. No YAML, no metric registry, one import.

## Evaluate your app

Your app is any function that takes a user message and returns an answer.
Works with the OpenAI SDK, OpenAI Agents SDK, LangGraph, Pydantic AI, Agno,
CrewAI, or anything OTel-instrumented:

```python
from parsbench.appeval import AppEvaluator, Golden, ToolCall

evaluator = AppEvaluator(goldens=[
    Golden(
        input="بلیط تهران-مشهد برای ۵ مهر می‌خوام. قیمتش چنده؟",
        tools=[ToolCall("search_flights", date="2026-09-27")],  # Jalali == Gregorian
        contains=["250 هزار تومان"],                            # rials == tomans
        forbidden_tools=["book_flight"],
    ),
])
evaluator.evaluate(my_bot).assert_passed()
```

Then open the local viewer, with live runs, full traces, RTL simulation
replay, and run-vs-run diffs. No extra dependencies, nothing to configure:

```bash
parsbench view
```

<p align="center">
    <img src="https://raw.githubusercontent.com/ParsBench/ParsBench/main/docs/imgs/viewer.png" alt="parsbench view" width="760">
</p>

There is more: multi-turn simulation with an Iranian-user persona (taarof,
Finglish, toman/rial confusion), golden generation from your docs, judge
calibration against human labels, and pytest/CI integration. Start with the
[app evaluation docs](https://parsbench.github.io/ParsBench/app-eval/) or the
runnable [`examples/`](examples/), which cover every supported framework plus
industry scenarios (banking, e-commerce, healthcare, telecom, RAG).

## Benchmark a model

```python
from parsbench.benchmarks import CustomBenchmark
from parsbench.models import OpenAIModel
from parsbench.tasks import ParsiNLUMultipleChoice, PersianMath

benchmark = CustomBenchmark(
    models=[OpenAIModel(api_base_url=..., api_secret_key=..., model=...)],
    tasks=[ParsiNLUMultipleChoice, PersianMath],
)
result = benchmark.run(prompt_lang="fa", prompt_shots=[0, 3])
result.show_radar_plot()
```

<p align="center">
    <img src="https://raw.githubusercontent.com/ParsBench/ParsBench/main/docs/imgs/radarplot.png" alt="Benchmark radar plot" width="560">
</p>

Any OpenAI-compatible API works (OpenAI, AvalAI, OpenRouter, a local Ollama),
and `PreTrainedTransformerModel` evaluates a HuggingFace checkpoint directly,
including one you just fine-tuned. See the
[benchmarking tutorial](https://parsbench.github.io/ParsBench/tutorial/models/)
and the
[task list](https://parsbench.github.io/ParsBench/tutorial/tasks/), or these
Colab notebooks benchmarking real Persian-capable models:
[Aya](https://colab.research.google.com/drive/1aPayB9AaheDxT7zS4A_4SAMH3a7mIDFX?usp=sharing),
[Ava](https://drive.google.com/file/d/1ToJ8gTQz1ifU70EBAM7fZG2LIOY4zAp0/view?usp=sharing),
[Dorna](https://drive.google.com/file/d/1f64d0GnmcQIZ-tlN8cg49pPdiwlVlWvi/view?usp=sharing),
[MaralGPT](https://drive.google.com/file/d/1ZfjxPa4CfAZdQgtPaEt3nnX180A825ZF/view?usp=sharing).

## Installation

> **Requires Python ≥ 3.12.** ParsBench 0.2+ targets current library versions
> (transformers 5, datasets 5, numpy 2), which need Python 3.12+. If you're on
> Python 3.10/3.11, pin the previous release: `pip install "parsbench==0.1.7"`.

```bash
pip install parsbench
```

`pip install 'parsbench[test]'` adds pytest for running golden suites with
`parsbench test` in CI. The Persian Math benchmark task additionally needs
`pip install git+https://github.com/hendrycks/math.git`.

## Documentation

Full documentation lives at
[parsbench.github.io/ParsBench](https://parsbench.github.io/ParsBench/). It is
also published in LLM-friendly form:
[`llms.txt`](https://parsbench.github.io/ParsBench/llms.txt) (index) and
[`llms-full.txt`](https://parsbench.github.io/ParsBench/llms-full.txt)
(everything inlined). Paste either into your assistant to give it the whole
API.

## Motivation

I was trying to fine-tune an open-source LLM for the Persian language and
needed a way to measure whether it was actually any good. That led me to
[this paper](https://arxiv.org/abs/2404.02403), great work preparing datasets
and evaluation methods for testing ChatGPT on Persian, with the code shared in
[this repository](https://github.com/Ipouyall/Benchmarking_ChatGPT_for_Persian).

So I built a handy framework that packages various tasks and datasets for
evaluating LLMs on Persian, reusing parts of their work (datasets, metrics,
basic prompt templates). ParsBench powered the
[Open Persian LLM Leaderboard](https://huggingface.co/spaces/ParsBench/leaderboard),
and has since grown a second pillar: helping teams that *build* Persian AI
products ship them with confidence, not just ranking base models.

## Sponsors

Here are the companies/people who have helped keep this project maintained.
If you want to support the project, see the
[donation page](https://parsbench.github.io/ParsBench/donation/).

- [AvalAI](https://avalai.ir/): gave us free OpenAI API credit several times
  through their "AvalAward" program, which funded R&D and benchmarking GPT
  models.
- [Basalam](https://basalam.com/): voluntarily helped run the benchmarks on
  open-weight models and build the
  [ParsBench Leaderboard](https://huggingface.co/spaces/ParsBench/leaderboard).

## Contributing

Contributions are welcome! Please refer to the
[contribution guidelines](https://parsbench.github.io/ParsBench/contribution/)
for how to get involved.

## Citation

If you use ParsBench in your research, please cite it as follows:

```bibtex
@software{parsbench2025,
  author = {Shahriar Shariati Motlagh},
  title = {ParsBench: A Toolkit for Benchmarking Persian Language Models},
  url = {https://github.com/ParsBench/ParsBench},
  year = {2025},
}
```

Or in text format:

Shariati Motlagh, S. (2025). ParsBench: A Toolkit for Benchmarking Persian
Language Models. GitHub repository: https://github.com/ParsBench/ParsBench

## License

ParsBench is distributed under the Apache-2.0 license.

## Contact

For support or questions, contact
[shahriarshm81@gmail.com](mailto:shahriarshm81@gmail.com) or open an issue on
[GitHub](https://github.com/ParsBench/ParsBench/issues).
