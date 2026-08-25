# Model benchmarking

The second pillar: score LLMs on ready-made Persian tasks. Use this when the
user has a model (an API, an Ollama tag, or a HuggingFace checkpoint they
fine-tuned) and wants to know how good it is at Persian, not when they want
to test an app.

## Models

```python
from parsbench.models import OpenAIModel, AnthropicModel, PreTrainedTransformerModel

model = OpenAIModel(
    api_base_url="http://localhost:11434/v1/",   # any OpenAI-compatible API
    api_secret_key="ollama",
    model="qwen2:latest",
)
```

Shared optional parameters on API models: `instruction_prompt=`,
`model_parameters=`, `completion_parameters=`, `retry_on_ratelimit=`,
`cooldown_interval=` (default 10), `max_retries=` (default 1).

`PreTrainedTransformerModel(model=..., tokenizer=...)` evaluates a
HuggingFace `AutoModelForCausalLM` directly, including a just-fine-tuned
checkpoint. Custom backends subclass `Model` (implement `model_name`,
`get_prompt_completion`, `prompt_formatter`, `completion_formatter`).

## Tasks

13 tasks in `parsbench.tasks`, each bundling dataset, prompt templates, and
scorer: `ParsiNLUSentimentAnalysis`, `ParsiNLUEntailment`,
`ParsiNLUMachineTranslationEnFa`, `ParsiNLUMachineTranslationFaEn`,
`ParsiNLUMultipleChoice`, `ParsiNLUReadingComprehension`, `PersianNER`,
`PersianMath`, `ConjNLIEntailment`, `PersianMMLU` (Khayyam Challenge),
`FarsTailEntailment`, `PersianNewsSummary`, `XLSummary`.
`parsbench.tasks.utils.load_all_tasks()` returns instances of all.

Use a task as a context manager (loads the dataset on enter, frees on exit):

```python
from parsbench.tasks import ParsiNLUMultipleChoice

with ParsiNLUMultipleChoice() as task:
    results = task.evaluate(model=model, prompt_lang="fa", prompt_shots=[0, 5])
```

`evaluate()` parameters that matter: `prompt_lang=` ("fa" default, "en"
where available), `prompt_shots=[0, 5]` (each shot count is its own result),
`n_first=100` (default 200; cheap smoke runs), `sub_tasks=[...]` (Persian
MMLU and ParsiNLU Multiple Choice have sub-tasks),
`skip_existing_matches=True` (resume an interrupted run),
`prefer_concurrency=` (default True) with `n_workers=` (default 4).

Note: the `PersianMath` task needs
`pip install git+https://github.com/hendrycks/math.git`.

## Benchmarks

```python
from parsbench.benchmarks import CustomBenchmark

benchmark = CustomBenchmark(
    models=[qwen2_model, aya_model],
    tasks=[ParsiNLUMultipleChoice, PersianMath],
)
result = benchmark.run(prompt_lang="fa", prompt_shots=[0, 3], n_first=100, sort_by_score=True)
result.show_radar_plot()     # also show_bar_plot()
```

`ParsiNLUBenchmark` bundles all ParsiNLU tasks.

## Results, merging, leaderboards

All importable from `parsbench.benchmarks`:

```python
from parsbench.benchmarks import (
    BenchmarkResult, merge_benchmark_results, build_leaderboard_from_benchmark,
)

r = BenchmarkResult.from_matches_files("out/", rescore=False)  # rebuild from saved matches
merged = merge_benchmark_results([r1, r2], sort=True)
leaderboard = build_leaderboard_from_benchmark(merged)         # pandas DataFrame
```

## Iran network notes

- Any OpenAI-compatible gateway works for API models: AvalAI
  (`https://api.avalai.ir/v1`), OpenRouter, local Ollama.
- Task datasets download from the HuggingFace Hub; set `HF_ENDPOINT` to a
  mirror if unreachable.
