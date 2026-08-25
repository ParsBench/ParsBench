# Tasks

A task evaluates model responses on one dataset. It ships with the data, a
prompt template per language, and a scorer, so evaluating a model is: build
prompts from the data, get the model's completions, score them against the
targets.

## Available tasks

| Task Name | Score Name | Dataset |
|-----------------------------|------------------|--------------|
| ParsiNLU Sentiment Analysis | Exact Match (F1) | [ParsiNLU](https://huggingface.co/datasets/persiannlp/parsinlu_sentiment) |
| ParsiNLU Entailment | Exact Match (F1) | [ParsiNLU](https://huggingface.co/datasets/persiannlp/parsinlu_entailment) |
| ParsiNLU Machine Translation En -> Fa | Bleu | [ParsiNLU](https://huggingface.co/datasets/persiannlp/parsinlu_translation_en_fa) |
| ParsiNLU Machine Translation Fa -> En | Bleu | [ParsiNLU](https://huggingface.co/datasets/persiannlp/parsinlu_translation_fa_en) |
| ParsiNLU Multiple Choice | Exact Match (Accuracy) | [ParsiNLU](https://github.com/persiannlp/parsinlu) |
| ParsiNLU Reading Comprehension | Common Tokens (F1) | [ParsiNLU](https://huggingface.co/datasets/persiannlp/parsinlu_reading_comprehension) |
| Persian NER | NER Exact Match (F1) | [PersianNER](https://github.com/HaniehP/PersianNER) |
| Persian Math | Math Equivalence (Accuracy) | [Source](https://github.com/Ipouyall/Benchmarking_ChatGPT_for_Persian) |
| ConjNLI Entailment | Exact Match (F1) | [Source](https://github.com/Ipouyall/Benchmarking_ChatGPT_for_Persian) |
| Persian MMLU (Khayyam Challenge) | Exact Match (Accuracy) | [Khayyam Challenge](https://huggingface.co/datasets/raia-center/khayyam-challenge) |
| FarsTail Entailment | Exact Match (F1) | [FarsTail](https://github.com/dml-qom/FarsTail) |
| Persian News Summary | Rouge | [PNSummary](https://huggingface.co/datasets/HooshvareLab/pn_summary) |
| XL-Sum | Rouge | [XLSum](https://huggingface.co/datasets/csebuetnlp/xlsum) |

Import any of them from `parsbench.tasks`, or get instances of all of them
with `parsbench.tasks.utils.load_all_tasks()`.

## Evaluation

The evaluation process has 6 steps:

1. Loading data
2. Loading the prompt template
3. Generating matches (prompt-answer pairs)
4. Generating completions
5. Scoring completions
6. Storing the result (optional)

`evaluate()` runs all of them:

```python
from parsbench.models import OpenAIModel
from parsbench.tasks import ParsiNLUMultipleChoice

model = OpenAIModel(
    api_base_url="http://localhost:11434/v1/",
    api_secret_key="ollama",
    model="qwen2:latest",
)

with ParsiNLUMultipleChoice() as task:
    results = task.evaluate(
        model=model,
        prompt_lang="fa",
        prompt_shots=[0, 5],
    )
```

Use the task in a context manager. It loads the dataset on enter and frees
it on exit.

The parameters you'll actually reach for:

- `prompt_lang=` selects the prompt template language, `"fa"` (default) or
  `"en"` where a task ships both.
- `prompt_shots=[0, 5]` evaluates zero-shot and 5-shot in one run; each shot
  count produces its own result.
- `n_first=100` evaluates only the first 100 samples (default 200). Handy
  for cheap smoke runs before a full evaluation.
- `sub_tasks=["math_and_logic"]` restricts a task with sub-tasks (Persian
  MMLU, ParsiNLU Multiple Choice) to a subset.
- `skip_existing_matches=True` resumes an interrupted run: matches already
  generated and scored under `output_path` are not re-run.
- `prefer_concurrency=` (default True) fans completion calls out over
  threads when the model supports it; tune with `n_workers=` (default 4).

## Evaluation result

`evaluate()` returns a list of `EvaluationResult` objects, one per sub-task,
each holding the overall score per shot count. Use them directly or convert
to a pandas DataFrame:

```python
eval_result = results[0]
print(eval_result.to_pandas())
```

Output:

```txt
     model_name                 task_name task_category        sub_task  n_shots   score_name     score
0  qwen2:latest  ParsiNLU Multiple Choice     knowledge  math_and_logic        0  Exact Match  0.600000
1  qwen2:latest  ParsiNLU Multiple Choice     knowledge  math_and_logic        3  Exact Match  0.285714
```

## Saving results

Save manually with the `save` method of `EvaluationResult`, or pass
`save_evaluation=True` to `evaluate()`. `save_matches=True` also writes every
match (prompt, completion, target, score), which is what you want when you
need to inspect *why* a score is low:

```python
with PersianMath() as task:
    results = task.evaluate(
        model=model,
        prompt_lang="fa",
        prompt_shots=[0, 5],
        save_matches=True,
        save_evaluation=True,
        output_path="results/",
    )
```

The output directory structure:

```txt
results
└── qwen2:latest
    └── Persian_Math
        ├── evaluation.jsonl
        ├── matches_0_shot.jsonl
        └── matches_5_shot.jsonl
```
