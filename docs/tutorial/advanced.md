# Advanced tutorial

This section is for implementing your own tasks, or using the framework's
building blocks (scores, data loaders, prompt templates) for other purposes.

## Scores

Scores measure how good a completion is compared to the expected answer.

### Available scores

| Score Name            | Description                                                             |
|-----------------------|-------------------------------------------------------------------------|
| Exact Match           | `1` if the completion and target are equal, otherwise `0`.              |
| English Sentence Bleu | Bleu n-gram score with NLTK English word tokenizer. Between `0` and `1`. |
| Persian Sentence Bleu | Bleu n-gram score with Hazm Persian word tokenizer. Between `0` and `1`. |
| English Rouge         | Rouge score with NLTK English word tokenizer. Between `0` and `1`.       |
| Persian Rouge         | Rouge score with Hazm Persian word tokenizer. Between `0` and `1`.       |

### Make your own score

Write a plain function and wrap it with `wrap_scorer`:

```python
from parsbench.scores.base import wrap_scorer

@wrap_scorer
def my_exact_match(completion: str, target: str) -> float:
    return float(completion.strip() == target.strip())
```

The function's name becomes the score's name in results.

## Data loaders

A data loader loads the dataset a task evaluates on. All three load from a
local path or a URL.

### JSONLine

`JSONLineDataLoader` reads jsonlines files (`.jsonl`):

```python
from parsbench.tasks.base import JSONLineDataLoader

data_loader = JSONLineDataLoader(data_path="dataset.jsonl")
data = data_loader.load()
```

### CSV

`CSVDataLoader` reads CSV files:

```python
from parsbench.tasks.base import CSVDataLoader

data_loader = CSVDataLoader(data_path="dataset.csv")
data = data_loader.load()
```

### HuggingFace

`HuggingFaceDataLoader` uses HuggingFace's `datasets` library to load from
disk or download from the Hub:

```python
from parsbench.tasks.base import HuggingFaceDataLoader

data_loader = HuggingFaceDataLoader(
    data_path="persiannlp/parsinlu_entailment",
    split="validation",
)
data = data_loader.load()
```

## Prompt templates

`PromptTemplate` defines the prompt for each language, shot templates, shot
examples, and variable mappings.

### With a shot template

A sentiment-analysis template where few-shot examples are rendered from the
dataset:

```python
FA_TEMPLATE = """
جمله زیر نظر یک شخص است. این جمله به زبان فارسی است. بار یا احساس موجود در این جمله را شناسایی کن.
پاسخ‌ های ممکن حالت‌های روبرو هستند:
SAD
NEUTRAL
HAPPY

فقط کلمه مربوط به احساس نظر داده شده را خروجی بده.

{example_shots}

نظر: {review}
احساس:
"""

FA_SHOT_TEMPLATE = """
نظر: {review}
احساس: {label}
"""
```

```python
from parsbench.tasks.base import PromptTemplate

prompt_template = PromptTemplate(
    language_templates={"fa": FA_TEMPLATE},
    prompt_shot_templates={"fa": FA_SHOT_TEMPLATE},
    prompt_variables_mapping={"review": "review"},
    target_variables_mapping={"label": "label"},
)

prompt = prompt_template.get_prompt(
    prompt_lang="fa",
    data={"review": "غذا خیلی بد بود", "label": "SAD"},
    n_shots=3,
    sample_data=[
        {"review": "خوشمزه بود ممنونم", "label": "HAPPY"},
        {"review": "غذا خوب بود فقط کاش زودتر می‌رسید.", "label": "NEUTRAL"},
        {"review": "نوشابه گرم بود. پیتزا هم خیلی بد مزه بود.", "label": "SAD"},
    ],
)
```

The variable mappings translate between prompt placeholders and dataset
columns: `prompt_variables_mapping={"review": "review"}` fills `{review}` in
the template from the `review` column, and `target_variables_mapping` does
the same for the expected answer.

### With static shot examples

For complicated tasks where you want hand-written few-shot examples (for
instance chain-of-thought prompting), use static shot examples instead of a
shot template:

```python
from parsbench.tasks.base import PromptTemplate

prompt_template = PromptTemplate(
    language_templates={"fa": FA_TEMPLATE},
    prompt_shot_examples={"fa": {1: FA_1_SHOT, 3: FA_3_SHOT, 5: FA_5_SHOT}},
)
prompt = prompt_template.get_prompt(n_shots=5, ...)
```

### Load templates from files

`LazyLoadTemplates` reads templates from text files on first use, which
keeps long prompts out of your Python code:

```python
from parsbench.tasks.base import PromptTemplate, LazyLoadTemplates

prompt_template = PromptTemplate(
    language_templates=LazyLoadTemplates(
        fa="fa_math.txt",
        en="en_math.txt",
    ),
    ...
)
```

### Constant prompt variables

To fill a placeholder with a fixed value rather than a dataset column, use
`ConstantPromptVariable`:

```python
from parsbench.tasks.base import PromptTemplate, ConstantPromptVariable

prompt_template = PromptTemplate(
    language_templates={"fa": FA_TEMPLATE},
    prompt_shot_templates={"fa": FA_SHOT_TEMPLATE},
    prompt_variables_mapping={
        "input": "input",
        "first_name": ConstantPromptVariable("شهریار"),
    },
    target_variables_mapping={"label": "label"},
)
```

## Tasks

The task is the primary unit of the framework: a batteries-included
evaluator that runs the whole pipeline from loading data to scoring.

### Task data provider

Each task carries its dataset. `task.get_data()` returns it:

```python
from parsbench.tasks import ParsiNLUEntailment

with ParsiNLUEntailment() as task:  # the context manager loads the data
    data = task.get_data()
```

### Task match generator

A `TaskMatch` holds a prompt, the target answer, and (once generated) the
model's completion and its score. Fresh matches have `completion` and
`score` set to `None`:

```python
from parsbench.tasks import ParsiNLUEntailment

with ParsiNLUEntailment() as task:
    matches = task.generate_matches(prompt_lang="fa", n_shots=0, n_first=100)
```

### Generate completions and score

You can drive the pipeline step by step, which is useful when you want to
inspect or modify matches between steps:

```python
from parsbench.tasks import ParsiNLUEntailment

with ParsiNLUEntailment() as task:
    matches = task.generate_matches(prompt_lang="fa", n_shots=0, n_first=100)
    model.generate_completions(matches)   # the model fills in completions
    task.score_matches(matches)           # the task's scorer fills in scores
```

### Make your own task

For a task with your own dataset, prompts, and scoring, inherit `Task` (or
one of the existing tasks). Put prompt templates in text files and load them
with `LazyLoadTemplates`:

```python
from parsbench.scores.base import Scorer, wrap_scorer
from parsbench.tasks.base import (
    HuggingFaceDataLoader,
    LazyLoadTemplates,
    PromptTemplate,
    Task,
    TaskCategory,
    TaskMatchGroup,
)

@wrap_scorer
def my_custom_score(completion: str, target: str) -> float:
    return float(completion.strip() == target.strip())

class CustomTask(Task):
    task_name: str = "Custom Task"
    task_category: TaskCategory = TaskCategory.REASONING

    data_loader: HuggingFaceDataLoader = HuggingFaceDataLoader(
        data_path="org/custom_dataset",
        split="test",
    )
    data_target_key: str = "target"

    prompt_template: PromptTemplate = PromptTemplate(
        language_templates=LazyLoadTemplates(
            en="path/to/en_template.txt",
            fa="path/to/fa_template.txt",
        ),
        prompt_shot_templates=LazyLoadTemplates(
            en="path/to/en_shot_template.txt",
            fa="path/to/fa_shot_template.txt",
        ),
        prompt_variables_mapping={"prompt_variable1": "variable1", "prompt_variable2": "variable2"},
        target_variables_mapping={"prompt_target": "target"},
    )

    scorer: Scorer = my_custom_score

    def score_matches(self, matches: TaskMatchGroup) -> TaskMatchGroup:
        matches.format_completions(
            lambda c: c.strip().strip("'").lower()
        )
        return super().score_matches(matches)

    def get_overall_score(cls, matches: TaskMatchGroup) -> float:
        return sum(match.score for match in matches) / len(matches)
```

Any data loader, prompt template, and scorer combination works.

#### Sub-tasks

If your dataset covers several categories, declare them and ParsBench
reports a score per category:

```python
class CustomTask(Task):
    ...
    sub_task_key: str = "category"  # dataset column that holds the sub task
    sub_tasks: list[str] = ["math_and_logic", "common_knowledge", "literature"]
    ...
```

Both evaluation and benchmarks can then run a subset:

```python
with CustomTask() as task:
    results = task.evaluate(..., sub_tasks=["math_and_logic"])

# or inside a benchmark
benchmark = CustomBenchmark(
    ...,
    tasks=[
        PersianMath,
        CustomTask.select_sub_tasks(["math_and_logic"]),
    ],
)
```
