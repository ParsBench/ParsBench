# Advanced Tutorial

This section is for the ones who want to implement their own Tasks or want to use the framework APIs for other use cases.

## Scores

Scores are the methods that we use to measure the goodness of the completion of our model, comparing to the expected answer.

### Available Scores

| Score Name            | Description                                                             |
|-----------------------|-------------------------------------------------------------------------|
| Exact Match           | `1` if the completion and target are equal, otherwise `0`.              |
| English Sentence Bleu | Bleu n-gram score with NLTK English word tokenizer. Between `0` to `1`. |
| Persian Sentence Bleu | Bleu n-gram score with Hazm Persian word tokenizer. Between `0` to `1`. |
| English Rouge         | Rouge score with NLTK English word tokenizer. Between `0` to `1`.       |
| Persian Rouge         | Rouge score with Hazm Persian word tokenizer. Between `0` to `1`.       |

### Make Your Score

You can write your score function anywhere and just wrap it using `wrap_scorer`.

```python
import random
from parsbench.scores.base import wrap_scorer

@wrap_scorer
def random_score(completion: str, target: str) -> float:
    return random.random()
```

The function's name will be used as the name of your score in the benchmark result.

## Data Loader

Data Loader class will load the dataset needed for evaluation.

### JSONLine

`JSONLineDataLoader` is based on the `jsonlines` file format (`.jsonl`). It can load jsonline files from web or local.

```python
from parsbench.tasks.base import JSONLineDataLoader

data_loader = JSONLineDataLoader(data_path="dataset.jsonl")
data = data_loader.load()
```

### CSV

`CSVDataLoader` is based on the `CSV` file format (`.csv`). It can load CSV files from web or local.

```python
from parsbench.tasks.base import CSVDataLoader

data_loader = CSVDataLoader(data_path="dataset.csv")
data = data_loader.load()
```

### HuggingFace

`HuggingFaceDataLoader` is based on the `datasets` library of HuggingFace that loads datasets from local or downloads it from the HuggingFace Dataset Hub.

```python
from parsbench.tasks.base import HuggingFaceDataLoader

data_loader = HuggingFaceDataLoader(
    data_path="persiannlp/parsinlu_entailment"
    split="validation"
)
data = data_loader.load()
```

## Prompt Template

Using `PromptTemplate` class, you can define the prompt template for different lanugages, shot templates, shot examples, prompt variables, etc.

### With Shot Template

In this example, you can define a prompt template for sentiment analysis task with shot template.

Prompt and Shot Templates:

```python
FA_TEMPLATE = """
جمله زیر نظر یک شخص است. این جمله به زبان فارسی است. بار یا احساس موجود در این جمله را شناسایی کن.
پاسخ‌ های ممکن حالت‌های روبرو هستند:
SAD
NETURAL
HAPPY

فقط کلمه مروبط به احساس نظر داده شده را خروجی بده.

{example_shots}

نظر: {review}
احساس:
"""

FA_SHOT_TEMPLATE = """
نظر: {review}
احساس: {label}
"""
```

And for the task prompt template:

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
    data={"review": "غذا خیلی بد بود", "label": "SAD"}
    n_shots=3,
    sample_data=[
        {"review": "خوشمزه بود ممونم", "label": "HAPPY"},
        {"review": "غذا خوب بود فقط کاش زودتر می‌رسید.", "label": "NETURAL"},
        {"review": "نوشابه گرم بود. پیتزا هم خیلی بد مزه بود.", "label": "SAD"}
    ]
)
```

### With Shot Examples

If the task is complicated and you want to use methods like CoT (Chain of Thought) prompting,
You can use static shot examples.

```python
from parsbench.tasks.base import PromptTemplate

prompt_template = PromptTemplate(
    language_templates={"fa": FA_TEMPLATE},
    prompt_shot_examples={"fa": {1: FA_1_SHOT, 3: FA_3_SHOT, 5: FA_5_SHOT}}
)
prompt = prompt_template.get_prompt(prompt_shot=5, ...)
```

### Load Templates From File

You can also load prompts from text files using `LazyLoadTemplates`.

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

### Constant Prompt Variable

Sometimes you may wanna fill some prompt variables with constant data and you don't wanna put it in the prompt template text. You can use `ConstantPromptVariable`:

```python
from parsbench.tasks.base import PromptTemplate, ConstantPromptVariable

prompt_template = PromptTemplate(
    language_templates={"fa": FA_TEMPLATE},
    prompt_shot_templates={"fa": FA_SHOT_TEMPLATE},
    prompt_variables_mapping={
        "input": "input",
        "first_name": ConstantPromptVariable("شهریار")
    },
    target_variables_mapping={"label": "label"},
)
```

## Tasks

The primary unit of the ParsBench framework is a task. Tasks are battery-included evaluators which do all the process from loading dataset to evaluating models and outputting the result.

### Task Data Provider

Each task includes a dataset for evaluation. If you need to get the data of that task, you can use `task.get_data` function.

```python
from parsbench.tasks import ParsiNLUEntailment

with ParsiNLUEntailment() as task:  # Open in context manager to load data.
    data = task.get_data()
```

### Task Match Generator

A `TaskMatch` is an object which includes prompt, target answer, model completion and the score. Initially the matches doesn't have `completion` and `score` attributes (default to `None`). But you can generate completions and score them using the task class itself.

To generate matches based on the prompt template, you can use following code:

```python
from parsbench.tasks import ParsiNLUEntailment

with ParsiNLUEntailment() as task:
    matches = task.generate_matches(prompt_lang="fa", n_shots=0, n_first=100)
```

### Generate Completions and Score

After generating matches, now we should generate completions for each match prompt. Then we score them based on the defined scorer in the task.

```python
from parsbench.tasks import ParsiNLUEntailment

with ParsiNLUEntailment() as task:
    matches = task.generate_matches(prompt_lang="fa", n_shots=0, n_first=100)
    model.generate_completions(matches)  # Completions are generated by the model.
    tasks.score_matches(matches)
```

### Make Your Task

If you want to have a task with your own private dataset, prompts, setups, etc. You can inherit one of the existing tasks or create your own task from scratch.

We suggest to put your prompt templates in a text file and use `LazyLoadTemplates` to load them for a better performance.

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

You can use any data loader, prompt template and scorer you want.

#### Add Sub Tasks

Your task might have a couple of sub tasks. In that case you should specify `sub_task_key` and `sub_tasks` attributes in the task class.

```python
class CustomTask(Task):
    ...
    sub_task_key: str = "category"  # Column in the dataset which specify the sub tasks.
    sub_tasks: list[str] = ["math_and_logic", "common_knowledge", "literature"]  # Expected sub tasks.
    ...
```

And for generating matches or evaluating the model, you can specify a subset of the sub tasks.

```python
with CustomTask() as task:
    results = task.evaluate(..., sub_tasks=["math_and_logic"])

# OR

benchmark = CustomBenchmark(
    ...,
    tasks=[
        PersianMath,
        CustomTask.select_sub_tasks(["math_and_logic"]),
    ]
)
```
