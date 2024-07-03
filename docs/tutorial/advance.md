# Advance Tutorial

This section is for the ones who want to implement their own Tasks or want to use the framework APIs for other use cases.

## Scores

Scores are the methods that we use to measure the goodness of the completion of our model, comparing to the expected answer.

### Available Scores

| Score Name            | Description                                                             |
|-----------------------|-------------------------------------------------------------------------|
| Exact Match           | `1` if the completion and target are equal, otherwise `0`.              |
| English Sentence Bleu | Bleu n-gram score with NLTK English word tokenizer. Between `0` to `1`. |
| Persian Sentence Bleu | Bleu n-gram score with Hazm Persian word tokenizer. Between `0` to `1`. |

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
