# Tasks

Tasks are used to test and evaluate the Model responses. They come with a dataset of questions and expected answers. The task will generate prompts based on the prompt template and data, get the completion of that prompt using the Model, and then score it using the specified score.

## Available Tasks

| Task Name                   | Score Name       | Dataset      |
|-----------------------------|------------------|--------------|
| ParsiNLU Sentiment Analysis | Exact Match (F1) | [ParsiNLU](https://huggingface.co/datasets/persiannlp/parsinlu_sentiment) |
| ParsiNLU Entailment | Exact Match (F1) | [ParsiNLU](https://huggingface.co/datasets/persiannlp/parsinlu_entailment) |
| ParsiNLU Machine Translation En -> Fa | Bleu | [ParsiNLU](https://huggingface.co/datasets/persiannlp/parsinlu_translation_en_fa) |
| ParsiNLU Machine Translation Fa -> En | Bleu | [ParsiNLU](https://huggingface.co/datasets/persiannlp/parsinlu_translation_fa_en) |
| PersiNLU Multiple Choice | Exact Match (Accuracy) | [ParsiNLU](https://github.com/persiannlp/parsinlu) |
| ParsiNLU Reading Comprehension | Common Tokens (F1) | [ParsiNLU](https://huggingface.co/datasets/persiannlp/parsinlu_reading_comprehension) |
| Persian NER | NER Exact Match (F1) | [PersianNER](https://github.com/HaniehP/PersianNER) |
| Persian Math | Math Equivalence (Accuracy) | [Source](https://github.com/Ipouyall/Benchmarking_ChatGPT_for_Persian) |
| ConjNLI Entailment | Exact Match (F1) | [Source](https://github.com/Ipouyall/Benchmarking_ChatGPT_for_Persian) |
| Persian MMLU (Khayyam Challenge) | Exact Match (Accuracy) | [Khayyam Challenge](https://huggingface.co/datasets/raia-center/khayyam-challenge) |

You can import the class of above tasks from `parsbench.tasks` and use it for evaluating your model.

## Evaluation

The evaluation process has 6 steps:

1. Loading Data
2. Loading Prompt Template
3. Generating Matches (Prompt-Answer)
4. Generating Completions
5. Scoring Completions
6. Storing Result (Optional)

Here is an example of evaluating a pre-trained model on PersianMath:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from parsbench.models import PreTrainedTransformerModel
from parsbench.tasks import ParsiNLUMultipleChoice

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-72B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-72B-Instruct")

tf_model = PreTrainedTransformerModel(model=model, tokenizer=tokenizer)

with ParsiNLUMultipleChoice() as task:
    results = task.evaluate(
        model=tf_model,
        prompt_lang="fa",
        prompt_shots=[0, 5],
    )
```

You should use the task in a context manager. It manages data loading and offloading for a better performance.

### Evaluation Result

The `evaluate` function of a task will return a list of `EvaluationResult` data classe which contains the overall score for each sub task and n_shot prompt of the task.

You can directly use the class or convert it to a Pandas DataFrame with `to_pandas` function.

```python
eval_result = results[0]
print(eval_result.to_pandas())
```

Output:

```txt
     model_name                 task_name task_category        sub_task  n_shots   score_name     score
0  qwen2:latest  PersiNLU Multiple Choice     knowladge  math_and_logic        0  Exact Match  0.600000
1  qwen2:latest  PersiNLU Multiple Choice     knowladge  math_and_logic        3  Exact Match  0.285714
```

### Save Result

You can manually save the result using `save` function of the `EvaluationResult` or pass `save_evaluation=True` to the `evaluate` function.

You can also save task matches which contains prompt, completion, target, and score by passing `save_matches=True` to the `evaluate` function.

```python
with PersianMath() as task:
    results = task.evaluate(
        model=tf_model,
        prompt_lang="fa",
        prompt_shots=[0, 5],
        save_matches=True,
        save_evaluation=True,
        output_path="results/",
    )
```
