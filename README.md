# ParsBench

<div align="center">
    <a href="https://github.com/ParsBench/ParsBench">
        <img src="https://raw.githubusercontent.com/ParsBench/ParsBench/main/docs/imgs/banner-black.png" alt="Beanie" width="480" height="240">
    </a>
    <br>
    <a href="https://shahriarshm.github.io/parsbench/">
        <img src="https://shields.io/badge/-docs-blue" alt="docs">
    </a>
    <a href="https://pypi.python.org/pypi/parsbench">
        <img src="https://img.shields.io/pypi/v/parsbench.svg" alt="pypi">
    </a>
    <a href="https://huggingface.co/ParsBench">
        <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-me-on-HF-md-dark.svg" alt="huggingface">
    </a>
</div>

ParsBench provides toolkits for benchmarking Large Language Models (LLMs) based on the Persian language. It includes various tasks for evaluating LLMs on different topics, benchmarking tools to compare multiple models and rank them, and an easy, fully customizable API for developers to create custom models, tasks, scores, and benchmarks.

## Key Features

- **Variety of Tasks**: Evaluate LLMs across various topics.
- **Benchmarking Tools**: Compare and rank multiple models.
- **Customizable API**: Create custom models, tasks, scores, and benchmarks with ease.

## Motivation

I was trying to fine-tune an open-source LLM for the Persian language. I needed some evaluation to test the performance and utility of my LLM. It leads me to research and find [this paper](https://arxiv.org/abs/2404.02403). It's great work that they prepared some datasets and evaluation methods to test on ChatGPT. They even shared their code in this [repository](https://github.com/Ipouyall/Benchmarking_ChatGPT_for_Persian).

So, I thought that I should build a handy framework that includes various tasks and datasets for evaluating LLMs based on the Persian language. I used some parts of their work (Datasets, Metrics, Basic prompt templates) in this library.

## Installation

Install [Math Equivalence](https://github.com/hendrycks/math) package manually:

```bash
pip install git+https://github.com/hendrycks/math.git
```

Install ParsBench using pip:

```bash
pip install parsbench
```

## Usage

### Evaluating a PreTrained Model

Load the pre-trained model and tokenizer from the HuggingFace and then, evaluate the model using the PersianMath task:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from parsbench.models import PreTrainedTransformerModel
from parsbench.tasks import PersianMath

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-72B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-72B-Instruct")

tf_model = PreTrainedTransformerModel(model=model, tokenizer=tokenizer)

with PersianMath() as task:
    results = task.evaluate(tf_model)
```

### Benchmarking Multiple Models with Multiple Tasks

For example, we run our local models using Ollama:

```bash
ollama run qwen2
ollama run aya
```

Then we benchmark those models using the ParsBench.

```python
from parsbench.benchmarks import CustomBenchmark
from parsbench.models import OpenAIModel
from parsbench.tasks import ParsiNLUMultipleChoice, PersianMath, ParsiNLUReadingComprehension

qwen2_model = OpenAIModel(
    api_base_url="http://localhost:11434/v1/",
    api_secret_key="ollama",
    model="qwen2:latest",
)
aya_model = OpenAIModel(
    api_base_url="http://localhost:11434/v1/",
    api_secret_key="ollama",
    model="aya:latest",
)

benchmark = CustomBenchmark(
    models=[qwen2_model, aya_model],
    tasks=[
        ParsiNLUMultipleChoice,
        ParsiNLUReadingComprehension,
        PersianMath,
    ],
)
result = benchmark.run(
    prompt_lang="fa",
    prompt_shots=[0, 3],
    n_first=100,
    sort_by_score=True,
)
result.show_radar_plot()
```

![Benchmark Bar Plot](https://raw.githubusercontent.com/ParsBench/ParsBench/main/docs/imgs/radarplot.png)

## Available Tasks

| Task Name                   | Score Name       | Dataset      |
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

You can import the class of above tasks from `parsbench.tasks` and use it for evaluating your model.

## Example Notebooks

- Benchmark [Aya](https://huggingface.co/CohereForAI) models: [![aya](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aPayB9AaheDxT7zS4A_4SAMH3a7mIDFX?usp=sharing)
- Benchmark [Ava](https://huggingface.co/MehdiHosseiniMoghadam) models: [![ava](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1ToJ8gTQz1ifU70EBAM7fZG2LIOY4zAp0/view?usp=sharing)
- Benchmark [Dorna](https://huggingface.co/PartAI) models: [![dorna](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1f64d0GnmcQIZ-tlN8cg49pPdiwlVlWvi/view?usp=sharing)
- Benchmark [MaralGPT](https://huggingface.co/MaralGPT) models: [![maralgpt](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1ZfjxPa4CfAZdQgtPaEt3nnX180A825ZF/view?usp=sharing)

## Sponsors

Here are the names of companies/people who helped us to keep maintaining this project. If you want to donate this project, see [this page](https://shahriarshm.github.io/parsbench/donation/).

- [AvalAI](https://avalai.ir/): They gave us free OpenAI API credit several times in their "AvalAward" program. It helped us for doing R&D and benchmarking GPT models.
- [Basalam](https://basalam.com/): They voluntarily helped us to run the benchmarks on open-weight models and build the [ParsBench Leaderboard](https://huggingface.co/spaces/ParsBench/leaderboard).

## Contributing

Contributions are welcome! Please refer to the [contribution guidelines](docs/contribution.md) for more information on how to contribute.

## License

ParsBench is distributed under the Apache-2.0 license.

## Contact Information

For support or questions, please contact: [shahriarshm81@gmail.com](mailto:shahriarshm81@gmail.com)
Feel free to let me know if there are any additional details or changes you'd like to make!
