# Getting Started

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

![Benchmark Bar Plot](imgs/radarplot.png)
