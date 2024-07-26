# Models

Models are the interfaces that enable you to generate completions using LLMs.

## OpenAIModel

The `OpenAIModel` is an interface for using OpenAI-like APIs to generate completions.

```python
from parsbench.models import OpenAIModel

model = OpenAIModel(
    api_base_url="https://api.openai.com/v1/",
    api_secret_key="{SECRET_KEY}",
    model="gpt-4o",
)
```

Use can run your local model using for example `Ollama`:

```bash
ollama run llama3
```

And use its API:

```python
from parsbench.models import OpenAIModel

model = OpenAIModel(
    api_base_url="http://localhost:11434/v1/",
    api_secret_key="ollama",
    model="llama3:latest",
)
```

## AnthropicModel

The `AnthropicModel` is an interface for using Anthropic-like APIs to generate completions.

```python
from parsbench.models import AnthropicModel

model = AnthropicModel(
    api_secret_key="{SECRET_KEY}",
    model="claude3.5-sonnet",
)
```

## PreTrainedTransformerModel

The `PreTrainedTransformerModel` is an interface for the `PreTrainedModel` of the [transformers](https://huggingface.co/docs/transformers) framework.
You can load any pre-trained model and tokenizer you want and pass it to the `PreTrainedTransformerModel`. And it will generate completions using your own model.

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

## Create Your Own Interface

You can easily create your own model interface by inheriting the `Model` abstract class:

```python
from parsbench.models import Model

class CustomModel(Model):
    @property
    def model_name(self) -> str:
        return "My Custom Model"

    def get_prompt_completion(self, prompt: str) -> str:
        return f"Response to {prompt}"

    def prompt_formater(self, prompt: str) -> str | list[dict]:
        return prompt  # No format

    def completion_formatter(self, completion: str) -> str:
        return completion.strip().replace("'", "")
```
