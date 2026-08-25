# Models

Models are the interfaces that generate completions from LLMs. A benchmark
or task doesn't care where the completion comes from; it calls the model
interface and scores what comes back. Three interfaces ship with ParsBench,
and writing your own takes four methods.

## OpenAIModel

`OpenAIModel` speaks the OpenAI chat-completions protocol, which in practice
means most of the ecosystem: OpenAI itself, gateways like
[AvalAI](https://avalai.ir/) and [OpenRouter](https://openrouter.ai/), and
local runtimes like [Ollama](https://ollama.com/) and vLLM.

```python
from parsbench.models import OpenAIModel

model = OpenAIModel(
    api_base_url="https://api.openai.com/v1/",
    api_secret_key="{SECRET_KEY}",
    model="gpt-4.1",
)
```

A local model through Ollama is the same interface with a different base URL:

```bash
ollama run llama3
```

```python
model = OpenAIModel(
    api_base_url="http://localhost:11434/v1/",
    api_secret_key="ollama",
    model="llama3:latest",
)
```

Useful optional parameters, shared with `AnthropicModel`:

- `instruction_prompt=` replaces the default system prompt.
- `completion_parameters=` passes sampling parameters through to the API,
  e.g. `{"temperature": 0}`.
- `retry_on_ratelimit=True` with `cooldown_interval=` (seconds, default 10)
  and `max_retries=` (default 1) retries rate-limited calls instead of
  failing the run.

## AnthropicModel

`AnthropicModel` is the same idea for Anthropic-style APIs:

```python
from parsbench.models import AnthropicModel

model = AnthropicModel(
    api_secret_key="{SECRET_KEY}",
    model="claude-sonnet-4-5",
)
```

`api_base_url=` is optional and lets you point it at an Anthropic-compatible
gateway.

## PreTrainedTransformerModel

`PreTrainedTransformerModel` wraps a `PreTrainedModel` from the
[transformers](https://huggingface.co/docs/transformers) library, so you can
evaluate a checkpoint directly, including one you just fine-tuned, without
serving it behind an API:

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

## Create your own interface

Inherit the `Model` abstract class and implement four methods:

```python
from parsbench.models import Model

class CustomModel(Model):
    @property
    def model_name(self) -> str:
        return "My Custom Model"

    def get_prompt_completion(self, prompt: str) -> str:
        # call your API / model here
        return f"Response to {prompt}"

    def prompt_formatter(self, prompt: str) -> str | list[dict]:
        return prompt  # or wrap into chat messages

    def completion_formatter(self, completion: str) -> str:
        return completion.strip().replace("'", "")
```

- `model_name` labels the model in results and output directories.
- `get_prompt_completion` does the actual call.
- `prompt_formatter` turns the task's prompt string into whatever your API
  expects (a raw string, or an OpenAI-style message list).
- `completion_formatter` cleans the raw completion before scoring, stripping
  quotes, whitespace, or chatter that would break exact-match scores.
