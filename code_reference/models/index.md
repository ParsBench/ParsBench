# Models

Bases: `ABC`

An abstract base class representing a model.

Attributes:

| Name                  | Type       | Description                                          |
| --------------------- | ---------- | ---------------------------------------------------- |
| `model_name`          | `property` | A property representing the name of the model.       |
| `support_concurrency` | `bool`     | A flag indicating if the model supports concurrency. |

Methods:

| Name                    | Description                                                                                                                                                                 |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_name`            | Abstract method to return the name of the model.                                                                                                                            |
| `get_prompt_completion` | str) -> str: Abstract method to generate completion for a given prompt.                                                                                                     |
| `prompt_formatter`      | str) -> Union\[str, List[Dict]\]: Abstract method to format a prompt.                                                                                                       |
| `completion_formatter`  | str) -> str: Method to format the model completion.                                                                                                                         |
| `generate_completions`  | TaskMatchGroup, prefer_concurrency: bool = True, n_workers: int = 4) -> TaskMatchGroup: Method to generate completions for a list of matches, optionally using concurrency. |

Note

This class should be subclassed to implement the abstract methods.

Source code in `parsbench/models/base.py`

```
class Model(ABC):
    """
    An abstract base class representing a model.

    Attributes:
        model_name (property): A property representing the name of the model.
        support_concurrency (bool): A flag indicating if the model supports concurrency.

    Methods:
        model_name(self) -> str: Abstract method to return the name of the model.
        get_prompt_completion (self, prompt: str) -> str: Abstract method to generate completion for a given prompt.
        prompt_formatter (self, prompt: str) -> Union[str, List[Dict]]: Abstract method to format a prompt.
        completion_formatter (self, completion: str) -> str: Method to format the model completion.
        generate_completions (self, matches: TaskMatchGroup, prefer_concurrency: bool = True, n_workers: int = 4) -> TaskMatchGroup: Method to generate completions for a list of matches, optionally using concurrency.

    Note:
        This class should be subclassed to implement the abstract methods.
    """

    support_concurrency: bool = False

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def get_prompt_completion(self, prompt: str) -> str:
        pass

    @abstractmethod
    def prompt_formatter(self, prompt: str) -> str | list[dict]:
        pass

    def completion_formatter(self, completion: str) -> str:
        return completion

    def generate_completions(
        self,
        matches: "TaskMatchGroup",
        prefer_concurrency: bool = True,
        skip_existing: bool = False,
        n_workers: int = 4,
    ) -> "TaskMatchGroup":
        if prefer_concurrency and self.support_concurrency:
            matches = self._gen_with_concurrency(
                matches, n_workers=n_workers, skip_existing=skip_existing
            )
        else:
            for match in tqdm(
                matches, total=len(matches), desc="Generating completions"
            ):
                if match.completion is not None and skip_existing:
                    continue
                match.completion = self.completion_formatter(
                    self.get_prompt_completion(match.prompt)
                )
        return matches

    def _gen_with_concurrency(
        self,
        matches: "TaskMatchGroup",
        n_workers: int = 4,
        skip_existing: bool = False,
    ) -> "TaskMatchGroup":
        def _gen_single_match_completion(match: "TaskMatch") -> "TaskMatch":
            match.completion = self.completion_formatter(
                self.get_prompt_completion(match.prompt)
            )
            return match

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []

            for match in matches:
                if match.completion is not None and skip_existing:
                    continue
                future = executor.submit(
                    _gen_single_match_completion,
                    match,
                )
                futures.append(future)

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Generating completions"
            ):
                future.result()

        matches.matches.sort(key=lambda m: m.id)
        return matches
```

Bases: `Model`

A model interface for OpenAI-like APIs.

Attributes:

| Name                    | Type     | Description                                            |
| ----------------------- | -------- | ------------------------------------------------------ |
| `api_base_url`          | `str`    | The base URL for the OpenAI API.                       |
| `api_secret_key`        | `str`    | The secret key for accessing the OpenAI API.           |
| `model`                 | `str`    | The specific model being used for processing.          |
| `instruction_prompt`    | `str`    | The default instruction prompt for the model.          |
| `model_parameters`      | `dict`   | Additional parameters specific to the model.           |
| `completion_parameters` | `dict`   | Parameters for completion generation.                  |
| `retry_on_ratelimit`    |          | bool = False,                                          |
| `cooldown_interval`     |          | int = 10,                                              |
| `max_retries`           |          | int = 1,                                               |
| `client`                | `OpenAI` | An instance of the OpenAI client for API interactions. |

Methods:

| Name                    | Description                                                                     |
| ----------------------- | ------------------------------------------------------------------------------- |
| `model_name`            | Returns the name of the model.                                                  |
| `prompt_formatter`      | Formats a given prompt into a list of messages. Could be overloaded.            |
| `completion_formatter`  | Method to format the model completion. Could be overloaded.                     |
| `get_prompt_completion` | Generates completion for a given prompt using the OpenAI API.                   |
| `generate_completions`  | Generates completions for a list of TaskMatch objects using ThreadPoolExecutor. |

Source code in `parsbench/models/openai_interface.py`

```
class OpenAIModel(Model):
    """
    A model interface for OpenAI-like APIs.

    Attributes:
        api_base_url (str): The base URL for the OpenAI API.
        api_secret_key (str): The secret key for accessing the OpenAI API.
        model (str): The specific model being used for processing.
        instruction_prompt (str): The default instruction prompt for the model.
        model_parameters (dict): Additional parameters specific to the model.
        completion_parameters (dict): Parameters for completion generation.
        retry_on_ratelimit: bool = False,
        cooldown_interval: int = 10,
        max_retries: int = 1,
        client (OpenAI): An instance of the OpenAI client for API interactions.

    Methods:
        model_name: Returns the name of the model.
        prompt_formatter: Formats a given prompt into a list of messages. Could be overloaded.
        completion_formatter: Method to format the model completion. Could be overloaded.
        get_prompt_completion: Generates completion for a given prompt using the OpenAI API.
        generate_completions: Generates completions for a list of TaskMatch objects using ThreadPoolExecutor.
    """

    support_concurrency: bool = True

    def __init__(
        self,
        api_base_url: str,
        api_secret_key: str,
        model: str,
        instruction_prompt: str = DEFAULT_INSTRUCTION_PROMPT,
        model_parameters: dict = None,
        completion_parameters: dict = None,
        retry_on_ratelimit: bool = False,
        cooldown_interval: int = 10,
        max_retries: int = 1,
        **kwargs
    ):
        self.api_base_url = api_base_url
        self.api_secret_key = api_secret_key
        self.model = model
        self.instruction_prompt = instruction_prompt
        self.model_parameters = model_parameters or dict()
        self.completion_parameters = completion_parameters or dict(temperature=0.7)
        self.retry_on_ratelimit = retry_on_ratelimit
        self.cooldown_interval = cooldown_interval
        self.max_retries = max_retries

        self.client = OpenAI(
            base_url=self.api_base_url,
            api_key=self.api_secret_key,
            **self.model_parameters,
        )

    @property
    def model_name(self) -> str:
        return self.model

    def prompt_formatter(self, prompt: str) -> list[dict]:
        messages = [
            {"role": "system", "content": self.instruction_prompt},
            {"role": "user", "content": prompt},
        ]
        return messages

    def get_prompt_completion(self, prompt: str) -> str:
        messages = self.prompt_formatter(prompt)

        retries = 0
        while retries < self.max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **self.completion_parameters,
                    stream=False,  # Always override this parameter.
                )
                return completion.choices[0].message.content
            except RateLimitError as exc:
                if self.retry_on_ratelimit:
                    retries += 1
                    time.sleep(self.cooldown_interval)
                else:
                    raise exc

        raise Exception("Max retries exceeded.")
```

Bases: `Model`

A model interface for Anthropic-like APIs.

Attributes:

| Name                    | Type        | Description                                               |
| ----------------------- | ----------- | --------------------------------------------------------- |
| `api_base_url`          | `str`       | The base URL for the Anthropic API.                       |
| `api_secret_key`        | `str`       | The secret key for accessing the Anthropic API.           |
| `model`                 | `str`       | The name of the model.                                    |
| `instruction_prompt`    | `str`       | The default instruction prompt for the model.             |
| `model_parameters`      | `dict`      | Additional parameters specific to the model.              |
| `completion_parameters` | `dict`      | Parameters for generating completions.                    |
| `retry_on_ratelimit`    |             | bool = False,                                             |
| `cooldown_interval`     |             | int = 10,                                                 |
| `max_retries`           |             | int = 1,                                                  |
| `client`                | `Anthropic` | An instance of the Anthropic client for API interactions. |

Methods:

| Name                    | Description                                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `model_name`            | Returns the name of the model.                                                                                                       |
| `prompt_formatter`      | str) -> list\[dict\]: Formats the prompt into a list of messages.                                                                    |
| `get_prompt_completion` | str) -> str: Generates completion for a given prompt.                                                                                |
| `generate_completions`  | TaskMatchGroup, prefer_concurrency: bool = True, n_workers: int = 4) -> TaskMatchGroup: Generates completions for a list of matches. |

Source code in `parsbench/models/anthropic_interface.py`

```
class AnthropicModel(Model):
    """
    A model interface for Anthropic-like APIs.

    Attributes:
        api_base_url (str): The base URL for the Anthropic API.
        api_secret_key (str): The secret key for accessing the Anthropic API.
        model (str): The name of the model.
        instruction_prompt (str): The default instruction prompt for the model.
        model_parameters (dict): Additional parameters specific to the model.
        completion_parameters (dict): Parameters for generating completions.
        retry_on_ratelimit: bool = False,
        cooldown_interval: int = 10,
        max_retries: int = 1,
        client (Anthropic): An instance of the Anthropic client for API interactions.

    Methods:
        model_name(self) -> str: Returns the name of the model.
        prompt_formatter(self, prompt: str) -> list[dict]: Formats the prompt into a list of messages.
        get_prompt_completion(self, prompt: str) -> str: Generates completion for a given prompt.
        generate_completions(self, matches: TaskMatchGroup, prefer_concurrency: bool = True, n_workers: int = 4) -> TaskMatchGroup: Generates completions for a list of matches.
    """

    support_concurrency: bool = True

    def __init__(
        self,
        api_secret_key: str,
        model: str,
        api_base_url: str | None = None,
        instruction_prompt: str = DEFAULT_INSTRUCTION_PROMPT,
        model_parameters: dict = None,
        completion_parameters: dict = None,
        retry_on_ratelimit: bool = False,
        cooldown_interval: int = 10,
        max_retries: int = 1,
        **kwargs
    ):
        self.api_base_url = api_base_url
        self.api_secret_key = api_secret_key
        self.model = model
        self.instruction_prompt = instruction_prompt
        self.model_parameters = model_parameters or dict()
        self.completion_parameters = completion_parameters or dict(
            max_tokens=1024, temperature=0.7
        )
        self.retry_on_ratelimit = retry_on_ratelimit
        self.cooldown_interval = cooldown_interval
        self.max_retries = max_retries

        self.client = Anthropic(
            base_url=self.api_base_url,
            api_key=self.api_secret_key,
            **self.model_parameters,
        )

    @property
    def model_name(self) -> str:
        return self.model

    def prompt_formatter(self, prompt: str) -> list[dict]:
        messages = [
            {"role": "user", "content": prompt},
        ]
        return messages

    def get_prompt_completion(self, prompt: str) -> str:
        messages = self.prompt_formatter(prompt)

        retries = 0
        while retries < self.max_retries:
            try:
                message = self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    system=self.instruction_prompt,
                    **self.completion_parameters,
                    stream=False,  # Always override this parameter.
                )
                return message.content[0].text
            except RateLimitError as exc:
                if self.retry_on_ratelimit:
                    retries += 1
                    time.sleep(self.cooldown_interval)
                else:
                    raise exc

        raise Exception("Max retries exceeded.")
```

Bases: `Model`

A model interface for pre-trained transformer models.

Attributes:

| Name                      | Type                     | Description                                       |
| ------------------------- | ------------------------ | ------------------------------------------------- |
| `model`                   | `PreTrainedModel`        | The pre-trained transformer model.                |
| `tokenizer`               | `PreTrainedTokenizer`    | The tokenizer associated with the model.          |
| `generation_config`       | `GenerationConfig`       | The generation configuration for text generation. |
| `instruction_prompt`      | `str`                    | The default instruction prompt for the model.     |
| `custom_prompt_formatter` | \`Callable\[[str], str\] | None\`                                            |

Methods:

| Name                    | Description                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------- |
| `model_name`            | Returns the base model prefix of the transformer model.                               |
| `prompt_formatter`      | Formats a prompt by combining system instruction and user input. Could be overloaded. |
| `completion_formatter`  | Method to format the model completion. Could be overloaded.                           |
| `get_prompt_completion` | Generates a completion for a given prompt using the model and tokenizer.              |

Source code in `parsbench/models/transformers_interface.py`

```
class PreTrainedTransformerModel(Model):
    """
    A model interface for pre-trained transformer models.

    Attributes:
        model (PreTrainedModel): The pre-trained transformer model.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        generation_config (GenerationConfig): The generation configuration for text generation.
        instruction_prompt (str): The default instruction prompt for the model.
        custom_prompt_formatter (Callable[[str], str] | None): A custom prompt formatter function.

    Methods:
        model_name: Returns the base model prefix of the transformer model.
        prompt_formatter: Formats a prompt by combining system instruction and user input. Could be overloaded.
        completion_formatter: Method to format the model completion. Could be overloaded.
        get_prompt_completion: Generates a completion for a given prompt using the model and tokenizer.
    """

    support_concurrency: bool = False  # TODO: should support later.

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        generation_config: GenerationConfig = DEFAULT_GENERATION_CONFIG,
        instruction_prompt: str = DEFAULT_INSTRUCTION_PROMPT,
        custom_prompt_formatter: Callable[[str], str] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.instruction_prompt = instruction_prompt
        self.custom_prompt_formatter = custom_prompt_formatter

    @property
    def model_name(self) -> str:
        return self.model.config.name_or_path or "model"

    def prompt_formatter(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.instruction_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text

    def get_prompt_completion(self, prompt: str) -> str:
        if self.custom_prompt_formatter:
            input_text = self.custom_prompt_formatter(prompt)
        else:
            input_text = self.prompt_formatter(prompt)

        model_inputs = self.tokenizer([input_text], return_tensors="pt").to(
            self.model.device
        )

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            generation_config=self.generation_config,
            attention_mask=model_inputs.attention_mask,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response
```
