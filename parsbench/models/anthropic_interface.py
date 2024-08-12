import time

from anthropic import Anthropic, RateLimitError

from .base import DEFAULT_INSTRUCTION_PROMPT, Model


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
