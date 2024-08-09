import time

from openai import OpenAI, RateLimitError

from .base import DEFAULT_INSTRUCTION_PROMPT, Model


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
