from openai import OpenAI

from .base import DEFUALT_INSTRUCTION_PROMPT, Model


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
        client (OpenAI): An instance of the OpenAI client for API interactions.

    Methods:
        model_name: Returns the name of the model.
        prompt_formater: Formats a given prompt into a list of messages.
        get_prompt_completion: Generates completion for a given prompt using the OpenAI API.
        generate_completions: Generates completions for a list of TaskMatch objects using ThreadPoolExecutor.
    """

    def __init__(
        self,
        api_base_url: str,
        api_secret_key: str,
        model: str,
        intruction_prompt: str = DEFUALT_INSTRUCTION_PROMPT,
        model_parameters: dict = None,
        completion_parameters: dict = None,
        **kwargs
    ):
        self.api_base_url = api_base_url
        self.api_secret_key = api_secret_key
        self.model = model
        self.intruction_prompt = intruction_prompt
        self.model_parameters = model_parameters or dict()
        self.completion_parameters = completion_parameters or dict(temperature=0.7)

        self.client = OpenAI(
            base_url=self.api_base_url,
            api_key=self.api_secret_key,
            **self.model_parameters,
        )

    @property
    def model_name(self) -> str:
        return self.model

    def prompt_formater(self, prompt: str) -> list[dict]:
        messages = [
            {"role": "system", "content": self.intruction_prompt},
            {"role": "user", "content": prompt},
        ]
        return messages

    def get_prompt_completion(self, prompt: str) -> str:
        messages = self.prompt_formater(prompt)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.completion_parameters,
            stream=False,  # Alwas override this parameter.
        )
        return completion.choices[0].message.content
