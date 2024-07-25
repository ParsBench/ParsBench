from typing import Callable

from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer

from .base import DEFAULT_INSTRUCTION_PROMPT, Model

DEFAULT_GENERATION_CONFIG = GenerationConfig(
    max_length=1024,
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.0,
    do_sample=True,
)


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
