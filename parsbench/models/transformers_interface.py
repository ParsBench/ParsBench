from typing import Callable

from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer

from .base import DEFUALT_INSTRUCTION_PROMPT, Model

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
        custom_prompt_formater (Callable[[str], str] | None): A custom prompt formatter function.

    Methods:
        model_name: Returns the base model prefix of the transformer model.
        prompt_formater: Formats a prompt by combining system instruction and user input.
        get_prompt_completion: Generates a completion for a given prompt using the model and tokenizer.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        generation_config: GenerationConfig = DEFAULT_GENERATION_CONFIG,
        intruction_prompt: str = DEFUALT_INSTRUCTION_PROMPT,
        custom_prompt_formater: Callable[[str], str] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.intruction_prompt = intruction_prompt
        self.custom_prompt_formater = custom_prompt_formater

    @property
    def model_name(self) -> str:
        return self.model.base_model_prefix

    def prompt_formater(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.intruction_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text

    def get_prompt_completion(self, prompt: str) -> str:
        if self.custom_prompt_formater:
            input_text = self.custom_prompt_formater(prompt)
        else:
            input_text = self.prompt_formater(prompt)

        model_inputs = self.tokenizer([input_text], return_tensors="pt").to(
            self.model.device
        )

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            generation_config=self.generation_config,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response
