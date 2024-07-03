from collections.abc import Mapping


class ConstantPromptVariable(str):
    """
    A class representing a constant prompt variable.

    Attributes:
        value (str): The value of the constant prompt variable.
    """

    def __init__(self, value):
        self.value = value


class PromptTemplate:
    """
    A class representing a prompt template.

    Attributes:
        language_templates (dict[str, str]): A dictionary mapping language codes to prompt templates.
        prompt_variables_mapping (dict[str, str]): A dictionary mapping prompt variable names to corresponding data keys.
        target_variables_mapping (dict[str, str]): A dictionary mapping target variable names to corresponding data keys.
        prompt_shot_templates (dict[str, str] | None): A dictionary mapping prompt shot templates to language codes, or None if not provided.
        prompt_shot_examples (dict[str, dict[int, str]] | None): A dictionary mapping prompt shot examples to language codes and shot numbers, or None if not provided.
    """

    def __init__(
        self,
        language_templates: dict[str, str],
        prompt_variables_mapping: dict[str, str],
        target_variables_mapping: dict[str, str],
        prompt_shot_templates: dict[str, str] | None = None,
        prompt_shot_examples: dict[str, dict[int, str]] | None = None,
    ):
        self.language_templates = language_templates
        self.prompt_variables_mapping = prompt_variables_mapping
        self.target_variables_mapping = target_variables_mapping

        if prompt_shot_templates is not None and prompt_shot_examples is not None:
            raise ValueError("Cannot provide both prompt shot templates and examples")

        if prompt_shot_templates is None and prompt_shot_examples is None:
            raise ValueError("Must provide either prompt shot templates or examples")

        self.prompt_shot_templates = prompt_shot_templates
        self.prompt_shot_examples = prompt_shot_examples

    def get_prompt(
        self,
        prompt_lang: str,
        data: dict,
        n_shots: int = 0,
        sample_data: list[dict] | None = None,
    ):
        prompt_template = self.language_templates.get(prompt_lang, None)
        if not prompt_template:
            raise RuntimeError(
                f"There is no prompt template for language {prompt_lang}."
            )

        if n_shots > 0:
            if sample_data:
                example_text = self._gen_example_text(prompt_lang, n_shots, sample_data)
            else:
                example_text = self._get_static_example_text(prompt_lang, n_shots)
        else:
            example_text = ""

        prompt = prompt_template.format(
            example_shots=example_text, **self.get_prompt_variables(data)
        )
        prompt = prompt.replace("\n\n\n", "\n")

        return prompt

    def get_prompt_variables(self, data: dict) -> dict:
        mapped_data = {}
        for pk, dk in self.prompt_variables_mapping.items():
            if isinstance(dk, ConstantPromptVariable):
                mapped_data[pk] = dk.value
            else:
                if dk not in data:
                    raise ValueError(f"Key {dk} not in data.")
                mapped_data[pk] = data[dk]
        return mapped_data

    def get_target_variables(self, data: dict) -> dict:
        mapped_data = {}
        for tk, dk in self.target_variables_mapping.items():
            if dk not in data:
                raise ValueError(f"Key {dk} not in data.")
            mapped_data[tk] = data[dk]
        return mapped_data

    def _get_static_example_text(self, prompt_lang: str, n_shots: int) -> str:
        shot_examples = self.prompt_shot_examples.get(prompt_lang, None)
        if not shot_examples:
            raise RuntimeError(f"There is no shot example for language {prompt_lang}.")

        example_text = shot_examples.get(n_shots, "")
        if not example_text:
            raise RuntimeError(
                f"There is no {n_shots}-shot example for langauge {prompt_lang}. "
                f"You can only use {', '.join(map(str, shot_examples.keys()))} shot examples."
            )

        return example_text

    def _gen_example_text(
        self, prompt_lang: str, n_shots: int, sample_data: list[dict]
    ) -> str:
        if len(sample_data) != n_shots:
            raise RuntimeError(
                f"The number of samples ({len(sample_data)}) is not equal to the number of shots ({n_shots})."
            )

        if shot_template := self.prompt_shot_templates.get(prompt_lang):
            example_text = "\n".join(
                shot_template.format(
                    **self.get_prompt_variables(sample),
                    **self.get_target_variables(sample),
                )
                for sample in sample_data
            )
        else:
            sample_variables = [
                {
                    **self.get_prompt_variables(sample),
                    **self.get_target_variables(sample),
                }
                for sample in sample_data
            ]
            example_text = "\n".join(
                "\n".join(f"{k.capitalize()}:\n{v}" for k, v in variables)
                for variables in sample_variables
            )

        return example_text

    @property
    def has_shot_templates(self) -> bool:
        return bool(self.prompt_shot_templates)

    @property
    def has_shot_examples(self) -> bool:
        return bool(self.prompt_shot_examples)


class LazyLoadTemplates(Mapping):
    """
    A class representing lazy loading of templates.

    Inherits from Mapping.

    Attributes:
        template_paths (dict[str, str]): A dictionary mapping template keys to file paths.
    """

    def __init__(self, template_paths: dict[str, str] | None = None, **kwargs):
        super().__init__()
        self.template_paths = template_paths or kwargs or {}
        self._contents: dict[str, str | None] = {
            key: None for key in self.template_paths
        }

    def _load_content(self, key):
        if key in self.template_paths:
            with open(self.template_paths[key], "r") as file:
                self._contents[key] = file.read()
        else:
            raise KeyError(f"Key '{key}' not found in template_paths")

    def __getitem__(self, key) -> str:
        if key not in self._contents:
            raise KeyError(f"Key '{key}' not found")
        if self._contents[key] is None:
            self._load_content(key)
        return self._contents[key]

    def __getattr__(self, key) -> str:
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError(f"Attribute '{key}' not found")

    def __iter__(self):
        return iter(self.template_paths)

    def __len__(self):
        return len(self.template_paths)
