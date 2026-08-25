# Tasks

Bases: `TaskMatchGenerator`, `TaskScorer`

Task class represents a task that combines functionality from TaskMatchGenerator and TaskScorer.

Attributes:

| Name            | Type           | Description               |
| --------------- | -------------- | ------------------------- |
| `task_name`     | `str`          | The name of the task.     |
| `task_category` | `TaskCategory` | The category of the task. |

Methods:

| Name       | Description                                                                              |
| ---------- | ---------------------------------------------------------------------------------------- |
| `evaluate` | Method to evaluate the task by generating matches, scoring them, and saving the results. |

Source code in `parsbench/tasks/base/task.py`

```
class Task(TaskMatchGenerator, TaskScorer, metaclass=ABCMeta):
    """
    Task class represents a task that combines functionality from TaskMatchGenerator and TaskScorer.

    Attributes:
        task_name (str): The name of the task.
        task_category (TaskCategory): The category of the task.

    Methods:
        evaluate: Method to evaluate the task by generating matches, scoring them, and saving the results.
    """

    task_name: str
    task_category: TaskCategory

    def evaluate(
        self,
        model: Model,
        prompt_lang: str = "fa",
        prompt_shots: list[int] = None,
        n_first: int = 200,
        sub_tasks: list[str] | None = None,
        save_matches: bool = False,
        save_evaluation: bool = False,
        output_path: str = None,
        skip_existing_matches: bool = False,
        prefer_concurrency: bool = True,
        n_workers: int = 4,
    ) -> list[EvaluationResult]:
        """
        Method to evaluate the task by generating matches, scoring them, and saving the results.

        Parameters:
            model (Model): The model to be evaluated.
            prompt_lang (str, optional): The language of the prompt (default is "fa").
            prompt_shots (list[int], optional): The list of prompt shots to evaluate (default is None).
            n_first (int, optional): The number of initial prompts to consider (default is 200).
            sub_tasks (list[str], optional): The list of sub-tasks to evaluate (default is None).
            save_matches (bool, optional): Flag to save the generated matches (default is False).
            save_evaluation (bool, optional): Flag to save the evaluation results (default is False).
            output_path (str, optional): The output path to save the matches and evaluation results.
            skip_existing_matches (bool, optional): Flag to skip already generated matches in the output path (default is False).
            prefer_concurrency (bool, optional): The flag to use concurrent processing if the model and task support that (default is True).
            n_workers (int, optional): The number of workers for concurrent processing (default is 4).

        Returns:
            list[EvaluationResult]: A list of EvaluationResult objects representing the evaluation results.

        Raises:
            Exception: If output_path is not provided when saving matches or evaluation.
            Exception: If output_path is not provided when skipping existing matches.
            Exception: If sub tasks are not defined or if invalid sub tasks are provided.

        """
        if (save_matches or save_evaluation) and not output_path:
            raise Exception(
                "You should set the output path to save matches/evaluation."
            )

        if skip_existing_matches and not output_path:
            raise Exception(
                "Cannot find already generated matches when output_path is not set."
            )

        task_path = None
        if output_path:
            task_path = get_task_path(output_path, model.model_name, self.task_name)

        prompt_shots = [0] if prompt_shots is None else prompt_shots

        if sub_tasks:
            if not self.sub_tasks:
                raise Exception("Sub tasks are not defined.")

            invalid_sub_tasks = set(sub_tasks) - set(self.sub_tasks)
            if invalid_sub_tasks:
                raise Exception(f"Sub tasks {invalid_sub_tasks} are not defined.")

        sub_tasks = sub_tasks or self._selected_sub_tasks or self.sub_tasks

        evaluation_results: list[EvaluationResult] = []

        for sub_task in sub_tasks or [None]:
            match_groups: list[TaskMatchGroup] = []

            for shots in prompt_shots:
                if skip_existing_matches and check_task_matches_exists(
                    task_path, shots, sub_task=sub_task
                ):
                    match_group = TaskMatchGroup.from_file(
                        task_path, shots, sub_task=sub_task
                    )
                    match_group._loaded_locally = True
                else:
                    match_group = self.generate_matches(
                        prompt_lang,
                        n_shots=shots,
                        n_first=n_first,
                        sub_task=sub_task,
                    )
                match_groups.append(match_group)

            has_error = False
            for match_group in match_groups:
                eval_desc = f"{match_group.n_shots}-shot"
                if sub_task:
                    eval_desc = f"sub task '{sub_task}' with " + eval_desc
                desc = f"Evaluating {eval_desc} prompt:"
                print(desc)

                is_loaded_locally = getattr(match_group, "_loaded_locally", False)

                if is_loaded_locally:
                    total_skipped = sum(m.completion is not None for m in match_group)
                    print(
                        f"{total_skipped} of {len(match_group)} match completions will be loaded from local."
                    )

                try:
                    model.generate_completions(
                        match_group,
                        prefer_concurrency=prefer_concurrency,
                        skip_existing=is_loaded_locally,
                        n_workers=n_workers,
                    )
                    self.score_matches(match_group)
                except Exception:
                    has_error = True
                finally:
                    if save_matches:
                        match_group.save(task_path, sub_task=sub_task)

            if not has_error:
                evaluation_result = EvaluationResult(
                    model_name=model.model_name,
                    task_name=self.task_name,
                    task_category=self.task_category,
                    score_name=self.score_name,
                    sub_task=sub_task,
                    prompt_shot_results=[
                        PromptShotEvaluationResult(
                            n_shots=m.n_shots,
                            score=self.get_overall_score(m),
                        )
                        for m in match_groups
                    ],
                )
                evaluation_results.append(evaluation_result)

                if save_evaluation:
                    evaluation_result.save(task_path)

        return evaluation_results

    def __enter__(self) -> "Task":
        self.load_data()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._data = None
```

## `evaluate(model, prompt_lang='fa', prompt_shots=None, n_first=200, sub_tasks=None, save_matches=False, save_evaluation=False, output_path=None, skip_existing_matches=False, prefer_concurrency=True, n_workers=4)`

Method to evaluate the task by generating matches, scoring them, and saving the results.

Parameters:

| Name                    | Type        | Description                                                                                 | Default    |
| ----------------------- | ----------- | ------------------------------------------------------------------------------------------- | ---------- |
| `model`                 | `Model`     | The model to be evaluated.                                                                  | *required* |
| `prompt_lang`           | `str`       | The language of the prompt (default is "fa").                                               | `'fa'`     |
| `prompt_shots`          | `list[int]` | The list of prompt shots to evaluate (default is None).                                     | `None`     |
| `n_first`               | `int`       | The number of initial prompts to consider (default is 200).                                 | `200`      |
| `sub_tasks`             | `list[str]` | The list of sub-tasks to evaluate (default is None).                                        | `None`     |
| `save_matches`          | `bool`      | Flag to save the generated matches (default is False).                                      | `False`    |
| `save_evaluation`       | `bool`      | Flag to save the evaluation results (default is False).                                     | `False`    |
| `output_path`           | `str`       | The output path to save the matches and evaluation results.                                 | `None`     |
| `skip_existing_matches` | `bool`      | Flag to skip already generated matches in the output path (default is False).               | `False`    |
| `prefer_concurrency`    | `bool`      | The flag to use concurrent processing if the model and task support that (default is True). | `True`     |
| `n_workers`             | `int`       | The number of workers for concurrent processing (default is 4).                             | `4`        |

Returns:

| Type                     | Description                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------- |
| `list[EvaluationResult]` | list\[EvaluationResult\]: A list of EvaluationResult objects representing the evaluation results. |

Raises:

| Type        | Description                                                        |
| ----------- | ------------------------------------------------------------------ |
| `Exception` | If output_path is not provided when saving matches or evaluation.  |
| `Exception` | If output_path is not provided when skipping existing matches.     |
| `Exception` | If sub tasks are not defined or if invalid sub tasks are provided. |

Source code in `parsbench/tasks/base/task.py`

```
def evaluate(
    self,
    model: Model,
    prompt_lang: str = "fa",
    prompt_shots: list[int] = None,
    n_first: int = 200,
    sub_tasks: list[str] | None = None,
    save_matches: bool = False,
    save_evaluation: bool = False,
    output_path: str = None,
    skip_existing_matches: bool = False,
    prefer_concurrency: bool = True,
    n_workers: int = 4,
) -> list[EvaluationResult]:
    """
    Method to evaluate the task by generating matches, scoring them, and saving the results.

    Parameters:
        model (Model): The model to be evaluated.
        prompt_lang (str, optional): The language of the prompt (default is "fa").
        prompt_shots (list[int], optional): The list of prompt shots to evaluate (default is None).
        n_first (int, optional): The number of initial prompts to consider (default is 200).
        sub_tasks (list[str], optional): The list of sub-tasks to evaluate (default is None).
        save_matches (bool, optional): Flag to save the generated matches (default is False).
        save_evaluation (bool, optional): Flag to save the evaluation results (default is False).
        output_path (str, optional): The output path to save the matches and evaluation results.
        skip_existing_matches (bool, optional): Flag to skip already generated matches in the output path (default is False).
        prefer_concurrency (bool, optional): The flag to use concurrent processing if the model and task support that (default is True).
        n_workers (int, optional): The number of workers for concurrent processing (default is 4).

    Returns:
        list[EvaluationResult]: A list of EvaluationResult objects representing the evaluation results.

    Raises:
        Exception: If output_path is not provided when saving matches or evaluation.
        Exception: If output_path is not provided when skipping existing matches.
        Exception: If sub tasks are not defined or if invalid sub tasks are provided.

    """
    if (save_matches or save_evaluation) and not output_path:
        raise Exception(
            "You should set the output path to save matches/evaluation."
        )

    if skip_existing_matches and not output_path:
        raise Exception(
            "Cannot find already generated matches when output_path is not set."
        )

    task_path = None
    if output_path:
        task_path = get_task_path(output_path, model.model_name, self.task_name)

    prompt_shots = [0] if prompt_shots is None else prompt_shots

    if sub_tasks:
        if not self.sub_tasks:
            raise Exception("Sub tasks are not defined.")

        invalid_sub_tasks = set(sub_tasks) - set(self.sub_tasks)
        if invalid_sub_tasks:
            raise Exception(f"Sub tasks {invalid_sub_tasks} are not defined.")

    sub_tasks = sub_tasks or self._selected_sub_tasks or self.sub_tasks

    evaluation_results: list[EvaluationResult] = []

    for sub_task in sub_tasks or [None]:
        match_groups: list[TaskMatchGroup] = []

        for shots in prompt_shots:
            if skip_existing_matches and check_task_matches_exists(
                task_path, shots, sub_task=sub_task
            ):
                match_group = TaskMatchGroup.from_file(
                    task_path, shots, sub_task=sub_task
                )
                match_group._loaded_locally = True
            else:
                match_group = self.generate_matches(
                    prompt_lang,
                    n_shots=shots,
                    n_first=n_first,
                    sub_task=sub_task,
                )
            match_groups.append(match_group)

        has_error = False
        for match_group in match_groups:
            eval_desc = f"{match_group.n_shots}-shot"
            if sub_task:
                eval_desc = f"sub task '{sub_task}' with " + eval_desc
            desc = f"Evaluating {eval_desc} prompt:"
            print(desc)

            is_loaded_locally = getattr(match_group, "_loaded_locally", False)

            if is_loaded_locally:
                total_skipped = sum(m.completion is not None for m in match_group)
                print(
                    f"{total_skipped} of {len(match_group)} match completions will be loaded from local."
                )

            try:
                model.generate_completions(
                    match_group,
                    prefer_concurrency=prefer_concurrency,
                    skip_existing=is_loaded_locally,
                    n_workers=n_workers,
                )
                self.score_matches(match_group)
            except Exception:
                has_error = True
            finally:
                if save_matches:
                    match_group.save(task_path, sub_task=sub_task)

        if not has_error:
            evaluation_result = EvaluationResult(
                model_name=model.model_name,
                task_name=self.task_name,
                task_category=self.task_category,
                score_name=self.score_name,
                sub_task=sub_task,
                prompt_shot_results=[
                    PromptShotEvaluationResult(
                        n_shots=m.n_shots,
                        score=self.get_overall_score(m),
                    )
                    for m in match_groups
                ],
            )
            evaluation_results.append(evaluation_result)

            if save_evaluation:
                evaluation_result.save(task_path)

    return evaluation_results
```

Bases: `ABC`

An abstract base class for defining data loaders.

Attributes:

| Name        | Type  | Description                  |
| ----------- | ----- | ---------------------------- |
| `data_path` | `str` | The path to the data source. |

Methods:

| Name   | Description                                                       |
| ------ | ----------------------------------------------------------------- |
| `load` | Abstract method to be implemented by subclasses for loading data. |

Source code in `parsbench/tasks/base/data_loader.py`

```
class DataLoader(ABC):
    """
    An abstract base class for defining data loaders.

    Attributes:
        data_path (str): The path to the data source.

    Methods:
        load(self) -> list[dict]: Abstract method to be implemented by subclasses for loading data.
    """

    def __init__(self, data_path: str, **kwargs) -> None:
        self.data_path = data_path

    @abstractmethod
    def load(self) -> list[dict]:
        pass
```

Bases: `DataLoader`

A data loader class for loading JSON line data from either a local file or a URL.

Attributes:

| Name        | Type  | Description                            |
| ----------- | ----- | -------------------------------------- |
| `data_path` | `str` | The path to the JSON line data source. |

Methods:

| Name   | Description                                         |
| ------ | --------------------------------------------------- |
| `load` | Loads the JSON line data from the specified source. |

Source code in `parsbench/tasks/base/data_loader.py`

```
class JSONLineDataLoader(DataLoader):
    """
    A data loader class for loading JSON line data from either a local file or a URL.

    Attributes:
        data_path (str): The path to the JSON line data source.

    Methods:
        load(self) -> list[dict]: Loads the JSON line data from the specified source.
    """

    def load(self) -> list[dict]:
        content = _fetch_text_file(self.data_path)

        reader = jsonlines.Reader(content.split("\n"))
        return list(reader.iter(type=dict, skip_invalid=True, skip_empty=True))
```

Bases: `DataLoader`

A data loader class for loading datasets using the Hugging Face library.

Attributes:

| Name        | Type  | Description                  |
| ----------- | ----- | ---------------------------- |
| `data_path` | `str` | The path to the data source. |
| `split`     | \`str | None\`                       |

Methods:

| Name          | Description                                                                                                |
| ------------- | ---------------------------------------------------------------------------------------------------------- |
| `load`        | Loads the dataset from the specified data path and split.                                                  |
| `with_filter` | Callable[..., bool]) -> "HuggingFaceDataLoader": Adds a filter function to apply when loading the dataset. |

Source code in `parsbench/tasks/base/data_loader.py`

```
class HuggingFaceDataLoader(DataLoader):
    """
    A data loader class for loading datasets using the Hugging Face library.

    Attributes:
        data_path (str): The path to the data source.
        split (str | None): The split of the dataset to load.

    Methods:
        load(self) -> list[dict]: Loads the dataset from the specified data path and split.
        with_filter(self, func: Callable[..., bool]) -> "HuggingFaceDataLoader": Adds a filter function to apply when loading the dataset.
    """

    def __init__(
        self,
        data_path: str,
        split: str | None = None,
        **optional_parameters: dict[str, Any],
    ) -> None:
        super().__init__(data_path)
        self.split = split
        self.optional_parameters = optional_parameters
        self._filters = []

    def load(self) -> list[dict]:
        dataset = datasets.load_dataset(
            self.data_path, split=self.split, **self.optional_parameters
        )
        if len(self._filters):
            for filter_ in self._filters:
                dataset = dataset.filter(filter_)
        return dataset.to_list()

    def with_filter(self, func: Callable[..., bool]) -> "HuggingFaceDataLoader":
        self._filters.append(func)
        return self
```

Bases: `DataLoader`

A data loader class for loading CSV line data from either a local file or a URL.

Attributes:

| Name        | Type  | Description                           |
| ----------- | ----- | ------------------------------------- |
| `data_path` | `str` | The path to the CSV line data source. |

Methods:

| Name   | Description                                        |
| ------ | -------------------------------------------------- |
| `load` | Loads the CSV line data from the specified source. |

Source code in `parsbench/tasks/base/data_loader.py`

```
class CSVDataLoader(DataLoader):
    """
    A data loader class for loading CSV line data from either a local file or a URL.

    Attributes:
        data_path (str): The path to the CSV line data source.

    Methods:
        load(self) -> list[dict]: Loads the CSV line data from the specified source.
    """

    def __init__(self, data_path: str, csv_arguments: dict | None = None, **kwargs):
        super().__init__(data_path)
        self.csv_arguments = csv_arguments or {}

    def load(self) -> list[dict]:
        content = _fetch_text_file(self.data_path)

        csv_reader = csv.DictReader(content.split("\n"), **self.csv_arguments)
        return list(csv_reader)
```

A class representing a prompt template.

Attributes:

| Name                       | Type                          | Description                                                            |
| -------------------------- | ----------------------------- | ---------------------------------------------------------------------- |
| `language_templates`       | `dict[str, str]`              | A dictionary mapping language codes to prompt templates.               |
| `prompt_variables_mapping` | `dict[str, str]`              | A dictionary mapping prompt variable names to corresponding data keys. |
| `target_variables_mapping` | `dict[str, str]`              | A dictionary mapping target variable names to corresponding data keys. |
| `prompt_shot_templates`    | \`dict[str, str]              | None\`                                                                 |
| `prompt_shot_examples`     | \`dict\[str, dict[int, str]\] | None\`                                                                 |

Source code in `parsbench/tasks/base/prompt_template.py`

```
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
```

Bases: `Mapping`

A class representing lazy loading of templates.

Inherits from Mapping.

Attributes:

| Name             | Type             | Description                                       |
| ---------------- | ---------------- | ------------------------------------------------- |
| `template_paths` | `dict[str, str]` | A dictionary mapping template keys to file paths. |

Source code in `parsbench/tasks/base/prompt_template.py`

```
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
```

A data class representing the evaluation result for a prompt shot, including the number of shots and the corresponding score.

Attributes:

| Name      | Type    | Description                                        |
| --------- | ------- | -------------------------------------------------- |
| `n_shots` | `int`   | The number of shots for the evaluation.            |
| `score`   | `float` | The score obtained for the prompt shot evaluation. |

Source code in `parsbench/tasks/base/evaluation_result.py`

```
@dataclass
class PromptShotEvaluationResult:
    """
    A data class representing the evaluation result for a prompt shot, including the number of shots and the corresponding score.

    Attributes:
        n_shots (int): The number of shots for the evaluation.
        score (float): The score obtained for the prompt shot evaluation.
    """

    n_shots: int
    score: float

    @classmethod
    def from_dict(cls, data: dict) -> "PromptShotEvaluationResult":
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame([self])

    def __str__(self) -> str:
        return f"{self.n_shots}-shot score: {self.score:.4f}"
```

A data class representing the evaluation result for a model on a specific task, including the model name, task name, task category, score name, prompt shot results, and optional sub-task.

Attributes:

| Name                  | Type                               | Description                                                                                        |
| --------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------- |
| `model_name`          | `str`                              | The name of the model being evaluated.                                                             |
| `task_name`           | `str`                              | The name of the task for which the model is being evaluated.                                       |
| `task_category`       | `TaskCategory`                     | The category of the task (e.g., CLASSIC, REASONING, MATH, KNOWLEDGE).                              |
| `score_name`          | `str`                              | The name of the score obtained for the evaluation.                                                 |
| `prompt_shot_results` | `list[PromptShotEvaluationResult]` | A list of PromptShotEvaluationResult objects representing the evaluation results for prompt shots. |
| `sub_task`            | `str`                              | The name of the sub-task being evaluated, if applicable.                                           |

Source code in `parsbench/tasks/base/evaluation_result.py`

```
@dataclass
class EvaluationResult:
    """
    A data class representing the evaluation result for a model on a specific task, including the model name, task name, task category, score name, prompt shot results, and optional sub-task.

    Attributes:
        model_name (str): The name of the model being evaluated.
        task_name (str): The name of the task for which the model is being evaluated.
        task_category (TaskCategory): The category of the task (e.g., CLASSIC, REASONING, MATH, KNOWLEDGE).
        score_name (str): The name of the score obtained for the evaluation.
        prompt_shot_results (list[PromptShotEvaluationResult]): A list of PromptShotEvaluationResult objects representing the evaluation results for prompt shots.
        sub_task (str, optional): The name of the sub-task being evaluated, if applicable.
    """

    model_name: str
    task_name: str
    task_category: TaskCategory
    score_name: str
    prompt_shot_results: list[PromptShotEvaluationResult]
    sub_task: str | None = None

    @classmethod
    def from_file(cls, path: str) -> "EvaluationResult":
        with jsonlines.open(path, "r") as reader:
            data = reader.read(type=dict)
            return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationResult":
        prompt_shot_results = [
            PromptShotEvaluationResult.from_dict(psr)
            for psr in data.pop("prompt_shot_results")
        ]
        data["task_category"] = TaskCategory[data["task_category"].upper()]
        return cls(**data, prompt_shot_results=prompt_shot_results)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "prompt_shot_results": [e.to_dict() for e in self.prompt_shot_results],
        }

    def to_pandas(self) -> pd.DataFrame:
        data = [
            {
                "model_name": self.model_name,
                "task_name": self.task_name,
                "task_category": self.task_category.value,
                "sub_task": self.sub_task,
                "n_shots": psr.n_shots,
                "score_name": self.score_name,
                "score": psr.score,
            }
            for psr in self.prompt_shot_results
        ]
        return pd.DataFrame(data)

    def save(self, path: str):
        file_name = (
            f"evaluation_{self.sub_task}.jsonl" if self.sub_task else "evaluation.jsonl"
        )
        task_path = path / file_name

        with jsonlines.open(task_path, "w") as writer:
            writer.write(self.to_dict())

    def __str__(self) -> str:
        text = f"Model: {self.model_name}\nTask: {self.task_name}"

        if self.sub_task:
            text += f" ({self.sub_task})"

        text += "\nScore:\n"

        for psr in self.prompt_shot_results:
            text += f" - {psr.n_shots}-shot prompt: {psr.score:.4f}\n"
        return text.strip("\n")

    @property
    def average_score(self) -> float:
        return sum([psr.score for psr in self.prompt_shot_results]) / len(
            self.prompt_shot_results
        )

    @property
    def max_score(self) -> float:
        return max([psr.score for psr in self.prompt_shot_results])
```

Source code in `parsbench/tasks/base/task_match.py`

```
@dataclass
class TaskMatch:
    id: int
    prompt: str
    target: str
    completion: str | None = None
    formatted_completion: str | None = None
    score: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "TaskMatch":
        return cls(**data)

    def format_completion(self, formatter: Callable[[str], str]):
        self.formatted_completion = formatter(self.completion)

    def format_prompt(self, formatter: Callable[[str], str]):
        self.prompt = formatter(self.prompt)

    def format_target(self, formatter: Callable[[str], str]):
        self.target = formatter(self.target)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame([self])

    @property
    def cleaned_completion(self) -> str | None:
        if self.formatted_completion is not None:
            return self.formatted_completion
        return self.completion
```

Source code in `parsbench/tasks/base/task_match.py`

```
@dataclass
class TaskMatchGroup:
    n_shots: int
    matches: list[TaskMatch]

    def __iter__(self):
        yield from iter(self.matches)

    def __len__(self) -> int:
        return len(self.matches)

    @classmethod
    def from_file(
        cls, path: str, n_shots: int, sub_task: str | None
    ) -> "TaskMatchGroup":
        if sub_task:
            matches_path = path / f"matches_{sub_task}_{n_shots}_shot.jsonl"
        else:
            matches_path = path / f"matches_{n_shots}_shot.jsonl"

        with jsonlines.open(matches_path, "r") as reader:
            matches: list[TaskMatch] = []
            for row in reader.iter(type=dict, skip_invalid=True):
                matches.append(TaskMatch.from_dict(row))

        return cls(n_shots=n_shots, matches=matches)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskMatchGroup":
        matches = [TaskMatch.from_dict(m) for m in data.pop("matches")]
        return cls(**data, matches=matches)

    def format_completions(self, formatter: Callable[[str], str]):
        for m in self.matches:
            m.format_completion(formatter)

    def format_prompts(self, formatter: Callable[[str], str]):
        for m in self.matches:
            m.format_prompt(formatter)

    def format_targets(self, formatter: Callable[[str], str]):
        for m in self.matches:
            m.format_target(formatter)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "matches": [match.to_dict() for match in self.matches],
        }

    def to_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame(
            [
                {
                    **asdict(match),
                    "n_shots": self.n_shots,
                }
                for match in self.matches
            ]
        )
        return df

    def save(self, path: str, sub_task: str | None):
        if sub_task:
            matches_path = path / f"matches_{sub_task}_{self.n_shots}_shot.jsonl"
        else:
            matches_path = path / f"matches_{self.n_shots}_shot.jsonl"

        with jsonlines.open(matches_path, "w") as writer:
            writer.write_all(self.to_dict()["matches"])

    @property
    def prompts(self) -> list[str]:
        return [m.prompt for m in self.matches]

    @property
    def targets(self) -> list[str]:
        return [m.target for m in self.matches]

    @property
    def completions(self) -> list[str | None]:
        return [m.cleaned_completion for m in self.matches]

    @property
    def scores(self) -> list[int | None]:
        return [m.score for m in self.matches]
```

Load all tasks from the 'parsbench.tasks' package and return a list of Task objects.

Returns:

| Type         | Description                                                                                         |
| ------------ | --------------------------------------------------------------------------------------------------- |
| `list[Task]` | list\[Task\]: A list of Task objects representing all tasks found in the 'parsbench.tasks' package. |

Source code in `parsbench/tasks/utils.py`

```
def load_all_tasks() -> list[Task]:
    """
    Load all tasks from the 'parsbench.tasks' package and return a list of Task objects.

    Returns:
        list[Task]: A list of Task objects representing all tasks found in the 'parsbench.tasks' package.

    """
    tasks: list[Task] = []

    module = importlib.import_module("parsbench.tasks")
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, Task) and attr is not Task:
            tasks.append(attr)

    return tasks
```
