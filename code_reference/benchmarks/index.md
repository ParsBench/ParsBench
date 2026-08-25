# Benchmarks

Bases: `ABC`

This abstract class defines the structure for a benchmarking task. Subclasses of Benchmark must implement the 'run' method, which takes in various parameters related to the benchmarking task and returns a BenchmarkResult object.

Methods:

| Name  | Description                                                                                                                                           |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run` | Abstract method that must be implemented by subclasses. It runs the benchmarking task with the given parameters and returns a BenchmarkResult object. |

Source code in `parsbench/benchmarks/base.py`

```
class Benchmark(ABC):
    """
    This abstract class defines the structure for a benchmarking task. Subclasses of Benchmark must implement the 'run' method, which takes in various parameters related to the benchmarking task and returns a BenchmarkResult object.

    Methods:
        run: Abstract method that must be implemented by subclasses. It runs the benchmarking task with the given parameters and returns a BenchmarkResult object.
    """

    @abstractmethod
    def run(
        self,
        prompt_lang: str = "fa",
        prompt_shots: list[int] | None = None,
        n_first: int | None = None,
        sort_by_score: bool = True,
        save_matches: bool = False,
        save_evaluation: bool = False,
        save_benchmark: bool = False,
        output_path: str = None,
        skip_existing_matches: bool = False,
        prefer_concurrency: bool = True,
        n_workers: int = 4,
    ) -> BenchmarkResult:
        """
        Abstract method that must be implemented by subclasses. It runs the benchmarking task with the given parameters and returns a BenchmarkResult object.

        Parameters:
            prompt_lang (str, optional): The language of the prompt (default is "fa").
            prompt_shots (list[int], optional): The list of prompt shots to evaluate (default is None).
            n_first (int, optional): The number of initial prompts to consider (default is 200).
            sort_by_score (bool, optional): Whether to sort the model benchmarks by average score (default is True).
            save_matches (bool, optional): Flag to save the generated matches (default is False).
            save_evaluation (bool, optional): Flag to save the evaluation results (default is False).
            skip_existing_matches (bool, optional): Flag to skip already generated matches in the output path (default is False).
            output_path (str, optional): The output path to save the matches and evaluation results.
            prefer_concurrency (bool, optional): The flag to use concurrent processing if the model and task support that (default is True).
            n_workers (int, optional): The number of workers for concurrent processing (default is 4).

        Returns:
            BenchmarkResult: An object containing the benchmarking results.
        """
        pass
```

## `run(prompt_lang='fa', prompt_shots=None, n_first=None, sort_by_score=True, save_matches=False, save_evaluation=False, save_benchmark=False, output_path=None, skip_existing_matches=False, prefer_concurrency=True, n_workers=4)`

Abstract method that must be implemented by subclasses. It runs the benchmarking task with the given parameters and returns a BenchmarkResult object.

Parameters:

| Name                    | Type        | Description                                                                                 | Default |
| ----------------------- | ----------- | ------------------------------------------------------------------------------------------- | ------- |
| `prompt_lang`           | `str`       | The language of the prompt (default is "fa").                                               | `'fa'`  |
| `prompt_shots`          | `list[int]` | The list of prompt shots to evaluate (default is None).                                     | `None`  |
| `n_first`               | `int`       | The number of initial prompts to consider (default is 200).                                 | `None`  |
| `sort_by_score`         | `bool`      | Whether to sort the model benchmarks by average score (default is True).                    | `True`  |
| `save_matches`          | `bool`      | Flag to save the generated matches (default is False).                                      | `False` |
| `save_evaluation`       | `bool`      | Flag to save the evaluation results (default is False).                                     | `False` |
| `skip_existing_matches` | `bool`      | Flag to skip already generated matches in the output path (default is False).               | `False` |
| `output_path`           | `str`       | The output path to save the matches and evaluation results.                                 | `None`  |
| `prefer_concurrency`    | `bool`      | The flag to use concurrent processing if the model and task support that (default is True). | `True`  |
| `n_workers`             | `int`       | The number of workers for concurrent processing (default is 4).                             | `4`     |

Returns:

| Name              | Type              | Description                                    |
| ----------------- | ----------------- | ---------------------------------------------- |
| `BenchmarkResult` | `BenchmarkResult` | An object containing the benchmarking results. |

Source code in `parsbench/benchmarks/base.py`

```
@abstractmethod
def run(
    self,
    prompt_lang: str = "fa",
    prompt_shots: list[int] | None = None,
    n_first: int | None = None,
    sort_by_score: bool = True,
    save_matches: bool = False,
    save_evaluation: bool = False,
    save_benchmark: bool = False,
    output_path: str = None,
    skip_existing_matches: bool = False,
    prefer_concurrency: bool = True,
    n_workers: int = 4,
) -> BenchmarkResult:
    """
    Abstract method that must be implemented by subclasses. It runs the benchmarking task with the given parameters and returns a BenchmarkResult object.

    Parameters:
        prompt_lang (str, optional): The language of the prompt (default is "fa").
        prompt_shots (list[int], optional): The list of prompt shots to evaluate (default is None).
        n_first (int, optional): The number of initial prompts to consider (default is 200).
        sort_by_score (bool, optional): Whether to sort the model benchmarks by average score (default is True).
        save_matches (bool, optional): Flag to save the generated matches (default is False).
        save_evaluation (bool, optional): Flag to save the evaluation results (default is False).
        skip_existing_matches (bool, optional): Flag to skip already generated matches in the output path (default is False).
        output_path (str, optional): The output path to save the matches and evaluation results.
        prefer_concurrency (bool, optional): The flag to use concurrent processing if the model and task support that (default is True).
        n_workers (int, optional): The number of workers for concurrent processing (default is 4).

    Returns:
        BenchmarkResult: An object containing the benchmarking results.
    """
    pass
```

Bases: `Benchmark`

CustomBenchmark class represents a custom benchmarking task that extends the Benchmark abstract class. It defines the run method to execute the benchmarking process for a given list of models and tasks.

Attributes:

| Name     | Type          | Description                                              |
| -------- | ------------- | -------------------------------------------------------- |
| `models` | `list[Model]` | The list of models to evaluate in the benchmarking task. |
| `tasks`  | `list[Task]`  | The list of tasks to evaluate with the models.           |

Methods:

| Name  | Description                                                                                                                      |
| ----- | -------------------------------------------------------------------------------------------------------------------------------- |
| `run` | Executes the benchmarking process for the specified models and tasks, generating evaluation results for each model on each task. |

Source code in `parsbench/benchmarks/custom_benchmark.py`

```
class CustomBenchmark(Benchmark):
    """
    CustomBenchmark class represents a custom benchmarking task that extends the Benchmark abstract class. It defines the run method to execute the benchmarking process for a given list of models and tasks.

    Attributes:
        models (list[Model]): The list of models to evaluate in the benchmarking task.
        tasks (list[Task]): The list of tasks to evaluate with the models.

    Methods:
        run: Executes the benchmarking process for the specified models and tasks, generating evaluation results for each model on each task.
    """

    def __init__(
        self,
        models: list[Model],
        tasks: list[Task],
    ):
        self.models = models
        self.tasks = tasks

    def run(
        self,
        prompt_lang: str = "fa",
        prompt_shots: list[int] | None = None,
        n_first: int | None = None,
        sort_by_score: bool = True,
        save_matches: bool = False,
        save_evaluation: bool = False,
        save_benchmark: bool = False,
        output_path: str = None,
        skip_existing_matches: bool = False,
        prefer_concurrency: bool = True,
        n_workers: int = 4,
    ) -> BenchmarkResult:
        """
        Run the benchmarking process for the given models and tasks.

        Parameters:
            prompt_lang (str, optional): The language of the prompt (default is "fa").
            prompt_shots (list[int], optional): The list of prompt shots to evaluate (default is None).
            n_first (int, optional): The number of initial prompts to consider (default is 200).
            sort_by_score (bool, optional): Whether to sort the model benchmarks by average score (default is True).
            save_matches (bool, optional): Flag to save the generated matches (default is False).
            save_evaluation (bool, optional): Flag to save the evaluation results (default is False).
            output_path (str, optional): The output path to save the matches and evaluation results.
            skip_existing_matches (bool, optional): Flag to skip already generated matches in the output path (default is False).
            prefer_concurrency (bool, optional): The flag to use concurrent processing if the model and task support that (default is True).
            n_workers (int, optional): The number of workers for concurrent processing (default is 4).

        Returns:
            BenchmarkResult: The result of the benchmarking process.
        """

        model_evaluations: dict[str, list[EvaluationResult]] = defaultdict(list)

        for task in self.tasks:
            print(f"Evaluating {task.task_name}:")

            if inspect.isclass(task):
                if issubclass(task, Task):
                    task: Task = task()
                else:
                    raise TypeError(
                        f"{task} is not a subclass/instance of the Task class."
                    )

            with task:
                for model in self.models:
                    print(f"Model: {model.model_name}")

                    evaluation_results = task.evaluate(
                        model=model,
                        prompt_lang=prompt_lang,
                        prompt_shots=prompt_shots,
                        n_first=n_first,
                        save_matches=save_matches,
                        save_evaluation=save_evaluation,
                        output_path=output_path,
                        skip_existing_matches=skip_existing_matches,
                        prefer_concurrency=prefer_concurrency,
                        n_workers=n_workers,
                    )
                    model_evaluations[model.model_name].extend(evaluation_results)

        model_benchmarks = [
            ModelBenchmarkResult(
                model_name=model_name,
                evaluation_results=evaluation_results,
            )
            for model_name, evaluation_results in model_evaluations.items()
        ]

        if sort_by_score:
            model_benchmarks.sort(key=lambda mb: mb.average_score, reverse=True)

        benchmark_result = BenchmarkResult(model_benchmarks=model_benchmarks)

        if save_benchmark:
            benchmark_result.save(output_path)

        return benchmark_result
```

## `run(prompt_lang='fa', prompt_shots=None, n_first=None, sort_by_score=True, save_matches=False, save_evaluation=False, save_benchmark=False, output_path=None, skip_existing_matches=False, prefer_concurrency=True, n_workers=4)`

Run the benchmarking process for the given models and tasks.

Parameters:

| Name                    | Type        | Description                                                                                 | Default |
| ----------------------- | ----------- | ------------------------------------------------------------------------------------------- | ------- |
| `prompt_lang`           | `str`       | The language of the prompt (default is "fa").                                               | `'fa'`  |
| `prompt_shots`          | `list[int]` | The list of prompt shots to evaluate (default is None).                                     | `None`  |
| `n_first`               | `int`       | The number of initial prompts to consider (default is 200).                                 | `None`  |
| `sort_by_score`         | `bool`      | Whether to sort the model benchmarks by average score (default is True).                    | `True`  |
| `save_matches`          | `bool`      | Flag to save the generated matches (default is False).                                      | `False` |
| `save_evaluation`       | `bool`      | Flag to save the evaluation results (default is False).                                     | `False` |
| `output_path`           | `str`       | The output path to save the matches and evaluation results.                                 | `None`  |
| `skip_existing_matches` | `bool`      | Flag to skip already generated matches in the output path (default is False).               | `False` |
| `prefer_concurrency`    | `bool`      | The flag to use concurrent processing if the model and task support that (default is True). | `True`  |
| `n_workers`             | `int`       | The number of workers for concurrent processing (default is 4).                             | `4`     |

Returns:

| Name              | Type              | Description                             |
| ----------------- | ----------------- | --------------------------------------- |
| `BenchmarkResult` | `BenchmarkResult` | The result of the benchmarking process. |

Source code in `parsbench/benchmarks/custom_benchmark.py`

```
def run(
    self,
    prompt_lang: str = "fa",
    prompt_shots: list[int] | None = None,
    n_first: int | None = None,
    sort_by_score: bool = True,
    save_matches: bool = False,
    save_evaluation: bool = False,
    save_benchmark: bool = False,
    output_path: str = None,
    skip_existing_matches: bool = False,
    prefer_concurrency: bool = True,
    n_workers: int = 4,
) -> BenchmarkResult:
    """
    Run the benchmarking process for the given models and tasks.

    Parameters:
        prompt_lang (str, optional): The language of the prompt (default is "fa").
        prompt_shots (list[int], optional): The list of prompt shots to evaluate (default is None).
        n_first (int, optional): The number of initial prompts to consider (default is 200).
        sort_by_score (bool, optional): Whether to sort the model benchmarks by average score (default is True).
        save_matches (bool, optional): Flag to save the generated matches (default is False).
        save_evaluation (bool, optional): Flag to save the evaluation results (default is False).
        output_path (str, optional): The output path to save the matches and evaluation results.
        skip_existing_matches (bool, optional): Flag to skip already generated matches in the output path (default is False).
        prefer_concurrency (bool, optional): The flag to use concurrent processing if the model and task support that (default is True).
        n_workers (int, optional): The number of workers for concurrent processing (default is 4).

    Returns:
        BenchmarkResult: The result of the benchmarking process.
    """

    model_evaluations: dict[str, list[EvaluationResult]] = defaultdict(list)

    for task in self.tasks:
        print(f"Evaluating {task.task_name}:")

        if inspect.isclass(task):
            if issubclass(task, Task):
                task: Task = task()
            else:
                raise TypeError(
                    f"{task} is not a subclass/instance of the Task class."
                )

        with task:
            for model in self.models:
                print(f"Model: {model.model_name}")

                evaluation_results = task.evaluate(
                    model=model,
                    prompt_lang=prompt_lang,
                    prompt_shots=prompt_shots,
                    n_first=n_first,
                    save_matches=save_matches,
                    save_evaluation=save_evaluation,
                    output_path=output_path,
                    skip_existing_matches=skip_existing_matches,
                    prefer_concurrency=prefer_concurrency,
                    n_workers=n_workers,
                )
                model_evaluations[model.model_name].extend(evaluation_results)

    model_benchmarks = [
        ModelBenchmarkResult(
            model_name=model_name,
            evaluation_results=evaluation_results,
        )
        for model_name, evaluation_results in model_evaluations.items()
    ]

    if sort_by_score:
        model_benchmarks.sort(key=lambda mb: mb.average_score, reverse=True)

    benchmark_result = BenchmarkResult(model_benchmarks=model_benchmarks)

    if save_benchmark:
        benchmark_result.save(output_path)

    return benchmark_result
```

Bases: `CustomBenchmark`

This benchmark class includes all existing tasks which use ParsiNLU datasets.

Attributes:

| Name     | Type          | Description                                              |
| -------- | ------------- | -------------------------------------------------------- |
| `models` | `list[Model]` | The list of models to evaluate in the benchmarking task. |

Methods:

| Name  | Description                                                                                                            |
| ----- | ---------------------------------------------------------------------------------------------------------------------- |
| `run` | Executes the benchmarking process for the specified models, generating evaluation results for each model on each task. |

Source code in `parsbench/benchmarks/parsinlu_benchmark.py`

```
class ParsiNLUBenchmark(CustomBenchmark):
    """
    This benchmark class includes all existing tasks which use ParsiNLU datasets.

    Attributes:
        models (list[Model]): The list of models to evaluate in the benchmarking task.

    Methods:
        run: Executes the benchmarking process for the specified models, generating evaluation results for each model on each task.
    """

    def __init__(self, models: list[Model]):
        tasks = [
            ParsiNLUEntailment,
            ParsiNLUMachineTranslationEnFa,
            ParsiNLUMachineTranslationFaEn,
            ParsiNLUMultipleChoice,
            ParsiNLUReadingComprehension,
            ParsiNLUSentimentAnalysis,
        ]
        super().__init__(models, tasks)
```

Represents the results of benchmarking a model across multiple evaluations.

Attributes:

| Name                 | Type                     | Description                                                                           |
| -------------------- | ------------------------ | ------------------------------------------------------------------------------------- |
| `model_name`         | `str`                    | The name of the model being benchmarked.                                              |
| `evaluation_results` | `list[EvaluationResult]` | A list of EvaluationResult objects representing the evaluation results for the model. |

Source code in `parsbench/benchmarks/benchmark_result.py`

```
@dataclass
class ModelBenchmarkResult:
    """
    Represents the results of benchmarking a model across multiple evaluations.

    Attributes:
        model_name (str): The name of the model being benchmarked.
        evaluation_results (list[EvaluationResult]): A list of EvaluationResult objects representing the evaluation results for the model.
    """

    model_name: str
    evaluation_results: list[EvaluationResult]

    @classmethod
    def from_dict(cls, data: dict) -> "ModelBenchmarkResult":
        evaluation_results = [
            EvaluationResult.from_dict(task) for task in data.pop("evaluation_results")
        ]
        return cls(**data, evaluation_results=evaluation_results)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "evaluation_results": [e.to_dict() for e in self.evaluation_results],
        }

    def to_pandas(self) -> pd.DataFrame:
        return pd.concat([er.to_pandas() for er in self.evaluation_results])

    def __str__(self) -> str:
        text = f"Model: {self.model_name}\nEvaluation Results:\n"
        for er in self.evaluation_results:
            text += f"- {er.task_name}"
            if er.sub_task:
                text += f" ({er.sub_task}):\n"
            else:
                text += ":\n"
            for psr in er.prompt_shot_results:
                text += f"  - {psr.n_shots}-shot prompt: {psr.score:.4f}\n"
        return text.strip("\n")

    @property
    def average_score(self) -> float:
        return sum([er.average_score for er in self.evaluation_results]) / len(
            self.evaluation_results
        )
```

Represents the results of benchmarking multiple models across various evaluations.

Attributes:

| Name               | Type                         | Description                                                                               |
| ------------------ | ---------------------------- | ----------------------------------------------------------------------------------------- |
| `model_benchmarks` | `list[ModelBenchmarkResult]` | A list of ModelBenchmarkResult objects representing the benchmark results for each model. |

Source code in `parsbench/benchmarks/benchmark_result.py`

```
@dataclass
class BenchmarkResult:
    """
    Represents the results of benchmarking multiple models across various evaluations.

    Attributes:
        model_benchmarks (list[ModelBenchmarkResult]): A list of ModelBenchmarkResult objects representing the benchmark results for each model.
    """

    model_benchmarks: list[ModelBenchmarkResult]

    @classmethod
    def from_file(cls, path: str) -> "BenchmarkResult":
        with jsonlines.open(path, "r") as reader:
            model_benchmarks: list[ModelBenchmarkResult] = []
            for row in reader.iter(type=dict, skip_invalid=True):
                model_benchmarks.append(ModelBenchmarkResult.from_dict(row))

        return cls(model_benchmarks=model_benchmarks)

    @classmethod
    def from_evaluation_files(cls, path: str) -> "BenchmarkResult":
        models = [(d.name, d.path) for d in os.scandir(path) if d.is_dir()]
        model_benchmarks = []

        for model_name, model_path in models:
            eval_paths = [d.path for d in os.scandir(model_path) if d.is_dir()]

            evaluation_results = []

            for eval_path in eval_paths:
                eval_files = [
                    f
                    for f in os.scandir(eval_path)
                    if f.is_file() and f.name.startswith("evaluation")
                ]
                evaluation_results.extend(
                    [
                        EvaluationResult.from_file(eval_file.path)
                        for eval_file in eval_files
                    ]
                )

            model_benchmarks.append(
                ModelBenchmarkResult(
                    model_name=model_name,
                    evaluation_results=evaluation_results,
                )
            )

        return BenchmarkResult(model_benchmarks=model_benchmarks)

    @classmethod
    def from_matches_files(cls, path: str, rescore: bool = False) -> "BenchmarkResult":
        task_cls_mapping = {
            task_cls.task_name.replace("-", " "): task_cls
            for task_cls in load_all_tasks()
        }

        _with_subtask_pattern = re.compile(r"matches_([\w\s]+)_(\d+)_shot\.jsonl")
        _without_subtask_pattern = re.compile(r"matches_(\d+)_shot\.jsonl")

        matches_paths = glob.glob(f"{path}/*/*/matches*.jsonl")

        model_evals: list[tuple[str, str, str, TaskMatchGroup]] = []

        for match_path in matches_paths:
            match_file = os.path.basename(match_path)
            task_name = os.path.basename(os.path.dirname(match_path)).replace("_", " ")
            model_name = os.path.basename(os.path.dirname(os.path.dirname(match_path)))
            sub_task = None
            n_shots = 0

            if m := _with_subtask_pattern.match(match_file):
                sub_task = m.group(1)
                n_shots = int(m.group(2))
            elif m := _without_subtask_pattern.match(match_file):
                n_shots = int(m.group(1))
            else:
                raise Exception(
                    f"Matches file '{match_file}' doesn't match the expected pattern."
                )

            task_matches = TaskMatchGroup.from_file(
                Path(match_path).parent, n_shots=n_shots, sub_task=sub_task
            )
            assert (
                task_name in task_cls_mapping
            ), f"No task class found for '{task_name}'."

            model_evals.append((model_name, task_name, sub_task, task_matches))

        # groupby only groups consecutive keys, so sort by (model, task) first.
        model_evals.sort(key=lambda t: (t[0], t[1]))

        model_benchmarks: list[ModelBenchmarkResult] = []

        for model_name, task_evals in itertools.groupby(
            model_evals, key=lambda t: t[0]
        ):
            print(f"Model: {model_name}")
            evaluation_results: list[EvaluationResult] = []

            for task_name, task_matches_group in itertools.groupby(
                task_evals, key=lambda t: t[1]
            ):
                print(f"Re-scoring {task_name}:")
                task = task_cls_mapping[task_name]()
                prompt_shot_evals = defaultdict(list)

                for _, _, sub_task, task_matches in task_matches_group:
                    print(f"{sub_task} {task_matches.n_shots}-shot prompt:")
                    if rescore:
                        task_matches = task.score_matches(task_matches)

                    score = task.get_overall_score(task_matches)

                    prompt_shot_evals[sub_task].append(
                        PromptShotEvaluationResult(
                            n_shots=task_matches.n_shots, score=score
                        )
                    )

                evaluation_results.extend(
                    EvaluationResult(
                        model_name=model_name,
                        task_name=task_name,
                        task_category=task.task_category,
                        score_name=task.score_name,
                        prompt_shot_results=prompt_shot_results,
                        sub_task=sub_task,
                    )
                    for sub_task, prompt_shot_results in prompt_shot_evals.items()
                )

            model_benchmarks.append(
                ModelBenchmarkResult(
                    model_name=model_name, evaluation_results=evaluation_results
                )
            )
            print("-" * 10)

        return BenchmarkResult(model_benchmarks=model_benchmarks)

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkResult":
        model_benchmarks = [
            ModelBenchmarkResult.from_dict(mbr) for mbr in data.pop("model_benchmarks")
        ]
        return cls(**data, model_benchmarks=model_benchmarks)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "model_benchmarks": [mb.to_dict() for mb in self.model_benchmarks],
        }

    def to_pandas(self, pivot: bool = False) -> pd.DataFrame:
        df = pd.concat([mb.to_pandas() for mb in self.model_benchmarks])
        if pivot:
            return df.pivot(
                index=["task_category", "task_name", "sub_task", "score_name"],
                columns=["model_name", "n_shots"],
                values=["score"],
            )
        return df

    def show_radar_plot(self, title="Radar Plot"):
        data = []
        categories = set()

        for mb in self.model_benchmarks:
            values = []
            for _, evals in groupby(mb.evaluation_results, key=lambda e: e.task_name):
                evals = list(evals)
                score = sum(e.average_score for e in evals) / len(evals)
                values.append(score)

            data.append({"name": mb.model_name, "values": values})
            categories |= set(e.task_name for e in mb.evaluation_results)

        _radar_plot(data, categories, title)

    def show_bar_plot(self, title="Bar Plot"):
        data = []
        categories = set()

        for mb in self.model_benchmarks:
            values = []
            for _, evals in groupby(mb.evaluation_results, key=lambda e: e.task_name):
                evals = list(evals)
                score = sum(e.average_score for e in evals) / len(evals)
                values.append(score)

            data.append({"name": mb.model_name, "values": values})
            categories |= set(e.task_name for e in mb.evaluation_results)

        _bar_plot(data, categories, title)

    def save(self, path: str):
        benchmark_path = Path(path) / "benchmark.jsonl"

        with jsonlines.open(benchmark_path, "w") as writer:
            for mb in self.model_benchmarks:
                writer.write(mb.to_dict())

    def __str__(self) -> str:
        text = ""
        for mb in self.model_benchmarks:
            text += str(mb) + "\n" + "-" * 10 + "\n"
        return text.strip("\n")
```

Merge multiple BenchmarkResult objects into a single BenchmarkResult object.

Parameters:

| Name              | Type                    | Description                                                                              | Default    |
| ----------------- | ----------------------- | ---------------------------------------------------------------------------------------- | ---------- |
| `benchmarks`      | `list[BenchmarkResult]` | A list of BenchmarkResult objects to merge.                                              | *required* |
| `sort`            | `bool`                  | Whether to sort the merged ModelBenchmarkResult list by average score. Defaults to True. | `True`     |
| `keep_duplicates` | `bool`                  | Whether to keep duplicate model names in the merged list. Defaults to False.             | `False`    |

Returns:

| Name              | Type              | Description                                                                   |
| ----------------- | ----------------- | ----------------------------------------------------------------------------- |
| `BenchmarkResult` | `BenchmarkResult` | A new BenchmarkResult object containing the merged ModelBenchmarkResult list. |

Source code in `parsbench/benchmarks/benchmark_result.py`

```
def merge_benchmark_results(
    benchmarks: list[BenchmarkResult], sort: bool = True, keep_duplicates: bool = False
) -> BenchmarkResult:
    """
    Merge multiple BenchmarkResult objects into a single BenchmarkResult object.

    Parameters:
        benchmarks (list[BenchmarkResult]): A list of BenchmarkResult objects to merge.
        sort (bool, optional): Whether to sort the merged ModelBenchmarkResult list by average score. Defaults to True.
        keep_duplicates (bool, optional): Whether to keep duplicate model names in the merged list. Defaults to False.

    Returns:
        BenchmarkResult: A new BenchmarkResult object containing the merged ModelBenchmarkResult list.
    """
    model_benchmarks: list[ModelBenchmarkResult] = []
    for benchmark in benchmarks:
        model_benchmarks.extend(benchmark.model_benchmarks)

    if not keep_duplicates:
        model_names = set()
        deduped: list[ModelBenchmarkResult] = []
        for mbr in model_benchmarks:
            if mbr.model_name not in model_names:
                model_names.add(mbr.model_name)
                deduped.append(mbr)
        model_benchmarks = deduped

    if sort:
        model_benchmarks.sort(key=lambda m: m.average_score, reverse=True)

    return BenchmarkResult(model_benchmarks=model_benchmarks)
```

This function generates leaderboard data from the benchmark result object.

Parameters:

| Name               | Type              | Description                         | Default    |
| ------------------ | ----------------- | ----------------------------------- | ---------- |
| `benchmark_result` | `BenchmarkResult` | BenchmarkResult object.             | *required* |
| `leaderboard_path` | `str`             | Path to store the leaderboard data. | *required* |

Source code in `parsbench/benchmarks/benchmark_result.py`

```
def build_leaderboard_from_benchmark(
    benchmark_result: BenchmarkResult, leaderboard_path: str
):
    """
    This function generates leaderboard data from the benchmark result object.

    Parameters:
        benchmark_result (BenchmarkResult): BenchmarkResult object.
        leaderboard_path (str): Path to store the leaderboard data.
    """
    requests_path = Path(leaderboard_path) / "requests"
    results_path = Path(leaderboard_path) / "results"

    requests_path.mkdir(exist_ok=True)
    results_path.mkdir(exist_ok=True)

    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    for mb in benchmark_result.model_benchmarks:
        model_name = mb.model_name

        os.makedirs(results_path / model_name, exist_ok=True)

        request = {
            "model": model_name,
            "base_model": "",
            "revision": "main",
            "private": False,
            "precision": "?",
            "weight_type": "Original",
            "status": "FINISHED",
            "submitted_time": now,
            "model_type": "\ud83d\udfe2 : pretrained",
            "likes": 0,
            "params": 0.1,
            "license": "custom",
        }
        with open(
            requests_path / f"{model_name}_eval_request_nshot.json", "wt"
        ) as writer:
            writer.write(json.dumps(request))

        result = {
            "config": {"model_dtype": "", "model_name": model_name, "model_sha": ""},
            "results": {
                er.task_name: {er.score_name: round(er.max_score, 3)}
                for er in mb.evaluation_results
            },
        }

        with open(results_path / model_name / f"results_{now}.json", "wt") as writer:
            writer.write(json.dumps(result))
```
