import inspect
from collections import defaultdict

from parsbench.models import Model
from parsbench.tasks.base import EvaluationResult, Task

from .base import Benchmark
from .benchmark_result import BenchmarkResult, ModelBenchmarkResult


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
