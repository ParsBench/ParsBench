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
    ) -> BenchmarkResult:
        """
        Run the benchmarking process for the given models and tasks.

        Parameters:
            prompt_lang (str): The language of the prompt (default is "fa").
            prompt_shots (list[int] | None): The list of prompt shots to evaluate, or None to use default.
            n_first (int | None): The number of prompts to evaluate, or None to use default.
            sort_by_score (bool): Whether to sort the model benchmarks by average score (default is True).
            save_matches (bool): Whether to save the matches during evaluation (default is False).
            save_evaluation (bool): Whether to save the evaluation results (default is False).
            save_benchmark (bool): Whether to save the benchmark results (default is False).
            output_path (str): The path to save the results.

        Returns:
            BenchmarkResult: The result of the benchmarking process.
        """
        model_evaluations: dict[str, list[EvaluationResult]] = defaultdict(list)

        for task_cls in self.tasks:
            print(f"Evaluating {task_cls.task_name}:")

            task: Task = task_cls()
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
