from abc import ABC, abstractmethod

from .benchmark_result import BenchmarkResult


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
