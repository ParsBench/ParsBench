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
    ) -> BenchmarkResult:
        """
        Abstract method that must be implemented by subclasses. It runs the benchmarking task with the given parameters and returns a BenchmarkResult object.

        Parameters:
            prompt_lang (str): The language prompt for the benchmarking task. Default is "fa".
            prompt_shots (list[int] | None): The list of prompt shots for the benchmarking task. Default is None.
            n_first (int | None): The number of first items to consider. Default is None.
            sort_by_score (bool): A flag to indicate whether to sort the results by score. Default is True.
            save_matches (bool): A flag to indicate whether to save the matches. Default is False.
            save_evaluation (bool): A flag to indicate whether to save the evaluation. Default is False.
            save_benchmark (bool): A flag to indicate whether to save the benchmark. Default is False.
            output_path (str): The output path to save the benchmark results. Default is None.

        Returns:
            BenchmarkResult: An object containing the benchmarking results.
        """
        pass
