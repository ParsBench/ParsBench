from parsbench.models import Model
from parsbench.tasks import (
    ParsiNLUEntailment,
    ParsiNLUMachineTranslationEnFa,
    ParsiNLUMachineTranslationFaEn,
    ParsiNLUMultipleChoice,
    ParsiNLUReadingComprehension,
    ParsiNLUSentimentAnalysis,
)

from .custom_benchmark import CustomBenchmark


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
