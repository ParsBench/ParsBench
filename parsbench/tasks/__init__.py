from .base import Task
from .classification import ParsiNLUSentimentAnalysis
from .entailment import ConjNLIEntailment, FarsTailEntailment, ParsiNLUEntailment
from .machine_translation import (
    ParsiNLUMachineTranslationEnFa,
    ParsiNLUMachineTranslationFaEn,
)
from .math import PersianMath
from .multiple_choice import ParsiNLUMultipleChoice, PersianMMLU
from .ner import PersianNER
from .reading_comprehension import ParsiNLUReadingComprehension
from .summarization import PersianNewsSummary, XLSummary

__all__ = [
    "Task",
    "ConjNLIEntailment",
    "ParsiNLUEntailment",
    "FarsTailEntailment",
    "ParsiNLUMachineTranslationFaEn",
    "ParsiNLUMachineTranslationEnFa",
    "PersianMath",
    "ParsiNLUMultipleChoice",
    "PersianMMLU",
    "PersianNER",
    "ParsiNLUReadingComprehension",
    "ParsiNLUSentimentAnalysis",
    "PersianNewsSummary",
    "XLSummary",
]
