from .base import Task
from .classification import ParsiNLUSentimentAnalysis
from .entailment import ConjNLIEntailment, ParsiNLUEntailment
from .machine_translation import (
    ParsiNLUMachineTranslationEnFa,
    ParsiNLUMachineTranslationFaEn,
)
from .math import PersianMath
from .multiple_choice import ParsiNLUMultipleChoice
from .ner import PersianNER
from .reading_comprehension import ParsiNLUReadingComprehension

__all__ = [
    "Task",
    "ConjNLIEntailment",
    "ParsiNLUEntailment",
    "ParsiNLUMachineTranslationFaEn",
    "ParsiNLUMachineTranslationEnFa",
    "PersianMath",
    "ParsiNLUMultipleChoice",
    "PersianNER",
    "ParsiNLUReadingComprehension",
    "ParsiNLUSentimentAnalysis",
]
