from enum import Enum


class TaskCategory(str, Enum):
    CLASSIC = "classic"
    REASONING = "reasoning"
    MATH = "math"
    KNOWLEDGE = "knowledge"
    LANGUAGE = "language"
