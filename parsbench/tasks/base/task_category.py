from enum import Enum


class TaskCategory(str, Enum):
    CLASSIC = "classic"
    REASONING = "reasoning"
    MATH = "math"
    KNOWLADGE = "knowladge"
