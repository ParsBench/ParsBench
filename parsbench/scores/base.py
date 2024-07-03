from typing import Callable


class Scorer:
    """
    Class representing a Scorer object.

    Attributes:
        func (Callable[[str, str], float]): The scoring function to be wrapped.

    Methods:
        measure(completion: str, target: str) -> float:
            Calculates the score between the completion and target strings using the wrapped scoring function.

        name() -> str:
            Returns the name of the wrapped scoring function with underscores replaced by spaces and title-cased.
    """

    def __init__(self, func: Callable[[str, str], float]):
        self.func = func

    def measure(self, completion: str, target: str) -> float:
        return self.func(completion, target)

    @property
    def name(self) -> str:
        return self.func.__name__.replace("_", " ").title()


def wrap_scorer(func):
    """Wraps a scorer function inside the Scorer class."""
    return Scorer(func)
