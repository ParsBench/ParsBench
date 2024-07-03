from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from parsbench.tasks.base import TaskMatch, TaskMatchGroup

DEFUALT_INSTRUCTION_PROMPT = "You are a helpful assistant."


class Model(ABC):
    """
    An abstract base class representing a model.

    Attributes:
        model_name (property): A property representing the name of the model.

    Methods:
        get_prompt_completion: An abstract method to get the completion for a given prompt.
        prompt_formater: An abstract method to format a prompt.
        generate_completions: A method to generate completions for a list of TaskMatch objects using ThreadPoolExecutor.

    Raises:
        NotImplementedError: If the abstract methods are not implemented in the subclass.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def get_prompt_completion(self, prompt: str) -> str:
        pass

    @abstractmethod
    def prompt_formater(self, prompt: str) -> str | list[dict]:
        pass

    def generate_completions(
        self,
        matches: "TaskMatchGroup",
        n_workers: int = 4,
    ) -> "TaskMatchGroup":
        def _gen_single_match_completion(match: "TaskMatch") -> "TaskMatch":
            match.completion = self.get_prompt_completion(match.prompt)
            return match

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []

            for match in matches:
                future = executor.submit(
                    _gen_single_match_completion,
                    match,
                )
                futures.append(future)

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Generating completions"
            ):
                pass

        matches.matches.sort(key=lambda m: m.id)
        return matches
