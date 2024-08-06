from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from parsbench.tasks.base import TaskMatch, TaskMatchGroup

DEFAULT_INSTRUCTION_PROMPT = "You are a helpful assistant."


class Model(ABC):
    """
    An abstract base class representing a model.

    Attributes:
        model_name (property): A property representing the name of the model.
        support_concurrency (bool): A flag indicating if the model supports concurrency.

    Methods:
        model_name(self) -> str: Abstract method to return the name of the model.
        get_prompt_completion (self, prompt: str) -> str: Abstract method to generate completion for a given prompt.
        prompt_formatter (self, prompt: str) -> Union[str, List[Dict]]: Abstract method to format a prompt.
        completion_formatter (self, completion: str) -> str: Method to format the model completion.
        generate_completions (self, matches: TaskMatchGroup, prefer_concurrency: bool = True, n_workers: int = 4) -> TaskMatchGroup: Method to generate completions for a list of matches, optionally using concurrency.

    Note:
        This class should be subclassed to implement the abstract methods.
    """

    support_concurrency: bool = False

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def get_prompt_completion(self, prompt: str) -> str:
        pass

    @abstractmethod
    def prompt_formatter(self, prompt: str) -> str | list[dict]:
        pass

    def completion_formatter(self, completion: str) -> str:
        return completion

    def generate_completions(
        self,
        matches: "TaskMatchGroup",
        prefer_concurrency: bool = True,
        skip_existing: bool = False,
        n_workers: int = 4,
    ) -> "TaskMatchGroup":
        if prefer_concurrency and self.support_concurrency:
            matches = self._gen_with_concurrency(
                matches, n_workers=n_workers, skip_existing=skip_existing
            )
        else:
            for match in tqdm(
                matches, total=len(matches), desc="Generating completions"
            ):
                if match.completion is not None and skip_existing:
                    continue
                match.completion = self.completion_formatter(
                    self.get_prompt_completion(match.prompt)
                )
        return matches

    def _gen_with_concurrency(
        self,
        matches: "TaskMatchGroup",
        n_workers: int = 4,
        skip_existing: bool = False,
    ) -> "TaskMatchGroup":
        def _gen_single_match_completion(match: "TaskMatch") -> "TaskMatch":
            match.completion = self.completion_formatter(
                self.get_prompt_completion(match.prompt)
            )
            return match

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []

            for match in matches:
                if match.completion is not None and skip_existing:
                    continue
                future = executor.submit(
                    _gen_single_match_completion,
                    match,
                )
                futures.append(future)

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Generating completions"
            ):
                future.result()

        matches.matches.sort(key=lambda m: m.id)
        return matches
