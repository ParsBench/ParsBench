from abc import ABCMeta
from typing import Callable, Self

from tqdm import tqdm

from parsbench import scores
from parsbench.models import Model
from parsbench.scores.base import Scorer

from .data_loader import DataLoader
from .evaluation_result import EvaluationResult, PromptShotEvaluationResult
from .helpers import get_task_path
from .prompt_template import PromptTemplate
from .task_category import TaskCategory
from .task_match import TaskMatch, TaskMatchGroup


class TaskDataProvider:
    data_loader: DataLoader
    _data: list[dict] | None = None

    def load_data(self) -> list[dict]:
        if not self._check_data_is_loaded(raise_exc=False):
            self._data = self.data_loader.load()
        return self._data

    def get_data(
        self, n_first: int | None = None, filter_: Callable[..., bool] | None = None
    ) -> list[dict]:
        self._check_data_is_loaded()
        data = self._data
        if filter_:
            data = list(filter(filter_, data))
        if n_first is not None and n_first < len(data):
            data = data[:n_first]
        return data

    def _check_data_is_loaded(self, raise_exc: bool = True):
        if self._data is None:
            if raise_exc:
                raise RuntimeError(
                    "Data is not loaded. You should use this method inside a context manager or muanlly run load_data."
                )
            return False
        return True


class TaskMatchGenerator(TaskDataProvider):
    prompt_template: PromptTemplate
    data_target_key: str = "label"

    sub_task_key: str | None = None
    sub_tasks: list[str] | None = None

    def generate_matches(
        self,
        prompt_lang: str,
        n_shots: int = 0,
        n_first: int | None = None,
        sub_task: str | None = None,
    ) -> TaskMatchGroup:
        matches = []

        subtask_filter = None
        if sub_task:
            subtask_filter = lambda d: d.get(self.sub_task_key, "") == sub_task

        data = self.get_data(n_first=n_first, filter_=subtask_filter)

        if n_shots > len(data) - 1:
            raise ValueError("Data length should be more than n_shots.")

        sample_data = None
        if not self.prompt_template.has_shot_examples and n_shots > 0:
            sample_data = data[:n_shots]

        # Note: Should skip the first n_shots that are already used for prompts.
        for index, row in enumerate(data[n_shots:]):
            prompt = self.prompt_template.get_prompt(
                prompt_lang, row, n_shots, sample_data
            )
            target = row[self.data_target_key]
            match = TaskMatch(index + 1, prompt, target)
            matches.append(match)

        return TaskMatchGroup(n_shots=n_shots, matches=matches)


class TaskScorer:
    scorer: Scorer = scores.exact_match

    def score_completion(self, completion: str, target: str) -> float:
        return self.scorer.measure(completion, target)

    def score_matches(self, matches: TaskMatchGroup) -> list[TaskMatchGroup]:
        for match in tqdm(matches, desc="Scoring matches"):
            match.score = self.score_completion(match.completion, match.target)

        return matches

    def get_overall_score(self, matches: TaskMatchGroup) -> float:
        scores = matches.scores
        assert (
            None not in scores
        ), "Cannot calculate accuracy score when there are None score."

        return sum(scores) / len(scores)

    @property
    def score_name(self) -> str:
        return self.scorer.name


class Task(TaskMatchGenerator, TaskScorer, metaclass=ABCMeta):
    """
    Task class represents a task that combines functionality from TaskMatchGenerator and TaskScorer.

    Attributes:
        task_name (str): The name of the task.
        task_category (TaskCategory): The category of the task.

    Methods:
        evaluate: Method to evaluate the task by generating matches, scoring them, and saving the results.
    """

    task_name: str
    task_category: TaskCategory

    def evaluate(
        self,
        model: Model,
        prompt_lang: str = "fa",
        prompt_shots: list[int] = None,
        n_first: int = 200,
        sub_tasks: list[str] | None = None,
        save_matches: bool = False,
        save_evaluation: bool = False,
        output_path: str = None,
        n_workers: int = 4,
    ) -> list[EvaluationResult]:
        """
        Method to evaluate the task by generating matches, scoring them, and saving the results.

        Parameters:
            model (Model): The model to be evaluated.
            prompt_lang (str, optional): The language of the prompt (default is "fa").
            prompt_shots (list[int], optional): The list of prompt shots to evaluate (default is None).
            n_first (int, optional): The number of initial prompts to consider (default is 200).
            sub_tasks (list[str], optional): The list of sub-tasks to evaluate (default is None).
            save_matches (bool, optional): Flag to save the generated matches (default is False).
            save_evaluation (bool, optional): Flag to save the evaluation results (default is False).
            output_path (str, optional): The output path to save the matches and evaluation results.
            n_workers (int, optional): The number of workers for parallel processing (default is 4).

        Returns:
            list[EvaluationResult]: A list of EvaluationResult objects representing the evaluation results.

        Raises:
            Exception: If output_path is not provided when saving matches or evaluation.
            Exception: If sub tasks are not defined or if invalid sub tasks are provided.
        """
        if (save_matches or save_evaluation) and not output_path:
            raise Exception(
                "You should set the output path to save matches/evaluation."
            )

        task_path = None
        if output_path:
            task_path = get_task_path(output_path, model.model_name, self.task_name)

        prompt_shots = [0] if prompt_shots is None else prompt_shots

        if sub_tasks:
            if not self.sub_tasks:
                raise Exception("Sub tasks are not defined.")

            invalid_sub_tasks = set(self.sub_tasks) - set(sub_tasks)
            if invalid_sub_tasks:
                raise Exception(f"Sub tasks {invalid_sub_tasks} are not defined.")

        sub_tasks = sub_tasks or self.sub_tasks

        evaluation_results: list[EvaluationResult] = []

        for sub_task in sub_tasks or [None]:
            match_groups: list[TaskMatchGroup] = []

            for shots in prompt_shots:
                match_group = self.generate_matches(
                    prompt_lang,
                    n_shots=shots,
                    n_first=n_first,
                    sub_task=sub_task,
                )
                match_groups.append(match_group)

            for match_group in match_groups:
                eval_desc = f"{match_group.n_shots}-shot"
                if sub_task:
                    eval_desc = f"sub task '{sub_task}' with " + eval_desc
                desc = f"Evaluating {eval_desc} prompt:"
                print(desc)
                model.generate_completions(match_group, n_workers=n_workers)
                self.score_matches(match_group)

                if save_matches:
                    match_group.save(task_path, sub_task=sub_task)

            evaluation_result = EvaluationResult(
                model_name=model.model_name,
                task_name=self.task_name,
                task_category=self.task_category,
                score_name=self.score_name,
                sub_task=sub_task,
                prompt_shot_results=[
                    PromptShotEvaluationResult(
                        n_shots=m.n_shots,
                        score=self.get_overall_score(m),
                    )
                    for m in match_groups
                ],
            )
            evaluation_results.append(evaluation_result)

            if save_evaluation:
                evaluation_result.save(task_path)

        return evaluation_results

    def __enter__(self) -> Self:
        self.load_data()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._data = None
