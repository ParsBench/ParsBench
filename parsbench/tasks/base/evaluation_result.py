from dataclasses import asdict, dataclass
from typing import Self

import jsonlines
import pandas as pd

from .task_category import TaskCategory


@dataclass
class PromptShotEvaluationResult:
    """
    A data class representing the evaluation result for a prompt shot, including the number of shots and the corresponding score.

    Attributes:
        n_shots (int): The number of shots for the evaluation.
        score (float): The score obtained for the prompt shot evaluation.
    """

    n_shots: int
    score: float

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame([self])

    def __str__(self) -> str:
        return f"{self.n_shots}-shot score: {self.score:.4f}"


@dataclass
class EvaluationResult:
    """
    A data class representing the evaluation result for a model on a specific task, including the model name, task name, task category, score name, prompt shot results, and optional sub-task.

    Attributes:
        model_name (str): The name of the model being evaluated.
        task_name (str): The name of the task for which the model is being evaluated.
        task_category (TaskCategory): The category of the task (e.g., CLASSIC, REASONING, MATH, KNOWLEDGE).
        score_name (str): The name of the score obtained for the evaluation.
        prompt_shot_results (list[PromptShotEvaluationResult]): A list of PromptShotEvaluationResult objects representing the evaluation results for prompt shots.
        sub_task (str, optional): The name of the sub-task being evaluated, if applicable.
    """

    model_name: str
    task_name: str
    task_category: TaskCategory
    score_name: str
    prompt_shot_results: list[PromptShotEvaluationResult]
    sub_task: str | None = None

    @classmethod
    def from_file(cls, path: str) -> Self:
        with jsonlines.open(path, "r") as reader:
            data = reader.read(type=dict)
            return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        prompt_shot_results = [
            PromptShotEvaluationResult.from_dict(psr)
            for psr in data.pop("prompt_shot_results")
        ]
        return cls(**data, prompt_shot_results=prompt_shot_results)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "prompt_shot_results": [e.to_dict() for e in self.prompt_shot_results],
        }

    def to_pandas(self) -> pd.DataFrame:
        data = [
            {
                "model_name": self.model_name,
                "task_name": self.task_name,
                "task_category": self.task_category,
                "sub_task": self.sub_task,
                "n_shots": psr.n_shots,
                "score_name": self.score_name,
                "score": psr.score,
            }
            for psr in self.prompt_shot_results
        ]
        return pd.DataFrame(data)

    def save(self, path: str):
        file_name = (
            f"evaluation_{self.sub_task}.jsonl" if self.sub_task else "evaluation.jsonl"
        )
        task_path = path / file_name

        with jsonlines.open(task_path, "w") as writer:
            writer.write(self.to_dict())

    def __str__(self) -> str:
        text = f"Model: {self.model_name}\nTask: {self.task_name}\nScore:\n"
        for psr in self.prompt_shot_results:
            text += f" - {psr.n_shots}-shot prompt: {psr.score:.4f}\n"
        return text.strip("\n")

    @property
    def average_score(self) -> float:
        return sum([psr.score for psr in self.prompt_shot_results]) / len(
            self.prompt_shot_results
        )
