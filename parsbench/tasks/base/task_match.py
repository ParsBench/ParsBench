from dataclasses import asdict, dataclass
from typing import Callable, Self

import jsonlines
import pandas as pd


@dataclass
class TaskMatch:
    id: int
    prompt: str
    target: str
    completion: str | None = None
    score: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(**data)

    def format_completion(self, formatter: Callable[[str], str]):
        self.completion = formatter(self.completion)

    def format_prompt(self, formatter: Callable[[str], str]):
        self.prompt = formatter(self.prompt)

    def format_target(self, formatter: Callable[[str], str]):
        self.target = formatter(self.target)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame([self])


@dataclass
class TaskMatchGroup:
    n_shots: int
    matches: list[TaskMatch]

    def __iter__(self):
        yield from iter(self.matches)

    @classmethod
    def from_file(cls, path: str, n_shots: int) -> Self:
        with jsonlines.open(path, "r") as reader:
            matches: list[TaskMatch] = []
            for row in reader.iter(type=dict, skip_invalid=True):
                matches.append(TaskMatch.from_dict(row))

        return cls(n_shots=n_shots, matches=matches)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        matches = [TaskMatch.from_dict(m) for m in data.pop("matches")]
        return cls(**data, matches=matches)

    def format_completions(self, formatter: Callable[[str], str]):
        for m in self.matches:
            m.format_completion(formatter)

    def format_prompts(self, formatter: Callable[[str], str]):
        for m in self.matches:
            m.format_prompt(formatter)

    def format_targets(self, formatter: Callable[[str], str]):
        for m in self.matches:
            m.format_target(formatter)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "matches": [match.to_dict() for match in self.matches],
        }

    def to_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame(
            [
                {
                    **asdict(match),
                    "n_shots": self.n_shots,
                }
                for match in self.matches
            ]
        )
        return df

    def save(self, path: str, sub_task: str | None):
        if sub_task:
            matches_path = path / f"matches_{sub_task}_{self.n_shots}_shot.jsonl"
        else:
            matches_path = path / f"matches_{self.n_shots}_shot.jsonl"

        with jsonlines.open(matches_path, "w") as writer:
            writer.write_all(self.to_dict()["matches"])

    @property
    def prompts(self) -> list[str]:
        return [m.prompt for m in self.matches]

    @property
    def targets(self) -> list[str]:
        return [m.target for m in self.matches]

    @property
    def completions(self) -> list[str | None]:
        return [m.completion for m in self.matches]

    @property
    def scores(self) -> list[int | None]:
        return [m.score for m in self.matches]
