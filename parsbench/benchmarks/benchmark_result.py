from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path
from typing import Self

import jsonlines
import numpy as np
import pandas as pd

from parsbench.tasks.base import EvaluationResult


@dataclass
class ModelBenchmarkResult:
    """
    Represents the results of benchmarking a model across multiple evaluations.

    Attributes:
        model_name (str): The name of the model being benchmarked.
        evaluation_results (list[EvaluationResult]): A list of EvaluationResult objects representing the evaluation results for the model.
    """

    model_name: str
    evaluation_results: list[EvaluationResult]

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        evaluation_results = [
            EvaluationResult.from_dict(task) for task in data.pop("evaluation_results")
        ]
        return cls(**data, evaluation_results=evaluation_results)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "evaluation_results": [e.to_dict() for e in self.evaluation_results],
        }

    def to_pandas(self) -> pd.DataFrame:
        return pd.concat([er.to_pandas() for er in self.evaluation_results])

    def __str__(self) -> str:
        text = f"Model: {self.model_name}\nEvaluation Results:\n"
        for er in self.evaluation_results:
            text += f"- {er.task_name}:\n"
            for psr in er.prompt_shot_results:
                text += f"  - {psr.n_shots}-shot prompt: {psr.score:.4f}\n"
        return text.strip("\n")

    @property
    def average_score(self) -> float:
        return sum([er.average_score for er in self.evaluation_results]) / len(
            self.evaluation_results
        )


@dataclass
class BenchmarkResult:
    """
    Represents the results of benchmarking multiple models across various evaluations.

    Attributes:
        model_benchmarks (list[ModelBenchmarkResult]): A list of ModelBenchmarkResult objects representing the benchmark results for each model.
    """

    model_benchmarks: list[ModelBenchmarkResult]

    @classmethod
    def from_file(cls, path: str) -> Self:
        with jsonlines.open(path, "r") as reader:
            model_benchmarks: list[ModelBenchmarkResult] = []
            for row in reader.iter(type=dict, skip_invalid=True):
                model_benchmarks.append(ModelBenchmarkResult.from_dict(row))

        return cls(model_benchmarks=model_benchmarks)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        model_benchmarks = [
            ModelBenchmarkResult.from_dict(mbr) for mbr in data.pop("model_benchmarks")
        ]
        return cls(**data, model_benchmarks=model_benchmarks)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "model_benchmarks": [mb.to_dict() for mb in self.model_benchmarks],
        }

    def to_pandas(self, pivot: bool = False) -> pd.DataFrame:
        df = pd.concat([mb.to_pandas() for mb in self.model_benchmarks])
        if pivot:
            return df.pivot(
                index=["task_category", "task_name", "sub_task", "score_name"],
                columns=["model_name", "n_shots"],
                values=["score"],
            )
        return df

    def show_radar_plot(self, title="Radar Plot"):
        data = []
        for mb in self.model_benchmarks:
            values = []
            for _, evals in groupby(mb.evaluation_results, key=lambda e: e.task_name):
                evals = list(evals)
                score = sum(e.average_score for e in evals) / len(evals)
                values.append(score)

            data.append({"name": mb.model_name, "values": values})

        categories = set(e.task_name for e in mb.evaluation_results)

        _radar_plot(data, categories, title)

    def save(self, path: str):
        benchmark_path = Path(path) / "benchmark.jsonl"

        with jsonlines.open(benchmark_path, "w") as writer:
            for mb in self.model_benchmarks:
                writer.write(mb.to_dict())

    def __str__(self) -> str:
        text = ""
        for mb in self.model_benchmarks:
            text += str(mb) + "\n" + "-" * 10 + "\n"
        return text.strip("\n")


def merge_benchmark_results(
    benchmarks: list[BenchmarkResult], sort: bool = True, keep_duplicates: bool = False
) -> BenchmarkResult:
    """
    Merge multiple BenchmarkResult objects into a single BenchmarkResult object.

    Parameters:
        benchmarks (list[BenchmarkResult]): A list of BenchmarkResult objects to merge.
        sort (bool, optional): Whether to sort the merged ModelBenchmarkResult list by average score. Defaults to True.
        keep_duplicates (bool, optional): Whether to keep duplicate model names in the merged list. Defaults to False.

    Returns:
        BenchmarkResult: A new BenchmarkResult object containing the merged ModelBenchmarkResult list.
    """
    model_benchmarks: list[ModelBenchmarkResult] = []
    for benchmark in benchmarks:
        model_benchmarks.extend(benchmark.model_benchmarks)

    if not keep_duplicates:
        model_names = set()
        skipped = 0
        for index in range(len(model_benchmarks)):
            mbr = model_benchmarks[index - skipped]
            if mbr.model_name in model_names:
                skipped += 1
                model_benchmarks.pop(index - skipped)
            model_names.add(mbr.model_name)

    if sort:
        model_benchmarks.sort(key=lambda m: m.average_score, reverse=True)

    return BenchmarkResult(model_benchmarks=model_benchmarks)


def _radar_plot(data, categories, title="Radar Plot"):
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        raise RuntimeError("The matplotlib is not installed.")

    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Complete the loop
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories)
    ax.set_rscale("linear")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_ylim(0, 1)

    for d in data:
        values = d["values"]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=d["name"])
        ax.fill(angles, values, alpha=0.25)

    plt.title(title)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.show()
