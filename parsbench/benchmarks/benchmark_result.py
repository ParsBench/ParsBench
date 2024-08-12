import datetime
import glob
import itertools
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
import pytz

from parsbench.tasks.base import EvaluationResult, TaskMatchGroup
from parsbench.tasks.base.evaluation_result import PromptShotEvaluationResult
from parsbench.tasks.utils import load_all_tasks


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
    def from_dict(cls, data: dict) -> "ModelBenchmarkResult":
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
            text += f"- {er.task_name}"
            if er.sub_task:
                text += f" ({er.sub_task}):\n"
            else:
                text += ":\n"
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
    def from_file(cls, path: str) -> "BenchmarkResult":
        with jsonlines.open(path, "r") as reader:
            model_benchmarks: list[ModelBenchmarkResult] = []
            for row in reader.iter(type=dict, skip_invalid=True):
                model_benchmarks.append(ModelBenchmarkResult.from_dict(row))

        return cls(model_benchmarks=model_benchmarks)

    @classmethod
    def from_evaluation_files(cls, path: str) -> "BenchmarkResult":
        models = [(d.name, d.path) for d in os.scandir(path) if d.is_dir()]
        model_benchmarks = []

        for model_name, model_path in models:
            eval_paths = [d.path for d in os.scandir(model_path) if d.is_dir()]

            evaluation_results = []

            for eval_path in eval_paths:
                eval_files = [
                    f
                    for f in os.scandir(eval_path)
                    if f.is_file() and f.name.startswith("evaluation")
                ]
                evaluation_results.extend(
                    [
                        EvaluationResult.from_file(eval_file.path)
                        for eval_file in eval_files
                    ]
                )

            model_benchmarks.append(
                ModelBenchmarkResult(
                    model_name=model_name,
                    evaluation_results=evaluation_results,
                )
            )

        return BenchmarkResult(model_benchmarks=model_benchmarks)

    @classmethod
    def from_matches_files(cls, path: str, rescore: bool = False) -> "BenchmarkResult":
        task_cls_mapping = {
            task_cls.task_name.replace("-", " "): task_cls
            for task_cls in load_all_tasks()
        }

        _with_subtask_pattern = re.compile(r"matches_([\w\s]+)_(\d+)_shot\.jsonl")
        _without_subtask_pattern = re.compile(r"matches_(\d+)_shot\.jsonl")

        matches_paths = glob.glob(f"{path}/*/*/matches*.jsonl")

        model_evals: list[tuple[str, str, str, TaskMatchGroup]] = []

        for match_path in matches_paths:
            match_file = os.path.basename(match_path)
            task_name = os.path.basename(os.path.dirname(match_path)).replace("_", " ")
            model_name = os.path.basename(os.path.dirname(os.path.dirname(match_path)))
            sub_task = None
            n_shots = 0

            if m := _with_subtask_pattern.match(match_file):
                sub_task = m.group(1)
                n_shots = int(m.group(2))
            elif m := _without_subtask_pattern.match(match_file):
                n_shots = int(m.group(1))
            else:
                raise Exception(
                    f"Matches file '{match_file}' doesn't match the expected pattern."
                )

            task_matches = TaskMatchGroup.from_file(
                Path(match_path).parent, n_shots=n_shots, sub_task=sub_task
            )
            assert (
                task_name is not task_cls_mapping
            ), f"No task class found for '{task_name}'."

            model_evals.append((model_name, task_name, sub_task, task_matches))

        model_benchmarks: list[ModelBenchmarkResult] = []

        for model_name, task_evals in itertools.groupby(
            model_evals, key=lambda t: t[0]
        ):
            print(f"Model: {model_name}")
            evaluation_results: list[EvaluationResult] = []

            for task_name, task_matches_group in itertools.groupby(
                task_evals, key=lambda t: t[1]
            ):
                print(f"Re-scoring {task_name}:")
                task = task_cls_mapping[task_name]()
                prompt_shot_evals = defaultdict(list)

                for _, _, sub_task, task_matches in task_matches_group:
                    print(f"{sub_task} {task_matches.n_shots}-shot prompt:")
                    if rescore:
                        task_matches = task.score_matches(task_matches)

                    score = task.get_overall_score(task_matches)

                    prompt_shot_evals[sub_task].append(
                        PromptShotEvaluationResult(
                            n_shots=task_matches.n_shots, score=score
                        )
                    )

                evaluation_results.extend(
                    EvaluationResult(
                        model_name=model_name,
                        task_name=task_name,
                        task_category=task.task_category,
                        score_name=task.score_name,
                        prompt_shot_results=prompt_shot_results,
                        sub_task=sub_task,
                    )
                    for sub_task, prompt_shot_results in prompt_shot_evals.items()
                )

            model_benchmarks.append(
                ModelBenchmarkResult(
                    model_name=model_name, evaluation_results=evaluation_results
                )
            )
            print("-" * 10)

        return BenchmarkResult(model_benchmarks=model_benchmarks)

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkResult":
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
        categories = set()

        for mb in self.model_benchmarks:
            values = []
            for _, evals in groupby(mb.evaluation_results, key=lambda e: e.task_name):
                evals = list(evals)
                score = sum(e.average_score for e in evals) / len(evals)
                values.append(score)

            data.append({"name": mb.model_name, "values": values})
            categories |= set(e.task_name for e in mb.evaluation_results)

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


def build_leaderboard_from_benchmark(
    benchmark_result: BenchmarkResult, leaderboard_path: str
):
    """
    This function generates leaderboard data from the benchmark result object.

    Parameters:
        benchmark_result (BenchmarkResult): BenchmarkResult object.
        leaderboard_path (str): Path to store the leaderboard data.
    """
    requests_path = Path(leaderboard_path) / "requests"
    results_path = Path(leaderboard_path) / "results"

    requests_path.mkdir(exist_ok=True)
    results_path.mkdir(exist_ok=True)

    now = datetime.datetime.now(pytz.UTC).isoformat(timespec="seconds")

    for mb in benchmark_result.model_benchmarks:
        model_name = mb.model_name

        os.makedirs(results_path / model_name, exist_ok=True)

        request = {
            "model": model_name,
            "base_model": "",
            "revision": "main",
            "private": False,
            "precision": "?",
            "weight_type": "Original",
            "status": "FINISHED",
            "submitted_time": now,
            "model_type": "\ud83d\udfe2 : pretrained",
            "likes": 0,
            "params": 0.1,
            "license": "custom",
        }
        with open(
            requests_path / f"{model_name}_eval_request_nshot.json", "wt"
        ) as writer:
            writer.write(json.dumps(request))

        result = {
            "config": {"model_dtype": "", "model_name": model_name, "model_sha": ""},
            "results": {
                er.task_name: {er.score_name: round(er.max_score, 3)}
                for er in mb.evaluation_results
            },
        }

        with open(results_path / model_name / f"results_{now}.json", "wt") as writer:
            writer.write(json.dumps(result))


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
