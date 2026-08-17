"""Result data classes for app evaluations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pandas/jsonlines are import-time heavy; load them on use
    import pandas as pd

EVALUATION_FILE_NAME = "app_evaluation.jsonl"


@dataclass
class CheckResult:
    """
    The outcome of a single check on a single golden.

    Attributes:
        check (str): The name of the check (e.g. "contains", "tools:subset").
        score (float): The check score between 0 and 1.
        passed (bool): Whether the check passed.
        skipped (bool): Whether the check was skipped (e.g. no judge configured).
        reason (str, optional): A readable explanation for failures/skips.
    """

    check: str
    score: float = 0.0
    passed: bool = False
    skipped: bool = False
    reason: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "CheckResult":
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GoldenEvaluationResult:
    """
    The evaluation result for one golden: its check results and, when the
    golden was run more than once, the pass/fail of each repeated run.

    Attributes:
        golden_name (str): The label of the evaluated golden.
        check_results (list[CheckResult]): The results of each check.
        run_passes (list[bool]): Pass/fail of each repeated run (n_runs > 1);
            a single-element list for a single run.
        transcript (str, optional): The rendered conversation (simulation only).
    """

    golden_name: str
    check_results: list[CheckResult] = field(default_factory=list)
    run_passes: list[bool] = field(default_factory=list)
    transcript: str | None = None

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.check_results if not r.skipped)

    @classmethod
    def from_dict(cls, data: dict) -> "GoldenEvaluationResult":
        check_results = [CheckResult.from_dict(cr) for cr in data.pop("check_results")]
        return cls(**data, check_results=check_results)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "check_results": [cr.to_dict() for cr in self.check_results],
        }

    def to_pandas(self) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(
            [
                {"golden_name": self.golden_name, **cr.to_dict()}
                for cr in self.check_results
            ]
        )

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        text = f"[{mark}] {self.golden_name}\n"
        for cr in self.check_results:
            status = "skip" if cr.skipped else ("ok  " if cr.passed else "FAIL")
            text += f"  {status} {cr.check:<16} {cr.score:.2f}"
            if cr.reason and not cr.passed:
                text += f"  — {cr.reason}"
            text += "\n"
        return text.strip("\n")


@dataclass
class AppEvaluationResult:
    """
    The result of evaluating an app against a suite of goldens.

    Attributes:
        golden_results (list[GoldenEvaluationResult]): One result per golden.

    Methods:
        score: Mean score, optionally restricted to one check name.
        pass_hat_k: tau2-style pass^k consistency over repeated runs.
        save: Writes the result to `app_evaluation.jsonl` in the given path.
        diff: Prints per-check mean deltas vs a previously saved result.
        to_langfuse: Pushes per-check scores to a Langfuse instance.
        assert_passed: Raises AssertionError with the failing checks (pytest-friendly).
    """

    golden_results: list[GoldenEvaluationResult]

    @property
    def passed(self) -> bool:
        return all(gr.passed for gr in self.golden_results)

    @property
    def average_score(self) -> float:
        return self.score()

    def score(self, check: str | None = None) -> float:
        scores = [
            cr.score
            for gr in self.golden_results
            for cr in gr.check_results
            if not cr.skipped and (check is None or cr.check.startswith(check))
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def pass_hat_k(self, k: int | None = None) -> float:
        """
        tau2-style pass^k over repeated runs: C(c,k)/C(n,k) averaged over
        goldens, where n = runs done and c = runs passed.

        Parameters:
            k (int, optional): The consistency exponent (default is all runs).

        Returns:
            float: The pass^k score.
        """
        values = []
        for gr in self.golden_results:
            runs = gr.run_passes or [gr.passed]
            n, c = len(runs), sum(runs)
            kk = k or n
            if kk > n:
                raise ValueError(
                    f"k={kk} but only {n} runs were done (evaluate with n_runs={kk})."
                )
            values.append(math.comb(c, kk) / math.comb(n, kk))
        return sum(values) / len(values) if values else 0.0

    @classmethod
    def from_file(cls, path: str) -> "AppEvaluationResult":
        import jsonlines

        with jsonlines.open(path, "r") as reader:
            golden_results = [
                GoldenEvaluationResult.from_dict(row) for row in reader.iter(type=dict)
            ]
        return cls(golden_results=golden_results)

    @classmethod
    def from_dict(cls, data: dict) -> "AppEvaluationResult":
        golden_results = [
            GoldenEvaluationResult.from_dict(gr) for gr in data.pop("golden_results")
        ]
        return cls(**data, golden_results=golden_results)

    def to_dict(self) -> dict:
        return {"golden_results": [gr.to_dict() for gr in self.golden_results]}

    def to_pandas(self) -> pd.DataFrame:
        import pandas as pd

        return pd.concat([gr.to_pandas() for gr in self.golden_results])

    def save(self, path: str):
        evaluation_path = Path(path) / EVALUATION_FILE_NAME
        # create the directory up front — failing here after a full (paid,
        # judge-calling) evaluation would lose the finished result
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        import jsonlines

        with jsonlines.open(evaluation_path, "w") as writer:
            for gr in self.golden_results:
                writer.write(gr.to_dict())

    def diff(self, path: str):
        """
        Print per-check mean score deltas vs a previously saved result.

        Parameters:
            path (str): Path to a saved `app_evaluation.jsonl` file.
        """
        old = AppEvaluationResult.from_file(path)
        checks = {cr.check for gr in self.golden_results for cr in gr.check_results}
        for check in sorted(checks):
            delta = self.score(check) - old.score(check)
            if abs(delta) > 1e-9:
                print(f"{check}: {old.score(check):.2f} -> {self.score(check):.2f} ({delta:+.2f})")

    def to_langfuse(self, **kwargs) -> str:
        """Push per-check scores into Langfuse. Returns the created trace id."""
        from parsbench.integrations import langfuse

        return langfuse.push(self, **kwargs)

    def assert_passed(self):
        """Raise AssertionError listing every failing check, for pytest/CI."""
        failed = [
            f"{gr.golden_name} — {cr.check}: {cr.reason or f'score={cr.score:.2f}'}"
            for gr in self.golden_results
            for cr in gr.check_results
            if not cr.passed and not cr.skipped
        ]
        if failed:
            raise AssertionError("ParsBench checks failed:\n  " + "\n  ".join(failed))

    def __str__(self) -> str:
        text = ""
        for gr in self.golden_results:
            text += str(gr) + "\n"
        total = sum(len(gr.check_results) for gr in self.golden_results)
        failed = sum(
            1
            for gr in self.golden_results
            for cr in gr.check_results
            if not cr.passed and not cr.skipped
        )
        text += f"score={self.score():.2f}  checks={total}  failed={failed}"
        return text
