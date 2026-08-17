"""JudgeCalibrator: measure judge-vs-human agreement on labeled Persian
samples before trusting (or publishing) judge scores."""

from dataclasses import dataclass
from typing import Any

from .checks import JUDGE_CHECKS, run_checks
from .golden import Golden
from .judge import resolve_model
from .trace import Trace


@dataclass
class CalibrationResult:
    """
    The result of calibrating a judge against human labels.

    Attributes:
        n (int): The number of labeled items.
        agreement (float): Fraction where judge pass/fail == human label.
        kappa (float): Cohen's kappa vs human labels.
        disagreements (list[str]): Readable descriptions for error analysis.
    """

    n: int
    agreement: float
    kappa: float
    disagreements: list[str]

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationResult":
        return cls(**data)

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    def __str__(self) -> str:
        return (
            f"judge-vs-human on {self.n} items: agreement={self.agreement:.2f}, "
            f"kappa={self.kappa:.2f}, disagreements={len(self.disagreements)}"
        )


class JudgeCalibrator:
    """
    JudgeCalibrator measures how well a judge model agrees with human labels
    on a labeled sample, so judge scores can be trusted (or fixed) before
    they are published.

    Attributes:
        judge (Model | Callable | str, optional): The judge to calibrate
            (falls back to the PARSBENCH_JUDGE env var).

    Methods:
        calibrate: Scores the labeled items and computes agreement and kappa.
    """

    def __init__(self, judge: Any = None):
        self.judge = judge

    def calibrate(
        self,
        items: list[dict],
        prefer_concurrency: bool = False,
        n_workers: int = 4,
    ) -> CalibrationResult:
        """
        Score each labeled item with the judge and compare to the human label.

        Parameters:
            items (list[dict]): Items of the form
                `{"golden": Golden(...), "output": "...", "human": True}` where
                the golden triggers at least one judge check (output=, context=
                or refuses=).
            prefer_concurrency (bool, optional): Fan judge calls out over a
                thread pool (default is False); the judge callable must then
                be thread-safe.
            n_workers (int, optional): The number of workers for concurrent
                processing (default is 4).

        Returns:
            CalibrationResult: Agreement, Cohen's kappa, and disagreements.
        """
        if not items:
            raise ValueError("items is empty. You should provide at least one labeled item.")
        judge = resolve_model(self.judge, "PARSBENCH_JUDGE")
        if judge is None:
            raise ValueError(
                "calibrate needs a judge — pass judge= or set PARSBENCH_JUDGE."
            )

        def score_item(item) -> tuple[str, bool, bool, str | None]:
            golden = item["golden"]
            golden = golden if isinstance(golden, Golden) else Golden.from_dict(golden)
            trace = Trace(final_output=str(item["output"]))
            results = [
                r for r in run_checks(golden, trace, judge=judge)
                if r.check in JUDGE_CHECKS and not r.skipped
            ]
            if not results:
                raise ValueError(
                    f"golden {golden.label!r} triggers no judge check — it needs "
                    "output=, context= or refuses=."
                )
            judged = all(r.passed for r in results)
            reason = next((r.reason for r in results if not r.passed), results[0].reason)
            return golden.label, judged, bool(item["human"]), reason

        if prefer_concurrency and n_workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                scored = list(pool.map(score_item, items))
        else:
            scored = [score_item(item) for item in items]

        pairs = [(judged, human) for _, judged, human, _ in scored]
        disagreements = [
            f"{label}: judge={'pass' if judged else 'fail'} "
            f"human={'pass' if human else 'fail'} — {reason}"
            for label, judged, human, reason in scored
            if judged != human
        ]

        n = len(pairs)
        agreement = sum(j == h for j, h in pairs) / n
        # Cohen's kappa from marginals
        judge_yes = sum(j for j, _ in pairs) / n
        human_yes = sum(h for _, h in pairs) / n
        expected = judge_yes * human_yes + (1 - judge_yes) * (1 - human_yes)
        kappa = 0.0 if expected == 1.0 else (agreement - expected) / (1 - expected)
        return CalibrationResult(
            n=n, agreement=agreement, kappa=kappa, disagreements=disagreements
        )
