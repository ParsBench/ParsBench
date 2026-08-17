"""AppEvaluator: run golden expectations against your app (or against
pre-captured traces) and score them with checks inferred from each golden."""

import asyncio
import inspect
import time
from typing import Any, Callable

from tqdm import tqdm

from .checks import run_checks
from .evaluation_result import (
    AppEvaluationResult,
    CheckResult,
    GoldenEvaluationResult,
)
from .golden import Golden
from .judge import resolve_judge
from .recording import RunRecorder
from .trace import Trace


def _to_trace(result: Any) -> Trace:
    if isinstance(result, Trace):
        return result
    if isinstance(result, list):
        return Trace.from_messages(result)
    return Trace(final_output=str(result))


def _sync_await(awaitable: Any) -> Any:
    """Run an awaitable to completion from sync code, running loop or not."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_ensure_coro(awaitable))
    # called from inside a running loop (notebook, FastAPI handler) —
    # give the coroutine its own loop on a worker thread
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _ensure_coro(awaitable)).result()


async def _ensure_coro(awaitable: Any) -> Any:
    return await awaitable


def _run_app(app: Callable, golden: Golden) -> Trace:
    start = time.perf_counter()
    # inspect the returned value, not the callable: objects with an async
    # __call__ (and partial-wrapped coroutine functions) must be awaited too
    result = app(golden.input)
    if inspect.isawaitable(result):
        result = _sync_await(result)
    trace = _to_trace(result)
    if trace.latency is None:
        trace.latency = time.perf_counter() - start
    return trace


class AppEvaluator:
    """
    AppEvaluator evaluates a Persian AI app — any framework — against a suite
    of Golden expectations. Each filled Golden field switches on its check;
    there is no separate metric configuration.

    Attributes:
        goldens (list[Golden]): The golden expectations to evaluate.
        judge (Model | Callable | str, optional): The judge for judge-based
            checks — a parsbench Model, a callable `prompt -> str`, or a model
            name string (client built from PARSBENCH_JUDGE_* / OPENAI_* env).
            Defaults to the PARSBENCH_JUDGE env var; judge checks skip
            gracefully without one.
        metrics (list[str], optional): Filter/override of which checks run,
            e.g. ["tools:strict", "contains"].

    Methods:
        evaluate: Runs each golden against the app and scores the traces.
        score_traces: Scores pre-captured traces instead of running the app.
    """

    def __init__(
        self,
        goldens: list[Golden | dict],
        judge: Any = None,
        metrics: list[str] | None = None,
    ):
        self.goldens = [
            g if isinstance(g, Golden) else Golden.from_dict(g) for g in goldens
        ]
        if not self.goldens:
            raise ValueError("goldens is empty. You should provide at least one Golden.")
        self.judge = judge
        self.metrics = metrics

    def evaluate(
        self,
        app: Callable,
        n_runs: int = 1,
        prefer_concurrency: bool = False,
        n_workers: int = 4,
        save_evaluation: bool = False,
        output_path: str | None = None,
        record: bool = True,
    ) -> AppEvaluationResult:
        """
        Run each golden against the app and score the resulting traces.

        Parameters:
            app (Callable): The app under evaluation — a plain callable (sync
                or async), `input -> str | messages | Trace`. A crash inside
                the app is reported as a failing `app_error` check on that
                golden instead of aborting the whole run.
            n_runs (int, optional): Repeated runs per golden (default is 1);
                see `AppEvaluationResult.pass_hat_k`.
            prefer_concurrency (bool, optional): Evaluate goldens in parallel
                over a thread pool (default is False). The app and judge
                callables must then be thread-safe — most stateful bots are not,
                which is why this is off by default.
            n_workers (int, optional): The number of workers for concurrent
                processing (default is 4).
            save_evaluation (bool, optional): Flag to save the evaluation
                result (default is False).
            output_path (str, optional): The output path to save the
                evaluation result.
            record (bool, optional): Record this run into the local
                `.parsbench` store for `parsbench view` (default is True;
                also disabled by the PARSBENCH_NO_RECORD env var).

        Returns:
            AppEvaluationResult: The evaluation result over all goldens.
        """
        if n_runs < 1:
            raise ValueError("n_runs must be at least 1.")

        def runner(golden: Golden) -> Trace:
            return _run_app(app, golden)

        return self._score(
            runner,
            n_runs=n_runs,
            prefer_concurrency=prefer_concurrency,
            n_workers=n_workers,
            save_evaluation=save_evaluation,
            output_path=output_path,
            record=record,
            kind="evaluation",
            app_name=getattr(app, "__name__", type(app).__name__),
        )

    def score_traces(
        self,
        traces: list[Trace | list[dict]],
        save_evaluation: bool = False,
        output_path: str | None = None,
        record: bool = True,
    ) -> AppEvaluationResult:
        """
        Score pre-captured traces instead of running the app — run the app
        yourself (or in production) and evaluate what happened.

        Parameters:
            traces (list[Trace | list[dict]]): One trace per golden — a Trace
                or an OpenAI-format message list.
            save_evaluation (bool, optional): Flag to save the evaluation
                result (default is False).
            output_path (str, optional): The output path to save the
                evaluation result.
            record (bool, optional): Record this run into the local
                `.parsbench` store for `parsbench view` (default is True;
                also disabled by the PARSBENCH_NO_RECORD env var).

        Returns:
            AppEvaluationResult: The evaluation result over all goldens.
        """
        if len(traces) != len(self.goldens):
            raise ValueError(f"{len(traces)} traces for {len(self.goldens)} goldens.")
        # pair positionally — an id()-keyed dict would collapse when the same
        # Golden object appears twice and silently score the wrong trace
        traces_iter = iter([_to_trace(trace) for trace in traces])
        return self._score(
            lambda golden: next(traces_iter),
            n_runs=1,
            prefer_concurrency=False,
            n_workers=1,
            save_evaluation=save_evaluation,
            output_path=output_path,
            record=record,
            kind="score_traces",
            app_name="traces",
        )

    def _score(
        self,
        runner: Callable[[Golden], Trace],
        n_runs: int,
        prefer_concurrency: bool,
        n_workers: int,
        save_evaluation: bool,
        output_path: str | None,
        record: bool = True,
        kind: str = "evaluation",
        app_name: str = "app",
    ) -> AppEvaluationResult:
        if save_evaluation and not output_path:
            raise Exception("You should set the output path to save the evaluation.")

        judge = resolve_judge(self.judge)
        recorder = (
            RunRecorder.start(
                kind=kind,
                app_name=app_name,
                n_goldens=len(self.goldens),
                n_runs=n_runs,
                metrics=self.metrics,
            )
            if record
            else None
        )

        def evaluate_golden(item: tuple[int, Golden]) -> GoldenEvaluationResult:
            golden_index, golden = item
            run_passes = []
            detail: list[CheckResult] | None = None
            for run_index in range(n_runs):
                trace: Trace | None = None
                try:
                    trace = runner(golden)
                except Exception as exc:  # the app crashing is a finding, not a crash
                    run = [
                        CheckResult(
                            check="app_error",
                            passed=False,
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                    ]
                else:
                    run = run_checks(golden, trace, judge=judge, only=self.metrics)
                if recorder:
                    recorder.record_event(
                        golden, golden_index, run_index, run, trace=trace
                    )
                detail = detail or run  # first run carries the readable detail
                run_passes.append(all(r.passed for r in run if not r.skipped))
            return GoldenEvaluationResult(
                golden_name=golden.label,
                check_results=detail or [],
                run_passes=run_passes,
            )

        try:
            items = list(enumerate(self.goldens))
            if prefer_concurrency and n_workers > 1:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    golden_results = list(
                        tqdm(
                            pool.map(evaluate_golden, items),
                            total=len(items),
                            desc="Evaluating goldens",
                        )
                    )
            else:
                golden_results = [
                    evaluate_golden(item)
                    for item in tqdm(items, desc="Evaluating goldens")
                ]

            evaluation_result = AppEvaluationResult(golden_results=golden_results)
        except BaseException as exc:
            if recorder:
                recorder.crashed(exc)
            raise

        if recorder:
            recorder.finish(evaluation_result)

        if save_evaluation and output_path:
            evaluation_result.save(output_path)

        return evaluation_result
