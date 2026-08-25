# App Evaluation

AppEvaluator evaluates a Persian AI app — any framework — against a suite of Golden expectations. Each filled Golden field switches on its check; there is no separate metric configuration.

Attributes:

| Name      | Type           | Description                                                             |
| --------- | -------------- | ----------------------------------------------------------------------- |
| `goldens` | `list[Golden]` | The golden expectations to evaluate.                                    |
| `judge`   | \`Model        | Callable                                                                |
| `metrics` | `list[str]`    | Filter/override of which checks run, e.g. ["tools:strict", "contains"]. |

Methods:

| Name           | Description                                             |
| -------------- | ------------------------------------------------------- |
| `evaluate`     | Runs each golden against the app and scores the traces. |
| `score_traces` | Scores pre-captured traces instead of running the app.  |

Source code in `parsbench/appeval/evaluator.py`

```
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
```

## `evaluate(app, n_runs=1, prefer_concurrency=False, n_workers=4, save_evaluation=False, output_path=None, record=True)`

Run each golden against the app and score the resulting traces.

Parameters:

| Name                 | Type       | Description                                                                                                                                                                                 | Default  |
| -------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `app`                | `Callable` | The app under evaluation — a plain callable (sync or async), input -> str                                                                                                                   | messages |
| `n_runs`             | `int`      | Repeated runs per golden (default is 1); see AppEvaluationResult.pass_hat_k.                                                                                                                | `1`      |
| `prefer_concurrency` | `bool`     | Evaluate goldens in parallel over a thread pool (default is False). The app and judge callables must then be thread-safe — most stateful bots are not, which is why this is off by default. | `False`  |
| `n_workers`          | `int`      | The number of workers for concurrent processing (default is 4).                                                                                                                             | `4`      |
| `save_evaluation`    | `bool`     | Flag to save the evaluation result (default is False).                                                                                                                                      | `False`  |
| `output_path`        | `str`      | The output path to save the evaluation result.                                                                                                                                              | `None`   |
| `record`             | `bool`     | Record this run into the local .parsbench store for parsbench view (default is True; also disabled by the PARSBENCH_NO_RECORD env var).                                                     | `True`   |

Returns:

| Name                  | Type                  | Description                             |
| --------------------- | --------------------- | --------------------------------------- |
| `AppEvaluationResult` | `AppEvaluationResult` | The evaluation result over all goldens. |

Source code in `parsbench/appeval/evaluator.py`

```
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
```

## `score_traces(traces, save_evaluation=False, output_path=None, record=True)`

Score pre-captured traces instead of running the app — run the app yourself (or in production) and evaluate what happened.

Parameters:

| Name              | Type          | Description                                                                                                                             | Default                                                          |
| ----------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `traces`          | \`list\[Trace | list[dict]\]\`                                                                                                                          | One trace per golden — a Trace or an OpenAI-format message list. |
| `save_evaluation` | `bool`        | Flag to save the evaluation result (default is False).                                                                                  | `False`                                                          |
| `output_path`     | `str`         | The output path to save the evaluation result.                                                                                          | `None`                                                           |
| `record`          | `bool`        | Record this run into the local .parsbench store for parsbench view (default is True; also disabled by the PARSBENCH_NO_RECORD env var). | `True`                                                           |

Returns:

| Name                  | Type                  | Description                             |
| --------------------- | --------------------- | --------------------------------------- |
| `AppEvaluationResult` | `AppEvaluationResult` | The evaluation result over all goldens. |

Source code in `parsbench/appeval/evaluator.py`

```
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
```

One expectation for the evaluated app. Every field except `input` is optional; each filled field switches on its corresponding check.

Attributes:

| Name              | Type             | Description                                                                                 |
| ----------------- | ---------------- | ------------------------------------------------------------------------------------------- |
| `input`           | `str`            | The user message sent to the app.                                                           |
| `name`            | `str`            | A readable name for reports.                                                                |
| `output`          | `str`            | Reference answer; judged by the correctness check.                                          |
| `contains`        | `list[str]`      | Substrings the answer must state — normalized, money- and calendar-equivalent.              |
| `not_contains`    | `list[str]`      | Substrings the answer must not state.                                                       |
| `format`          | `type`           | A pydantic-style model class the answer must validate against (checked via model_validate). |
| `tools`           | `list[ToolCall]` | Tool calls the app must make; arguments are compared date -> number -> normalized text.     |
| `forbidden_tools` | `list[str]`      | Tools the app must not call.                                                                |
| `context`         | `list[str]`      | Grounding context; judged by the faithfulness check.                                        |
| `refuses`         | `bool`           | Whether the app must decline the request (refusal check).                                   |
| `max_steps`       | `int`            | Budget on assistant steps.                                                                  |
| `max_latency`     | `float`          | Budget on wall-clock seconds.                                                               |
| `max_cost`        | `float`          | Budget on cost.                                                                             |
| `check`           | `Callable`       | Custom Trace -> bool                                                                        |
| `tags`            | `list[str]`      | Free-form labels.                                                                           |

Methods:

| Name        | Description                                                                       |
| ----------- | --------------------------------------------------------------------------------- |
| `from_dict` | Builds a Golden from a dict, accepting the short in/out aliases for input/output. |

Source code in `parsbench/appeval/golden.py`

```
@dataclass
class Golden:
    """
    One expectation for the evaluated app. Every field except `input` is
    optional; each filled field switches on its corresponding check.

    Attributes:
        input (str): The user message sent to the app.
        name (str, optional): A readable name for reports.
        output (str, optional): Reference answer; judged by the `correctness` check.
        contains (list[str]): Substrings the answer must state — normalized,
            money- and calendar-equivalent.
        not_contains (list[str]): Substrings the answer must not state.
        format (type, optional): A pydantic-style model class the answer must
            validate against (checked via `model_validate`).
        tools (list[ToolCall]): Tool calls the app must make; arguments are
            compared date -> number -> normalized text.
        forbidden_tools (list[str]): Tools the app must not call.
        context (list[str]): Grounding context; judged by the `faithfulness` check.
        refuses (bool): Whether the app must decline the request (`refusal` check).
        max_steps (int, optional): Budget on assistant steps.
        max_latency (float, optional): Budget on wall-clock seconds.
        max_cost (float, optional): Budget on cost.
        check (Callable, optional): Custom `Trace -> bool | float` check.
        tags (list[str]): Free-form labels.

    Methods:
        from_dict: Builds a Golden from a dict, accepting the short `in`/`out`
            aliases for `input`/`output`.
    """

    input: str
    name: str | None = None
    # answer expectations
    output: str | None = None
    contains: list[str] = field(default_factory=list)
    not_contains: list[str] = field(default_factory=list)
    format: type | None = None
    # behavior expectations
    tools: list[ToolCall] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    refuses: bool = False
    # budgets
    max_steps: int | None = None
    max_latency: float | None = None
    max_cost: float | None = None
    # escape hatch
    check: Callable[[Trace], bool | float] | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        # a bare string for a list field must not be iterated character by
        # character — wrap it; dict tool specs are promoted to ToolCall
        self.contains = _as_list(self.contains)
        self.not_contains = _as_list(self.not_contains)
        self.forbidden_tools = _as_list(self.forbidden_tools)
        self.context = _as_list(self.context)
        self.tags = _as_list(self.tags)
        self.tools = [
            tc if isinstance(tc, ToolCall) else ToolCall(**tc) for tc in self.tools
        ]

    @classmethod
    def from_dict(cls, data: dict) -> "Golden":
        data = dict(data)
        if "in" in data:
            data["input"] = data.pop("in")
        if "out" in data:
            data["output"] = data.pop("out")
        return cls(**data)

    @property
    def label(self) -> str:
        return self.name or (self.input[:40] + "…" if len(self.input) > 40 else self.input)
```

A multi-turn expectation, consumed by SimulationEvaluator.

Attributes:

| Name               | Type        | Description                                          |
| ------------------ | ----------- | ---------------------------------------------------- |
| `goal`             | `str`       | What the simulated user wants from the conversation. |
| `name`             | `str`       | A readable name for reports.                         |
| `scenario`         | `str`       | Extra situation description for the user simulator.  |
| `expected_outcome` | `str`       | What "done well" looks like, for the judge.          |
| `criteria`         | `list[str]` | Behaviors judged each as their own check.            |
| `max_turns`        | `int`       | Turn cap for the conversation.                       |

Source code in `parsbench/appeval/golden.py`

```
@dataclass
class ConversationGolden:
    """
    A multi-turn expectation, consumed by SimulationEvaluator.

    Attributes:
        goal (str): What the simulated user wants from the conversation.
        name (str, optional): A readable name for reports.
        scenario (str): Extra situation description for the user simulator.
        expected_outcome (str): What "done well" looks like, for the judge.
        criteria (list[str]): Behaviors judged each as their own check.
        max_turns (int): Turn cap for the conversation.
    """

    goal: str
    name: str | None = None
    scenario: str = ""
    expected_outcome: str = ""
    criteria: list[str] = field(default_factory=list)
    max_turns: int = 8

    def __post_init__(self):
        self.criteria = _as_list(self.criteria)

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationGolden":
        return cls(**data)

    @property
    def label(self) -> str:
        return self.name or (self.goal[:40] + "…" if len(self.goal) > 40 else self.goal)
```

What the evaluated app actually did in response to one input.

Attributes:

| Name           | Type            | Description                               |
| -------------- | --------------- | ----------------------------------------- |
| `messages`     | `list[Message]` | The conversation messages, OpenAI-style.  |
| `final_output` | `str`           | The final assistant answer.               |
| `latency`      | `float`         | Wall-clock seconds for the run.           |
| `cost`         | `float`         | Cost of the run, if known.                |
| `raw`          | `Any`           | The original framework object, untouched. |

Methods:

| Name            | Description                                      |
| --------------- | ------------------------------------------------ |
| `from_messages` | Builds a Trace from OpenAI-format message dicts. |

Source code in `parsbench/appeval/trace.py`

```
@dataclass
class Trace:
    """
    What the evaluated app actually did in response to one input.

    Attributes:
        messages (list[Message]): The conversation messages, OpenAI-style.
        final_output (str): The final assistant answer.
        latency (float, optional): Wall-clock seconds for the run.
        cost (float, optional): Cost of the run, if known.
        raw (Any, optional): The original framework object, untouched.

    Methods:
        from_messages: Builds a Trace from OpenAI-format message dicts.
    """

    messages: list[Message] = field(default_factory=list)
    final_output: str = ""
    latency: float | None = None
    cost: float | None = None
    raw: Any = None

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [tc for m in self.messages if m.role == "assistant" for tc in m.tool_calls]

    @property
    def n_steps(self) -> int:
        return sum(1 for m in self.messages if m.role == "assistant")

    @classmethod
    def from_messages(
        cls, messages: list[dict], *, final_output: Any = None, raw: Any = None
    ) -> "Trace":
        """
        Build a Trace from OpenAI-format message dicts.

        Parameters:
            messages (list[dict]): OpenAI chat-format messages.
            final_output (Any, optional): Overrides the derived final answer,
                but only when it is a non-empty string.
            raw (Any, optional): The original framework object to keep around.

        Returns:
            Trace: The parsed trace.
        """
        parsed: list[Message] = []
        calls_by_id: dict[str, ToolCall] = {}
        for m in messages:
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                call = ToolCall(fn.get("name", ""), arguments=args)
                calls.append(call)
                if tc.get("id"):
                    calls_by_id[tc["id"]] = call
            role = m.get("role", "assistant")
            role = _ROLE_ALIASES.get(role, role if role in _ROLES else "assistant")
            parsed.append(
                Message(
                    role=role,
                    content=m.get("content"),
                    tool_calls=calls,
                    tool_call_id=m.get("tool_call_id"),
                )
            )
            if role == "tool" and m.get("tool_call_id") in calls_by_id:
                calls_by_id[m["tool_call_id"]].result = m.get("content")
        final = next(
            (m.content for m in reversed(parsed) if m.role == "assistant" and m.content),
            "",
        )
        if isinstance(final_output, str) and final_output:
            final = final_output
        return cls(messages=parsed, final_output=final, raw=raw)
```

## `from_messages(messages, *, final_output=None, raw=None)`

Build a Trace from OpenAI-format message dicts.

Parameters:

| Name           | Type         | Description                                                                 | Default    |
| -------------- | ------------ | --------------------------------------------------------------------------- | ---------- |
| `messages`     | `list[dict]` | OpenAI chat-format messages.                                                | *required* |
| `final_output` | `Any`        | Overrides the derived final answer, but only when it is a non-empty string. | `None`     |
| `raw`          | `Any`        | The original framework object to keep around.                               | `None`     |

Returns:

| Name    | Type    | Description       |
| ------- | ------- | ----------------- |
| `Trace` | `Trace` | The parsed trace. |

Source code in `parsbench/appeval/trace.py`

```
@classmethod
def from_messages(
    cls, messages: list[dict], *, final_output: Any = None, raw: Any = None
) -> "Trace":
    """
    Build a Trace from OpenAI-format message dicts.

    Parameters:
        messages (list[dict]): OpenAI chat-format messages.
        final_output (Any, optional): Overrides the derived final answer,
            but only when it is a non-empty string.
        raw (Any, optional): The original framework object to keep around.

    Returns:
        Trace: The parsed trace.
    """
    parsed: list[Message] = []
    calls_by_id: dict[str, ToolCall] = {}
    for m in messages:
        calls = []
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            call = ToolCall(fn.get("name", ""), arguments=args)
            calls.append(call)
            if tc.get("id"):
                calls_by_id[tc["id"]] = call
        role = m.get("role", "assistant")
        role = _ROLE_ALIASES.get(role, role if role in _ROLES else "assistant")
        parsed.append(
            Message(
                role=role,
                content=m.get("content"),
                tool_calls=calls,
                tool_call_id=m.get("tool_call_id"),
            )
        )
        if role == "tool" and m.get("tool_call_id") in calls_by_id:
            calls_by_id[m["tool_call_id"]].result = m.get("content")
    final = next(
        (m.content for m in reversed(parsed) if m.role == "assistant" and m.content),
        "",
    )
    if isinstance(final_output, str) and final_output:
        final = final_output
    return cls(messages=parsed, final_output=final, raw=raw)
```

A single message in a conversation trace.

Attributes:

| Name           | Type             | Description                                                 |
| -------------- | ---------------- | ----------------------------------------------------------- |
| `role`         | `str`            | One of "system", "user", "assistant", or "tool".            |
| `content`      | `str`            | The text content of the message.                            |
| `tool_calls`   | `list[ToolCall]` | Tool calls made in this message.                            |
| `tool_call_id` | `str`            | For tool messages, the id of the call this message answers. |

Source code in `parsbench/appeval/trace.py`

```
@dataclass
class Message:
    """
    A single message in a conversation trace.

    Attributes:
        role (str): One of "system", "user", "assistant", or "tool".
        content (str, optional): The text content of the message.
        tool_calls (list[ToolCall]): Tool calls made in this message.
        tool_call_id (str, optional): For tool messages, the id of the call
            this message answers.
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
```

Represents an expected or observed tool call.

Extra keyword arguments become tool arguments, so the short form `ToolCall("search", date="1405-07-05")` is equivalent to `ToolCall(name="search", arguments={"date": "1405-07-05"})`.

Attributes:

| Name        | Type   | Description                                            |
| ----------- | ------ | ------------------------------------------------------ |
| `name`      | `str`  | The name of the tool.                                  |
| `arguments` | `dict` | The arguments the tool was (or should be) called with. |
| `result`    | `Any`  | The observed tool result, if any.                      |
| `error`     | `str`  | The observed tool error, if any.                       |

Source code in `parsbench/appeval/trace.py`

```
@dataclass(init=False)
class ToolCall:
    """
    Represents an expected or observed tool call.

    Extra keyword arguments become tool arguments, so the short form
    `ToolCall("search", date="1405-07-05")` is equivalent to
    `ToolCall(name="search", arguments={"date": "1405-07-05"})`.

    Attributes:
        name (str): The name of the tool.
        arguments (dict): The arguments the tool was (or should be) called with.
        result (Any, optional): The observed tool result, if any.
        error (str, optional): The observed tool error, if any.
    """

    name: str
    arguments: dict[str, Any]
    result: Any
    error: str | None

    def __init__(
        self,
        name: str = "",
        arguments: dict[str, Any] | None = None,
        result: Any = None,
        error: str | None = None,
        **extra_arguments: Any,
    ):
        self.name = name
        self.arguments = {**(arguments or {}), **extra_arguments}
        self.result = result
        self.error = error
```

The result of evaluating an app against a suite of goldens.

Attributes:

| Name             | Type                           | Description            |
| ---------------- | ------------------------------ | ---------------------- |
| `golden_results` | `list[GoldenEvaluationResult]` | One result per golden. |

Methods:

| Name            | Description                                                      |
| --------------- | ---------------------------------------------------------------- |
| `score`         | Mean score, optionally restricted to one check name.             |
| `pass_hat_k`    | tau2-style pass^k consistency over repeated runs.                |
| `save`          | Writes the result to app_evaluation.jsonl in the given path.     |
| `diff`          | Prints per-check mean deltas vs a previously saved result.       |
| `to_langfuse`   | Pushes per-check scores to a Langfuse instance.                  |
| `assert_passed` | Raises AssertionError with the failing checks (pytest-friendly). |

Source code in `parsbench/appeval/evaluation_result.py`

```
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
```

## `assert_passed()`

Raise AssertionError listing every failing check, for pytest/CI.

Source code in `parsbench/appeval/evaluation_result.py`

```
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
```

## `diff(path)`

Print per-check mean score deltas vs a previously saved result.

Parameters:

| Name   | Type  | Description                                | Default    |
| ------ | ----- | ------------------------------------------ | ---------- |
| `path` | `str` | Path to a saved app_evaluation.jsonl file. | *required* |

Source code in `parsbench/appeval/evaluation_result.py`

```
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
```

## `pass_hat_k(k=None)`

tau2-style pass^k over repeated runs: C(c,k)/C(n,k) averaged over goldens, where n = runs done and c = runs passed.

Parameters:

| Name | Type  | Description                                     | Default |
| ---- | ----- | ----------------------------------------------- | ------- |
| `k`  | `int` | The consistency exponent (default is all runs). | `None`  |

Returns:

| Name    | Type    | Description       |
| ------- | ------- | ----------------- |
| `float` | `float` | The pass^k score. |

Source code in `parsbench/appeval/evaluation_result.py`

```
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
```

## `to_langfuse(**kwargs)`

Push per-check scores into Langfuse. Returns the created trace id.

Source code in `parsbench/appeval/evaluation_result.py`

```
def to_langfuse(self, **kwargs) -> str:
    """Push per-check scores into Langfuse. Returns the created trace id."""
    from parsbench.integrations import langfuse

    return langfuse.push(self, **kwargs)
```

The evaluation result for one golden: its check results and, when the golden was run more than once, the pass/fail of each repeated run.

Attributes:

| Name            | Type                | Description                                                                          |
| --------------- | ------------------- | ------------------------------------------------------------------------------------ |
| `golden_name`   | `str`               | The label of the evaluated golden.                                                   |
| `check_results` | `list[CheckResult]` | The results of each check.                                                           |
| `run_passes`    | `list[bool]`        | Pass/fail of each repeated run (n_runs > 1); a single-element list for a single run. |
| `transcript`    | `str`               | The rendered conversation (simulation only).                                         |

Source code in `parsbench/appeval/evaluation_result.py`

```
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
```

The outcome of a single check on a single golden.

Attributes:

| Name      | Type    | Description                                               |
| --------- | ------- | --------------------------------------------------------- |
| `check`   | `str`   | The name of the check (e.g. "contains", "tools:subset").  |
| `score`   | `float` | The check score between 0 and 1.                          |
| `passed`  | `bool`  | Whether the check passed.                                 |
| `skipped` | `bool`  | Whether the check was skipped (e.g. no judge configured). |
| `reason`  | `str`   | A readable explanation for failures/skips.                |

Source code in `parsbench/appeval/evaluation_result.py`

```
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
```

SimulationEvaluator simulates Persian users against a bot and judges the finished conversations against each goal and its criteria.

Attributes:

| Name              | Type                       | Description                                                                                                                                                                  |
| ----------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `goldens`         | `list[ConversationGolden]` | The multi-turn expectations.                                                                                                                                                 |
| `user`            | `PersianUser`              | The simulated user's personality. Also accepts a string like "محاوره‌ای+finglish_switch" mixing a register with trap names (see TRAPS); free text becomes extra instructions. |
| `simulator_model` | \`Model                    | Callable                                                                                                                                                                     |
| `judge`           | \`Model                    | Callable                                                                                                                                                                     |

Methods:

| Name       | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| `evaluate` | Simulates the conversations against the app and scores them. |

Source code in `parsbench/appeval/simulation.py`

```
class SimulationEvaluator:
    """
    SimulationEvaluator simulates Persian users against a bot and judges the
    finished conversations against each goal and its criteria.

    Attributes:
        goldens (list[ConversationGolden]): The multi-turn expectations.
        user (PersianUser): The simulated user's personality. Also accepts a
            string like "محاوره‌ای+finglish_switch" mixing a register with trap
            names (see TRAPS); free text becomes extra instructions.
        simulator_model (Model | Callable | str, optional): The user-simulator
            LLM (falls back to PARSBENCH_SIMULATOR, then PARSBENCH_JUDGE env).
        judge (Model | Callable | str, optional): The conversation judge
            (falls back to the PARSBENCH_JUDGE env var).

    Methods:
        evaluate: Simulates the conversations against the app and scores them.
    """

    def __init__(
        self,
        goldens: list[ConversationGolden | dict] | None = None,
        goal: str | None = None,
        criteria: list[str] | None = None,
        user: PersianUser | str | None = None,
        simulator_model: Any = None,
        judge: Any = None,
    ):
        self.goldens = [
            g if isinstance(g, ConversationGolden) else ConversationGolden.from_dict(g)
            for g in (goldens or [])
        ]
        if goal:
            self.goldens.append(ConversationGolden(goal=goal, criteria=criteria or []))
        if not self.goldens:
            raise ValueError(
                "goldens is empty. You should provide a goal or at least one "
                "ConversationGolden."
            )
        self.user = _parse_user_spec(user)
        self.simulator_model = simulator_model
        self.judge = judge

    def evaluate(
        self,
        app: Callable,
        n_runs: int = 1,
        max_turns: int | None = None,
        save_evaluation: bool = False,
        output_path: str | None = None,
        record: bool = True,
    ) -> AppEvaluationResult:
        """
        Simulate each golden's conversation against the app and judge it.

        Parameters:
            app (Callable): The bot under evaluation — `fn(message)` for
                stateful bots or `fn(message, history)`.
            n_runs (int, optional): Repeated conversations per golden
                (default is 1); see `AppEvaluationResult.pass_hat_k`.
            max_turns (int, optional): Turn cap override; defaults to each
                golden's own `max_turns`.
            save_evaluation (bool, optional): Flag to save the evaluation
                result (default is False).
            output_path (str, optional): The output path to save the
                evaluation result.
            record (bool, optional): Record this run into the local
                `.parsbench` store for `parsbench view` (default is True;
                also disabled by the PARSBENCH_NO_RECORD env var).

        Returns:
            AppEvaluationResult: The evaluation result over all conversations.
        """
        if n_runs < 1:
            raise ValueError("n_runs must be at least 1.")
        if save_evaluation and not output_path:
            raise Exception("You should set the output path to save the evaluation.")

        # the user simulator stays at the provider's default temperature —
        # pinning it to 0 would replay near-identical conversations and blind
        # pass_hat_k
        simulator = resolve_model(
            self.simulator_model, "PARSBENCH_SIMULATOR", "PARSBENCH_JUDGE",
            temperature=None,
        )
        if simulator is None:
            raise ValueError(
                "simulator model not configured — pass simulator_model= or set "
                "PARSBENCH_SIMULATOR."
            )
        judge = resolve_model(self.judge, "PARSBENCH_JUDGE")

        recorder = (
            RunRecorder.start(
                kind="simulation",
                app_name=getattr(app, "__name__", type(app).__name__),
                n_goldens=len(self.goldens),
                n_runs=n_runs,
                extra_meta={
                    "user": {
                        "style": self.user.style,
                        "persona": self.user.persona,
                        "traps": self.user.traps,
                    }
                },
            )
            if record
            else None
        )

        arity = _app_arity(app)
        golden_results = []
        try:
            for golden_index, golden in enumerate(
                tqdm(self.goldens, desc="Simulating conversations")
            ):
                turn_cap = max_turns or golden.max_turns
                run_passes: list[bool] = []
                detail: list[CheckResult] | None = None
                transcript = None
                for run_index in range(n_runs):
                    rendered: str | None = None
                    run_trace: Trace | None = None
                    converged: bool | None = None
                    try:
                        history, converged = self._run_conversation(
                            app, arity, simulator, golden, turn_cap
                        )
                    except Exception as exc:  # a crashing bot is a finding
                        checks = [
                            CheckResult(
                                check="app_error",
                                passed=False,
                                reason=f"{type(exc).__name__}: {exc}",
                            )
                        ]
                        converged = None
                    else:
                        checks = self._score_conversation(
                            history, converged, golden, judge
                        )
                        rendered = _render(history)
                        run_trace = Trace.from_messages(history)
                        if transcript is None:
                            transcript = rendered
                    if recorder:
                        recorder.record_event(
                            golden, golden_index, run_index, checks,
                            trace=run_trace, transcript=rendered,
                            converged=converged,
                        )
                    if detail is None:
                        detail = checks
                    run_passes.append(all(c.passed for c in checks if not c.skipped))
                golden_results.append(
                    GoldenEvaluationResult(
                        golden_name=golden.label,
                        check_results=detail or [],
                        run_passes=run_passes,
                        transcript=transcript,
                    )
                )

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

    def _run_conversation(
        self,
        app: Callable,
        arity: int,
        simulator: Callable,
        golden: ConversationGolden,
        max_turns: int,
    ) -> tuple[list[dict], bool]:
        system = self.user.system_prompt(golden)
        history: list[dict] = []
        converged = False
        for _ in range(max_turns):
            prompt = system
            if history:
                prompt += "\n\nگفتگو تا این لحظه:\n" + _render(history)
            prompt += "\n\nپیام بعدی کاربر:"
            user_msg = str(simulator(prompt)).strip()
            if DONE in user_msg:
                converged = True
                break
            history.append({"role": "user", "content": user_msg})
            history.append(
                {"role": "assistant",
                 "content": _call_app(app, arity, user_msg, history[:-1])}
            )
        return history, converged

    def _score_conversation(
        self,
        history: list[dict],
        converged: bool,
        golden: ConversationGolden,
        judge: Any,
    ) -> list[CheckResult]:
        transcript = _render(history)
        specs = [
            ("goal", GOAL_EVAL_FA.format(
                transcript=transcript, goal=golden.goal,
                expected=f"نتیجهٔ مطلوب: {golden.expected_outcome}\n"
                if golden.expected_outcome else "",
            ))
        ]
        specs += [
            (f"criterion:{c[:24]}",
             CRITERION_EVAL_FA.format(transcript=transcript, criterion=c))
            for c in golden.criteria
        ]
        judged = run_judge_specs(judge, specs)
        goal = judged[0]
        # a chatty simulator that never emits DONE must not fail a conversation
        # the judge scored as successful — the turn cap merely cut the chat short
        reached = converged or (not goal.skipped and goal.passed)
        if not reached:
            reason = "شبیه‌ساز کاربر به هدف نرسید (سقف نوبت‌ها)."
        elif not converged:
            reason = "سقف نوبت‌ها پر شد ولی داور هدف را برآورده‌شده ارزیابی کرد."
        else:
            reason = None
        out = [CheckResult(check="converged", score=float(reached), passed=reached,
                           reason=reason)]
        out.extend(judged)
        return out
```

## `evaluate(app, n_runs=1, max_turns=None, save_evaluation=False, output_path=None, record=True)`

Simulate each golden's conversation against the app and judge it.

Parameters:

| Name              | Type       | Description                                                                                                                             | Default    |
| ----------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `app`             | `Callable` | The bot under evaluation — fn(message) for stateful bots or fn(message, history).                                                       | *required* |
| `n_runs`          | `int`      | Repeated conversations per golden (default is 1); see AppEvaluationResult.pass_hat_k.                                                   | `1`        |
| `max_turns`       | `int`      | Turn cap override; defaults to each golden's own max_turns.                                                                             | `None`     |
| `save_evaluation` | `bool`     | Flag to save the evaluation result (default is False).                                                                                  | `False`    |
| `output_path`     | `str`      | The output path to save the evaluation result.                                                                                          | `None`     |
| `record`          | `bool`     | Record this run into the local .parsbench store for parsbench view (default is True; also disabled by the PARSBENCH_NO_RECORD env var). | `True`     |

Returns:

| Name                  | Type                  | Description                                   |
| --------------------- | --------------------- | --------------------------------------------- |
| `AppEvaluationResult` | `AppEvaluationResult` | The evaluation result over all conversations. |

Source code in `parsbench/appeval/simulation.py`

```
def evaluate(
    self,
    app: Callable,
    n_runs: int = 1,
    max_turns: int | None = None,
    save_evaluation: bool = False,
    output_path: str | None = None,
    record: bool = True,
) -> AppEvaluationResult:
    """
    Simulate each golden's conversation against the app and judge it.

    Parameters:
        app (Callable): The bot under evaluation — `fn(message)` for
            stateful bots or `fn(message, history)`.
        n_runs (int, optional): Repeated conversations per golden
            (default is 1); see `AppEvaluationResult.pass_hat_k`.
        max_turns (int, optional): Turn cap override; defaults to each
            golden's own `max_turns`.
        save_evaluation (bool, optional): Flag to save the evaluation
            result (default is False).
        output_path (str, optional): The output path to save the
            evaluation result.
        record (bool, optional): Record this run into the local
            `.parsbench` store for `parsbench view` (default is True;
            also disabled by the PARSBENCH_NO_RECORD env var).

    Returns:
        AppEvaluationResult: The evaluation result over all conversations.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1.")
    if save_evaluation and not output_path:
        raise Exception("You should set the output path to save the evaluation.")

    # the user simulator stays at the provider's default temperature —
    # pinning it to 0 would replay near-identical conversations and blind
    # pass_hat_k
    simulator = resolve_model(
        self.simulator_model, "PARSBENCH_SIMULATOR", "PARSBENCH_JUDGE",
        temperature=None,
    )
    if simulator is None:
        raise ValueError(
            "simulator model not configured — pass simulator_model= or set "
            "PARSBENCH_SIMULATOR."
        )
    judge = resolve_model(self.judge, "PARSBENCH_JUDGE")

    recorder = (
        RunRecorder.start(
            kind="simulation",
            app_name=getattr(app, "__name__", type(app).__name__),
            n_goldens=len(self.goldens),
            n_runs=n_runs,
            extra_meta={
                "user": {
                    "style": self.user.style,
                    "persona": self.user.persona,
                    "traps": self.user.traps,
                }
            },
        )
        if record
        else None
    )

    arity = _app_arity(app)
    golden_results = []
    try:
        for golden_index, golden in enumerate(
            tqdm(self.goldens, desc="Simulating conversations")
        ):
            turn_cap = max_turns or golden.max_turns
            run_passes: list[bool] = []
            detail: list[CheckResult] | None = None
            transcript = None
            for run_index in range(n_runs):
                rendered: str | None = None
                run_trace: Trace | None = None
                converged: bool | None = None
                try:
                    history, converged = self._run_conversation(
                        app, arity, simulator, golden, turn_cap
                    )
                except Exception as exc:  # a crashing bot is a finding
                    checks = [
                        CheckResult(
                            check="app_error",
                            passed=False,
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                    ]
                    converged = None
                else:
                    checks = self._score_conversation(
                        history, converged, golden, judge
                    )
                    rendered = _render(history)
                    run_trace = Trace.from_messages(history)
                    if transcript is None:
                        transcript = rendered
                if recorder:
                    recorder.record_event(
                        golden, golden_index, run_index, checks,
                        trace=run_trace, transcript=rendered,
                        converged=converged,
                    )
                if detail is None:
                    detail = checks
                run_passes.append(all(c.passed for c in checks if not c.skipped))
            golden_results.append(
                GoldenEvaluationResult(
                    golden_name=golden.label,
                    check_results=detail or [],
                    run_passes=run_passes,
                    transcript=transcript,
                )
            )

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
```

The simulated user's personality. Editable, composable.

Attributes:

| Name      | Type        | Description                                    |
| --------- | ----------- | ---------------------------------------------- |
| `style`   | `str`       | The speech register (e.g. "محاوره‌ای", "رسمی"). |
| `persona` | `str`       | Extra free-text persona description.           |
| `traps`   | `list[str]` | Keys of TRAPS or free-text instructions.       |

Methods:

| Name            | Description                                       |
| --------------- | ------------------------------------------------- |
| `system_prompt` | Renders the simulator system prompt for a golden. |

Source code in `parsbench/appeval/simulation.py`

```
@dataclass
class PersianUser:
    """
    The simulated user's personality. Editable, composable.

    Attributes:
        style (str): The speech register (e.g. "محاوره‌ای", "رسمی").
        persona (str): Extra free-text persona description.
        traps (list[str]): Keys of TRAPS or free-text instructions.

    Methods:
        system_prompt: Renders the simulator system prompt for a golden.
    """

    style: str = "محاوره‌ای"
    persona: str = ""
    traps: list[str] = field(default_factory=list)

    def system_prompt(self, golden: ConversationGolden) -> str:
        trap_lines = "\n".join(f"- {TRAPS.get(t, t)}" for t in self.traps)
        parts = [
            "تو نقش یک کاربر واقعی ایرانی را بازی می‌کنی که با یک دستیار هوشمند گفتگو می‌کند.",
            f"هدف تو از این گفتگو: {golden.goal}",
            f"موقعیت: {golden.scenario}" if golden.scenario else "",
            f"سبک گفتار: {self.style}. {self.persona}".strip(),
            trap_lines,
            "قواعد: هر بار فقط پیامِ بعدیِ کاربر را بنویس، کوتاه و طبیعی. "
            "به محض این که به هدفت رسیدی یا مطمئن شدی به نتیجه نمی‌رسی، دیگر "
            f"سؤال تازه‌ای نپرس و فقط بنویس: {DONE}",
        ]
        return "\n".join(p for p in parts if p)
```

GoldenGenerator turns the user's own documentation into Golden expectations, ready to feed an AppEvaluator. Generated goldens carry their source chunk as `context`, so faithfulness is judged automatically.

Attributes:

| Name          | Type        | Description                                                                           |
| ------------- | ----------- | ------------------------------------------------------------------------------------- |
| `model`       | \`Model     | Callable                                                                              |
| `registers`   | `list[str]` | Question registers to rotate through (default is formal and colloquial Persian).      |
| `adversarial` | `bool`      | Mix digit scripts, Jalali dates, and Finglish into some questions (default is False). |

Methods:

| Name       | Description                       |
| ---------- | --------------------------------- |
| `generate` | Generates goldens from documents. |

Source code in `parsbench/appeval/generator.py`

```
class GoldenGenerator:
    """
    GoldenGenerator turns the user's own documentation into Golden
    expectations, ready to feed an AppEvaluator. Generated goldens carry
    their source chunk as `context`, so faithfulness is judged automatically.

    Attributes:
        model (Model | Callable | str, optional): The generator LLM (falls
            back to PARSBENCH_GENERATOR, then PARSBENCH_JUDGE env).
        registers (list[str], optional): Question registers to rotate through
            (default is formal and colloquial Persian).
        adversarial (bool): Mix digit scripts, Jalali dates, and Finglish
            into some questions (default is False).

    Methods:
        generate: Generates goldens from documents.
    """

    def __init__(
        self,
        model: Any = None,
        registers: list[str] | None = None,
        adversarial: bool = False,
    ):
        self.model = model
        self.registers = registers or ["رسمی", "محاوره‌ای"]
        self.adversarial = adversarial

    def generate(self, docs, n: int = 20) -> list[Golden]:
        """
        Generate goldens from the given documents.

        Parameters:
            docs (str | Path | list): A file path, glob pattern, directory, or
                a list of those. Text-like files only (.txt, .md, .rst, .html,
                .json).
            n (int, optional): The number of goldens to generate (default is 20).

        Returns:
            list[Golden]: The generated goldens.
        """
        llm = resolve_model(self.model, "PARSBENCH_GENERATOR", "PARSBENCH_JUDGE")
        if llm is None:
            raise ValueError(
                "generator model not configured — pass model= or set PARSBENCH_JUDGE."
            )
        chunks = [c for text in _read_docs(docs) for c in _chunks(text)]
        per_chunk = max(1, -(-n // len(chunks)))
        goldens: list[Golden] = []
        for i, chunk in enumerate(chunks):
            if len(goldens) >= n:
                break
            prompt = TESTGEN_FA.format(
                chunk=chunk,
                count=min(per_chunk, n - len(goldens)),
                register=self.registers[i % len(self.registers)],
                adversarial=_ADVERSARIAL_FA if self.adversarial else "",
            )
            reply = str(llm(prompt))
            match = re.search(r"\[.*\]", reply, re.DOTALL)
            if not match:
                continue
            try:
                rows = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            for row in rows:
                if not isinstance(row, dict) or not row.get("input"):
                    continue
                goldens.append(
                    Golden(
                        input=row["input"],
                        output=row.get("output"),
                        contains=row.get("contains") or [],
                        context=[chunk],
                        tags=["generated"],
                    )
                )
        return goldens[:n]
```

## `generate(docs, n=20)`

Generate goldens from the given documents.

Parameters:

| Name   | Type  | Description                                        | Default |
| ------ | ----- | -------------------------------------------------- | ------- |
| `docs` | \`str | Path                                               | list\`  |
| `n`    | `int` | The number of goldens to generate (default is 20). | `20`    |

Returns:

| Type           | Description                            |
| -------------- | -------------------------------------- |
| `list[Golden]` | list\[Golden\]: The generated goldens. |

Source code in `parsbench/appeval/generator.py`

```
def generate(self, docs, n: int = 20) -> list[Golden]:
    """
    Generate goldens from the given documents.

    Parameters:
        docs (str | Path | list): A file path, glob pattern, directory, or
            a list of those. Text-like files only (.txt, .md, .rst, .html,
            .json).
        n (int, optional): The number of goldens to generate (default is 20).

    Returns:
        list[Golden]: The generated goldens.
    """
    llm = resolve_model(self.model, "PARSBENCH_GENERATOR", "PARSBENCH_JUDGE")
    if llm is None:
        raise ValueError(
            "generator model not configured — pass model= or set PARSBENCH_JUDGE."
        )
    chunks = [c for text in _read_docs(docs) for c in _chunks(text)]
    per_chunk = max(1, -(-n // len(chunks)))
    goldens: list[Golden] = []
    for i, chunk in enumerate(chunks):
        if len(goldens) >= n:
            break
        prompt = TESTGEN_FA.format(
            chunk=chunk,
            count=min(per_chunk, n - len(goldens)),
            register=self.registers[i % len(self.registers)],
            adversarial=_ADVERSARIAL_FA if self.adversarial else "",
        )
        reply = str(llm(prompt))
        match = re.search(r"\[.*\]", reply, re.DOTALL)
        if not match:
            continue
        try:
            rows = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        for row in rows:
            if not isinstance(row, dict) or not row.get("input"):
                continue
            goldens.append(
                Golden(
                    input=row["input"],
                    output=row.get("output"),
                    contains=row.get("contains") or [],
                    context=[chunk],
                    tags=["generated"],
                )
            )
    return goldens[:n]
```

JudgeCalibrator measures how well a judge model agrees with human labels on a labeled sample, so judge scores can be trusted (or fixed) before they are published.

Attributes:

| Name    | Type    | Description |
| ------- | ------- | ----------- |
| `judge` | \`Model | Callable    |

Methods:

| Name        | Description                                                |
| ----------- | ---------------------------------------------------------- |
| `calibrate` | Scores the labeled items and computes agreement and kappa. |

Source code in `parsbench/appeval/calibration.py`

```
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
```

## `calibrate(items, prefer_concurrency=False, n_workers=4)`

Score each labeled item with the judge and compare to the human label.

Parameters:

| Name                 | Type         | Description                                                                                                                                                   | Default    |
| -------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `items`              | `list[dict]` | Items of the form {"golden": Golden(...), "output": "...", "human": True} where the golden triggers at least one judge check (output=, context= or refuses=). | *required* |
| `prefer_concurrency` | `bool`       | Fan judge calls out over a thread pool (default is False); the judge callable must then be thread-safe.                                                       | `False`    |
| `n_workers`          | `int`        | The number of workers for concurrent processing (default is 4).                                                                                               | `4`        |

Returns:

| Name                | Type                | Description                                  |
| ------------------- | ------------------- | -------------------------------------------- |
| `CalibrationResult` | `CalibrationResult` | Agreement, Cohen's kappa, and disagreements. |

Source code in `parsbench/appeval/calibration.py`

```
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
```

The result of calibrating a judge against human labels.

Attributes:

| Name            | Type        | Description                                    |
| --------------- | ----------- | ---------------------------------------------- |
| `n`             | `int`       | The number of labeled items.                   |
| `agreement`     | `float`     | Fraction where judge pass/fail == human label. |
| `kappa`         | `float`     | Cohen's kappa vs human labels.                 |
| `disagreements` | `list[str]` | Readable descriptions for error analysis.      |

Source code in `parsbench/appeval/calibration.py`

```
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
```

Unify codepoints/digits, read ZWNJ as a space, drop thousands commas, collapse whitespace.

Source code in `parsbench/appeval/normalize.py`

```
def normalize(text: str) -> str:
    """Unify codepoints/digits, read ZWNJ as a space, drop thousands commas,
    collapse whitespace."""
    text = str(text).translate(_CHAR_MAP).replace(ZWNJ, " ")
    text = _DIGIT_COMMA.sub("", text)
    return re.sub(r"\s+", " ", text).strip()
```

Substring check that survives digit scripts, ZWNJ and spacing variants.

Source code in `parsbench/appeval/normalize.py`

```
def contains_normalized(haystack: str, needle: str) -> bool:
    """Substring check that survives digit scripts, ZWNJ and spacing variants."""
    return _contains_norm(normalize(haystack), needle)
```

Parse '۲۵۰ هزار تومان' → (2_500_000.0, 'rial'). Returns (value, unit|None).

Source code in `parsbench/appeval/normalize.py`

```
def parse_number(value) -> tuple[float, str | None] | None:
    """Parse '۲۵۰ هزار تومان' → (2_500_000.0, 'rial'). Returns (value, unit|None)."""
    parsed = _parse_amount(value)
    return None if parsed is None else (parsed[0], parsed[1])
```

Source code in `parsbench/appeval/normalize.py`

```
def numbers_equal(a, b) -> bool:
    pa, pb = _parse_amount(a), _parse_amount(b)
    if pa is None or pb is None:
        return False
    (va, ua, _), (vb, ub, _) = pa, pb
    if ua and ub:
        return va == vb
    # unit missing on one side — accept either rial/toman reading
    if ua or ub:
        return va == vb or va == vb * 10 or vb == va * 10
    return va == vb
```

Does the text state this amount, in any unit/scale/digit-script? amount_in('قیمت ۲٬۵۰۰٬۰۰۰ ریال است', '250 هزار تومان') → True.

Source code in `parsbench/appeval/normalize.py`

```
def amount_in(haystack: str, needle) -> bool:
    """Does the text state this amount, in any unit/scale/digit-script?
    amount_in('قیمت ۲٬۵۰۰٬۰۰۰ ریال است', '250 هزار تومان') → True."""
    return _amount_in_norm(normalize(haystack), needle)
```

Parse a date(-time) string to a Gregorian (y, m, d). Year \<1600 → Jalali. The whole string must be the date — ranges and prose return None.

Source code in `parsbench/appeval/normalize.py`

```
def parse_date(value) -> tuple[int, int, int] | None:
    """Parse a date(-time) string to a Gregorian (y, m, d). Year <1600 → Jalali.
    The whole string must be the date — ranges and prose return None."""
    m = _DATE_ONLY.match(normalize(value))
    if not m:
        return None
    return _to_gregorian(int(m.group(1)), int(m.group(2)), int(m.group(3)))
```

Source code in `parsbench/appeval/normalize.py`

```
def dates_equal(a, b) -> bool:
    pa, pb = parse_date(a), parse_date(b)
    return pa is not None and pa == pb
```

Does the text mention this date, in either calendar?

Source code in `parsbench/appeval/normalize.py`

```
def date_in(haystack: str, needle) -> bool:
    """Does the text mention this date, in either calendar?"""
    return _date_in_norm(normalize(haystack), needle)
```

Equivalence chain for text expectations (contains / not_contains): normalized substring, then money-equivalence, then calendar-equivalence. Pass normalized=True when the haystack is already normalize()d.

Source code in `parsbench/appeval/normalize.py`

```
def text_matches(haystack: str, needle, *, normalized: bool = False) -> bool:
    """Equivalence chain for text expectations (contains / not_contains):
    normalized substring, then money-equivalence, then calendar-equivalence.
    Pass normalized=True when the haystack is already normalize()d."""
    h = haystack if normalized else normalize(haystack)
    n = needle if isinstance(needle, str) else str(needle)
    if _contains_norm(h, n):
        return True
    if _amount_in_norm(h, n):
        return True
    return _date_in_norm(h, n)
```

Equivalence chain for tool arguments: date → number → normalized string.

Source code in `parsbench/appeval/normalize.py`

```
def values_equal(a, b) -> bool:
    """Equivalence chain for tool arguments: date → number → normalized string."""
    da, db = parse_date(a), parse_date(b)
    if da or db:
        return da == db
    if numbers_equal(a, b):
        return True
    return normalize(a) == normalize(b)
```
