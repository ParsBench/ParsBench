# CI and regression tracking

The point of goldens is that they run on every commit. `assert_passed()` raises with the failing checks, which makes goldens plain pytest cases with readable failure messages:

```
import pytest
from parsbench.appeval import AppEvaluator

@pytest.mark.parametrize("golden", GOLDENS, ids=lambda g: g.label)
def test_bot(golden):
    AppEvaluator(goldens=[golden]).evaluate(bot).assert_passed()
```

Run them with `pytest`, or with `parsbench test`, a thin pytest wrapper that ships in the `parsbench[test]` extra and passes its arguments straight through:

```
pip install 'parsbench[test]'
parsbench test tests/ -k booking -x
```

A complete working file is [`examples/ci_with_pytest.py`](https://github.com/ParsBench/ParsBench/blob/main/examples/ci_with_pytest.py). In CI you'll usually also set `PARSBENCH_NO_RECORD=1` so runs don't write into the [local run store](https://parsbench.github.io/ParsBench/app-eval/viewer/index.md).

## Flaky agents: n_runs and pass^k

Agents are sampled, so a golden that passes once may fail the next run. Repeat each golden and measure consistency instead of luck:

```
result = evaluator.evaluate(bot, n_runs=5)
result.pass_hat_k()      # pass^k: probability all k sampled runs pass
result.pass_hat_k(3)     # same, for a subsample of k=3
```

`pass_hat_k` is the unbiased pass^k estimator: for each golden, the probability that `k` randomly chosen runs out of `n_runs` all pass, averaged over goldens. `pass_hat_k()` with no argument uses `k = n_runs`. A bot with `average_score` 0.9 but pass^5 of 0.4 works most of the time and fails somebody every day; the second number is the one your support team feels.

## Concurrency

Independent goldens fan out over threads:

```
result = evaluator.evaluate(bot, prefer_concurrency=True, n_workers=8)
```

Your app and judge callables must then be thread-safe. Most stateful bots are not, which is why this is off by default. `JudgeCalibrator.calibrate` takes the same two arguments.

## Tracking regressions between runs

Results are plain dataclasses with the same conveniences as benchmark results:

```
result.to_pandas()                       # one row per check
result.save("out/")                      # writes out/app_evaluation.jsonl
result = evaluator.evaluate(bot, save_evaluation=True, output_path="out/")  # same
```

`diff()` compares against a saved baseline and prints how each check's mean score moved:

```
result.diff("baseline/app_evaluation.jsonl")
# contains: 0.90 -> 0.80 (-0.10)
```

Unchanged checks print nothing. For a per-golden view of what regressed, use the [viewer's compare page](https://parsbench.github.io/ParsBench/app-eval/viewer/index.md).

A practical CI recipe: save the result on every main-branch run, and in PR builds diff against the latest main artifact before `assert_passed()`, so the log answers "what moved" and not just "something broke".

## Exporting results

`result.to_langfuse()` pushes per-check scores to a Langfuse instance, so eval scores land next to your production traces:

```
export LANGFUSE_HOST=...
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
```

```
result.to_langfuse()          # returns the created trace's id
```

For anything else, `result.to_dict()` / `to_pandas()` serialize the whole run, and the [viewer](https://parsbench.github.io/ParsBench/app-eval/viewer/index.md) exports JSON, CSV, and Markdown reports.
