# CI, regression tracking, and the viewer

## pytest

`assert_passed()` raises with the failing checks, so goldens are plain
pytest cases:

```python
import pytest
from parsbench.appeval import AppEvaluator

@pytest.mark.parametrize("golden", GOLDENS, ids=lambda g: g.label)
def test_bot(golden):
    AppEvaluator(goldens=[golden]).evaluate(bot).assert_passed()
```

Run with `pytest`, or with `parsbench test`, a thin pytest wrapper that
passes arguments straight through. It needs pytest installed
(`pip install pytest`, or the `parsbench[test]` extra which pins it):

```bash
pip install 'parsbench[test]'
parsbench test tests/ -k booking -x
```

In CI set `PARSBENCH_NO_RECORD=1` so runs don't write into the local
`.parsbench/` store. A complete working file: `examples/ci_with_pytest.py`.

## Flaky agents: n_runs and pass^k

Agents are sampled; a golden that passes once may fail next run. Repeat and
measure consistency:

```python
result = evaluator.evaluate(bot, n_runs=5)
result.pass_hat_k()      # pass^k: probability all k sampled runs pass
result.pass_hat_k(3)     # same, for a subsample of k=3
```

`pass_hat_k` is the unbiased pass^k estimator: per golden, the probability
that k randomly chosen runs out of n_runs all pass, averaged over goldens.
No argument means k = n_runs. A bot with `average_score` 0.9 but pass^5 of
0.4 fails somebody every day; quote the second number.

## Concurrency

```python
result = evaluator.evaluate(bot, prefer_concurrency=True, n_workers=8)
```

The app and judge callables must then be thread-safe; most stateful bots
are not, so this is off by default. `JudgeCalibrator.calibrate` takes the
same two arguments.

## Tracking regressions between runs

```python
result.to_pandas()                       # one row per check
result.save("out/")                      # writes out/app_evaluation.jsonl
result = evaluator.evaluate(bot, save_evaluation=True, output_path="out/")  # same
```

`diff()` compares against a saved baseline and prints how each check's mean
score moved (unchanged checks print nothing, returns None):

```python
result.diff("baseline/app_evaluation.jsonl")
# contains: 0.90 -> 0.80 (-0.10)
```

Practical CI recipe: save the result on every main-branch run; in PR builds
diff against the latest main artifact before `assert_passed()`. For a
per-golden view of what regressed, use the viewer's compare page.

## The viewer: parsbench view

Every `evaluate()`/`score_traces()` call records its run (full traces) into
a project-local, self-gitignored `.parsbench/` store. The viewer is a local
web UI over it, zero extra dependencies:

```bash
parsbench view                     # serves 127.0.0.1:1404, opens the browser
parsbench view path/to/project     # a .parsbench store or a dir containing one
parsbench view --port 8080 --host 0.0.0.0 --no-open
```

It shows live runs, a goldens-by-checks matrix (failure-first, judge reasons
in Persian one click away), trace detail with per-run tabs when n_runs > 1,
RTL simulation replay, run-vs-run compare, and JSON/CSV/Markdown export.

Recording is on by default; turn off per call with
`evaluate(bot, record=False)` or globally with `PARSBENCH_NO_RECORD=1`.
`.parsbench/` is safe to delete; only the viewer reads it.
