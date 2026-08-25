# The judge

Deterministic checks always run. Judge checks (`output=`, `context=`,
`refuses=`) need a judge model and **skip gracefully** without one; a suite
with no API key still runs its deterministic checks. Don't let a user think
their judged checks passed when they were skipped: check the printed result.

## Configuration

Environment variables (any OpenAI-compatible gateway works: OpenAI, AvalAI,
OpenRouter, local Ollama):

```bash
export PARSBENCH_JUDGE=gpt-4.1-mini      # model name
export OPENAI_BASE_URL=...
export OPENAI_API_KEY=...
```

Judge-specific credentials win over the `OPENAI_*` ones (useful when the app
under test and the judge use different gateways):

```bash
export PARSBENCH_JUDGE_BASE_URL=...
export PARSBENCH_JUDGE_API_KEY=...
```

`judge=` on the evaluator overrides the environment and accepts three forms:

```python
# 1. model name string, resolved against the env vars above
AppEvaluator(goldens, judge="gpt-4.1-mini")

# 2. any parsbench Model
from parsbench.models import OpenAIModel
AppEvaluator(goldens, judge=OpenAIModel(api_base_url=..., api_secret_key=..., model=...))

# 3. a plain callable, prompt -> str (own client, local model, test stub)
AppEvaluator(goldens, judge=lambda prompt: my_client.complete(prompt))
```

The built-in client retries transient failures and bounds each call.
`PARSBENCH_MAX_RETRIES` (default 5) and `PARSBENCH_TIMEOUT` (default 120
seconds) override.

## Prompts

Judge prompts are authored in Persian (a judge reasoning about Persian in
Persian makes fewer mistakes than one translating on the fly). They're
importable and editable:

```python
from parsbench.appeval import prompts_fa
print(prompts_fa.CORRECTNESS)   # also FAITHFULNESS, REFUSAL
```

The judge's verdict and reasoning land in each `CheckResult.reason`.

## Calibrating the judge

Before trusting judge scores, measure judge-vs-human agreement on a labeled
sample:

```python
from parsbench.appeval import Golden, JudgeCalibrator

calibration = JudgeCalibrator(judge="gpt-4.1-mini").calibrate(
    [
        {"golden": Golden(input="...", output="..."), "output": "...", "human": True},
        {"golden": Golden(input="...", output="..."), "output": "...", "human": False},
        # 30-50 labeled cases is a reasonable start
    ],
    prefer_concurrency=True, n_workers=8,
)
print(calibration)     # agreement rate, Cohen's kappa, readable disagreements
```

Each item is a golden, the app output the judge will score, and the human
verdict. Kappa near zero means the judge agrees with humans no more than
chance; fix the rubric or pick a stronger judge model before publishing
scores.
