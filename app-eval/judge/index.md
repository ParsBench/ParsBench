# The judge

Deterministic checks always run. Judge checks (`output=`, `context=`, `refuses=`) need a judge model, and they **skip gracefully** without one, so a suite with no API key still runs its deterministic checks.

## Configuration

The quickest setup is environment variables:

```
export PARSBENCH_JUDGE=gpt-4.1-mini      # any OpenAI-compatible model name
export OPENAI_BASE_URL=...               # AvalAI, OpenRouter, Ollama, ...
export OPENAI_API_KEY=...
```

When the judge should use different credentials than your app (common when the app under test and the judge run through different gateways), use the judge-specific variables. They win over the `OPENAI_*` ones:

```
export PARSBENCH_JUDGE_BASE_URL=...
export PARSBENCH_JUDGE_API_KEY=...
```

`judge=` on the evaluator overrides the environment and accepts three forms:

```
# 1. a model name string, resolved against the env vars above
AppEvaluator(goldens, judge="gpt-4.1-mini")

# 2. any parsbench Model
from parsbench.models import OpenAIModel
AppEvaluator(goldens, judge=OpenAIModel(api_base_url=..., api_secret_key=..., model=...))

# 3. a plain callable, prompt -> str  (your own client, a local model, a stub in tests)
AppEvaluator(goldens, judge=lambda prompt: my_client.complete(prompt))
```

The built-in client retries transient failures and bounds each call. `PARSBENCH_MAX_RETRIES` and `PARSBENCH_TIMEOUT` override the defaults of 5 retries and 120 seconds.

## The prompts

Judge prompts are authored in Persian, because a judge reasoning about Persian text in Persian makes fewer normalization mistakes than one translating on the fly. They're importable, so you can read or edit the rubrics:

```
from parsbench.appeval import prompts_fa
print(prompts_fa.CORRECTNESS)
```

The judge's verdict and its reasoning land in each `CheckResult.reason`, and `parsbench view` shows them one click away from the checks matrix.

## Calibrating the judge

An LLM judge is a measurement instrument, and you should know its error rate before you trust it. `JudgeCalibrator` measures judge-vs-human agreement on a labeled sample:

```
from parsbench.appeval import Golden, JudgeCalibrator

calibration = JudgeCalibrator(judge="gpt-4.1-mini").calibrate(
    [
        {"golden": Golden(input="...", output="..."), "output": "...", "human": True},
        {"golden": Golden(input="...", output="..."), "output": "...", "human": False},
        # ... 30-50 labeled cases is a reasonable start
    ],
    prefer_concurrency=True, n_workers=8,
)
print(calibration)     # agreement rate, Cohen's kappa, readable disagreements
```

Each item is a golden, the app output the judge will score, and your human verdict. The printed report lists every disagreement so you can see whether the judge is too strict, too lenient, or confused by a specific phrasing. Kappa near zero means the judge agrees with humans no more than chance would; fix the rubric or pick a stronger judge model before publishing scores.
