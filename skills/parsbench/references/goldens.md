# Goldens and checks

A `Golden` is one expectation: the user message plus what a good response
looks like. Only `input` is required. Every filled field switches on its
check; empty fields don't run.

```python
from pydantic import BaseModel
from parsbench.appeval import AppEvaluator, Golden, ToolCall

class FlightAnswer(BaseModel):
    flight_no: str
    price_toman: int

golden = Golden(
    input="بلیط تهران-مشهد برای ۵ مهر ۱۴۰۵ می‌خوام. قیمتش چنده؟",
    name="flight search, Jalali date",
    output="پرواز ساعت ۸ صبح به قیمت ۲۵۰ هزار تومان موجود است.",
    contains=["250 هزار تومان"],
    not_contains=["رزرو شد"],
    tools=[ToolCall("search_flights", date="2026-09-27")],
    forbidden_tools=["book_flight"],
    max_steps=4,
    tags=["booking"],
)
```

## Field reference

| Field | Check | Passes when |
|---|---|---|
| `output=` | `correctness` | a judge scores the answer consistent with the reference |
| `contains=[...]` | `contains` | each needle appears, normalized, money- and date-equivalent |
| `not_contains=[...]` | `not_contains` | no needle appears (same equivalences) |
| `format=Model` | `format` | the output parses and validates against a pydantic schema |
| `tools=[ToolCall(...)]` | `tools` | expected calls happened; args compared date, then number, then text |
| `forbidden_tools=[...]` | `forbidden_tools` | none of these tools were called |
| `context=[...]` | `faithfulness` | a judge finds every claim supported by the context |
| `refuses=True` | `refusal` | a judge confirms the request was declined |
| `max_steps` / `max_latency` / `max_cost` | `budget:*` | the trace stayed within the budget |
| `check=fn` | `custom` | your `Trace -> bool \| float` returns truthy / 1.0 |

Notes:

- `name=` labels the golden in reports and the viewer; without it the first
  40 characters of the input are used. `golden.label` returns whichever
  applies (useful as a pytest id).
- List fields accept a bare string: `contains="۲۵۰ هزار تومان"` works.
- `ToolCall("search", date="...")` is shorthand for
  `ToolCall(name="search", arguments={"date": "..."})`. Plain dicts also
  work. An expected call matches an observed call when names match and every
  expected argument is present and equivalent; extra observed arguments are
  allowed.
- `check=` is the escape hatch: it receives the full `Trace` (messages, tool
  calls, latency, cost) and returns a bool or 0..1 score.
- Goldens can be plain dicts; `AppEvaluator` promotes them via
  `Golden.from_dict`, which accepts `in`/`out` as aliases for
  `input`/`output`. Handy for goldens stored in JSON.
- `output=`, `context=`, and `refuses=` need a judge model (see
  `judge.md`); the rest are deterministic and run with no API key.

## Filtering checks with metrics=

```python
AppEvaluator(goldens, metrics=["tools:strict", "contains"])
```

- Only the named checks run.
- `tools:strict` fails on any unexpected tool call, not just missing ones.
- Unknown names raise `ValueError` instead of silently passing. Golden field
  names (`max_steps`, `refuses`, ...) work as aliases for check names.

## Reading results

`evaluate()` and `score_traces()` return an `AppEvaluationResult`:

```python
result.passed              # True when every check on every golden passed
result.average_score       # mean score over all checks
result.score("contains")   # mean score for one check
print(result)              # readable failure-first summary
result.to_pandas()         # one row per check
result.assert_passed()     # raises with failing checks (pytest-ready)
```

Each golden's entry holds a `CheckResult` per check with `check` (the check
name), `passed`, `score`, `reason`, and `skipped`; judge checks carry the
judge's reasoning in Persian. `skipped=True` is how you tell "no judge
configured" apart from a failure.

## Generating goldens from the user's docs

`GoldenGenerator` bootstraps a suite from a product FAQ, knowledge base, or
policy pages:

```python
from parsbench.appeval import AppEvaluator, GoldenGenerator

goldens = GoldenGenerator(model="gpt-4.1-mini", adversarial=True).generate("kb/", n=30)
result = AppEvaluator(goldens).evaluate(my_bot)
```

- `generate()` accepts a file path, glob, directory, or a list of those.
  Text-like files only (`.txt`, `.md`, `.rst`, `.html`, `.json`).
- Questions rotate through registers (formal and colloquial Persian by
  default; override with `registers=[...]`). `adversarial=True` mixes digit
  scripts, Jalali dates, and Finglish into some questions.
- Each generated golden carries its source chunk as `context=`, so the
  faithfulness check runs automatically.
- Model resolution: `model=` arg, then `PARSBENCH_GENERATOR`, then
  `PARSBENCH_JUDGE`. Generation raises without a model (unlike judge checks,
  which skip).

Generated goldens need human review. A practical loop (`Golden` is a
dataclass, not a pydantic model, so use `dataclasses.asdict`):

```python
import json
from dataclasses import asdict

json.dump([asdict(g) for g in goldens], f, ensure_ascii=False, indent=2, default=str)
```

Let the user edit the JSON, then load it back with
`AppEvaluator(goldens=json.load(f))` (dicts are accepted directly).
