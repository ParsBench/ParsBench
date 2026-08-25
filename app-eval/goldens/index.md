# Goldens and checks

A `Golden` is one expectation: the user message you send, plus what a good response looks like. Only `input` is required. Every other field you fill switches on its corresponding check. Leave a field empty and that check doesn't run.

```
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

| Field                                    | Check             | Passes when                                                         |
| ---------------------------------------- | ----------------- | ------------------------------------------------------------------- |
| `output=`                                | `correctness`     | a judge scores the answer consistent with the reference             |
| `contains=[...]`                         | `contains`        | each needle appears, normalized, money- and date-equivalent         |
| `not_contains=[...]`                     | `not_contains`    | no needle appears (same equivalences)                               |
| `format=Model`                           | `format`          | the output parses and validates against a pydantic schema           |
| `tools=[ToolCall(...)]`                  | `tools`           | expected calls happened; args compared date, then number, then text |
| `forbidden_tools=[...]`                  | `forbidden_tools` | none of these tools were called                                     |
| `context=[...]`                          | `faithfulness`    | a judge finds every claim supported by the context                  |
| `refuses=True`                           | `refusal`         | a judge confirms the request was declined                           |
| `max_steps` / `max_latency` / `max_cost` | `budget:*`        | the trace stayed within the budget                                  |
| `check=fn`                               | `custom`          | your `Trace -> bool \| float` returns truthy / 1.0                  |

Notes on individual fields:

- `name=` labels the golden in reports and in `parsbench view`. Without it, the first 40 characters of the input are used.
- List fields accept a bare string, so `contains="۲۵۰ هزار تومان"` works.
- `tools=` also accepts plain dicts. `ToolCall("search", date="...")` is shorthand for `ToolCall(name="search", arguments={"date": "..."})`. A golden's expected call matches an observed call when the names match and every expected argument is present and equivalent. Extra observed arguments are allowed.
- `check=` is the escape hatch. It receives the full `Trace` (messages, tool calls, latency, cost) and returns a bool or a 0..1 score. Use it for anything the built-in checks don't cover.
- Goldens can also be plain dicts. `AppEvaluator` promotes them via `Golden.from_dict`, which accepts `in`/`out` as short aliases for `input`/`output`. Handy when goldens live in a JSON file.

`output=`, `context=`, and `refuses=` are judged by an LLM and need a [judge model](https://parsbench.github.io/ParsBench/app-eval/judge/index.md). The rest are deterministic and run with no API key.

## Filtering checks with metrics=

By default every check implied by the filled fields runs. `metrics=` restricts or re-modes them:

```
AppEvaluator(goldens, metrics=["tools:strict", "contains"])
```

- Only the named checks run; here, tool calls and `contains`.
- `tools:strict` fails on any tool call that wasn't expected, not just on missing ones.
- Unknown names raise a `ValueError` instead of silently passing. Golden field names (`max_steps`, `refuses`, ...) work as aliases for their check names.

## Reading results

`evaluate()` and `score_traces()` return an `AppEvaluationResult`:

```
result = evaluator.evaluate(my_bot)

result.passed              # True when every check on every golden passed
result.average_score       # mean score over all checks
result.score("contains")   # mean score for one check
print(result)              # readable failure-first summary
```

Each golden's entry holds a `CheckResult` per check with `name`, `passed`, `score`, and `reason`; judge checks carry the judge's reasoning in Persian. `result.to_pandas()` gives one row per check for analysis, and `result.assert_passed()` raises with the failing checks, which makes goldens [plain pytest cases](https://parsbench.github.io/ParsBench/app-eval/ci/index.md).
