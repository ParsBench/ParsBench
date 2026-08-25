# Multi-turn simulation

Single-turn goldens catch a lot, but real users don't speak in goldens. They open with taarof, quote prices in the wrong unit, switch to Finglish mid-conversation, and change their mind. `SimulationEvaluator` drives your bot with an LLM playing an Iranian user, then judges the finished conversation against the goal.

```
from parsbench.appeval import SimulationEvaluator

evaluator = SimulationEvaluator(
    goal="خرید بسته اینترنت یک‌ماهه و دانستن قیمت آن",
    user="محاوره‌ای+toman_rial_confusion+finglish_switch",
    criteria=["قیمت به کاربر اعلام شود"],
    simulator_model="gpt-4.1-mini",           # the user simulator
    judge="gpt-4.1-mini",
)
result = evaluator.evaluate(my_bot)           # fn(message) or fn(message, history)
```

Your bot is a callable again. `fn(message)` works for stateful bots that track their own history; `fn(message, history)` receives the conversation so far as OpenAI-format dicts.

Simulation needs two models: the simulator (falls back to the `PARSBENCH_SIMULATOR` env var, then `PARSBENCH_JUDGE`) and the [judge](https://parsbench.github.io/ParsBench/app-eval/judge/index.md) (`PARSBENCH_JUDGE`).

## The simulated user

`user=` is a string mixing a register with trap names, joined by `+`. Free text in the string becomes extra instructions for the simulated user, so `"محاوره‌ای+typos+اهل اصفهان است"` works. The named traps live in `parsbench.appeval.TRAPS`:

| Trap                   | The simulated user will                                                  |
| ---------------------- | ------------------------------------------------------------------------ |
| `taarof_opening`       | open with taarof and hold back the real request in the first message     |
| `toman_rial_confusion` | quote amounts in toman even when the system looks rial-based             |
| `jalali_date`          | give dates in Jalali with Persian month names («۵ مهر»)                  |
| `finglish_switch`      | write some mid-conversation messages in Finglish («merci, hamin khoobe») |
| `typos`                | make natural typos now and then                                          |
| `impatient`            | complain when answers are slow or vague                                  |
| `iran_formats`         | use Iranian phone/address formats (۰۹۱۲…، خیابان/کوچه/پلاک)              |

For finer control, pass a `PersianUser(style=..., persona=..., traps=[...])` object instead of the string.

## Multiple scenarios

`goal=` is shorthand for a single scenario. A real suite is a list of `ConversationGolden` objects:

```
from parsbench.appeval import ConversationGolden, SimulationEvaluator

evaluator = SimulationEvaluator(
    goldens=[
        ConversationGolden(
            goal="خرید بسته اینترنت یک‌ماهه",
            scenario="کاربر قبلاً یک بسته دارد که هفتهٔ بعد منقضی می‌شود.",
            expected_outcome="بسته مناسب پیشنهاد و قیمت اعلام شود.",
            criteria=["قیمت به کاربر اعلام شود", "بدون تأیید کاربر خریدی انجام نشود"],
            max_turns=8,
        ),
        ConversationGolden(goal="لغو اشتراک", criteria=["فرایند لغو کامل توضیح داده شود"]),
    ],
    user="محاوره‌ای+impatient",
)
```

Each criterion is judged as its own check, so the result shows exactly which behavior failed. `max_turns` caps the conversation per golden; `evaluate(app, max_turns=...)` overrides it for a run. Hitting the turn cap does not fail a conversation the goal judge scored as successful. A chatty simulator that never stops talking shouldn't punish the app.

## Reading the run

The result is the same `AppEvaluationResult` as everywhere else, so `assert_passed()`, `to_pandas()`, and `n_runs=` for [consistency scoring](https://parsbench.github.io/ParsBench/app-eval/ci/index.md) all apply. In [`parsbench view`](https://parsbench.github.io/ParsBench/app-eval/viewer/index.md), simulation runs get a replay page that shows the conversation as an RTL chat with the goal and active traps alongside.
