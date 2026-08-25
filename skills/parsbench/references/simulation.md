# Multi-turn simulation

`SimulationEvaluator` drives the bot with an LLM playing an Iranian user
(taarof, wrong currency units, Finglish mid-conversation), then judges the
finished conversation against a goal.

```python
from parsbench.appeval import SimulationEvaluator

evaluator = SimulationEvaluator(
    goal="خرید بسته اینترنت یک‌ماهه و دانستن قیمت آن",
    user="محاوره‌ای+toman_rial_confusion+finglish_switch",
    criteria=["قیمت به کاربر اعلام شود"],
    simulator_model="gpt-4.1-mini",
    judge="gpt-4.1-mini",
)
result = evaluator.evaluate(my_bot)
```

The bot is a callable: `fn(message)` for stateful bots that track their own
history, or `fn(message, history)` to receive the conversation so far as
OpenAI-format dicts.

Two models are needed: the simulator (falls back to `PARSBENCH_SIMULATOR`,
then `PARSBENCH_JUDGE`) and the judge (`PARSBENCH_JUDGE`).

## The simulated user

`user=` mixes a register with trap names joined by `+`. Free text in the
string becomes extra instructions, so `"محاوره‌ای+typos+اهل اصفهان است"`
works. Named traps in `parsbench.appeval.TRAPS`:

| Trap | The simulated user will |
|---|---|
| `taarof_opening` | open with taarof and hold back the real request in the first message |
| `toman_rial_confusion` | quote amounts in toman even when the system looks rial-based |
| `jalali_date` | give dates in Jalali with Persian month names («۵ مهر») |
| `finglish_switch` | write some messages in Finglish («merci, hamin khoobe») |
| `typos` | make natural typos now and then |
| `impatient` | complain when answers are slow or vague |
| `iran_formats` | use Iranian phone/address formats (۰۹۱۲…، خیابان/کوچه/پلاک) |

For finer control pass a `PersianUser(style=..., persona=..., traps=[...])`
object instead of the string.

## Multiple scenarios

`goal=` is shorthand for one scenario. A real suite is a list of
`ConversationGolden` objects:

```python
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

- Each criterion is judged as its own check, so results show exactly which
  behavior failed.
- `max_turns` caps the conversation per golden; `evaluate(app, max_turns=...)`
  overrides per run. Hitting the cap does not fail a conversation the goal
  judge scored as successful.

The result is the same `AppEvaluationResult` as everywhere else:
`assert_passed()`, `to_pandas()`, and `n_runs=` all apply. In
`parsbench view`, simulation runs get an RTL chat replay with the goal and
active traps alongside.
