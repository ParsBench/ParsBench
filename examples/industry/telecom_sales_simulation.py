"""Telecom: simulate Iranian users against a package-sales bot.

    pip install parsbench openai
    export OPENAI_API_KEY=...   # optional: OPENAI_BASE_URL for any compatible gateway
    python examples/industry/telecom_sales_simulation.py

What this demonstrates for an operator/ISP product:
- multi-turn simulation: an LLM plays a colloquial Iranian user who quotes
  prices in toman (while your system thinks in rials) and drifts into Finglish
- goal + criteria judging over the finished conversation, in Persian
- the transcript comes back for error analysis
"""

import os

from openai import OpenAI

from parsbench.appeval import ConversationGolden, PersianUser, SimulationEvaluator

MODEL = os.getenv("MODEL", "gpt-4o-mini")
client = OpenAI()

SYSTEM = (
    "تو دستیار فروش اپراتور اینترنت «نت‌یار» هستی. فقط دو بسته داری: "
    "یک‌ماههٔ ۵۰ گیگ به قیمت ۲٬۵۰۰٬۰۰۰ ریال و سه‌ماههٔ ۱۵۰ گیگ به قیمت ۶٬۰۰۰٬۰۰۰ ریال. "
    "فعال‌سازی با ارسال عدد بسته به ۷۷۷. کوتاه، دقیق و مؤدبانه پاسخ بده."
)


def sales_bot(message, history):
    msgs = [{"role": "system", "content": SYSTEM}, *history,
            {"role": "user", "content": message}]
    return client.chat.completions.create(model=MODEL, messages=msgs).choices[0].message.content


evaluator = SimulationEvaluator(
    goldens=[ConversationGolden(
        name="خرید بستهٔ یک‌ماهه",
        goal="خرید بستهٔ اینترنت یک‌ماهه و دانستن قیمت دقیق و روش فعال‌سازی آن",
        expected_outcome="کاربر قیمت بستهٔ یک‌ماهه و روش فعال‌سازی (ارسال عدد به ۷۷۷) را بداند",
        criteria=[
            "قیمت بسته به‌روشنی اعلام شود",
            "لحن مؤدب بماند حتی وقتی کاربر فینگلیش می‌نویسد",
        ],
        max_turns=5,
    )],
    user=PersianUser(style="محاوره‌ای",
                     traps=["toman_rial_confusion", "finglish_switch"]),
    simulator_model=MODEL,   # the user simulator (PARSBENCH_SIMULATOR overrides)
    judge=MODEL,             # the conversation judge (PARSBENCH_JUDGE overrides)
)
result = evaluator.evaluate(sales_bot)
print(result)
print("\n--- متن گفتگو ---")
print(result.golden_results[0].transcript)
