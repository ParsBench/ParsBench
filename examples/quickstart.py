"""ParsBench quickstart — no framework, no API key. Run: python examples/quickstart.py

Your app is any function that takes a user message and returns a string,
a Trace, or an OpenAI-format message list (the lingua franca).
"""

from parsbench.appeval import AppEvaluator, Golden, ToolCall


# Stand-in for your app. Note what it does: calls the tool with a *Jalali*
# date and quotes the price in *rials* with Persian digits.
def my_bot(message):
    return [
        {"role": "user", "content": message},
        {"role": "assistant", "tool_calls": [
            {"id": "1", "function": {"name": "search_flights",
                                     "arguments": '{"origin": "THR", "dest": "MHD", "date": "1405-07-05"}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "پرواز PY-101 ساعت ۰۸:۰۰"},
        {"role": "assistant", "content": "پرواز PY-101 ساعت ۸ صبح موجود است، قیمت ۲٬۵۰۰٬۰۰۰ ریال"},
    ]


evaluator = AppEvaluator(goldens=[
    Golden(
        name="جستجوی پرواز — تقویم و واحد پول را ParsBench تطبیق می‌دهد",
        input="سلام، یه بلیط از تهران به مشهد برای ۵ مهر ۱۴۰۵ می‌خوام. قیمتش چنده؟",
        # golden says Gregorian, the bot calls Jalali — same day, passes
        tools=[ToolCall("search_flights", date="2026-09-27")],
        # golden says toman + Latin digits, the bot says rial + Persian digits — same money, passes
        contains=["250 هزار تومان"],
        # must not book before the user confirms
        forbidden_tools=["book_flight"],
        max_steps=3,
    ),
    Golden(
        name="این یکی عمداً رد می‌شود — دلیل شکست را ببینید",
        input="بلیط برای ۶ مهر",
        tools=[ToolCall("search_flights", date="1405-07-06")],  # the bot books the 5th
    ),
])
print(evaluator.evaluate(my_bot))
