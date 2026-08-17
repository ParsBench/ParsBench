"""Evaluate a plain OpenAI SDK tool-calling bot with ParsBench.

    pip install parsbench openai
    export OPENAI_API_KEY=...   # optional: OPENAI_BASE_URL for any compatible gateway
    python examples/with_openai_sdk.py
"""

import json
import os

from openai import OpenAI

from parsbench.appeval import AppEvaluator, Golden, ToolCall

MODEL = os.getenv("MODEL", "gpt-4o-mini")
client = OpenAI()

SYSTEM = (
    "تو «پروازیار» هستی، دستیار فارسی‌زبان فروش بلیط هواپیما. "
    "برای یافتن پرواز از ابزار search_flights استفاده کن و حتماً قیمت را به کاربر اعلام کن. "
    "هرگز بدون تأیید صریح کاربر book_flight را صدا نزن."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "search_flights",
        "description": "جستجوی پرواز بین دو شهر. تاریخ به صورت YYYY-MM-DD (شمسی یا میلادی).",
        "parameters": {"type": "object", "properties": {
            "origin": {"type": "string"}, "dest": {"type": "string"},
            "date": {"type": "string"}}, "required": ["origin", "dest", "date"]}}},
    {"type": "function", "function": {
        "name": "book_flight",
        "description": "رزرو قطعی پرواز — فقط پس از تأیید صریح کاربر.",
        "parameters": {"type": "object", "properties": {
            "flight_id": {"type": "string"}}, "required": ["flight_id"]}}},
]


def search_flights(origin, dest, date):
    return f"پرواز {origin} به {dest} در {date}: پرواز PY-101 ساعت ۰۸:۰۰، قیمت ۲٬۵۰۰٬۰۰۰ ریال"


def book_flight(flight_id):
    return f"پرواز {flight_id} رزرو شد"


def bot(user_message):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_message}]
    while True:
        msg = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS).choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return messages  # the OpenAI message list IS a parsbench trace
        for tc in msg.tool_calls:
            fn = {"search_flights": search_flights, "book_flight": book_flight}[tc.function.name]
            result = fn(**json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})


evaluator = AppEvaluator(goldens=[
    Golden(
        name="جستجوی پرواز — تقویم و واحد پول را ParsBench تطبیق می‌دهد",
        input="سلام، یه بلیط از تهران به مشهد برای ۵ مهر ۱۴۰۵ می‌خوام. قیمتش چنده؟",
        tools=[ToolCall("search_flights", date="2026-09-27")],  # مدل شمسی صدا می‌زند — یکسان است
        contains=["250 هزار تومان"],                            # مدل ریال می‌گوید — یکسان است
        forbidden_tools=["book_flight"],
    ),
])
print(evaluator.evaluate(bot))
