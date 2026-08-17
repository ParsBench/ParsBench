"""Evaluate an Agno agent with ParsBench.

    pip install parsbench agno openai
    export OPENAI_API_KEY=...
    python examples/with_agno.py
"""

import os

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from parsbench.appeval import AppEvaluator, Golden, ToolCall
from parsbench.integrations import agno as pb_agno

MODEL = os.getenv("MODEL", "gpt-4o-mini")

SYSTEM = (
    "تو «پروازیار» هستی، دستیار فارسی‌زبان فروش بلیط هواپیما. "
    "برای یافتن پرواز از ابزار search_flights استفاده کن و حتماً قیمت را به کاربر اعلام کن. "
    "هرگز بدون تأیید صریح کاربر book_flight را صدا نزن."
)


def search_flights(origin: str, dest: str, date: str) -> str:
    """جستجوی پرواز بین دو شهر. تاریخ به صورت YYYY-MM-DD (شمسی یا میلادی)."""
    return f"پرواز {origin} به {dest} در {date}: پرواز PY-101 ساعت ۰۸:۰۰، قیمت ۲٬۵۰۰٬۰۰۰ ریال"


def book_flight(flight_id: str) -> str:
    """رزرو قطعی پرواز — فقط پس از تأیید صریح کاربر."""
    return f"پرواز {flight_id} رزرو شد"


agent = Agent(model=OpenAIChat(id=MODEL), instructions=SYSTEM,
              tools=[search_flights, book_flight], markdown=False)

golden = Golden(
    name="جستجوی پرواز — تقویم و واحد پول را ParsBench تطبیق می‌دهد",
    input="سلام، یه بلیط از تهران به مشهد برای ۵ مهر ۱۴۰۵ می‌خوام. قیمتش چنده؟",
    tools=[ToolCall("search_flights", date="2026-09-27")],  # مدل شمسی صدا می‌زند — یکسان است
    contains=["250 هزار تومان"],                            # مدل ریال می‌گوید — یکسان است
    forbidden_tools=["book_flight"],
)

run = agent.run(golden.input)
trace = pb_agno.to_trace(run)

print(AppEvaluator(goldens=[golden]).score_traces([trace]))
