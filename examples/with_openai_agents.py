"""Evaluate an OpenAI Agents SDK agent with ParsBench.

    pip install parsbench openai-agents
    export OPENAI_API_KEY=...
    export OPENAI_AGENTS_DISABLE_TRACING=1  # if the key belongs to a non-OpenAI gateway
    python examples/with_openai_agents.py
"""

import os

from agents import Agent, Runner, function_tool

from parsbench.appeval import AppEvaluator, Golden, ToolCall
from parsbench.integrations import openai_agents

MODEL = os.getenv("MODEL", "gpt-4o-mini")

SYSTEM = (
    "تو «پروازیار» هستی، دستیار فارسی‌زبان فروش بلیط هواپیما. "
    "برای یافتن پرواز از ابزار search_flights استفاده کن و حتماً قیمت را به کاربر اعلام کن. "
    "هرگز بدون تأیید صریح کاربر book_flight را صدا نزن."
)


@function_tool
def search_flights(origin: str, dest: str, date: str) -> str:
    """جستجوی پرواز بین دو شهر. تاریخ به صورت YYYY-MM-DD (شمسی یا میلادی)."""
    return f"پرواز {origin} به {dest} در {date}: پرواز PY-101 ساعت ۰۸:۰۰، قیمت ۲٬۵۰۰٬۰۰۰ ریال"


@function_tool
def book_flight(flight_id: str) -> str:
    """رزرو قطعی پرواز — فقط پس از تأیید صریح کاربر."""
    return f"پرواز {flight_id} رزرو شد"


agent = Agent(name="پروازیار", instructions=SYSTEM, model=MODEL,
              tools=[search_flights, book_flight])

golden = Golden(
    name="جستجوی پرواز — تقویم و واحد پول را ParsBench تطبیق می‌دهد",
    input="سلام، یه بلیط از تهران به مشهد برای ۵ مهر ۱۴۰۵ می‌خوام. قیمتش چنده؟",
    tools=[ToolCall("search_flights", date="2026-09-27")],  # مدل شمسی صدا می‌زند — یکسان است
    contains=["250 هزار تومان"],                            # مدل ریال می‌گوید — یکسان است
    forbidden_tools=["book_flight"],
)

run_result = Runner.run_sync(agent, golden.input)
trace = openai_agents.to_trace(run_result)

print(AppEvaluator(goldens=[golden]).score_traces([trace]))
