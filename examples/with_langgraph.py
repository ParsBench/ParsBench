"""Evaluate a LangGraph (LangChain) ReAct agent with ParsBench.

    pip install parsbench langchain langchain-openai
    export OPENAI_API_KEY=...
    python examples/with_langgraph.py
"""

import os

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from parsbench.appeval import AppEvaluator, Golden, ToolCall
from parsbench.integrations import langgraph as pb_langgraph

MODEL = os.getenv("MODEL", "gpt-4o-mini")

SYSTEM = (
    "تو «پروازیار» هستی، دستیار فارسی‌زبان فروش بلیط هواپیما. "
    "برای یافتن پرواز از ابزار search_flights استفاده کن و حتماً قیمت را به کاربر اعلام کن. "
    "هرگز بدون تأیید صریح کاربر book_flight را صدا نزن."
)


@tool
def search_flights(origin: str, dest: str, date: str) -> str:
    """جستجوی پرواز بین دو شهر. تاریخ به صورت YYYY-MM-DD (شمسی یا میلادی)."""
    return f"پرواز {origin} به {dest} در {date}: پرواز PY-101 ساعت ۰۸:۰۰، قیمت ۲٬۵۰۰٬۰۰۰ ریال"


@tool
def book_flight(flight_id: str) -> str:
    """رزرو قطعی پرواز — فقط پس از تأیید صریح کاربر."""
    return f"پرواز {flight_id} رزرو شد"


graph = create_agent(ChatOpenAI(model=MODEL),
                     tools=[search_flights, book_flight], system_prompt=SYSTEM)

golden = Golden(
    name="جستجوی پرواز — تقویم و واحد پول را ParsBench تطبیق می‌دهد",
    input="سلام، یه بلیط از تهران به مشهد برای ۵ مهر ۱۴۰۵ می‌خوام. قیمتش چنده؟",
    tools=[ToolCall("search_flights", date="2026-09-27")],  # مدل شمسی صدا می‌زند — یکسان است
    contains=["250 هزار تومان"],                            # مدل ریال می‌گوید — یکسان است
    forbidden_tools=["book_flight"],
)

state = graph.invoke({"messages": [{"role": "user", "content": golden.input}]})
trace = pb_langgraph.to_trace(state)

print(AppEvaluator(goldens=[golden]).score_traces([trace]))
