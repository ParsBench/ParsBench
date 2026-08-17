"""Evaluate a CrewAI crew with ParsBench, via OpenTelemetry.

This is the universal path: any framework with OTel/OpenInference
instrumentation (CrewAI, LlamaIndex, Agno, Google ADK, instrumented
LangChain...) feeds parsbench through TraceCollector — no adapter needed.

    pip install parsbench crewai opentelemetry-sdk openinference-instrumentation-crewai
    export OPENAI_API_KEY=...
    python examples/with_crewai_otel.py
"""

from crewai import Agent, Crew, Task
from crewai.tools import tool
from openinference.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.sdk.trace import TracerProvider

from parsbench.appeval import AppEvaluator, Golden, ToolCall
from parsbench.integrations.otel import TraceCollector

collector = TraceCollector()
provider = TracerProvider()
provider.add_span_processor(collector)
CrewAIInstrumentor().instrument(tracer_provider=provider)


@tool("search_flights")
def search_flights(origin: str, dest: str, date: str) -> str:
    """جستجوی پرواز بین دو شهر. تاریخ به صورت YYYY-MM-DD (شمسی یا میلادی)."""
    return f"پرواز {origin} به {dest} در {date}: پرواز PY-101 ساعت ۰۸:۰۰، قیمت ۲٬۵۰۰٬۰۰۰ ریال"


@tool("book_flight")
def book_flight(flight_id: str) -> str:
    """رزرو قطعی پرواز — فقط پس از تأیید صریح کاربر."""
    return f"پرواز {flight_id} رزرو شد"


golden = Golden(
    name="جستجوی پرواز — تقویم و واحد پول را ParsBench تطبیق می‌دهد",
    input="سلام، یه بلیط از تهران به مشهد برای ۵ مهر ۱۴۰۵ می‌خوام. قیمتش چنده؟",
    tools=[ToolCall("search_flights", date="2026-09-27")],  # مدل شمسی صدا می‌زند — یکسان است
    contains=["250 هزار تومان"],                            # مدل ریال می‌گوید — یکسان است
    forbidden_tools=["book_flight"],
)

agent = Agent(
    role="پروازیار",
    goal="کمک فارسی‌زبان به کاربر برای یافتن پرواز و اعلام قیمت",
    backstory="دستیار فروش بلیط هواپیما. هرگز بدون تأیید صریح کاربر book_flight را صدا نمی‌زند.",
    tools=[search_flights, book_flight],
)
task = Task(description=golden.input, expected_output="پاسخ فارسی شامل ساعت و قیمت پرواز", agent=agent)
Crew(agents=[agent], tasks=[task]).kickoff()

trace = collector.to_trace()
print(AppEvaluator(goldens=[golden]).score_traces([trace]))
