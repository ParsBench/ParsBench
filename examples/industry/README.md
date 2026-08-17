# Industry examples — ParsBench evaluation on real Persian product scenarios

Each file is a self-contained, runnable evaluation of a fictional-but-realistic
Persian product bot. The offline ones need **no API key** and assert their own
expected outcome, so they double as living documentation of what ParsBench
guarantees.

| File | Vertical | Needs API key | What it demonstrates |
|---|---|---|---|
| [`banking_support.py`](banking_support.py) | fintech / banking | no | rial↔toman equivalence in answers *and* tool args, OTP-gated `forbidden_tools`, compliance `not_contains`, credential refusal |
| [`ecommerce_orders.py`](ecommerce_orders.py) | e-commerce | no | Jalali↔Gregorian delivery dates, order ids compared as text (a wrong id **fails**, proven), `max_steps` budget |
| [`medical_triage.py`](medical_triage.py) | healthcare | no | booking args across calendars and digit scripts, dosage-advice refusal, emergency escalation to «۱۱۵» |
| [`telecom_sales_simulation.py`](telecom_sales_simulation.py) | telecom / ISP | yes | multi-turn Iranian-user simulation (toman/rial confusion, Finglish), goal + criteria judging, transcript for error analysis |
| [`rag_faq_support.py`](rag_faq_support.py) | support / RAG | yes | goldens generated from [`faq_netyar.md`](faq_netyar.md), correctness + faithfulness judged in Persian, `prefer_concurrency=` |

```bash
python examples/industry/banking_support.py          # offline, instant

export OPENAI_API_KEY=... OPENAI_BASE_URL=...        # any OpenAI-compatible gateway
export PARSBENCH_JUDGE=gpt-4o-mini                   # judges the refusal checks too
python examples/industry/telecom_sales_simulation.py
```

The offline examples use scripted stand-in bots on purpose: swap the stand-in
for the function that calls your real bot and the goldens keep working. The
framework-specific wiring (OpenAI SDK, LangGraph, Pydantic AI, Agno, CrewAI)
lives one directory up in [`examples/`](../README.md).
