"""Customer-support RAG: generate goldens from your docs, judge faithfulness.

    pip install parsbench openai
    export OPENAI_API_KEY=...   # optional: OPENAI_BASE_URL for any compatible gateway
    python examples/industry/rag_faq_support.py

What this demonstrates for a knowledge-base support product:
- GoldenGenerator: Persian test cases synthesized from your own FAQ,
  in both formal and colloquial register (adversarial: Finglish, mixed digits)
- generated goldens carry their source chunk as context=, so every answer is
  judged for faithfulness (no unsupported claims) on top of correctness
"""

import os
from pathlib import Path

from openai import OpenAI

from parsbench.appeval import AppEvaluator, GoldenGenerator

MODEL = os.getenv("MODEL", "gpt-4o-mini")
client = OpenAI()

KB = Path(__file__).with_name("faq_netyar.md")
SYSTEM = (
    "تو پشتیبان «نت‌یار» هستی. فقط بر اساس متن راهنمای زیر پاسخ بده و اگر پاسخ "
    "در راهنما نیست، صادقانه بگو نمی‌دانی.\n\n--- راهنما ---\n" + KB.read_text(encoding="utf-8")
)


def support_bot(message):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}]
    return client.chat.completions.create(model=MODEL, messages=msgs).choices[0].message.content


goldens = GoldenGenerator(model=MODEL, adversarial=True).generate(KB, n=4)
print(f"{len(goldens)} golden generated from {KB.name}:")
for g in goldens:
    print(f"  - {g.input}")

# for generated suites, judge-based checks are the robust signal — generated
# contains-needles are hints and can be brittle against paraphrase
evaluator = AppEvaluator(goldens, judge=MODEL,
                         metrics=["correctness", "faithfulness"])
result = evaluator.evaluate(support_bot, prefer_concurrency=True, n_workers=4)
print(result)
print(f"\ncorrectness={result.score('correctness'):.2f}  "
      f"faithfulness={result.score('faithfulness'):.2f}")
