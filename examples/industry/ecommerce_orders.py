"""E-commerce: evaluate a Persian order-tracking bot. Runs offline: no API key.

    python examples/industry/ecommerce_orders.py

What this demonstrates for a marketplace/shop product:
- calendar equivalence: the golden expects a Gregorian delivery date, the bot
  answers with the Jalali one — same day, passes
- reference-code safety: order ids are compared as text, so a *wrong* id can
  never sneak past the tools check (the second golden proves it fails)
- budget checks: the whole flow must stay within max_steps
"""

from parsbench.appeval import AppEvaluator, Golden, ToolCall

ORDERS = {"CHB-4021": "۱۴۰۵/۰۶/۰۳"}  # order id → Jalali delivery date


# Stand-in for your app: replace with the function that talks to your bot.
def shop_bot(message):
    order_id = next((oid for oid in ORDERS if oid in message), "CHB-0000")
    return [
        {"role": "user", "content": message},
        {"role": "assistant", "tool_calls": [
            {"id": "1", "function": {"name": "get_order",
                                     "arguments": f'{{"order_id": "{order_id}"}}'}}]},
        {"role": "tool", "tool_call_id": "1",
         "content": f"وضعیت: در حال ارسال — تحویل {ORDERS.get(order_id, 'نامشخص')}"},
        {"role": "assistant",
         "content": f"سفارش {order_id} در حال ارسال است و {ORDERS.get(order_id, '؟')} تحویل می‌شود."},
    ]


evaluator = AppEvaluator(goldens=[
    Golden(
        name="پیگیری سفارش — تاریخ شمسی و میلادی یکی است",
        input="سفارش CHB-4021 من کی می‌رسه؟",
        tools=[ToolCall("get_order", order_id="CHB-4021")],
        contains=["2026-08-25"],   # the bot answers «۱۴۰۵/۰۶/۰۳» — same day
        max_steps=2,
    ),
    Golden(
        name="این یکی عمداً رد می‌شود — شناسهٔ اشتباه هرگز قبول نمی‌شود",
        input="سفارش CHB-4021 من کی می‌رسه؟",
        tools=[ToolCall("get_order", order_id="CHB-4029")],  # different order!
    ),
])
result = evaluator.evaluate(shop_bot)
print(result)

passed_by_name = {gr.golden_name: gr.passed for gr in result.golden_results}
assert passed_by_name["پیگیری سفارش — تاریخ شمسی و میلادی یکی است"]
assert not passed_by_name["این یکی عمداً رد می‌شود — شناسهٔ اشتباه هرگز قبول نمی‌شود"], \
    "a wrong order id must never match"
