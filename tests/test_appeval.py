"""Core app-eval checks: normalization, traces, and AppEvaluator."""

import pytest

from parsbench.appeval import AppEvaluator, Golden, ToolCall, Trace
from parsbench.appeval.normalize import (
    contains_normalized,
    dates_equal,
    gregorian_to_jalali,
    jalali_to_gregorian,
    normalize,
    numbers_equal,
    values_equal,
)


# --- normalization -----------------------------------------------------------

def test_char_and_digit_normalization():
    assert normalize("كتاب") == "کتاب"          # Arabic kaf
    assert normalize("علي") == "علی"            # Arabic yeh
    assert normalize("۵ مهر ١٤٠٥") == "5 مهر 1405"  # both digit scripts


def test_zwnj_contains():
    assert contains_normalized("من می‌روم خانه", "می روم")
    assert contains_normalized("من می روم", "می‌روم")
    assert contains_normalized("من میروم خانه", "می‌روم")  # fully joined typing


def test_contains_does_not_fuse_across_word_boundaries():
    assert not contains_normalized("او به من گفت", "بهمن")  # 'به من' is not the month
    assert not contains_normalized("گفت 2 و بعد 5", "25")


def test_amount_in_running_text():
    from parsbench.appeval.normalize import amount_in

    assert amount_in("قیمت ۲٬۵۰۰٬۰۰۰ ریال است", "250 هزار تومان")
    assert amount_in("میشه ۲۵۰ هزار تومان", "2500000 ریال")
    assert not amount_in("قیمت ۳٬۰۰۰٬۰۰۰ ریال است", "250 هزار تومان")


def test_compound_persian_amounts():
    from parsbench.appeval.normalize import amount_in

    assert numbers_equal("۲ میلیون و ۵۰۰ هزار تومان", "25000000 ریال")
    assert amount_in("قیمت ۲ میلیون و ۵۰۰ هزار تومان است", "25,000,000 ریال")


def test_persian_decimal_forms():
    from parsbench.appeval.normalize import amount_in

    assert numbers_equal("۲٫۵ میلیون تومان", "25000000 ریال")  # U+066B decimal sep
    assert numbers_equal("۲/۵ میلیون تومان", "25000000 ریال")  # slash decimal
    assert amount_in("قیمت ۲۵۰۰۰۰۰۰ ریال است", "۲/۵ میلیون تومان")


def test_scale_words_match_whole_words_only():
    from parsbench.appeval.normalize import amount_in

    # 'هزاران' (plural noun) is not the scale word 'هزار'
    assert not amount_in("هزاران نفر آمدند و قیمت ۵ تومان بود", "5000 تومان")
    assert amount_in("قیمت ۵ هزار تومان بود", "5000 تومان")


def test_bare_numbers_need_exact_amount_match():
    from parsbench.appeval.normalize import amount_in

    assert not amount_in("کد سفارش: 250000", "250 هزار تومان")  # order id is not money
    assert amount_in("قیمت نهایی: 2500000", "250 هزار تومان")   # bare rial value, exact


def test_non_money_digit_needles_dont_fuzzy_match():
    from parsbench.appeval.normalize import text_matches

    assert not text_matches("رستوران ساعت 7 باز است", "سالن 7")
    assert text_matches("جلسه در سالن 7 برگزار می‌شود", "سالن 7")


def test_numbers_and_currency():
    assert numbers_equal("۲٬۵۰۰٬۰۰۰", 2500000)
    assert numbers_equal("۲۵۰ هزار تومان", "2,500,000 ریال")
    assert numbers_equal("2.5 میلیون ریال", "۲۵۰ هزار تومان")
    assert not numbers_equal("۳۰۰ هزار تومان", "2,500,000 ریال")


def test_jalali_gregorian():
    assert jalali_to_gregorian(1405, 7, 5) == (2026, 9, 27)
    assert gregorian_to_jalali(2026, 9, 27) == (1405, 7, 5)
    assert dates_equal("1405-07-05", "2026-09-27")
    assert dates_equal("۱۴۰۵/۰۷/۰۵", "2026-09-27")
    assert not dates_equal("1405-07-06", "2026-09-27")


def test_values_equal_chain():
    assert values_equal("1405-07-05", "2026/9/27")
    assert not values_equal("1405-07-05", "1405")  # date vs number is not equal
    assert values_equal("تهران", "تهران‌")         # trailing ZWNJ
    assert values_equal("۱۲۳", 123)


def test_date_pattern_ignores_longer_numbers():
    # reference codes are not dates — a date must not match inside a longer number
    assert not values_equal("کد پیگیری 2026-01-123456", "کد پیگیری 2026-01-129999")


def test_values_equal_respects_surrounding_text():
    assert not values_equal("خیابان ولیعصر پلاک 12", "کوچه انقلاب 12")  # addresses differ
    assert not values_equal("1404/05/25", "از 1404/05/25 تا 1404/06/01")  # date vs range


def test_multi_number_strings_are_not_number_equal():
    assert not numbers_equal("0912 345 6789", "0912 111 1111")  # different phones
    assert values_equal("۰۹۱۲ ۳۴۵ ۶۷۸۹", "0912 345 6789")       # same phone, other digits


# --- trace & tool matching ---------------------------------------------------

def _flight_messages(date="1405-07-05"):
    return [
        {"role": "user", "content": "بلیط می‌خوام"},
        {"role": "assistant", "tool_calls": [
            {"id": "1", "function": {"name": "search_flights",
                                     "arguments": f'{{"origin": "THR", "date": "{date}"}}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "پرواز ۸ صبح"},
        {"role": "assistant", "content": "پرواز ساعت ۸ صبح موجود است، ۲٬۵۰۰٬۰۰۰ ریال"},
    ]


def test_trace_from_messages():
    trace = Trace.from_messages(_flight_messages())
    assert trace.final_output.startswith("پرواز")
    assert trace.tool_calls[0].name == "search_flights"
    assert trace.tool_calls[0].arguments["origin"] == "THR"
    assert trace.n_steps == 2


def test_trace_tolerates_developer_and_function_roles():
    trace = Trace.from_messages([
        {"role": "developer", "content": "فقط فارسی جواب بده"},
        {"role": "user", "content": "سلام"},
        {"role": "function", "name": "lookup", "content": "نتیجه"},
        {"role": "assistant", "content": "درود"},
    ])
    assert trace.final_output == "درود"
    assert trace.n_steps == 1  # developer/function messages are not assistant steps


def test_trace_attaches_tool_results():
    trace = Trace.from_messages(_flight_messages())
    assert trace.tool_calls[0].result == "پرواز ۸ صبح"


def test_toolcall_kwargs_form():
    tc = ToolCall("search_flights", origin="THR", date="1405-07-05")
    assert tc.name == "search_flights"
    assert tc.arguments == {"origin": "THR", "date": "1405-07-05"}


def test_evaluate_scores_tools_across_calendars_and_digits():
    # golden in Gregorian, bot called with Jalali — must pass
    golden = Golden(
        input="بلیط برای ۵ مهر",
        tools=[ToolCall("search_flights", date="2026-09-27")],
        contains=["2500000 ریال"],   # bot said ۲٬۵۰۰٬۰۰۰ — digit-script equivalence
    )
    result = AppEvaluator([golden]).evaluate(lambda _: _flight_messages())
    assert result.passed, [cr for gr in result.golden_results for cr in gr.check_results]


def test_evaluate_fails_on_wrong_arg():
    golden = Golden(input="...", tools=[ToolCall("search_flights", date="1405-07-06")])
    result = AppEvaluator([golden]).evaluate(lambda _: _flight_messages())
    assert not result.passed


def test_forbidden_tools_and_budget():
    golden = Golden(
        input="...",
        forbidden_tools=["transfer_money"],
        max_steps=3,
        check=lambda t: t.tool_calls[0].name == "search_flights",
    )
    result = AppEvaluator([golden]).evaluate(lambda _: _flight_messages())
    assert result.passed


def test_evaluator_rejects_empty_goldens():
    with pytest.raises(ValueError, match="goldens"):
        AppEvaluator([])


def test_evaluate_rejects_nonpositive_n_runs():
    with pytest.raises(ValueError, match="n_runs"):
        AppEvaluator([Golden(input="س")]).evaluate(lambda _: "x", n_runs=0)


def test_evaluate_captures_app_crash_as_finding():
    def bot(message):
        raise RuntimeError("خطای داخلی ربات")

    result = AppEvaluator([Golden(input="سلام", contains=["الف"])]).evaluate(bot)
    (gr,) = result.golden_results
    assert not gr.passed and not result.passed
    (cr,) = gr.check_results
    assert cr.check == "app_error"
    assert "RuntimeError" in cr.reason and "خطای داخلی" in cr.reason


def test_evaluate_async_app_inside_running_loop():
    import asyncio

    async def bot(message):
        return "پاسخ الف"

    async def run():
        return AppEvaluator([Golden(input="س", contains=["الف"])]).evaluate(bot)

    result = asyncio.run(run())  # notebooks/servers evaluate inside a loop
    assert result.passed


def test_evaluate_concurrency_preserves_order_and_results():
    def bot(message):
        return "پاسخ " + message

    goldens = [Golden(input=f"مورد {i}", contains=[f"مورد {i}"]) for i in range(6)]
    result = AppEvaluator(goldens).evaluate(bot, prefer_concurrency=True, n_workers=4)
    assert result.passed
    assert [gr.golden_name for gr in result.golden_results] == [g.label for g in goldens]


def test_judge_checks_skip_without_judge_and_run_with_one():
    golden = Golden(input="سلام", output="پاسخ مرجع")
    result = AppEvaluator([golden]).evaluate(lambda _: "پاسخی")
    assert result.golden_results[0].check_results[0].skipped

    result = AppEvaluator(
        [golden], judge=lambda prompt: "پاسخ هم‌معناست.\nنمره: ۵",
    ).evaluate(lambda _: "پاسخی")
    (cr,) = result.golden_results[0].check_results
    assert cr.passed and cr.score == 1.0


def test_dict_goldens_and_assert_passed():
    AppEvaluator(
        [{"in": "بلیط", "tools": [ToolCall("search_flights")]}]
    ).evaluate(lambda _: _flight_messages()).assert_passed()
    with pytest.raises(AssertionError, match="بلیط — contains"):
        AppEvaluator([{"in": "بلیط", "contains": ["فاکتور"]}]).evaluate(
            lambda _: "هیچ"
        ).assert_passed()


def test_golden_wraps_bare_strings_for_list_fields():
    # a bare string must not be iterated character by character
    golden = Golden(input="س", contains="فاکتور", not_contains="ممنوع", context="زمینه")
    assert golden.contains == ["فاکتور"]
    assert golden.not_contains == ["ممنوع"]
    assert golden.context == ["زمینه"]
    result = AppEvaluator([golden], metrics=["contains", "not_contains"]).score_traces(
        [Trace(final_output="هیچ")]
    )
    contains = next(
        cr for gr in result.golden_results for cr in gr.check_results
        if cr.check == "contains"
    )
    assert not contains.passed  # «فاکتور» absent — must fail, not char-pass


def test_score_traces_and_metrics_filter():
    golden = Golden(input="...", tools=[ToolCall("nonexistent")], contains=["پرواز"])
    result = AppEvaluator([golden], metrics=["contains"]).score_traces(
        [_flight_messages()]
    )
    assert result.passed  # tools check filtered out
    assert {cr.check for gr in result.golden_results for cr in gr.check_results} == {"contains"}


def test_score_traces_rejects_length_mismatch():
    with pytest.raises(ValueError, match="traces"):
        AppEvaluator([Golden(input="س"), Golden(input="ب")]).score_traces(
            [Trace(final_output="x")]
        )


def test_unknown_metric_names_raise_instead_of_vacuous_pass():
    golden = Golden(input="...", max_steps=2)
    with pytest.raises(ValueError, match="unknown metric"):
        AppEvaluator([golden], metrics=["max_stepz"]).score_traces(
            [Trace(final_output="x")]
        )


def test_metric_aliases_match_golden_field_names():
    golden = Golden(input="...", max_steps=2, contains=["ندارد"])
    result = AppEvaluator([golden], metrics=["max_steps"]).score_traces(
        [Trace(final_output="x")]
    )
    assert {cr.check for gr in result.golden_results for cr in gr.check_results} == {"budget:max_steps"}


def test_not_contains_gets_money_equivalence():
    golden = Golden(input="...", not_contains=["250 هزار تومان"])
    trace = Trace(final_output="قیمت ۲٬۵۰۰٬۰۰۰ ریال است")
    result = AppEvaluator([golden]).score_traces([trace])
    assert not result.passed  # the forbidden amount was stated, just in rials


def test_contains_matches_dates_across_calendars():
    golden = Golden(input="...", contains=["2026-09-27"])
    trace = Trace(final_output="پرواز در تاریخ ۱۴۰۵/۰۷/۰۵ انجام می‌شود")
    result = AppEvaluator([golden]).score_traces([trace])
    assert result.passed


def test_result_round_trips_through_save_and_diff(tmp_path, capsys):
    golden = Golden(input="س", contains=["الف"])
    result = AppEvaluator([golden]).score_traces([Trace(final_output="الف ب")])
    result.save(str(tmp_path))
    from parsbench.appeval import AppEvaluationResult
    from parsbench.appeval.evaluation_result import EVALUATION_FILE_NAME

    loaded = AppEvaluationResult.from_file(str(tmp_path / EVALUATION_FILE_NAME))
    assert loaded.to_dict() == result.to_dict()
    assert loaded.passed and loaded.average_score == 1.0
    assert set(result.to_pandas().columns) >= {"golden_name", "check", "score"}
    result.diff(str(tmp_path / EVALUATION_FILE_NAME))
    assert capsys.readouterr().out == ""  # identical run — no deltas printed


def test_root_import_stays_clean_and_appeval_needs_no_pydantic():
    import subprocess
    import sys

    code = (
        "import sys, parsbench; "
        "assert 'parsbench.appeval' not in sys.modules, 'appeval imported eagerly'; "
        "from parsbench.appeval import Golden; "
        "assert 'pydantic' not in sys.modules, 'pydantic imported'; "
        "print(Golden(input='سلام').label)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "سلام" in out.stdout


def test_from_messages_final_and_raw_override():
    trace = Trace.from_messages(_flight_messages(), final_output="خروجی نهایی",
                                raw={"x": 1})
    assert trace.final_output == "خروجی نهایی"
    assert trace.raw == {"x": 1}
