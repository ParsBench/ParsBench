"""App-eval simulation, golden generation, and judge calibration."""

import pytest

from parsbench.appeval import (
    ConversationGolden,
    Golden,
    GoldenGenerator,
    JudgeCalibrator,
    PersianUser,
    SimulationEvaluator,
)
from parsbench.appeval.evaluator import AppEvaluator
from parsbench.appeval.simulation import DONE, TRAPS


def scripted_sim(replies):
    """User-simulator fake: returns queued user messages, then DONE forever."""
    queue = list(replies)

    def sim(prompt: str) -> str:
        return queue.pop(0) if queue else DONE

    return sim


FAKE_JUDGE = lambda prompt: "قابل قبول است.\nنمره: ۵"


def test_simulation_end_to_end():
    sim = scripted_sim(["سلام، بلیط تهران-مشهد می‌خوام", "برای ۵ مهر لطفاً"])
    bot_seen = []

    def bot(message, history):
        bot_seen.append((message, len(history)))
        return "باشه، انجام شد"

    evaluator = SimulationEvaluator(
        goal="رزرو بلیط تهران-مشهد",
        criteria=["دستیار مؤدب بود"],
        simulator_model=sim,
        judge=FAKE_JUDGE,
    )
    result = evaluator.evaluate(bot, max_turns=5)
    assert result.passed
    checks = {cr.check for gr in result.golden_results for cr in gr.check_results}
    assert "converged" in checks and "goal" in checks
    assert any(c.startswith("criterion:") for c in checks)
    assert "بلیط" in result.golden_results[0].transcript
    assert bot_seen[0][1] == 0 and bot_seen[1][1] == 2  # history grows


def test_simulation_stateful_single_arg_bot_and_n_runs():
    def bot(message):  # stateful bots take only the message
        return "چشم"

    evaluator = SimulationEvaluator(
        goal="تست", simulator_model=scripted_sim(["پیام"]), judge=FAKE_JUDGE
    )
    result = evaluator.evaluate(bot, n_runs=2)
    assert result.golden_results[0].run_passes == [True, True]
    assert result.pass_hat_k() == 1.0


def test_simulation_rejects_empty_goldens():
    with pytest.raises(ValueError, match="goldens"):
        SimulationEvaluator()


def test_not_converged_with_goal_met_passes():
    # a chatty simulator that never says DONE must not fail a conversation
    # the judge scored as successful — the turn cap cut the chat, that's all
    stubborn = lambda prompt: "باز هم سوال دارم"
    result = SimulationEvaluator(
        goal="تست", simulator_model=stubborn, judge=FAKE_JUDGE
    ).evaluate(lambda m: "پاسخ", max_turns=3)
    (gr,) = result.golden_results
    converged = next(cr for cr in gr.check_results if cr.check == "converged")
    assert converged.passed and "سقف نوبت" in converged.reason
    assert result.passed


def test_not_converged_without_goal_fails():
    stubborn = lambda prompt: "باز هم سوال دارم"  # never says DONE
    failing_judge = lambda prompt: "برآورده نشد.\nنمره: ۱"
    result = SimulationEvaluator(
        goal="تست", simulator_model=stubborn, judge=failing_judge
    ).evaluate(lambda m: "پاسخ", max_turns=3)
    (gr,) = result.golden_results
    converged = next(cr for cr in gr.check_results if cr.check == "converged")
    assert not converged.passed and not result.passed


def test_not_converged_without_judge_fails():
    stubborn = lambda prompt: "باز هم سوال دارم"
    result = SimulationEvaluator(
        goal="تست", simulator_model=stubborn, judge=None
    ).evaluate(lambda m: "پاسخ", max_turns=3)
    (gr,) = result.golden_results
    converged = next(cr for cr in gr.check_results if cr.check == "converged")
    assert not converged.passed  # no judge to vouch for the goal — cap is a failure


def test_persian_user_prompt_and_string_spec():
    golden = ConversationGolden(goal="خرید شارژ", scenario="اپراتور همراه اول")
    user = PersianUser(style="رسمی", traps=["taarof_opening", "toman_rial_confusion"])
    prompt = user.system_prompt(golden)
    assert "خرید شارژ" in prompt and TRAPS["taarof_opening"] in prompt
    # string form: register + trap names
    result = SimulationEvaluator(
        goal="تست", user="رسمی+jalali_date",
        simulator_model=scripted_sim(["سلام"]), judge=FAKE_JUDGE,
    ).evaluate(lambda m: "بله")
    assert result.passed


def test_user_string_spec_keeps_free_text_traps():
    prompts = []

    def sim(prompt):
        prompts.append(prompt)
        return DONE

    SimulationEvaluator(
        goal="تست",
        user="رسمی+عجول باش و زود ناراضی شو+jalali_date",
        simulator_model=sim, judge=FAKE_JUDGE,
    ).evaluate(lambda m: "بله")
    assert "عجول باش و زود ناراضی شو" in prompts[0]  # free text becomes an instruction
    assert TRAPS["jalali_date"] in prompts[0]
    assert "سبک گفتار: رسمی" in prompts[0]


def test_generate_from_docs(tmp_path):
    doc = tmp_path / "kb.md"
    doc.write_text("ساعت کاری پشتیبانی از ۸ صبح تا ۵ عصر است.", encoding="utf-8")

    def fake_llm(prompt):
        assert "ساعت کاری" in prompt
        return ('[{"input": "پشتیبانی کی بازه؟", "output": "از ۸ صبح تا ۵ عصر",'
                ' "contains": ["۸ صبح"]}]')

    goldens = GoldenGenerator(model=fake_llm).generate(str(doc), n=5)
    assert len(goldens) == 1
    g = goldens[0]
    assert g.output == "از ۸ صبح تا ۵ عصر"
    assert g.context and "ساعت کاری" in g.context[0]
    assert "generated" in g.tags
    # generated goldens run straight through an AppEvaluator
    result = AppEvaluator(goldens, metrics=["contains"]).evaluate(
        lambda _: "از ۸ صبح تا ۵ عصر باز است"
    )
    assert result.passed


def test_generate_reads_a_docs_directory(tmp_path):
    (tmp_path / "a.md").write_text("بسته یک‌ماهه ۵۰ گیگ.", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("پشتیبانی همه‌روزه.", encoding="utf-8")

    goldens = GoldenGenerator(
        model=lambda p: '[{"input": "سوال؟", "output": "پاسخ", "contains": []}]',
    ).generate(str(tmp_path), n=2)
    assert len(goldens) == 2  # both files found, recursively


def test_generate_rejects_binary_docs(tmp_path):
    pdf = tmp_path / "kb.pdf"
    pdf.write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="pdf"):
        GoldenGenerator(model=lambda p: "[]").generate(str(pdf))


def test_calibrate_agreement_and_kappa():
    def judge(prompt):
        return "نمره: ۵" if "درست" in prompt else "نمره: ۱"

    items = [
        {"golden": Golden(input="۱+۱؟", output="جواب درست دو"), "output": "جواب درست دو",
         "human": True},
        {"golden": Golden(input="۲+۲؟", output="پاسخ چهار"), "output": "چیز دیگری",
         "human": False},
        {"golden": Golden(input="۳+۳؟", output="پاسخ شش"), "output": "غلط", "human": True},
    ]
    result = JudgeCalibrator(judge=judge).calibrate(items)
    assert result.n == 3
    assert result.agreement == pytest.approx(2 / 3)
    assert len(result.disagreements) == 1
    assert "judge-vs-human" in str(result)


def test_simulation_accepts_partial_apps():
    from functools import partial

    def bot(prefix, message):
        return prefix + message

    result = SimulationEvaluator(
        goal="تست", simulator_model=scripted_sim(["سلام"]), judge=FAKE_JUDGE
    ).evaluate(partial(bot, "پ:"))
    assert result.passed


def test_simulator_resolves_without_pinned_temperature(monkeypatch):
    import parsbench.appeval.simulation as simulation

    seen = {}
    real = simulation.resolve_model

    def spy(spec, *envs, **kwargs):
        if "PARSBENCH_SIMULATOR" in envs:
            seen.update(kwargs)
        return real(spec, *envs, **kwargs)

    monkeypatch.setattr(simulation, "resolve_model", spy)
    SimulationEvaluator(
        goal="ت", simulator_model=scripted_sim(["س"]), judge=FAKE_JUDGE
    ).evaluate(lambda m: "ب")
    assert seen.get("temperature", "missing") is None


def test_calibrate_concurrency_matches_sequential():
    def judge(prompt):
        return "نمره: ۵" if "درست" in prompt else "نمره: ۱"

    items = [
        {"golden": Golden(input="۱+۱؟", output="جواب درست دو"), "output": "جواب درست دو",
         "human": True},
        {"golden": Golden(input="۲+۲؟", output="پاسخ چهار"), "output": "چیز دیگری",
         "human": False},
        {"golden": Golden(input="۳+۳؟", output="پاسخ شش"), "output": "غلط", "human": True},
    ]
    result = JudgeCalibrator(judge=judge).calibrate(
        items, prefer_concurrency=True, n_workers=4
    )
    assert result.n == 3
    assert result.agreement == pytest.approx(2 / 3)
    assert len(result.disagreements) == 1


def test_calibrate_requires_judgeable_golden():
    with pytest.raises(ValueError, match="judge check"):
        JudgeCalibrator(judge=FAKE_JUDGE).calibrate(
            [{"golden": Golden(input="سلام"), "output": "x", "human": True}]
        )
