"""RunRecorder: the .parsbench run store that `parsbench view` reads."""

import json
import threading
import warnings

import pytest

from parsbench.appeval.evaluation_result import (
    AppEvaluationResult,
    CheckResult,
    GoldenEvaluationResult,
)
from parsbench.appeval.recording import RunRecorder


@pytest.fixture
def store_cwd(tmp_path, monkeypatch):
    """A tmp cwd with recording enabled (conftest disables it suite-wide)."""
    monkeypatch.delenv("PARSBENCH_NO_RECORD", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _result(passed=True):
    score = 1.0 if passed else 0.0
    return AppEvaluationResult(
        golden_results=[
            GoldenEvaluationResult(
                golden_name="g",
                run_passes=[passed],
                check_results=[
                    CheckResult(check="contains", score=score, passed=passed)
                ],
            )
        ]
    )


# --- lifecycle ---------------------------------------------------------------

def test_start_creates_store_layout(store_cwd):
    rec = RunRecorder.start(kind="evaluation", app_name="My Bot", n_goldens=2)
    assert rec is not None
    root = store_cwd / ".parsbench"
    assert (root / ".gitignore").read_text() == "*\n"
    meta = json.loads((root / "runs" / rec.run_id / "run.json").read_text())
    assert meta["status"] == "running"
    assert meta["kind"] == "evaluation"
    assert meta["app_name"] == "My Bot"
    assert meta["n_goldens"] == 2
    assert meta["n_runs"] == 1
    assert meta["started_at"] and meta["finished_at"] is None
    assert meta["summary"] is None
    assert "my-bot" in rec.run_id


def test_run_ids_are_unique(store_cwd):
    ids = {RunRecorder.start(kind="evaluation", app_name="b", n_goldens=1).run_id
           for _ in range(5)}
    assert len(ids) == 5


def test_extra_meta_lands_in_run_json(store_cwd):
    rec = RunRecorder.start(
        kind="simulation", app_name="b", n_goldens=1,
        extra_meta={"user": {"style": "محاوره‌ای", "traps": ["jalali_date"]}},
    )
    meta = json.loads(
        (store_cwd / ".parsbench" / "runs" / rec.run_id / "run.json").read_text()
    )
    assert meta["user"]["traps"] == ["jalali_date"]


def test_finish_rewrites_meta(store_cwd):
    rec = RunRecorder.start(kind="evaluation", app_name="bot", n_goldens=1)
    rec.finish(_result(passed=False))
    meta = json.loads(
        (store_cwd / ".parsbench" / "runs" / rec.run_id / "run.json").read_text()
    )
    assert meta["status"] == "finished"
    assert meta["finished_at"]
    assert meta["summary"] == {"score": 0.0, "passed": False, "pass_rate": 0.0}


def test_crashed_records_error(store_cwd):
    rec = RunRecorder.start(kind="evaluation", app_name="bot", n_goldens=1)
    rec.crashed(ValueError("boom"))
    meta = json.loads(
        (store_cwd / ".parsbench" / "runs" / rec.run_id / "run.json").read_text()
    )
    assert meta["status"] == "crashed"
    assert meta["summary"] == {"error": "ValueError: boom"}


# --- opt-outs and failure ----------------------------------------------------

def test_no_record_env_disables(store_cwd, monkeypatch):
    monkeypatch.setenv("PARSBENCH_NO_RECORD", "1")
    assert RunRecorder.start(kind="evaluation", app_name="b", n_goldens=1) is None
    assert not (store_cwd / ".parsbench").exists()


def test_parsbench_dir_env_overrides_root(store_cwd, monkeypatch):
    monkeypatch.setenv("PARSBENCH_DIR", str(store_cwd / "elsewhere"))
    rec = RunRecorder.start(kind="evaluation", app_name="b", n_goldens=1)
    assert (store_cwd / "elsewhere" / "runs" / rec.run_id / "run.json").exists()


def test_unwritable_store_warns_and_returns_none(store_cwd, monkeypatch):
    blocked = store_cwd / "blocked"
    blocked.write_text("a file, not a dir")
    monkeypatch.setenv("PARSBENCH_DIR", str(blocked))
    with pytest.warns(UserWarning, match="recording disabled"):
        assert RunRecorder.start(kind="evaluation", app_name="b", n_goldens=1) is None


# --- events ------------------------------------------------------------------

from parsbench.appeval.golden import Golden
from parsbench.appeval.recording import TRUNCATE_AT, golden_to_dict, trace_to_dict
from parsbench.appeval.trace import Message, ToolCall, Trace


def my_check(trace):
    return True


class OrderReply:
    pass


def _events(store_cwd, rec):
    path = store_cwd / ".parsbench" / "runs" / rec.run_id / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_record_event_appends_full_event(store_cwd):
    rec = RunRecorder.start(kind="evaluation", app_name="b", n_goldens=1)
    golden = Golden(
        input="سفارش من کجاست؟", name="پیگیری", contains=["ارسال"],
        tools=[ToolCall("get_order", order_id="۱۲۳")],
        check=my_check, format=OrderReply,
    )
    trace = Trace(
        messages=[
            Message(role="assistant",
                    tool_calls=[ToolCall("get_order", arguments={"order_id": "۱۲۳"},
                                         result="ok")]),
            Message(role="assistant", content="ارسال شد"),
        ],
        final_output="ارسال شد", latency=1.5, raw=object(),
    )
    rec.record_event(golden, golden_index=0, run_index=0,
                     check_results=[CheckResult(check="contains", score=1.0,
                                                passed=True)],
                     trace=trace)
    (event,) = _events(store_cwd, rec)
    assert event["seq"] == 1 and event["ts"]
    assert event["golden_name"] == "پیگیری"
    assert event["golden_index"] == 0 and event["run_index"] == 0
    assert event["golden"]["check"] == "my_check"
    assert event["golden"]["format"] == "OrderReply"
    assert event["golden"]["tools"] == [{"name": "get_order",
                                         "arguments": {"order_id": "۱۲۳"}}]
    assert "output" not in event["golden"]          # empty fields dropped
    assert event["check_results"][0]["passed"] is True
    assert event["trace"]["final_output"] == "ارسال شد"
    assert event["trace"]["messages"][0]["tool_calls"][0]["result"] == "ok"
    assert event["trace"]["latency"] == 1.5
    assert "raw" not in event["trace"]
    assert event["transcript"] is None and event["converged"] is None


def test_event_persian_stays_readable(store_cwd):
    rec = RunRecorder.start(kind="evaluation", app_name="b", n_goldens=1)
    rec.record_event(Golden(input="سلام"), 0, 0, [], trace=Trace(final_output="درود"))
    raw = (store_cwd / ".parsbench" / "runs" / rec.run_id / "events.jsonl").read_text()
    assert "سلام" in raw and "درود" in raw  # ensure_ascii=False


def test_oversized_strings_truncated(store_cwd):
    rec = RunRecorder.start(kind="evaluation", app_name="b", n_goldens=1)
    big = "x" * (TRUNCATE_AT + 100)
    trace = Trace(
        messages=[Message(role="assistant",
                          tool_calls=[ToolCall("t", arguments={}, result=big)])],
        final_output=big,
    )
    rec.record_event(Golden(input="q"), 0, 0, [], trace=trace)
    (event,) = _events(store_cwd, rec)
    out = event["trace"]["final_output"]
    assert len(out) < TRUNCATE_AT + 50 and out.endswith("[truncated by parsbench]")
    assert event["trace"]["messages"][0]["tool_calls"][0]["result"].endswith(
        "[truncated by parsbench]"
    )


def test_unserializable_values_stringified(store_cwd):
    rec = RunRecorder.start(kind="evaluation", app_name="b", n_goldens=1)
    trace = Trace(
        messages=[Message(role="assistant",
                          tool_calls=[ToolCall("t", arguments={"x": object()})])],
        final_output="ok",
    )
    rec.record_event(Golden(input="q"), 0, 0, [], trace=trace)  # must not raise
    (event,) = _events(store_cwd, rec)
    assert "object" in event["trace"]["messages"][0]["tool_calls"][0]["arguments"]["x"]


def test_appends_are_thread_safe(store_cwd):
    rec = RunRecorder.start(kind="evaluation", app_name="b", n_goldens=32)
    golden = Golden(input="q")

    def write(i):
        rec.record_event(golden, i, 0, [], trace=Trace(final_output=str(i)))

    threads = [threading.Thread(target=write, args=(i,)) for i in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    events = _events(store_cwd, rec)   # every line parses — no interleaving
    assert sorted(e["seq"] for e in events) == list(range(1, 33))


def test_recorder_failure_warns_once_then_stays_silent(store_cwd):
    rec = RunRecorder.start(kind="evaluation", app_name="b", n_goldens=1)
    rec._events.close()  # force the next append to fail
    with pytest.warns(UserWarning, match="recording disabled"):
        rec.record_event(Golden(input="q"), 0, 0, [], trace=Trace(final_output="a"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a second warning would fail the test
        rec.record_event(Golden(input="q"), 0, 0, [], trace=Trace(final_output="a"))
        rec.finish(_result())


# --- AppEvaluator integration ------------------------------------------------

from parsbench.appeval import AppEvaluator


def support_bot(message):
    return "سفارش شما ارسال شد"


def test_evaluate_records_a_run(store_cwd):
    goldens = [Golden(input="سفارش؟", contains=["ارسال"]),
               Golden(input="سلام", contains=["نیست"])]
    result = AppEvaluator(goldens=goldens).evaluate(support_bot, n_runs=2)
    runs_dir = store_cwd / ".parsbench" / "runs"
    (run_dir,) = list(runs_dir.iterdir())
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["kind"] == "evaluation"
    assert meta["app_name"] == "support_bot"
    assert meta["status"] == "finished"
    assert meta["summary"]["passed"] is result.passed
    events = [json.loads(l)
              for l in (run_dir / "events.jsonl").read_text().splitlines()]
    assert len(events) == 4                      # 2 goldens × 2 runs
    assert {e["run_index"] for e in events} == {0, 1}   # every run's detail
    assert {e["golden_index"] for e in events} == {0, 1}
    assert all(e["trace"]["final_output"] for e in events)


def test_app_crash_records_app_error_event(store_cwd):
    def broken(message):
        raise RuntimeError("down")

    AppEvaluator(goldens=[Golden(input="hi")]).evaluate(broken)
    (run_dir,) = list((store_cwd / ".parsbench" / "runs").iterdir())
    (event,) = [json.loads(l)
                for l in (run_dir / "events.jsonl").read_text().splitlines()]
    assert event["trace"] is None
    assert event["check_results"][0]["check"] == "app_error"
    assert "RuntimeError" in event["check_results"][0]["reason"]


def test_evaluator_crash_marks_run_crashed(store_cwd):
    evaluator = AppEvaluator(goldens=[Golden(input="hi")], metrics=["not-a-metric"])
    with pytest.raises(ValueError):
        evaluator.evaluate(support_bot)  # unknown metric raises inside run_checks
    (run_dir,) = list((store_cwd / ".parsbench" / "runs").iterdir())
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["status"] == "crashed"
    assert "not-a-metric" in meta["summary"]["error"]


def test_record_false_writes_nothing(store_cwd):
    AppEvaluator(goldens=[Golden(input="hi")]).evaluate(support_bot, record=False)
    assert not (store_cwd / ".parsbench").exists()


def test_score_traces_records_too(store_cwd):
    AppEvaluator(goldens=[Golden(input="hi", contains=["درود"])]).score_traces(
        [Trace(final_output="درود")]
    )
    (run_dir,) = list((store_cwd / ".parsbench" / "runs").iterdir())
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["kind"] == "score_traces"
    assert meta["app_name"] == "traces"


def test_concurrent_evaluate_records_all_events(store_cwd):
    goldens = [Golden(input=f"q{i}", contains=["جواب"]) for i in range(8)]
    AppEvaluator(goldens=goldens).evaluate(
        lambda m: "جواب", prefer_concurrency=True, n_workers=4
    )
    (run_dir,) = list((store_cwd / ".parsbench" / "runs").iterdir())
    lines = (run_dir / "events.jsonl").read_text().splitlines()
    assert len(lines) == 8
    assert all(json.loads(l) for l in lines)


# --- SimulationEvaluator integration -----------------------------------------

from parsbench.appeval import SimulationEvaluator
from parsbench.appeval.simulation import DONE


def _scripted_simulator():
    lines = iter(["سلام، سفارشم کجاست؟", DONE])

    def simulator(prompt):
        return next(lines)

    return simulator


def sim_bot(message):
    return "سفارش شما فردا می‌رسد"


def test_simulation_records_run(store_cwd):
    evaluator = SimulationEvaluator(
        goal="پیگیری سفارش",
        user="محاوره‌ای+jalali_date",
        simulator_model=_scripted_simulator(),
    )
    evaluator.evaluate(sim_bot)  # judge unset -> goal check skips gracefully
    (run_dir,) = list((store_cwd / ".parsbench" / "runs").iterdir())
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["kind"] == "simulation"
    assert meta["app_name"] == "sim_bot"
    assert meta["user"]["style"] == "محاوره‌ای"
    assert meta["user"]["traps"] == ["jalali_date"]
    (event,) = [json.loads(l)
                for l in (run_dir / "events.jsonl").read_text().splitlines()]
    assert event["converged"] is True
    assert "کاربر:" in event["transcript"]
    assert event["trace"]["messages"][0]["role"] == "user"
    assert event["trace"]["final_output"] == "سفارش شما فردا می‌رسد"


def test_simulation_bot_crash_records_app_error(store_cwd):
    def broken(message):
        raise RuntimeError("down")

    SimulationEvaluator(
        goal="هدف", simulator_model=_scripted_simulator()
    ).evaluate(broken)
    (run_dir,) = list((store_cwd / ".parsbench" / "runs").iterdir())
    (event,) = [json.loads(l)
                for l in (run_dir / "events.jsonl").read_text().splitlines()]
    assert event["trace"] is None and event["transcript"] is None
    assert event["check_results"][0]["check"] == "app_error"


def test_simulation_record_false(store_cwd):
    SimulationEvaluator(
        goal="هدف", simulator_model=_scripted_simulator()
    ).evaluate(sim_bot, record=False)
    assert not (store_cwd / ".parsbench").exists()
