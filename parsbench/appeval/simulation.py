"""SimulationEvaluator: a Persian user simulator drives YOUR bot.

The simulator LLM plays an Iranian user (register, taarof, Finglish, traps),
the bot under evaluation replies, and a judge scores the finished
conversation against the goal and criteria.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from tqdm import tqdm

from .checks import run_judge_specs
from .evaluation_result import (
    AppEvaluationResult,
    CheckResult,
    GoldenEvaluationResult,
)
from .evaluator import _sync_await
from .golden import ConversationGolden
from .judge import resolve_model
from .recording import RunRecorder
from .trace import Trace

DONE = "###تمام###"

# Named traps — realistic Iranian-user behaviors that break English-shaped bots.
TRAPS = {
    "taarof_opening": "مکالمه را با تعارف شروع کن و درخواست اصلی را در پیام اول مستقیم نگو.",
    "toman_rial_confusion": "مبلغ‌ها را به تومان بگو، حتی اگر سیستم ریالی به نظر برسد.",
    "jalali_date": "تاریخ‌ها را شمسی و با نام ماه فارسی بگو (مثلاً «۵ مهر»).",
    "finglish_switch": "از میانهٔ گفتگو بعضی پیام‌ها را فینگلیش بنویس (مثلاً «merci, hamin khoobe»).",
    "typos": "گاهی غلط تایپی طبیعی داشته باش.",
    "impatient": "عجول باش؛ اگر پاسخ کند یا مبهم بود، ابراز نارضایتی کن.",
    "iran_formats": "شماره تلفن و آدرس را به قالب رایج ایران بگو (۰۹۱۲…، خیابان/کوچه/پلاک).",
}


@dataclass
class PersianUser:
    """
    The simulated user's personality. Editable, composable.

    Attributes:
        style (str): The speech register (e.g. "محاوره‌ای", "رسمی").
        persona (str): Extra free-text persona description.
        traps (list[str]): Keys of TRAPS or free-text instructions.

    Methods:
        system_prompt: Renders the simulator system prompt for a golden.
    """

    style: str = "محاوره‌ای"
    persona: str = ""
    traps: list[str] = field(default_factory=list)

    def system_prompt(self, golden: ConversationGolden) -> str:
        trap_lines = "\n".join(f"- {TRAPS.get(t, t)}" for t in self.traps)
        parts = [
            "تو نقش یک کاربر واقعی ایرانی را بازی می‌کنی که با یک دستیار هوشمند گفتگو می‌کند.",
            f"هدف تو از این گفتگو: {golden.goal}",
            f"موقعیت: {golden.scenario}" if golden.scenario else "",
            f"سبک گفتار: {self.style}. {self.persona}".strip(),
            trap_lines,
            "قواعد: هر بار فقط پیامِ بعدیِ کاربر را بنویس، کوتاه و طبیعی. "
            "به محض این که به هدفت رسیدی یا مطمئن شدی به نتیجه نمی‌رسی، دیگر "
            f"سؤال تازه‌ای نپرس و فقط بنویس: {DONE}",
        ]
        return "\n".join(p for p in parts if p)


GOAL_EVAL_FA = (
    "متن کامل گفتگوی یک کاربر با یک دستیار را می‌بینی.\n\n{transcript}\n\n"
    "هدف کاربر: {goal}\n{expected}"
    "آیا دستیار هدف کاربر را برآورده کرد؟ دلیل را کوتاه بنویس و در خط آخر فقط "
    "بنویس: نمره: X که X از ۱ (اصلاً) تا ۵ (کاملاً) است."
)
CRITERION_EVAL_FA = (
    "متن کامل گفتگوی یک کاربر با یک دستیار را می‌بینی.\n\n{transcript}\n\n"
    "معیار: «{criterion}»\n"
    "آیا رفتار دستیار در این گفتگو با معیار سازگار بود؟ در خط آخر فقط بنویس: "
    "نمره: X که X از ۱ (ناسازگار) تا ۵ (کاملاً سازگار) است."
)


def _app_arity(app: Callable) -> int:
    """Bot contract: fn(message) for stateful bots, fn(message, history) otherwise."""
    try:
        return len(
            [
                p
                for p in inspect.signature(app).parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
        )
    except (TypeError, ValueError):
        return 1


def _call_app(app: Callable, arity: int, message: str, history: list[dict]) -> str:
    result = app(message) if arity < 2 else app(message, history)
    if inspect.isawaitable(result):  # async bots work here too, not just in AppEvaluator
        result = _sync_await(result)
    if isinstance(result, Trace):
        return result.final_output
    if isinstance(result, list):
        return Trace.from_messages(result).final_output
    return str(result)


def _render(history: list[dict]) -> str:
    names = {"user": "کاربر", "assistant": "دستیار"}
    return "\n".join(f"{names[m['role']]}: {m['content']}" for m in history)


def _parse_user_spec(user: "PersianUser | str | None") -> PersianUser:
    if isinstance(user, PersianUser):
        return user
    if isinstance(user, str):
        parts = [p.strip() for p in user.split("+") if p.strip()]
        style, traps = None, []
        for part in parts:
            if part in TRAPS:
                traps.append(part)
            elif style is None:
                style = part  # first free-text part is the register
            else:
                traps.append(part)  # later free-text parts are extra instructions
        return PersianUser(style=style or "محاوره‌ای", traps=traps)
    return PersianUser()


class SimulationEvaluator:
    """
    SimulationEvaluator simulates Persian users against a bot and judges the
    finished conversations against each goal and its criteria.

    Attributes:
        goldens (list[ConversationGolden]): The multi-turn expectations.
        user (PersianUser): The simulated user's personality. Also accepts a
            string like "محاوره‌ای+finglish_switch" mixing a register with trap
            names (see TRAPS); free text becomes extra instructions.
        simulator_model (Model | Callable | str, optional): The user-simulator
            LLM (falls back to PARSBENCH_SIMULATOR, then PARSBENCH_JUDGE env).
        judge (Model | Callable | str, optional): The conversation judge
            (falls back to the PARSBENCH_JUDGE env var).

    Methods:
        evaluate: Simulates the conversations against the app and scores them.
    """

    def __init__(
        self,
        goldens: list[ConversationGolden | dict] | None = None,
        goal: str | None = None,
        criteria: list[str] | None = None,
        user: PersianUser | str | None = None,
        simulator_model: Any = None,
        judge: Any = None,
    ):
        self.goldens = [
            g if isinstance(g, ConversationGolden) else ConversationGolden.from_dict(g)
            for g in (goldens or [])
        ]
        if goal:
            self.goldens.append(ConversationGolden(goal=goal, criteria=criteria or []))
        if not self.goldens:
            raise ValueError(
                "goldens is empty. You should provide a goal or at least one "
                "ConversationGolden."
            )
        self.user = _parse_user_spec(user)
        self.simulator_model = simulator_model
        self.judge = judge

    def evaluate(
        self,
        app: Callable,
        n_runs: int = 1,
        max_turns: int | None = None,
        save_evaluation: bool = False,
        output_path: str | None = None,
        record: bool = True,
    ) -> AppEvaluationResult:
        """
        Simulate each golden's conversation against the app and judge it.

        Parameters:
            app (Callable): The bot under evaluation — `fn(message)` for
                stateful bots or `fn(message, history)`.
            n_runs (int, optional): Repeated conversations per golden
                (default is 1); see `AppEvaluationResult.pass_hat_k`.
            max_turns (int, optional): Turn cap override; defaults to each
                golden's own `max_turns`.
            save_evaluation (bool, optional): Flag to save the evaluation
                result (default is False).
            output_path (str, optional): The output path to save the
                evaluation result.
            record (bool, optional): Record this run into the local
                `.parsbench` store for `parsbench view` (default is True;
                also disabled by the PARSBENCH_NO_RECORD env var).

        Returns:
            AppEvaluationResult: The evaluation result over all conversations.
        """
        if n_runs < 1:
            raise ValueError("n_runs must be at least 1.")
        if save_evaluation and not output_path:
            raise Exception("You should set the output path to save the evaluation.")

        # the user simulator stays at the provider's default temperature —
        # pinning it to 0 would replay near-identical conversations and blind
        # pass_hat_k
        simulator = resolve_model(
            self.simulator_model, "PARSBENCH_SIMULATOR", "PARSBENCH_JUDGE",
            temperature=None,
        )
        if simulator is None:
            raise ValueError(
                "simulator model not configured — pass simulator_model= or set "
                "PARSBENCH_SIMULATOR."
            )
        judge = resolve_model(self.judge, "PARSBENCH_JUDGE")

        recorder = (
            RunRecorder.start(
                kind="simulation",
                app_name=getattr(app, "__name__", type(app).__name__),
                n_goldens=len(self.goldens),
                n_runs=n_runs,
                extra_meta={
                    "user": {
                        "style": self.user.style,
                        "persona": self.user.persona,
                        "traps": self.user.traps,
                    }
                },
            )
            if record
            else None
        )

        arity = _app_arity(app)
        golden_results = []
        try:
            for golden_index, golden in enumerate(
                tqdm(self.goldens, desc="Simulating conversations")
            ):
                turn_cap = max_turns or golden.max_turns
                run_passes: list[bool] = []
                detail: list[CheckResult] | None = None
                transcript = None
                for run_index in range(n_runs):
                    rendered: str | None = None
                    run_trace: Trace | None = None
                    converged: bool | None = None
                    try:
                        history, converged = self._run_conversation(
                            app, arity, simulator, golden, turn_cap
                        )
                    except Exception as exc:  # a crashing bot is a finding
                        checks = [
                            CheckResult(
                                check="app_error",
                                passed=False,
                                reason=f"{type(exc).__name__}: {exc}",
                            )
                        ]
                        converged = None
                    else:
                        checks = self._score_conversation(
                            history, converged, golden, judge
                        )
                        rendered = _render(history)
                        run_trace = Trace.from_messages(history)
                        if transcript is None:
                            transcript = rendered
                    if recorder:
                        recorder.record_event(
                            golden, golden_index, run_index, checks,
                            trace=run_trace, transcript=rendered,
                            converged=converged,
                        )
                    if detail is None:
                        detail = checks
                    run_passes.append(all(c.passed for c in checks if not c.skipped))
                golden_results.append(
                    GoldenEvaluationResult(
                        golden_name=golden.label,
                        check_results=detail or [],
                        run_passes=run_passes,
                        transcript=transcript,
                    )
                )

            evaluation_result = AppEvaluationResult(golden_results=golden_results)
        except BaseException as exc:
            if recorder:
                recorder.crashed(exc)
            raise

        if recorder:
            recorder.finish(evaluation_result)

        if save_evaluation and output_path:
            evaluation_result.save(output_path)

        return evaluation_result

    def _run_conversation(
        self,
        app: Callable,
        arity: int,
        simulator: Callable,
        golden: ConversationGolden,
        max_turns: int,
    ) -> tuple[list[dict], bool]:
        system = self.user.system_prompt(golden)
        history: list[dict] = []
        converged = False
        for _ in range(max_turns):
            prompt = system
            if history:
                prompt += "\n\nگفتگو تا این لحظه:\n" + _render(history)
            prompt += "\n\nپیام بعدی کاربر:"
            user_msg = str(simulator(prompt)).strip()
            if DONE in user_msg:
                converged = True
                break
            history.append({"role": "user", "content": user_msg})
            history.append(
                {"role": "assistant",
                 "content": _call_app(app, arity, user_msg, history[:-1])}
            )
        return history, converged

    def _score_conversation(
        self,
        history: list[dict],
        converged: bool,
        golden: ConversationGolden,
        judge: Any,
    ) -> list[CheckResult]:
        transcript = _render(history)
        specs = [
            ("goal", GOAL_EVAL_FA.format(
                transcript=transcript, goal=golden.goal,
                expected=f"نتیجهٔ مطلوب: {golden.expected_outcome}\n"
                if golden.expected_outcome else "",
            ))
        ]
        specs += [
            (f"criterion:{c[:24]}",
             CRITERION_EVAL_FA.format(transcript=transcript, criterion=c))
            for c in golden.criteria
        ]
        judged = run_judge_specs(judge, specs)
        goal = judged[0]
        # a chatty simulator that never emits DONE must not fail a conversation
        # the judge scored as successful — the turn cap merely cut the chat short
        reached = converged or (not goal.skipped and goal.passed)
        if not reached:
            reason = "شبیه‌ساز کاربر به هدف نرسید (سقف نوبت‌ها)."
        elif not converged:
            reason = "سقف نوبت‌ها پر شد ولی داور هدف را برآورده‌شده ارزیابی کرد."
        else:
            reason = None
        out = [CheckResult(check="converged", score=float(reached), passed=reached,
                           reason=reason)]
        out.extend(judged)
        return out
