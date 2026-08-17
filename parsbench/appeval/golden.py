"""Golden expectations: the inputs of an app evaluation."""

from dataclasses import dataclass, field
from typing import Any, Callable

from .trace import ToolCall, Trace


def _as_list(value: Any) -> list:
    if isinstance(value, str):
        return [value]
    return list(value)


@dataclass
class Golden:
    """
    One expectation for the evaluated app. Every field except `input` is
    optional; each filled field switches on its corresponding check.

    Attributes:
        input (str): The user message sent to the app.
        name (str, optional): A readable name for reports.
        output (str, optional): Reference answer; judged by the `correctness` check.
        contains (list[str]): Substrings the answer must state — normalized,
            money- and calendar-equivalent.
        not_contains (list[str]): Substrings the answer must not state.
        format (type, optional): A pydantic-style model class the answer must
            validate against (checked via `model_validate`).
        tools (list[ToolCall]): Tool calls the app must make; arguments are
            compared date -> number -> normalized text.
        forbidden_tools (list[str]): Tools the app must not call.
        context (list[str]): Grounding context; judged by the `faithfulness` check.
        refuses (bool): Whether the app must decline the request (`refusal` check).
        max_steps (int, optional): Budget on assistant steps.
        max_latency (float, optional): Budget on wall-clock seconds.
        max_cost (float, optional): Budget on cost.
        check (Callable, optional): Custom `Trace -> bool | float` check.
        tags (list[str]): Free-form labels.

    Methods:
        from_dict: Builds a Golden from a dict, accepting the short `in`/`out`
            aliases for `input`/`output`.
    """

    input: str
    name: str | None = None
    # answer expectations
    output: str | None = None
    contains: list[str] = field(default_factory=list)
    not_contains: list[str] = field(default_factory=list)
    format: type | None = None
    # behavior expectations
    tools: list[ToolCall] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    refuses: bool = False
    # budgets
    max_steps: int | None = None
    max_latency: float | None = None
    max_cost: float | None = None
    # escape hatch
    check: Callable[[Trace], bool | float] | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        # a bare string for a list field must not be iterated character by
        # character — wrap it; dict tool specs are promoted to ToolCall
        self.contains = _as_list(self.contains)
        self.not_contains = _as_list(self.not_contains)
        self.forbidden_tools = _as_list(self.forbidden_tools)
        self.context = _as_list(self.context)
        self.tags = _as_list(self.tags)
        self.tools = [
            tc if isinstance(tc, ToolCall) else ToolCall(**tc) for tc in self.tools
        ]

    @classmethod
    def from_dict(cls, data: dict) -> "Golden":
        data = dict(data)
        if "in" in data:
            data["input"] = data.pop("in")
        if "out" in data:
            data["output"] = data.pop("out")
        return cls(**data)

    @property
    def label(self) -> str:
        return self.name or (self.input[:40] + "…" if len(self.input) > 40 else self.input)


@dataclass
class ConversationGolden:
    """
    A multi-turn expectation, consumed by SimulationEvaluator.

    Attributes:
        goal (str): What the simulated user wants from the conversation.
        name (str, optional): A readable name for reports.
        scenario (str): Extra situation description for the user simulator.
        expected_outcome (str): What "done well" looks like, for the judge.
        criteria (list[str]): Behaviors judged each as their own check.
        max_turns (int): Turn cap for the conversation.
    """

    goal: str
    name: str | None = None
    scenario: str = ""
    expected_outcome: str = ""
    criteria: list[str] = field(default_factory=list)
    max_turns: int = 8

    def __post_init__(self):
        self.criteria = _as_list(self.criteria)

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationGolden":
        return cls(**data)

    @property
    def label(self) -> str:
        return self.name or (self.goal[:40] + "…" if len(self.goal) > 40 else self.goal)
