"""Data classes describing what the evaluated app actually did: tool calls,
messages, and the whole trace. OpenAI chat format is the lingua franca."""

import json
from dataclasses import dataclass, field
from typing import Any

_ROLES = {"system", "user", "assistant", "tool"}
# roles other providers emit; anything else unknown is treated as the speaker
_ROLE_ALIASES = {"developer": "system", "function": "tool", "model": "assistant"}


@dataclass(init=False)
class ToolCall:
    """
    Represents an expected or observed tool call.

    Extra keyword arguments become tool arguments, so the short form
    `ToolCall("search", date="1405-07-05")` is equivalent to
    `ToolCall(name="search", arguments={"date": "1405-07-05"})`.

    Attributes:
        name (str): The name of the tool.
        arguments (dict): The arguments the tool was (or should be) called with.
        result (Any, optional): The observed tool result, if any.
        error (str, optional): The observed tool error, if any.
    """

    name: str
    arguments: dict[str, Any]
    result: Any
    error: str | None

    def __init__(
        self,
        name: str = "",
        arguments: dict[str, Any] | None = None,
        result: Any = None,
        error: str | None = None,
        **extra_arguments: Any,
    ):
        self.name = name
        self.arguments = {**(arguments or {}), **extra_arguments}
        self.result = result
        self.error = error


@dataclass
class Message:
    """
    A single message in a conversation trace.

    Attributes:
        role (str): One of "system", "user", "assistant", or "tool".
        content (str, optional): The text content of the message.
        tool_calls (list[ToolCall]): Tool calls made in this message.
        tool_call_id (str, optional): For tool messages, the id of the call
            this message answers.
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass
class Trace:
    """
    What the evaluated app actually did in response to one input.

    Attributes:
        messages (list[Message]): The conversation messages, OpenAI-style.
        final_output (str): The final assistant answer.
        latency (float, optional): Wall-clock seconds for the run.
        cost (float, optional): Cost of the run, if known.
        raw (Any, optional): The original framework object, untouched.

    Methods:
        from_messages: Builds a Trace from OpenAI-format message dicts.
    """

    messages: list[Message] = field(default_factory=list)
    final_output: str = ""
    latency: float | None = None
    cost: float | None = None
    raw: Any = None

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [tc for m in self.messages if m.role == "assistant" for tc in m.tool_calls]

    @property
    def n_steps(self) -> int:
        return sum(1 for m in self.messages if m.role == "assistant")

    @classmethod
    def from_messages(
        cls, messages: list[dict], *, final_output: Any = None, raw: Any = None
    ) -> "Trace":
        """
        Build a Trace from OpenAI-format message dicts.

        Parameters:
            messages (list[dict]): OpenAI chat-format messages.
            final_output (Any, optional): Overrides the derived final answer,
                but only when it is a non-empty string.
            raw (Any, optional): The original framework object to keep around.

        Returns:
            Trace: The parsed trace.
        """
        parsed: list[Message] = []
        calls_by_id: dict[str, ToolCall] = {}
        for m in messages:
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                call = ToolCall(fn.get("name", ""), arguments=args)
                calls.append(call)
                if tc.get("id"):
                    calls_by_id[tc["id"]] = call
            role = m.get("role", "assistant")
            role = _ROLE_ALIASES.get(role, role if role in _ROLES else "assistant")
            parsed.append(
                Message(
                    role=role,
                    content=m.get("content"),
                    tool_calls=calls,
                    tool_call_id=m.get("tool_call_id"),
                )
            )
            if role == "tool" and m.get("tool_call_id") in calls_by_id:
                calls_by_id[m["tool_call_id"]].result = m.get("content")
        final = next(
            (m.content for m in reversed(parsed) if m.role == "assistant" and m.content),
            "",
        )
        if isinstance(final_output, str) and final_output:
            final = final_output
        return cls(messages=parsed, final_output=final, raw=raw)
