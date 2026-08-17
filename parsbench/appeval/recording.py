"""RunRecorder: append finished goldens to the project-local run store.

The store (.parsbench/runs/<run_id>/, two files: run.json + events.jsonl)
is the whole contract between the evaluators and `parsbench view`. A
recording failure must never break an evaluation: `start` returns None on
failure and every method on a live recorder swallows its own errors,
warns once, and disables itself.
"""

import importlib.metadata
import json
import os
import re
import secrets
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from .evaluation_result import AppEvaluationResult, CheckResult

TRUNCATE_AT = 16 * 1024
_TRUNCATE_MARK = "… [truncated by parsbench]"


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > TRUNCATE_AT:
        return value[:TRUNCATE_AT] + _TRUNCATE_MARK
    return value


def golden_to_dict(golden: Any) -> dict:
    """Golden/ConversationGolden -> JSON-safe dict; callables/types by name."""
    out: dict[str, Any] = {}
    for key, value in vars(golden).items():
        if not value:  # drop empties/None/False so events stay small
            continue
        if key in ("check", "format"):
            out[key] = getattr(value, "__name__", str(value))
        elif key == "tools":
            out[key] = [{"name": t.name, "arguments": t.arguments} for t in value]
        else:
            out[key] = value
    return out


def trace_to_dict(trace: Any) -> dict:
    """Trace -> JSON-safe dict; `raw` dropped, long strings truncated."""
    return {
        "messages": [
            {
                "role": m.role,
                "content": _truncate(m.content),
                "tool_calls": [
                    {
                        "name": t.name,
                        "arguments": t.arguments,
                        "result": _truncate(t.result),
                        "error": t.error,
                    }
                    for t in m.tool_calls
                ],
                "tool_call_id": m.tool_call_id,
            }
            for m in trace.messages
        ],
        "final_output": _truncate(trace.final_output),
        "latency": trace.latency,
        "cost": trace.cost,
    }


def store_root() -> Path:
    """The run-store root: $PARSBENCH_DIR or ./.parsbench."""
    return Path(os.environ.get("PARSBENCH_DIR") or ".parsbench")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9؀-ۿ]+", "-", name).strip("-").lower()
    return slug[:32] or "run"


def _version() -> str:
    try:
        return importlib.metadata.version("parsbench")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class RunRecorder:
    """
    RunRecorder writes one evaluation run into the local run store, one
    appended event per finished golden×run, so `parsbench view` can show
    it live and after the fact.

    Methods:
        start: Opens a run directory and returns a recorder (or None).
        record_event: Appends one finished golden×run to events.jsonl.
        finish: Marks the run finished and writes the summary.
        crashed: Marks the run crashed with the error.
    """

    def __init__(self, run_dir: Path, meta: dict):
        self._dir = run_dir
        self._meta = meta
        self._lock = threading.Lock()
        self._seq = 0
        self._disabled = False
        self._events: TextIO = (run_dir / "events.jsonl").open("a", encoding="utf-8")

    @classmethod
    def start(
        cls,
        kind: str,
        app_name: str,
        n_goldens: int,
        n_runs: int = 1,
        metrics: list[str] | None = None,
        extra_meta: dict | None = None,
    ) -> "RunRecorder | None":
        if os.environ.get("PARSBENCH_NO_RECORD"):
            return None
        try:
            root = store_root()
            (root / "runs").mkdir(parents=True, exist_ok=True)
            gitignore = root / ".gitignore"
            if not gitignore.exists():  # the pytest-cache trick: self-ignoring dir
                gitignore.write_text("*\n", encoding="utf-8")
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_id = f"{stamp}-{_slug(app_name)}-{secrets.token_hex(2)}"
            run_dir = root / "runs" / run_id
            run_dir.mkdir()
            meta = {
                "run_id": run_id,
                "status": "running",
                "kind": kind,
                "app_name": app_name,
                "started_at": _now(),
                "finished_at": None,
                "n_goldens": n_goldens,
                "n_runs": n_runs,
                "metrics": metrics,
                "parsbench_version": _version(),
                "summary": None,
                **(extra_meta or {}),
            }
            recorder = cls(run_dir, meta)
            recorder._write_meta()
            return recorder
        except Exception as exc:
            warnings.warn(f"parsbench recording disabled: {exc}")
            return None

    @property
    def run_id(self) -> str:
        return self._meta["run_id"]

    def _write_meta(self) -> None:
        # atomic rewrite: a viewer polling run.json must never read a torn file
        tmp = self._dir / "run.json.tmp"
        tmp.write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self._dir / "run.json")

    def record_event(
        self,
        golden: Any,
        golden_index: int,
        run_index: int,
        check_results: list[CheckResult],
        trace: Any = None,
        transcript: str | None = None,
        converged: bool | None = None,
    ) -> None:
        if self._disabled:
            return
        try:
            event = {
                "seq": 0,  # patched under the lock below
                "ts": _now(),
                "golden_name": golden.label,
                "golden_index": golden_index,
                "run_index": run_index,
                "golden": golden_to_dict(golden),
                "check_results": [c.to_dict() for c in check_results],
                "trace": trace_to_dict(trace) if trace is not None else None,
                "transcript": _truncate(transcript) if transcript else transcript,
                "converged": converged,
            }
            with self._lock:
                self._seq += 1
                event["seq"] = self._seq
                self._events.write(
                    json.dumps(event, ensure_ascii=False, default=str) + "\n"
                )
                self._events.flush()  # whole lines land promptly — live tailing
        except Exception as exc:
            self._disable(exc)

    def _disable(self, exc: Exception) -> None:
        self._disabled = True
        warnings.warn(f"parsbench recording disabled: {exc}")

    def finish(self, result: AppEvaluationResult) -> None:
        if self._disabled:
            return
        try:
            n = len(result.golden_results)
            pass_rate = (
                sum(gr.passed for gr in result.golden_results) / n if n else 0.0
            )
            self._meta.update(
                status="finished",
                finished_at=_now(),
                summary={
                    "score": result.score(),
                    "passed": result.passed,
                    "pass_rate": pass_rate,
                },
            )
            self._write_meta()
            self._events.close()
        except Exception as exc:
            self._disable(exc)

    def crashed(self, exc_in: BaseException) -> None:
        if self._disabled:
            return
        try:
            self._meta.update(
                status="crashed",
                finished_at=_now(),
                summary={"error": f"{type(exc_in).__name__}: {exc_in}"},
            )
            self._write_meta()
            self._events.close()
        except Exception as exc:
            self._disable(exc)
