"""Read-only HTTP server over the local run store — `parsbench view`.

Stdlib only. Serves the bundled SPA, a tiny JSON API, and an SSE stream
that tails the store; the .parsbench directory is the only input and
nothing here ever writes to it.
"""

import json
import mimetypes
import re
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlsplit

STALE_AFTER = 300.0  # seconds without event growth before a "running" run is stale
_SAFE_ID = re.compile(r"[\w.-]+")


class RunStore:
    """
    RunStore reads the .parsbench run store: run metadata, events, and
    liveness (a "running" run whose events stopped growing is stale, not
    live — the process may have died without finishing).

    Methods:
        list_runs: All runs' metadata, newest first.
        get_run: One run's metadata plus all its events.
        get_events: One run's events, optionally after a sequence number.
    """

    def __init__(self, root: Path | str):
        root = Path(root)
        if (root / ".parsbench").is_dir():  # accept the containing directory too
            root = root / ".parsbench"
        self.root = root

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def list_runs(self) -> list[dict]:
        if not self.runs_dir.is_dir():
            return []
        metas = [
            self._run_meta(run_dir)
            for run_dir in self.runs_dir.iterdir()
            if run_dir.is_dir()
        ]
        return sorted(metas, key=lambda m: m.get("started_at") or "", reverse=True)

    def get_run(self, run_id: str) -> dict | None:
        run_dir = self._run_dir(run_id)
        if run_dir is None:
            return None
        return {"run": self._run_meta(run_dir), "events": self.get_events(run_id)}

    def get_events(self, run_id: str, since: int = 0) -> list[dict] | None:
        run_dir = self._run_dir(run_id)
        if run_dir is None:
            return None
        events = []
        path = run_dir / "events.jsonl"
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # torn tail line — only the tail can be damaged
                    if event.get("seq", 0) > since:
                        events.append(event)
        return events

    def _run_dir(self, run_id: str) -> Path | None:
        if not _SAFE_ID.fullmatch(run_id) or run_id in (".", ".."):
            return None
        run_dir = self.runs_dir / run_id
        return run_dir if run_dir.is_dir() else None

    def _run_meta(self, run_dir: Path) -> dict:
        try:
            meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        except Exception:
            return {"run_id": run_dir.name, "status": "unreadable", "stale": False}
        meta["stale"] = self._is_stale(run_dir, meta)
        return meta

    def _is_stale(self, run_dir: Path, meta: dict) -> bool:
        if meta.get("status") != "running":
            return False
        events = run_dir / "events.jsonl"
        ref = events if events.exists() else run_dir / "run.json"
        try:
            return (time.time() - ref.stat().st_mtime) > STALE_AFTER
        except OSError:
            return True


class Handler(BaseHTTPRequestHandler):
    """Routes /api/runs, /api/runs/<id>[/events], /api/stream, and static files."""

    store: ClassVar[RunStore]
    poll_interval: ClassVar[float] = 0.5

    def log_message(self, format: str, *args: Any) -> None:
        pass  # a local viewer should not spam the terminal per request

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        try:
            if parts == ["api", "runs"]:
                self._json(self.store.list_runs())
            elif len(parts) == 3 and parts[:2] == ["api", "runs"]:
                run = self.store.get_run(parts[2])
                if run is None:
                    self._json({"error": "run not found"}, 404)
                else:
                    self._json(run)
            elif (len(parts) == 4 and parts[:2] == ["api", "runs"]
                  and parts[3] == "events"):
                since = int(parse_qs(parsed.query).get("since", ["0"])[0] or 0)
                events = self.store.get_events(parts[2], since=since)
                if events is None:
                    self._json({"error": "run not found"}, 404)
                else:
                    self._json(events)
            elif parts == ["api", "stream"]:
                self._stream()
            else:
                self._static(parts)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away — routine for SSE and page refreshes

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        # snapshot what already exists — the stream only pushes what changes
        known: dict[str, tuple[str, int]] = {}
        for meta in self.store.list_runs():
            rid = meta.get("run_id", "")
            events = self.store.get_events(rid) or []
            known[rid] = (
                meta.get("status", ""),
                max((e.get("seq", 0) for e in events), default=0),
            )
        # snapshot done — anything appearing from here on gets pushed. The
        # comment lets clients (and tests) know the boundary deterministically.
        self.wfile.write(b": connected\n\n")
        self.wfile.flush()
        last_beat = time.monotonic()
        while True:
            for meta in self.store.list_runs():
                rid = meta.get("run_id", "")
                status = meta.get("status", "")
                prev_status, prev_seq = known.get(rid, (None, 0))
                if prev_status is None:
                    self._sse("run-started", meta)
                for event in self.store.get_events(rid, since=prev_seq) or []:
                    self._sse("golden-finished", {"run_id": rid, "event": event})
                    prev_seq = max(prev_seq, event.get("seq", 0))
                if prev_status not in (None, status) and status != "running":
                    self._sse("run-finished", meta)
                known[rid] = (status, prev_seq)
            if time.monotonic() - last_beat > 15:
                self.wfile.write(b": beat\n\n")
                self.wfile.flush()
                last_beat = time.monotonic()
            time.sleep(self.poll_interval)

    def _sse(self, event: str, data: Any) -> None:
        payload = json.dumps(data, ensure_ascii=False)
        self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode())
        self.wfile.flush()

    def _static(self, parts: list[str]) -> None:
        rel = parts or ["index.html"]
        if any(p == ".." for p in rel):
            self._json({"error": "bad path"}, 400)
            return
        resource = resources.files("parsbench.ui").joinpath("static")
        for part in rel:
            resource = resource.joinpath(part)
        if not resource.is_file():
            self._json({"error": "not found"}, 404)
            return
        body = resource.read_bytes()
        ctype = mimetypes.guess_type(rel[-1])[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # revalidate every load — a browser must never show a stale UI after
        # a parsbench upgrade (the assets ship inside the wheel)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def make_server(
    root: Path | str,
    host: str = "127.0.0.1",
    port: int = 0,
    poll_interval: float = 0.5,
) -> ThreadingHTTPServer:
    """Build (but do not start) a viewer server bound to host:port."""
    store = RunStore(root)

    class BoundHandler(Handler):
        pass

    BoundHandler.store = store
    BoundHandler.poll_interval = poll_interval
    return ThreadingHTTPServer((host, port), BoundHandler)


def _bind(root: Path | str, host: str, port: int) -> ThreadingHTTPServer:
    """Bind to the first free port at or above `port`."""
    while True:
        try:
            return make_server(root, host=host, port=port)
        except OSError:
            port += 1


def serve(
    root: Path | str,
    host: str = "127.0.0.1",
    port: int = 1404,
    open_browser: bool = True,
) -> None:
    """Serve the viewer until interrupted (the `parsbench view` entry point)."""
    server = _bind(root, host, port)
    url = f"http://{host}:{server.server_address[1]}"
    print(f"ParsBench viewer at {url}  (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
