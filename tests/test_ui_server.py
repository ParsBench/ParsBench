"""The parsbench view server: RunStore reading + JSON API + SSE + static."""

import http.client
import json
import os
import threading
import time

import pytest

from parsbench.ui.server import STALE_AFTER, RunStore, make_server


def _make_run(root, run_id, status="finished", started_at="2026-08-17T10:00:00+03:30",
              summary=None, events=()):
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    meta = {
        "run_id": run_id, "status": status, "kind": "evaluation",
        "app_name": "bot", "started_at": started_at, "finished_at": None,
        "n_goldens": 1, "n_runs": 1, "metrics": None,
        "parsbench_version": "0.3.0", "summary": summary,
    }
    (run_dir / "run.json").write_text(json.dumps(meta, ensure_ascii=False),
                                      encoding="utf-8")
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return run_dir


def _event(seq, name="گلدن", passed=True):
    return {
        "seq": seq, "ts": "2026-08-17T10:00:01+03:30", "golden_name": name,
        "golden_index": 0, "run_index": seq - 1, "golden": {"input": name},
        "check_results": [{"check": "contains", "score": 1.0 if passed else 0.0,
                           "passed": passed, "skipped": False, "reason": None}],
        "trace": {"messages": [], "final_output": "جواب", "latency": 0.1,
                  "cost": None},
        "transcript": None, "converged": None,
    }


@pytest.fixture
def store_root(tmp_path):
    root = tmp_path / ".parsbench"
    _make_run(root, "20260817-100000-old-aaaa",
              started_at="2026-08-17T10:00:00+03:30", events=[_event(1)])
    _make_run(root, "20260817-110000-new-bbbb",
              started_at="2026-08-17T11:00:00+03:30",
              events=[_event(1), _event(2, passed=False)])
    return root


@pytest.fixture
def server(store_root):
    srv = make_server(store_root, port=0, poll_interval=0.05)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _get(server, path):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1],
                                      timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body


def _get_json(server, path):
    status, body = _get(server, path)
    return status, json.loads(body)


# --- RunStore ----------------------------------------------------------------

def test_store_accepts_containing_directory(store_root):
    assert len(RunStore(store_root.parent).list_runs()) == 2  # finds .parsbench


def test_list_runs_newest_first_with_stale_flag(store_root):
    runs = RunStore(store_root).list_runs()
    assert [r["run_id"] for r in runs] == ["20260817-110000-new-bbbb",
                                           "20260817-100000-old-aaaa"]
    assert all(r["stale"] is False for r in runs)


def test_running_run_goes_stale_without_growth(store_root):
    run_dir = _make_run(store_root, "20260817-120000-live-cccc",
                        status="running", events=[_event(1)])
    old = time.time() - STALE_AFTER - 60
    os.utime(run_dir / "events.jsonl", (old, old))
    run = next(r for r in RunStore(store_root).list_runs()
               if r["run_id"] == "20260817-120000-live-cccc")
    assert run["stale"] is True


def test_unreadable_run_listed_not_fatal(store_root):
    bad = store_root / "runs" / "20260817-130000-bad-dddd"
    bad.mkdir()
    (bad / "run.json").write_text("{not json", encoding="utf-8")
    runs = RunStore(store_root).list_runs()
    assert any(r["status"] == "unreadable" for r in runs)
    assert len(runs) == 3


def test_torn_tail_line_skipped(store_root):
    path = store_root / "runs" / "20260817-100000-old-aaaa" / "events.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write('{"seq": 99, "golden')  # crash mid-write
    events = RunStore(store_root).get_events("20260817-100000-old-aaaa")
    assert [e["seq"] for e in events] == [1]


def test_since_filters_events(store_root):
    events = RunStore(store_root).get_events("20260817-110000-new-bbbb", since=1)
    assert [e["seq"] for e in events] == [2]


def test_run_id_traversal_rejected(store_root):
    store = RunStore(store_root)
    assert store.get_run("..") is None
    assert store.get_run("../../etc") is None
    assert store.get_run("x/y") is None


# --- HTTP API ----------------------------------------------------------------

def test_api_runs(server):
    status, runs = _get_json(server, "/api/runs")
    assert status == 200
    assert [r["run_id"] for r in runs] == ["20260817-110000-new-bbbb",
                                           "20260817-100000-old-aaaa"]


def test_api_run_detail_is_utf8_persian(server):
    status, body = _get(server, "/api/runs/20260817-110000-new-bbbb")
    assert status == 200
    assert "گلدن".encode() in body  # ensure_ascii=False on the wire
    data = json.loads(body)
    assert data["run"]["run_id"] == "20260817-110000-new-bbbb"
    assert len(data["events"]) == 2


def test_api_run_events_since(server):
    status, events = _get_json(
        server, "/api/runs/20260817-110000-new-bbbb/events?since=1")
    assert status == 200
    assert [e["seq"] for e in events] == [2]


def test_api_unknown_run_404(server):
    status, _ = _get_json(server, "/api/runs/nope")
    assert status == 404


# --- SSE + static ------------------------------------------------------------

def test_stream_announces_new_runs_and_events(server, store_root):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1],
                                      timeout=10)
    conn.request("GET", "/api/stream")
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("Content-Type").startswith("text/event-stream")
    # wait for the snapshot boundary — a run created before it would be
    # treated as pre-existing and never announced
    connected = b""
    while b"connected" not in connected:
        connected += resp.fp.readline()

    _make_run(store_root, "20260817-140000-live-eeee", status="running",
              events=[_event(1)])

    seen = b""
    deadline = time.time() + 8
    while time.time() < deadline and b"golden-finished" not in seen:
        seen += resp.fp.readline()
    conn.close()
    assert b"event: run-started" in seen
    assert b"event: golden-finished" in seen
    assert b"20260817-140000-live-eeee" in seen


def test_index_served(server):
    status, body = _get(server, "/")
    assert status == 200
    assert b"ParsBench" in body


def test_unknown_static_404(server):
    status, _ = _get(server, "/nope.js")
    assert status == 404


def test_static_traversal_rejected(server):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1],
                                      timeout=5)
    conn.request("GET", "/../pyproject.toml")
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    assert resp.status in (400, 404)
    assert b"[tool.poetry]" not in body


# --- serve/CLI/packaging -----------------------------------------------------

from importlib import resources

from parsbench.ui.server import _bind


def test_bind_walks_past_busy_ports(store_root):
    s1 = _bind(store_root, "127.0.0.1", 43217)
    try:
        start = s1.server_address[1]  # 43217 unless something else holds it
        s2 = _bind(store_root, "127.0.0.1", start)
        try:
            assert s2.server_address[1] > start  # walked past the busy port
        finally:
            s2.server_close()
    finally:
        s1.server_close()


def test_cli_view_calls_serve(monkeypatch, tmp_path):
    import parsbench.cli
    import parsbench.ui.server

    called = {}

    def fake_serve(root, host, port, open_browser):
        called.update(root=root, host=host, port=port, open_browser=open_browser)

    monkeypatch.setattr(parsbench.ui.server, "serve", fake_serve)
    parsbench.cli.main(["view", str(tmp_path), "--no-open", "--port", "2000"])
    assert called == {"root": str(tmp_path), "host": "127.0.0.1",
                      "port": 2000, "open_browser": False}


def test_ui_assets_ship_with_the_package():
    static = resources.files("parsbench.ui").joinpath("static")
    assert static.joinpath("index.html").is_file()


def test_spa_assets_ship_and_serve(server):
    for path, must_contain in [
        ("/app.js", b"render"),
        ("/api.js", b"EventSource"),
        ("/style.css", b"#009485"),
        ("/vendor/preact.standalone.module.js", b"useState"),
        ("/export.js", b"downloadRun"),
        ("/views/charts.js", b"svg"),
        ("/util.js", b"checkMeans"),
        ("/logo.svg", b"<svg"),
        ("/fonts/Vazirmatn.woff2", b"wOF2"),
    ]:
        status, body = _get(server, path)
        assert status == 200, path
        assert must_contain in body, path
