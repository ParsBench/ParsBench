"""CLI: `parsbench test` wraps pytest."""

import parsbench.cli as cli


def test_cli_test_runs_pytest(monkeypatch):
    seen = {}

    def fake_call(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    assert cli.main(["test", "tests/my_evals.py"]) == 0
    assert seen["args"][-1] == "tests/my_evals.py"


def test_cli_test_defaults_to_pytest_discovery(monkeypatch):
    import sys

    seen = {}

    def fake_call(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    assert cli.main(["test"]) == 0
    assert seen["args"] == [sys.executable, "-m", "pytest"]  # pytest's own discovery


def test_cli_test_without_pytest_explains(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_pytest_available", lambda: False)
    assert cli.main(["test"]) == 1
    assert "pip install pytest" in capsys.readouterr().err
