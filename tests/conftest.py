"""Shared test config."""

import pytest


@pytest.fixture(autouse=True)
def _no_recording(monkeypatch):
    # keep the suite from writing .parsbench stores into the repo cwd;
    # recording tests opt back in with monkeypatch.delenv + chdir(tmp_path)
    monkeypatch.setenv("PARSBENCH_NO_RECORD", "1")
