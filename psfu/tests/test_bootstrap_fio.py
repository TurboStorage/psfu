"""S4 bootstrap tests: ensure_fio smoke-test path.

ensure_fio() is called in ``ensure_runtime_dependencies`` and is
responsible for validating the vendored fio binary before any FioRunner
tries to use it. The smoke test is just ``fio --version``; we cover the
happy path and the version-string-missing failure so a broken build
can't silently pass the bootstrap and then blow up the phase loop.
"""
from __future__ import annotations

from typing import List

from psfu.bootstrap.build_tools import ensure_fio, fio_smoke_test


def test_fio_smoke_test_parses_version_output(monkeypatch) -> None:
    monkeypatch.setattr(
        "psfu.bootstrap.build_tools.run_capture_env",
        lambda cmd, timeout=None, **kw: (0, "fio-3.36", ""),
    )
    ok, detail = fio_smoke_test("/usr/bin/fio")
    assert ok is True
    assert "fio-3.36" in detail


def test_fio_smoke_test_rejects_missing_version_string(monkeypatch) -> None:
    monkeypatch.setattr(
        "psfu.bootstrap.build_tools.run_capture_env",
        lambda cmd, timeout=None, **kw: (0, "gibberish output", ""),
    )
    ok, detail = fio_smoke_test("/usr/bin/fio")
    assert ok is False
    assert "version output looked wrong" in detail


def test_fio_smoke_test_rejects_nonzero_exit(monkeypatch) -> None:
    monkeypatch.setattr(
        "psfu.bootstrap.build_tools.run_capture_env",
        lambda cmd, timeout=None, **kw: (127, "", "command not found"),
    )
    ok, detail = fio_smoke_test("/usr/bin/fio")
    assert ok is False
    assert "rc=127" in detail


def test_ensure_fio_reports_available_when_binary_present(monkeypatch) -> None:
    monkeypatch.setattr("psfu.bootstrap.build_tools.is_linux", lambda: True)
    monkeypatch.setattr(
        "psfu.bootstrap.build_tools.fio_binary_path",
        lambda: "/usr/bin/fio",
    )
    actions: List[str] = []
    ensure_fio(actions, build=False)
    assert any("fio available" in a for a in actions)


def test_ensure_fio_reports_missing_without_build_flag(monkeypatch) -> None:
    monkeypatch.setattr("psfu.bootstrap.build_tools.is_linux", lambda: True)
    monkeypatch.setattr("psfu.bootstrap.build_tools.fio_binary_path", lambda: None)
    actions: List[str] = []
    ensure_fio(actions, build=False)
    assert any("fio not found" in a for a in actions)


def test_ensure_fio_build_validates_existing_binary_with_smoke(monkeypatch) -> None:
    """With build=True and a present binary, ensure_fio smoke-tests it and
    reports ``usable`` rather than forcing a rebuild."""
    monkeypatch.setattr("psfu.bootstrap.build_tools.is_linux", lambda: True)
    monkeypatch.setattr(
        "psfu.bootstrap.build_tools.fio_binary_path",
        lambda: "/usr/bin/fio",
    )
    monkeypatch.setattr(
        "psfu.bootstrap.build_tools.run_capture_env",
        lambda cmd, timeout=None, **kw: (0, "fio-3.37", ""),
    )
    actions: List[str] = []
    ensure_fio(actions, build=True)
    assert any("fio usable" in a and "smoke test passed" in a for a in actions)
