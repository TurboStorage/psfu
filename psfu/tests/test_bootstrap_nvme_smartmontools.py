"""Bootstrap tests for ensure_nvme_cli and ensure_smartmontools.

Covers the smoke-test functions and the run-mode / bootstrap-mode
behaviour of ensure_nvme_cli / ensure_smartmontools, mirroring the
pattern in test_bootstrap_fio.py.
"""
from __future__ import annotations

from typing import List

from psfu.bootstrap.build_tools import (
    ensure_nvme_cli,
    ensure_smartmontools,
    nvme_cli_smoke_test,
    smartctl_smoke_test,
)


# ---------------------------------------------------------------------------
# nvme_cli_smoke_test
# ---------------------------------------------------------------------------


def test_nvme_cli_smoke_test_parses_version_output(monkeypatch) -> None:
    monkeypatch.setattr(
        "psfu.bootstrap.build_tools.run_capture_env",
        lambda cmd, timeout=None, **kw: (0, "nvme version 2.10", ""),
    )
    ok, detail = nvme_cli_smoke_test("/usr/bin/nvme")
    assert ok is True
    assert "nvme version 2.10" in detail


def test_nvme_cli_smoke_test_rejects_missing_version_string(monkeypatch) -> None:
    monkeypatch.setattr(
        "psfu.bootstrap.build_tools.run_capture_env",
        lambda cmd, timeout=None, **kw: (0, "gibberish output", ""),
    )
    ok, detail = nvme_cli_smoke_test("/usr/bin/nvme")
    assert ok is False
    assert "version output looked wrong" in detail


def test_nvme_cli_smoke_test_rejects_nonzero_exit(monkeypatch) -> None:
    monkeypatch.setattr(
        "psfu.bootstrap.build_tools.run_capture_env",
        lambda cmd, timeout=None, **kw: (127, "", "command not found"),
    )
    ok, detail = nvme_cli_smoke_test("/usr/bin/nvme")
    assert ok is False
    assert "rc=127" in detail


# ---------------------------------------------------------------------------
# ensure_nvme_cli
# ---------------------------------------------------------------------------


def test_ensure_nvme_cli_reports_available_when_binary_present(monkeypatch) -> None:
    monkeypatch.setattr("psfu.bootstrap.build_tools.is_linux", lambda: True)
    monkeypatch.setattr("psfu.bootstrap.build_tools.nvme_cli_binary_path", lambda: "/usr/bin/nvme")
    actions: List[str] = []
    ensure_nvme_cli(actions, build=False)
    assert any("nvme-cli available" in a for a in actions)


def test_ensure_nvme_cli_reports_missing_without_build_flag(monkeypatch) -> None:
    monkeypatch.setattr("psfu.bootstrap.build_tools.is_linux", lambda: True)
    monkeypatch.setattr("psfu.bootstrap.build_tools.nvme_cli_binary_path", lambda: None)
    actions: List[str] = []
    ensure_nvme_cli(actions, build=False)
    assert any("nvme-cli not found" in a for a in actions)


def test_ensure_nvme_cli_build_validates_existing_binary_with_smoke(monkeypatch) -> None:
    monkeypatch.setattr("psfu.bootstrap.build_tools.is_linux", lambda: True)
    monkeypatch.setattr("psfu.bootstrap.build_tools.nvme_cli_binary_path", lambda: "/usr/bin/nvme")
    monkeypatch.setattr(
        "psfu.bootstrap.build_tools.run_capture_env",
        lambda cmd, timeout=None, **kw: (0, "nvme version 2.10", ""),
    )
    actions: List[str] = []
    ensure_nvme_cli(actions, build=True)
    assert any("nvme-cli usable" in a and "smoke test passed" in a for a in actions)


# ---------------------------------------------------------------------------
# smartctl_smoke_test
# ---------------------------------------------------------------------------


def test_smartctl_smoke_test_parses_version_output(monkeypatch) -> None:
    monkeypatch.setattr(
        "psfu.bootstrap.build_tools.run_capture_env",
        lambda cmd, timeout=None, **kw: (0, "smartctl 7.4 2023-08-01", ""),
    )
    ok, detail = smartctl_smoke_test("/usr/bin/smartctl")
    assert ok is True
    assert "smartctl 7.4" in detail


def test_smartctl_smoke_test_rejects_missing_version_string(monkeypatch) -> None:
    monkeypatch.setattr(
        "psfu.bootstrap.build_tools.run_capture_env",
        lambda cmd, timeout=None, **kw: (0, "gibberish output", ""),
    )
    ok, detail = smartctl_smoke_test("/usr/bin/smartctl")
    assert ok is False
    assert "unexpected output" in detail


def test_smartctl_smoke_test_rejects_bad_exit(monkeypatch) -> None:
    monkeypatch.setattr(
        "psfu.bootstrap.build_tools.run_capture_env",
        lambda cmd, timeout=None, **kw: (127, "", "command not found"),
    )
    ok, detail = smartctl_smoke_test("/usr/bin/smartctl")
    assert ok is False
    assert "rc=127" in detail


# ---------------------------------------------------------------------------
# ensure_smartmontools
# ---------------------------------------------------------------------------


def test_ensure_smartmontools_reports_available_when_binary_present(monkeypatch) -> None:
    monkeypatch.setattr("psfu.bootstrap.build_tools.is_linux", lambda: True)
    monkeypatch.setattr("psfu.bootstrap.build_tools.smartctl_binary_path", lambda: "/usr/bin/smartctl")
    actions: List[str] = []
    ensure_smartmontools(actions, build=False)
    assert any("smartmontools available" in a for a in actions)


def test_ensure_smartmontools_reports_missing_without_build_flag(monkeypatch) -> None:
    monkeypatch.setattr("psfu.bootstrap.build_tools.is_linux", lambda: True)
    monkeypatch.setattr("psfu.bootstrap.build_tools.smartctl_binary_path", lambda: None)
    actions: List[str] = []
    ensure_smartmontools(actions, build=False)
    assert any("smartmontools not found" in a for a in actions)


def test_ensure_smartmontools_build_validates_existing_binary_with_smoke(monkeypatch) -> None:
    monkeypatch.setattr("psfu.bootstrap.build_tools.is_linux", lambda: True)
    monkeypatch.setattr("psfu.bootstrap.build_tools.smartctl_binary_path", lambda: "/usr/bin/smartctl")
    monkeypatch.setattr(
        "psfu.bootstrap.build_tools.run_capture_env",
        lambda cmd, timeout=None, **kw: (0, "smartctl 7.4 2023-08-01", ""),
    )
    actions: List[str] = []
    ensure_smartmontools(actions, build=True)
    assert any("smartmontools usable" in a and "smoke test passed" in a for a in actions)
