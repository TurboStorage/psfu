from __future__ import annotations

import json
from pathlib import Path

from psfu.storage.selftest import (
    SELFTEST_PASSED,
    SELFTEST_FAILED,
    SELFTEST_TIMEOUT,
    SELFTEST_ABORTED,
    SELFTEST_ERROR,
    run_selftest,
)


def _make_log(op_code: int = 0, result_code: int = 0, completion: int = 100) -> dict:
    return {
        "current_operation": {"value": op_code},
        "current_completion": completion,
        "table": [
            {"self_test_result": {"value": result_code}},
        ],
    }


def test_run_selftest_returns_error_when_nvme_cli_missing(monkeypatch) -> None:
    monkeypatch.setattr("psfu.storage.selftest.nvme_cli_binary_path", lambda: None)
    result, log = run_selftest("/dev/nvme0n1", verbose=False)
    assert result == SELFTEST_ERROR
    assert log is None


def test_run_selftest_returns_error_when_trigger_fails(monkeypatch) -> None:
    monkeypatch.setattr("psfu.storage.selftest.nvme_cli_binary_path", lambda: "/usr/bin/nvme")
    monkeypatch.setattr(
        "psfu.storage.selftest.run_capture",
        lambda cmd, timeout=15: (1, "", "io error"),
    )
    result, log = run_selftest("/dev/nvme0n1", verbose=False)
    assert result == SELFTEST_ERROR


def test_run_selftest_returns_passed_on_completion_code_zero(monkeypatch, tmp_path: Path) -> None:
    calls = {"n": 0}

    def _fake_run(cmd, timeout=None):
        calls["n"] += 1
        if "device-self-test" in cmd:
            return 0, "Device self-test started", ""
        # self-test-log: first call returns in-progress, second returns done
        if calls["n"] == 2:
            return 0, json.dumps(_make_log(op_code=1, completion=50)), ""
        return 0, json.dumps(_make_log(op_code=0, result_code=0)), ""

    monkeypatch.setattr("psfu.storage.selftest.nvme_cli_binary_path", lambda: "/usr/bin/nvme")
    monkeypatch.setattr("psfu.storage.selftest.run_capture", _fake_run)
    monkeypatch.setattr("psfu.storage.selftest._POLL_INTERVAL_SEC", 0)

    result, log = run_selftest("/dev/nvme0n1", verbose=False, out_dir=tmp_path)
    assert result == SELFTEST_PASSED
    assert log is not None
    artifact = tmp_path / "nvme-selftest-dev-nvme0n1.json"
    assert artifact.exists()


def test_run_selftest_returns_failed_on_nonzero_result(monkeypatch) -> None:
    def _fake_run(cmd, timeout=None):
        if "device-self-test" in cmd:
            return 0, "started", ""
        return 0, json.dumps(_make_log(op_code=0, result_code=0x3)), ""

    monkeypatch.setattr("psfu.storage.selftest.nvme_cli_binary_path", lambda: "/usr/bin/nvme")
    monkeypatch.setattr("psfu.storage.selftest.run_capture", _fake_run)
    monkeypatch.setattr("psfu.storage.selftest._POLL_INTERVAL_SEC", 0)

    result, _ = run_selftest("/dev/nvme0n1", verbose=False)
    assert result == SELFTEST_FAILED


def test_run_selftest_returns_aborted_on_code_0xf(monkeypatch) -> None:
    def _fake_run(cmd, timeout=None):
        if "device-self-test" in cmd:
            return 0, "started", ""
        return 0, json.dumps(_make_log(op_code=0, result_code=0xF)), ""

    monkeypatch.setattr("psfu.storage.selftest.nvme_cli_binary_path", lambda: "/usr/bin/nvme")
    monkeypatch.setattr("psfu.storage.selftest.run_capture", _fake_run)
    monkeypatch.setattr("psfu.storage.selftest._POLL_INTERVAL_SEC", 0)

    result, _ = run_selftest("/dev/nvme0n1", verbose=False)
    assert result == SELFTEST_ABORTED


def test_run_selftest_returns_timeout_when_operation_never_completes(monkeypatch) -> None:
    def _fake_run(cmd, timeout=None):
        if "device-self-test" in cmd:
            return 0, "started", ""
        return 0, json.dumps(_make_log(op_code=1, completion=10)), ""

    monkeypatch.setattr("psfu.storage.selftest.nvme_cli_binary_path", lambda: "/usr/bin/nvme")
    monkeypatch.setattr("psfu.storage.selftest.run_capture", _fake_run)
    monkeypatch.setattr("psfu.storage.selftest._POLL_INTERVAL_SEC", 0)
    monkeypatch.setattr("psfu.storage.selftest._SHORT_TIMEOUT_SEC", 0)

    result, _ = run_selftest("/dev/nvme0n1", short=True, verbose=False)
    assert result == SELFTEST_TIMEOUT


def test_run_selftest_short_flag_uses_type_0x1(monkeypatch) -> None:
    triggered = {"type": None}

    def _fake_run(cmd, timeout=None):
        if "device-self-test" in cmd:
            idx = cmd.index("-s") if "-s" in cmd else -1
            if idx >= 0:
                triggered["type"] = cmd[idx + 1]
            return 0, "started", ""
        return 0, json.dumps(_make_log(op_code=0, result_code=0)), ""

    monkeypatch.setattr("psfu.storage.selftest.nvme_cli_binary_path", lambda: "/usr/bin/nvme")
    monkeypatch.setattr("psfu.storage.selftest.run_capture", _fake_run)
    monkeypatch.setattr("psfu.storage.selftest._POLL_INTERVAL_SEC", 0)

    run_selftest("/dev/nvme0n1", short=True, verbose=False)
    assert triggered["type"] == "0x1"


def test_run_selftest_extended_uses_type_0x2(monkeypatch) -> None:
    triggered = {"type": None}

    def _fake_run(cmd, timeout=None):
        if "device-self-test" in cmd:
            idx = cmd.index("-s") if "-s" in cmd else -1
            if idx >= 0:
                triggered["type"] = cmd[idx + 1]
            return 0, "started", ""
        return 0, json.dumps(_make_log(op_code=0, result_code=0)), ""

    monkeypatch.setattr("psfu.storage.selftest.nvme_cli_binary_path", lambda: "/usr/bin/nvme")
    monkeypatch.setattr("psfu.storage.selftest.run_capture", _fake_run)
    monkeypatch.setattr("psfu.storage.selftest._POLL_INTERVAL_SEC", 0)

    run_selftest("/dev/nvme0n1", short=False, verbose=False)
    assert triggered["type"] == "0x2"
