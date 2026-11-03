from __future__ import annotations

import json
from pathlib import Path

from psfu.collectors.storage_ocp_latency import (
    capture_storage_ocp_latency,
    storage_ocp_latency_events,
    write_ocp_latency_artifacts,
)


def _fake_latency_log() -> dict:
    return {
        "bucket_read_latencies": [100, 200, 50],
        "bucket_write_latencies": [80, 160, 30],
    }


def test_capture_returns_not_supported_when_nvme_cli_missing(monkeypatch) -> None:
    monkeypatch.setattr("psfu.collectors.storage_ocp_latency.nvme_cli_binary_path", lambda: None)
    payload = capture_storage_ocp_latency(["/dev/nvme0n1"])
    assert payload["devices"][0]["supported"] is False
    assert payload["devices"][0]["data"] is None


def test_capture_returns_not_supported_when_command_fails(monkeypatch) -> None:
    monkeypatch.setattr("psfu.collectors.storage_ocp_latency.nvme_cli_binary_path", lambda: "/usr/bin/nvme")
    monkeypatch.setattr(
        "psfu.collectors.storage_ocp_latency.run_capture",
        lambda cmd, timeout=15: (1, "", "not supported"),
    )
    payload = capture_storage_ocp_latency(["/dev/nvme0n1"])
    assert payload["devices"][0]["supported"] is False


def test_capture_returns_supported_when_command_succeeds(monkeypatch) -> None:
    monkeypatch.setattr("psfu.collectors.storage_ocp_latency.nvme_cli_binary_path", lambda: "/usr/bin/nvme")
    monkeypatch.setattr(
        "psfu.collectors.storage_ocp_latency.run_capture",
        lambda cmd, timeout=15: (0, json.dumps(_fake_latency_log()), ""),
    )
    payload = capture_storage_ocp_latency(["/dev/nvme0n1"])
    assert payload["devices"][0]["supported"] is True
    assert payload["devices"][0]["data"]["bucket_read_latencies"] == [100, 200, 50]


def test_capture_returns_not_supported_for_invalid_json(monkeypatch) -> None:
    monkeypatch.setattr("psfu.collectors.storage_ocp_latency.nvme_cli_binary_path", lambda: "/usr/bin/nvme")
    monkeypatch.setattr(
        "psfu.collectors.storage_ocp_latency.run_capture",
        lambda cmd, timeout=15: (0, "not json at all", ""),
    )
    payload = capture_storage_ocp_latency(["/dev/nvme0n1"])
    assert payload["devices"][0]["supported"] is False


def test_events_not_supported_when_start_or_end_missing(monkeypatch) -> None:
    start = {"devices": [{"device": "/dev/nvme0n1", "supported": False, "data": None}]}
    end = {"devices": [{"device": "/dev/nvme0n1", "supported": True, "data": _fake_latency_log()}]}
    events = storage_ocp_latency_events(start, end)
    assert len(events) == 1
    assert events[0][1] == "ssd_ocp_latency_not_supported"
    assert events[0][0] == "info"


def test_events_captured_when_both_snapshots_supported(monkeypatch) -> None:
    start = {"devices": [{"device": "/dev/nvme0n1", "supported": True, "data": _fake_latency_log()}]}
    end = {"devices": [{"device": "/dev/nvme0n1", "supported": True, "data": _fake_latency_log()}]}
    events = storage_ocp_latency_events(start, end)
    assert len(events) == 1
    assert events[0][1] == "ssd_ocp_latency_captured"
    assert events[0][0] == "info"


def test_write_ocp_latency_artifacts_creates_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("psfu.collectors.storage_ocp_latency.nvme_cli_binary_path", lambda: "/usr/bin/nvme")
    monkeypatch.setattr(
        "psfu.collectors.storage_ocp_latency.run_capture",
        lambda cmd, timeout=15: (0, json.dumps(_fake_latency_log()), ""),
    )
    payload = write_ocp_latency_artifacts(tmp_path, ["/dev/nvme0n1"], "start")
    artifact = tmp_path / "storage" / "nvme" / "ocp-latency-monitor-start.json"
    assert artifact.exists()
    data = json.loads(artifact.read_text(encoding="utf-8"))
    assert data["devices"][0]["supported"] is True
    assert payload["devices"][0]["supported"] is True


def test_multiple_devices_each_get_entry(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_run(cmd, timeout=15):
        calls["count"] += 1
        if "/dev/nvme0n1" in cmd:
            return 0, json.dumps(_fake_latency_log()), ""
        return 1, "", "not supported"

    monkeypatch.setattr("psfu.collectors.storage_ocp_latency.nvme_cli_binary_path", lambda: "/usr/bin/nvme")
    monkeypatch.setattr("psfu.collectors.storage_ocp_latency.run_capture", _fake_run)
    payload = capture_storage_ocp_latency(["/dev/nvme0n1", "/dev/nvme1n1"])
    assert len(payload["devices"]) == 2
    by_dev = {d["device"]: d for d in payload["devices"]}
    assert by_dev["/dev/nvme0n1"]["supported"] is True
    assert by_dev["/dev/nvme1n1"]["supported"] is False
