from __future__ import annotations

import json

from psfu.collectors.nvme_telemetry import NvmeTelemetryCollector, _parse_smart_log_sample


def test_parse_smart_log_sample() -> None:
    sample = _parse_smart_log_sample({
        "temperature": 47,
        "warning_temp_time": 3,
        "critical_comp_time": 1,
        "percent_used": 8,
        "avail_spare": 100,
        "spare_thresh": 10,
        "data_units_read": 1234,
        "data_units_written": 5678,
        "host_reads": 90,
        "host_writes": 45,
        "controller_busy_time": 12,
        "power_cycles": 4,
        "unsafe_shutdowns": 2,
        "media_errors": 0,
        "num_err_log_entries": 6,
    })
    assert sample["nvme_composite_temp_c"] == 47.0
    assert sample["nvme_warning_temp_time_min"] == 3.0
    assert sample["nvme_data_units_written"] == 5678
    assert sample["nvme_host_read_cmds"] == 90


def test_parse_handles_missing_optional_fields() -> None:
    sample = _parse_smart_log_sample({"temperature": 38})
    assert sample["nvme_composite_temp_c"] == 38.0
    assert sample["nvme_warning_temp_time_min"] is None
    assert sample["nvme_num_err_log_entries"] is None


def test_collector_snapshot_returns_none_dict_when_drive_missing() -> None:
    collector = NvmeTelemetryCollector(["/dev/nvme0n1"])
    snap = collector.snapshot()
    assert all(value is None for value in snap.values())


def test_state_machine_resilient_to_transient_errors(monkeypatch, tmp_path) -> None:
    calls = {"count": 0}

    def _fake_run_capture(cmd, timeout=5):
        calls["count"] += 1
        if calls["count"] == 1:
            return 0, json.dumps({"temperature": 40, "data_units_read": 1}), ""
        return 1, "", "io error"

    monkeypatch.setattr("psfu.collectors.nvme_telemetry.nvme_cli_binary_path", lambda: "/usr/bin/nvme")
    monkeypatch.setattr("psfu.collectors.nvme_telemetry.run_capture", _fake_run_capture)

    collector = NvmeTelemetryCollector(["/dev/nvme0n1"], run_dir=tmp_path)
    assert collector.start() is None
    snap = collector.snapshot()
    collector.stop()
    assert snap["nvme_composite_temp_c"] == 40.0


def test_multi_drive_snapshot_pins_first(monkeypatch, tmp_path) -> None:
    def _fake_run_capture(cmd, timeout=5):
        if cmd[2] == "/dev/nvme0n1":
            return 0, json.dumps({"temperature": 41, "percent_used": 1}), ""
        return 0, json.dumps({"temperature": 55, "percent_used": 9}), ""

    monkeypatch.setattr("psfu.collectors.nvme_telemetry.nvme_cli_binary_path", lambda: "/usr/bin/nvme")
    monkeypatch.setattr("psfu.collectors.nvme_telemetry.run_capture", _fake_run_capture)

    collector = NvmeTelemetryCollector(["/dev/nvme0n1", "/dev/nvme1n1"], run_dir=tmp_path)
    assert collector.start() is None
    snap = collector.snapshot()
    collector.stop()
    assert snap["nvme_composite_temp_c"] == 41.0
    assert snap["nvme_percentage_used"] == 1.0
