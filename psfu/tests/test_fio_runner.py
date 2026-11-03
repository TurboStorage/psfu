from __future__ import annotations

import json
from pathlib import Path

from psfu.runners.workers.fio_runner import (
    FioRunner,
    extract_interval_display_metrics,
    parse_json_plus_output,
)


def _job(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "job.fio"
    path.write_text(body, encoding="utf-8")
    return path


def test_build_cmd_respects_readonly_flag(tmp_path, monkeypatch) -> None:
    job = _job(
        tmp_path,
        "[global]\nreadonly=1\n[read]\nrw=randread\n",
    )
    monkeypatch.setattr("psfu.runners.workers.fio_runner.fio_binary_path", lambda: "/usr/bin/fio")
    runner = FioRunner(job, device="/dev/nvme0n1", phase_name="steady", run_dir=tmp_path)
    cmd = runner.build_command()
    assert cmd[0] == "/usr/bin/fio"
    assert "--filename=/dev/nvme0n1" in cmd
    assert "--output-format=json+" in cmd
    assert str(job) == cmd[-1]


def test_refuses_to_run_without_readonly_if_profile_not_destructive(tmp_path, monkeypatch) -> None:
    job = _job(
        tmp_path,
        "[global]\n[write]\nrw=randwrite\n",
    )
    monkeypatch.setattr("psfu.runners.workers.fio_runner.fio_binary_path", lambda: "/usr/bin/fio")
    runner = FioRunner(job, device="/dev/nvme0n1", phase_name="steady", run_dir=tmp_path)
    err = runner.start()
    assert err is not None
    assert "non-readonly fio job" in err


def test_parse_json_plus_output_populates_summary() -> None:
    rows = parse_json_plus_output(
        {
            "jobs": [
                {
                    "jobname": "qd1_4k_randread",
                    "job_runtime": 120000,
                    "error": 0,
                    "read": {
                        "bw": 7123,
                        "iops": 1780.5,
                        "lat_ns": {"mean": 12345.0, "stddev": 120.0, "max": 77777.0},
                        "percentile": {
                            "50.000000": 11000.0,
                            "99.000000": 22000.0,
                            "99.900000": 33000.0,
                            "99.990000": 44000.0,
                            "99.999000": 55000.0,
                        },
                    },
                    "write": {
                        "bw": 0,
                        "iops": 0.0,
                        "lat_ns": {"mean": 0.0, "stddev": 0.0, "max": 0.0},
                        "percentile": {"99.999000": 66000.0},
                    },
                    "verify": {"errors": 0},
                }
            ]
        },
        phase_name="steady",
        device="/dev/nvme0n1",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["job_name"] == "qd1_4k_randread"
    assert row["device"] == "/dev/nvme0n1"
    assert row["read_bw_kbps"] == 7123
    assert row["read_lat_p99_9_ns"] == 33000.0
    assert row["read_lat_p99_999_ns"] == 55000.0
    assert row["write_lat_p99_999_ns"] == 66000.0
    assert row["runtime_ms"] == 120000


def test_build_cmd_includes_status_interval_for_s6_stall_detection(tmp_path, monkeypatch) -> None:
    """S6 — fio must run with `--status-interval=1` so the stanza reader
    thread has per-second snapshots to feed check_nvme_fio_stall. Removing
    this flag silently disables the controller-hang watchdog."""
    job = _job(tmp_path, "[global]\nreadonly=1\n[read]\nrw=randread\n")
    monkeypatch.setattr("psfu.runners.workers.fio_runner.fio_binary_path", lambda: "/usr/bin/fio")
    runner = FioRunner(job, device="/dev/nvme0n1", phase_name="steady", run_dir=tmp_path)
    cmd = runner.build_command()
    assert "--status-interval=1" in cmd


def test_extract_interval_display_metrics_aggregates_live_tail_latency() -> None:
    metrics = extract_interval_display_metrics(
        {
            "jobs": [
                {
                    "read": {"percentile": {"99.000000": 125000.0, "99.999000": 900000.0}},
                    "write": {"percentile": {"99.000000": 250000.0, "99.999000": 1500000.0}},
                },
                {
                    "read": {"percentile": {"99.000000": 150000.0, "99.999000": 1000000.0}},
                    "write": {"percentile": {"99.000000": 200000.0, "99.999000": 1250000.0}},
                },
            ]
        }
    )
    assert metrics["fio_read_lat_p99_us"] == 150.0
    assert metrics["fio_write_lat_p99_us"] == 250.0
    assert metrics["fio_read_lat_p99_999_us"] == 1000.0
    assert metrics["fio_write_lat_p99_999_us"] == 1500.0


def test_reader_loop_extracts_multi_line_stanza(tmp_path, monkeypatch) -> None:
    """The reader thread accumulates chunks until a full JSON object is
    parseable. fio emits stanzas across many lines; the parser must stream,
    not buffer-per-line. Simulates two complete stanzas arriving back-to-back
    in a single chunk."""
    job = _job(tmp_path, "[global]\nreadonly=1\n[read]\nrw=randread\n")
    monkeypatch.setattr("psfu.runners.workers.fio_runner.fio_binary_path", lambda: "/usr/bin/fio")
    runner = FioRunner(job, device="/dev/nvme0n1", phase_name="steady", run_dir=tmp_path)

    first = {"jobs": [{"read": {"total_ios": 100}, "write": {"total_ios": 0}}]}
    second = {"jobs": [{"read": {"total_ios": 250}, "write": {"total_ios": 0}}]}
    combined = json.dumps(first, indent=2) + "\n" + json.dumps(second, indent=2) + "\n"

    # Feed one chunk at a time to simulate line-buffered stdout.
    class _FakeStdout:
        def __init__(self, payload: str) -> None:
            self._lines = payload.splitlines(keepends=True)
            self._i = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._i >= len(self._lines):
                raise StopIteration
            line = self._lines[self._i]
            self._i += 1
            return line

    class _FakeProc:
        def __init__(self, payload: str) -> None:
            self.stdout = _FakeStdout(payload)

    runner.process = _FakeProc(combined)  # type: ignore[assignment]
    runner._reader_loop()
    latest = runner.latest_interval()
    assert latest is not None
    assert latest["jobs"][0]["read"]["total_ios"] == 250


def test_ios_progress_snapshot_sums_jobs(tmp_path, monkeypatch) -> None:
    job = _job(tmp_path, "[global]\nreadonly=1\n[read]\nrw=randread\n")
    monkeypatch.setattr("psfu.runners.workers.fio_runner.fio_binary_path", lambda: "/usr/bin/fio")
    runner = FioRunner(job, device="/dev/nvme0n1", phase_name="steady", run_dir=tmp_path)
    runner._latest_interval = {
        "jobs": [
            {"read": {"total_ios": 100}, "write": {"total_ios": 0}},
            {"read": {"total_ios": 50}, "write": {"total_ios": 20}},
        ]
    }
    assert runner.ios_progress_snapshot() == (150, 20)


def test_ios_progress_snapshot_returns_none_before_first_stanza(tmp_path, monkeypatch) -> None:
    job = _job(tmp_path, "[global]\nreadonly=1\n[read]\nrw=randread\n")
    monkeypatch.setattr("psfu.runners.workers.fio_runner.fio_binary_path", lambda: "/usr/bin/fio")
    runner = FioRunner(job, device="/dev/nvme0n1", phase_name="steady", run_dir=tmp_path)
    assert runner.ios_progress_snapshot() is None


def test_stop_drains_output_before_parse(tmp_path, monkeypatch) -> None:
    job = _job(
        tmp_path,
        "[global]\nreadonly=1\n[read]\nrw=randread\n",
    )
    monkeypatch.setattr("psfu.runners.workers.fio_runner.fio_binary_path", lambda: "/usr/bin/fio")
    runner = FioRunner(job, device="/dev/nvme0n1", phase_name="steady", run_dir=tmp_path)
    runner.output_dir.mkdir(parents=True, exist_ok=True)
    runner.output_json.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "jobname": "qd1_4k_randread",
                        "job_runtime": 1000,
                        "error": 0,
                        "read": {
                            "bw": 1,
                            "iops": 1.0,
                            "lat_ns": {"mean": 2.0, "stddev": 3.0, "max": 4.0},
                            "percentile": {"50.000000": 2.0, "99.000000": 3.0},
                        },
                        "write": {"bw": 0, "iops": 0.0, "lat_ns": {"mean": 0.0, "stddev": 0.0, "max": 0.0}},
                        "verify": {"errors": 0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runner.stop()
    assert len(runner.summary_rows) == 1
    assert runner.summary_rows[0]["read_bw_kbps"] == 1
