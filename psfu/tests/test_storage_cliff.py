from __future__ import annotations

import json
from pathlib import Path

from psfu.runners.phase_runners import FioDutRunner
from psfu.storage.cliff import (
    analyze_write_cliff,
    build_write_bw_windows,
    classify_write_workload,
    extract_fio_interval_row,
    ssd_baseline_path_for_identity,
)


def test_extract_fio_interval_row_sums_jobs() -> None:
    row = extract_fio_interval_row(
        {
            "jobs": [
                {"read": {"bw": 100, "total_ios": 10}, "write": {"bw": 500, "total_ios": 20}},
                {"read": {"bw": 50, "total_ios": 5}, "write": {"bw": 250, "total_ios": 8}},
            ]
        }
    )
    assert row is not None
    assert row["read_bw_kbps"] == 150.0
    assert row["write_bw_kbps"] == 750.0
    assert row["read_total_ios"] == 15
    assert row["write_total_ios"] == 28


def test_build_write_bw_windows_uses_full_windows_only() -> None:
    rows = [{"write_bw_kbps": 1000.0}] * 25
    windows = build_write_bw_windows(rows, window_sec=10)
    assert len(windows) == 2
    assert windows[0]["avg_write_bw_kbps"] == 1000.0
    assert windows[1]["avg_write_bw_kbps"] == 1000.0


def test_analyze_write_cliff_flags_severe_ratio() -> None:
    rows = [{"write_bw_kbps": 1000.0}] * 10
    rows.extend([{"write_bw_kbps": 900.0}] * 10)
    rows.extend([{"write_bw_kbps": 90.0}] * 10)
    rows.extend([{"write_bw_kbps": 90.0}] * 10)
    analysis = analyze_write_cliff(rows, phase_name="cliff-write", device="/dev/nvme0n1")
    assert analysis is not None
    assert analysis["full_window_count"] == 4
    assert analysis["severe"] is True
    assert round(analysis["cliff_ratio"], 3) == 0.090
    assert analysis["workload_stats"]["major_drop_count"] >= 1
    assert analysis["workload_stats"]["changepoint_count"] >= 1


def test_analyze_write_cliff_returns_none_without_enough_windows() -> None:
    rows = [{"write_bw_kbps": 1000.0}] * 30
    assert analyze_write_cliff(rows, phase_name="cliff-write", device="/dev/nvme0n1") is None


def test_analyze_write_cliff_loads_matching_baseline(tmp_path: Path, monkeypatch) -> None:
    baseline_dir = tmp_path / "baselines"
    monkeypatch.setattr("psfu.storage.cliff.BASELINE_DIR", baseline_dir, raising=False)
    monkeypatch.setattr("psfu.storage.cliff.BASELINES_DIR", baseline_dir)
    identity_dir = tmp_path / "storage" / "nvme" / "nvme0n1"
    identity_dir.mkdir(parents=True, exist_ok=True)
    (identity_dir / "start-id-ctrl.json").write_text(
        json.dumps({"payload": {"vid": 5197, "mn": "WD_BLACK SN850X", "fr": "620311WD"}}),
        encoding="utf-8",
    )
    (identity_dir / "start-id-ns.json").write_text(
        json.dumps({"payload": {"nsze": 2000398934016}}),
        encoding="utf-8",
    )
    baseline_path = ssd_baseline_path_for_identity(
        vendor="vid-144d",
        model="WD_BLACK SN850X",
        capacity="nsze-2000398934016",
        fw_rev="620311WD",
    )
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps({
            "cliff_ratio_p5": 0.42,
            "sigma": 0.03,
            "sample_count": 7,
        }),
        encoding="utf-8",
    )
    rows = [{"write_bw_kbps": 1000.0}] * 10
    rows.extend([{"write_bw_kbps": 900.0}] * 10)
    rows.extend([{"write_bw_kbps": 150.0}] * 10)
    rows.extend([{"write_bw_kbps": 150.0}] * 10)
    analysis = analyze_write_cliff(
        rows,
        phase_name="cliff-write",
        device="/dev/nvme0n1",
        run_dir=tmp_path,
    )
    assert analysis is not None
    assert analysis["baseline_loaded"] is True
    assert analysis["baseline"]["cliff_ratio_p5"] == 0.42
    assert analysis["baseline_comparison"]["severe_floor_ratio"] == 0.32999999999999996
    assert analysis["baseline_comparison"]["below_severe_floor"] is True


def test_analyze_write_cliff_reports_oscillation_and_post_peak_stats() -> None:
    rows = [{"write_bw_kbps": 1000.0}] * 10
    rows.extend([{"write_bw_kbps": 700.0}] * 10)
    rows.extend([{"write_bw_kbps": 400.0}] * 10)
    rows.extend([{"write_bw_kbps": 650.0}] * 10)
    rows.extend([{"write_bw_kbps": 380.0}] * 10)
    analysis = analyze_write_cliff(rows, phase_name="speed-verify-cliff", device="/dev/nvme0n1")
    assert analysis is not None
    workload = analysis["workload_stats"]
    assert workload["major_drop_count"] >= 2
    assert workload["major_rise_count"] >= 1
    assert workload["post_peak_window_count"] == 4
    assert workload["post_peak_write_bw_cv"] is not None
    assert workload["tail_window_write_bw_cv"] is not None
    assert analysis["classification"]["slug"] == "oscillatory_gc"


def test_classify_write_workload_stable_plateau() -> None:
    classified = classify_write_workload(
        {
            "cliff_ratio": 0.38,
            "workload_stats": {
                "major_drop_count": 1,
                "major_rise_count": 0,
                "tail_window_write_bw_cv": 0.03,
                "post_peak_write_bw_cv": 0.04,
            },
        }
    )
    assert classified["slug"] == "single_cliff_stable_plateau"


def test_fio_dut_runner_teardown_emits_info_for_severe_cliff(tmp_path: Path) -> None:
    class _StubFio:
        def __init__(self) -> None:
            self.device = "/dev/nvme0n1"
            self.job_file = Path("/tmp/job.fio")
            self.run_dir = tmp_path
            self.output_dir = tmp_path / "storage" / "fio" / "cliff-write" / "seq_write_full-nvme0n1"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.interval_rows = [{"write_bw_kbps": 1000.0}] * 10
            self.interval_rows.extend([{"write_bw_kbps": 900.0}] * 10)
            self.interval_rows.extend([{"write_bw_kbps": 80.0}] * 10)
            self.interval_rows.extend([{"write_bw_kbps": 80.0}] * 10)
            self.summary_rows = []
            self.stopped = False

        def start(self):
            return None

        def pause(self):
            return None

        def resume(self):
            return None

        def stop(self):
            self.stopped = True

        def state(self) -> str:
            return "storage:run"

    fio = _StubFio()
    events = []

    def emit(severity, phase, kind, message, *, tui_text=None):  # type: ignore[no-untyped-def]
        events.append({"severity": severity, "phase": phase, "kind": kind, "message": message})

    runner = FioDutRunner(fio, emit_event=emit, phase_name="cliff-write")  # type: ignore[arg-type]
    runner.teardown()

    assert fio.stopped is True
    assert any(event["kind"] == "fio_write_cliff_observed" for event in events)
    artifact = fio.output_dir / "write-cliff-analysis.json"
    assert artifact.exists()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["severe"] is True
    assert payload["phase_name"] == "cliff-write"
    assert payload["workload_stats"]["major_drop_count"] >= 1


def test_fio_dut_runner_non_cliff_phase_does_not_emit_cliff_event(tmp_path: Path) -> None:
    class _StubFio:
        def __init__(self) -> None:
            self.device = "/dev/nvme0n1"
            self.job_file = Path("/tmp/job.fio")
            self.run_dir = tmp_path
            self.output_dir = tmp_path / "storage" / "fio" / "full-write" / "seq_write_full-nvme0n1"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.interval_rows = [{"write_bw_kbps": 1000.0}] * 40
            self.summary_rows = []

        def start(self):
            return None

        def pause(self):
            return None

        def resume(self):
            return None

        def stop(self):
            return None

        def state(self) -> str:
            return "storage:run"

    events = []
    runner = FioDutRunner(
        _StubFio(),
        emit_event=lambda severity, phase, kind, message, *, tui_text=None: events.append(kind),
        phase_name="full-write",
    )  # type: ignore[arg-type]
    runner.teardown()
    assert "fio_write_cliff_severe" not in events


def test_fio_dut_runner_tick_updates_live_phase_extra(tmp_path: Path) -> None:
    class _StubFio:
        def __init__(self) -> None:
            self.device = "/dev/nvme0n1"
            self.job_file = Path("/tmp/job.fio")
            self.run_dir = tmp_path
            self.output_dir = tmp_path / "storage" / "fio" / "speed-verify" / "seq_write_full-nvme0n1"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.interval_rows = [{"write_bw_kbps": 1000.0}] * 10
            self.interval_rows.extend([{"write_bw_kbps": 950.0}] * 10)
            self.interval_rows.extend([{"write_bw_kbps": 340.0}] * 10)
            self.interval_rows.extend([{"write_bw_kbps": 335.0}] * 10)
            self.interval_rows.extend([{"write_bw_kbps": 330.0}] * 10)
            self.summary_rows = []

        def start(self):
            return None

        def pause(self):
            return None

        def resume(self):
            return None

        def stop(self):
            return None

        def state(self) -> str:
            return "storage:run"

    class _Status:
        def __init__(self) -> None:
            self.extra = ""

        def set_phase_extra(self, text: str) -> None:
            self.extra = text

        def scrollback_note(self, text: str) -> None:
            return None

    status = _Status()
    runner = FioDutRunner(
        _StubFio(),
        emit_event=lambda *a, **k: None,
        status=status,
        phase_name="speed-verify",
    )  # type: ignore[arg-type]
    runner.tick(0.0, 0.0)
    assert "cliff stable" in status.extra


def test_fio_dut_runner_teardown_emits_warn_for_baseline_backed_severe_cliff(
    tmp_path: Path, monkeypatch
) -> None:
    baseline_dir = tmp_path / "baselines"
    monkeypatch.setattr("psfu.storage.cliff.BASELINES_DIR", baseline_dir)
    identity_dir = tmp_path / "storage" / "nvme" / "nvme0n1"
    identity_dir.mkdir(parents=True, exist_ok=True)
    (identity_dir / "start-id-ctrl.json").write_text(
        json.dumps({"payload": {"vid": 5197, "mn": "WD_BLACK SN850X", "fr": "620311WD"}}),
        encoding="utf-8",
    )
    (identity_dir / "start-id-ns.json").write_text(
        json.dumps({"payload": {"nsze": 2000398934016}}),
        encoding="utf-8",
    )
    baseline_path = ssd_baseline_path_for_identity(
        vendor="vid-144d",
        model="WD_BLACK SN850X",
        capacity="nsze-2000398934016",
        fw_rev="620311WD",
    )
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps({"cliff_ratio_p5": 0.42, "sigma": 0.03}),
        encoding="utf-8",
    )

    class _StubFio:
        def __init__(self) -> None:
            self.device = "/dev/nvme0n1"
            self.job_file = Path("/tmp/job.fio")
            self.run_dir = tmp_path
            self.output_dir = tmp_path / "storage" / "fio" / "cliff-write" / "seq_write_full-nvme0n1"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.interval_rows = [{"write_bw_kbps": 1000.0}] * 10
            self.interval_rows.extend([{"write_bw_kbps": 900.0}] * 10)
            self.interval_rows.extend([{"write_bw_kbps": 80.0}] * 10)
            self.interval_rows.extend([{"write_bw_kbps": 80.0}] * 10)
            self.summary_rows = []

        def start(self):
            return None

        def pause(self):
            return None

        def resume(self):
            return None

        def stop(self):
            return None

        def state(self) -> str:
            return "storage:run"

    events = []

    def emit(severity, phase, kind, message, *, tui_text=None):  # type: ignore[no-untyped-def]
        events.append({"severity": severity, "phase": phase, "kind": kind, "message": message})

    runner = FioDutRunner(_StubFio(), emit_event=emit, phase_name="cliff-write")  # type: ignore[arg-type]
    runner.teardown()
    assert any(event["kind"] == "fio_write_cliff_severe" and event["severity"] == "warning" for event in events)
