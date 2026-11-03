from __future__ import annotations

from pathlib import Path

from psfu.batch import maybe_batch_pattern_event, write_batch_summary
from psfu.tests.conftest import _base_sample, make_run_db


def _seed_run(tmp_path: Path, name: str, *, serial: str, lot: str, verdict: str) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    db = make_run_db(
        run_dir,
        overrides={
            "psu_serial": serial,
            "psu_lot": lot,
            "verdict": verdict,
            "telemetry_status": "software-only",
        },
    )
    db.add_sample(_base_sample())
    db.close()
    return run_dir


def test_write_batch_summary_reports_loop3_trigger(monkeypatch, tmp_path: Path) -> None:
    _seed_run(tmp_path, "run-a", serial="SN-A", lot="LOT-42", verdict="FAIL")
    _seed_run(tmp_path, "run-b", serial="SN-B", lot="LOT-42", verdict="FAIL")
    _seed_run(tmp_path, "run-c", serial="SN-C", lot="LOT-42", verdict="PASS")
    _seed_run(tmp_path, "run-d", serial="SN-D", lot="LOT-99", verdict="FAIL")

    monkeypatch.setattr("psfu.batch.RUNS_DIR", tmp_path)
    result = write_batch_summary("LOT-42")
    text = result["output_path"].read_text(encoding="utf-8")

    assert len(result["rows"]) == 3
    assert result["fail_count"] == 2
    assert "| Loop 3 trigger | YES |" in text
    assert "Affected serials: SN-A, SN-B" in text
    assert "run-d" not in text


def test_maybe_batch_pattern_event_fires_on_second_same_lot_fail(monkeypatch, tmp_path: Path) -> None:
    _seed_run(tmp_path, "run-prev", serial="SN-A", lot="LOT-42", verdict="FAIL")

    run_dir = tmp_path / "run-current"
    run_dir.mkdir(parents=True, exist_ok=True)
    db = make_run_db(
        run_dir,
        overrides={
            "psu_serial": "SN-B",
            "psu_lot": "LOT-42",
            "verdict": "FAIL",
            "telemetry_status": "software-only",
        },
    )
    db.add_sample(_base_sample())

    monkeypatch.setattr("psfu.batch.RUNS_DIR", tmp_path)
    event = maybe_batch_pattern_event(run_dir, db)
    db.close()

    assert event is not None
    assert event["kind"] == "batch_pattern_detected"
    assert "SN-A" in event["message"]
    assert "SN-B" in event["message"]
