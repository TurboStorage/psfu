from __future__ import annotations

import json
from pathlib import Path

from psfu.reports.exporter import build_verdict_payload
from psfu.tests.conftest import _base_sample, make_run_db
from psfu.verdict.confidence import coverage_confidence


def _write_bootstrap(
    run_dir: Path,
    *,
    pynvml: str = "ready",
    turbostat: str = "ready",
) -> None:
    payload = {
        "steps": [
            {"name": "pynvml", "status": pynvml},
            {"name": "turbostat", "status": turbostat},
        ]
    }
    (run_dir / "bootstrap.json").write_text(json.dumps(payload), encoding="utf-8")


def test_confidence_full_coverage_scores_one(tmp_path: Path) -> None:
    run_dir = tmp_path
    _write_bootstrap(run_dir)
    db = make_run_db(tmp_path, overrides={"meter_available": 1})
    db.add_event("run", "info", "dut_selection", "armed DUTs: CPU, GPU, MEM, STORAGE")
    row = db.fetch_run()
    score = coverage_confidence(run_dir, row, db.fetch_events())
    assert score["score"] == 1.0
    assert score["factors"] == []


def test_confidence_no_meter_defaults_to_point_seven(tmp_path: Path) -> None:
    _write_bootstrap(tmp_path)
    db = make_run_db(tmp_path, overrides={"meter_available": 0})
    db.add_event("run", "info", "dut_selection", "armed DUTs: CPU, GPU, MEM, STORAGE")
    row = db.fetch_run()
    score = coverage_confidence(tmp_path, row, db.fetch_events())
    assert score["score"] == 0.7
    assert [factor["code"] for factor in score["factors"]] == ["no_multimeter"]


def test_confidence_applies_kernel_nvml_turbostat_and_disabled_dut_penalties(tmp_path: Path) -> None:
    _write_bootstrap(tmp_path, pynvml="missing", turbostat="missing")
    db = make_run_db(tmp_path, overrides={"meter_available": 0})
    db.add_event("run", "warn", "kernel_watcher_unavailable", "journalctl not available")
    db.add_event("run", "info", "dut_selection", "armed DUTs: CPU, MEM")
    row = db.fetch_run()
    score = coverage_confidence(tmp_path, row, db.fetch_events())
    assert score["score"] == 0.15
    assert [factor["code"] for factor in score["factors"]] == [
        "no_multimeter",
        "kernel_watcher_unavailable",
        "nvml_unavailable",
        "turbostat_unavailable",
        "dut_disabled_gpu",
        "dut_disabled_storage",
    ]


def test_verdict_payload_includes_confidence_and_factors(tmp_path: Path) -> None:
    _write_bootstrap(tmp_path, pynvml="missing")
    db = make_run_db(tmp_path, overrides={"meter_available": 0})
    row = db.fetch_run()
    db.update_run(
        row["id"],
        {
            "verdict": "PASS",
            "telemetry_status": "software-only",
            "verdict_reasons_json": json.dumps(["No fail or warning rules tripped in this run."]),
            "subsystem_results_json": json.dumps({
                "power": {"status": "PASS", "reasons": []},
                "thermal": {"status": "PASS", "reasons": []},
                "memory": {"status": "PASS", "reasons": []},
                "storage": {"status": "PASS", "reasons": []},
                "pcie": {"status": "PASS", "reasons": []},
            }),
        },
    )
    db.add_sample(_base_sample())
    db.add_event("run", "info", "dut_selection", "armed DUTs: CPU, GPU, MEM, STORAGE")
    payload = build_verdict_payload(tmp_path, db.fetch_run(), db.fetch_events())
    assert payload["overall"]["confidence"] == 0.55
    assert [factor["code"] for factor in payload["overall"]["confidence_factors"]] == [
        "no_multimeter",
        "nvml_unavailable",
    ]
    assert payload["overall"]["evidence"] == []


def test_verdict_payload_includes_storage_workload_summary_when_artifact_exists(tmp_path: Path) -> None:
    _write_bootstrap(tmp_path)
    artifact = tmp_path / "storage" / "fio" / "speed-verify" / "seq_write_verify_2m_nj16_qd8-nvme0n1" / "write-cliff-analysis.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "phase_name": "speed-verify",
                "cliff_ratio": 0.217,
                "tail_window_median_kbps": 1250000.0,
                "baseline_loaded": True,
                "peak_window": {"avg_write_bw_kbps": 5750000.0},
                "classification": {
                    "slug": "oscillatory_gc",
                    "label": "gc oscillating",
                },
                "workload_stats": {
                    "major_drop_count": 3,
                    "major_rise_count": 2,
                    "tail_window_write_bw_cv": 0.143,
                },
            }
        ),
        encoding="utf-8",
    )
    db = make_run_db(tmp_path, overrides={"meter_available": 0})
    db.add_event("run", "info", "dut_selection", "armed DUTs: CPU, GPU, MEM, STORAGE")
    payload = build_verdict_payload(tmp_path, db.fetch_run(), db.fetch_events())
    assert payload["storage_workload"]["classification"] == "oscillatory_gc"
    assert payload["storage_workload"]["classification_label"] == "gc oscillating"
    assert payload["storage_workload"]["cliff_ratio"] == 0.217
    assert payload["artifacts"]["storage_write_cliff_analysis"] == artifact.relative_to(tmp_path).as_posix()
