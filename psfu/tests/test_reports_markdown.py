from __future__ import annotations

import json
from pathlib import Path

from types import SimpleNamespace

from psfu.reports.markdown import (
    render_rma_summary,
    render_storage_workload_section,
)


def test_render_storage_workload_section_reads_cliff_artifact(tmp_path: Path) -> None:
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
                    "label": "gc oscillating",
                    "summary": "Repeated drops and recoveries after peak suggest GC or folding oscillation.",
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

    lines = render_storage_workload_section(tmp_path)
    rendered = "\n".join(lines)

    assert "## Storage Workload" in rendered
    assert "| Classification | gc oscillating |" in rendered
    assert "| Cliff ratio | 0.217 |" in rendered
    assert "| Baseline loaded | yes |" in rendered
    assert "Repeated drops and recoveries after peak" in rendered


def test_render_rma_summary_includes_storage_workload_line(tmp_path: Path) -> None:
    artifact = tmp_path / "storage" / "fio" / "speed-verify" / "seq_write_verify_2m_nj16_qd8-nvme0n1" / "write-cliff-analysis.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "cliff_ratio": 0.217,
                "classification": {"label": "gc oscillating"},
            }
        ),
        encoding="utf-8",
    )

    row = {
        "verdict_reasons_json": "[]",
        "psu_make": "Corsair",
        "psu_model": "HX1500i",
        "psu_wattage": 1500,
        "psu_serial": "ABC123",
        "psu_lot": "LOT1",
        "ended_at": "2026-04-20T12:00:00Z",
        "started_at": "2026-04-20T11:00:00Z",
        "ambient_c": None,
        "bench_name": "Bench-1",
        "post_result": "pass",
        "meter_available": 0,
        "known_good_swap_performed": 0,
        "known_good_swap_result": "",
        "telemetry_status": "full",
    }

    rendered = render_rma_summary(
        tmp_path,
        row,  # type: ignore[arg-type]
        [],
        [],
        lambda _row: "Bench-isolated result.",
    )

    assert "## Storage Workload Characterization" in rendered
    assert "matched `gc oscillating` with cliff ratio 0.217" in rendered
