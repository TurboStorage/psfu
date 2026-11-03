from __future__ import annotations

import json
from pathlib import Path

import pytest

from psfu.cli import main
from psfu.tests.conftest import _base_sample, make_run_db
from psfu.verdict.rules import VerdictEngine


def _seed_pass_run(run_dir: Path, *, bench_name: str, verdict: str = "PASS", soak_12v: float = 12.10, load_12v: float = 12.00) -> None:
    db = make_run_db(run_dir, overrides={"bench_name": bench_name})
    row = db.fetch_run()
    db.update_run(
        row["id"],
        {
            "verdict": verdict,
            "telemetry_status": "software-only",
            "verdict_reasons_json": json.dumps(["clean run"] if verdict == "PASS" else ["warn"]),
            "subsystem_results_json": json.dumps({
                "power": {"status": "PASS", "reasons": []},
                "thermal": {"status": "PASS", "reasons": []},
                "memory": {"status": "PASS", "reasons": []},
                "storage": {"status": "PASS", "reasons": []},
                "pcie": {"status": "PASS", "reasons": []},
            }),
            "hardware_json": json.dumps({
                "commands": {
                    "nvidia_smi_L": {
                        "stdout": "GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-test)"
                    }
                }
            }),
        },
    )
    for _ in range(5):
        db.add_sample(_base_sample(phase="soak", rail_12v=soak_12v, cpu_avg_mhz=5000.0))
    for _ in range(5):
        db.add_sample(_base_sample(phase="steady", rail_12v=load_12v, cpu_avg_mhz=5000.0))
    db.close()


def test_baseline_record_uses_latest_pass_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_dir = tmp_path / "runs"
    older = runs_dir / "2026-04-18_old"
    newer = runs_dir / "2026-04-19_new"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    _seed_pass_run(older, bench_name="bench-a")
    _seed_pass_run(newer, bench_name="bench-a")

    import psfu.cli as cli
    import psfu.baselines as baselines

    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(baselines, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(baselines, "BASELINES_DIR", tmp_path / "baselines")

    rc = main(["baseline", "record", "--profile", "test-profile"])
    assert rc == 0
    files = list((tmp_path / "baselines").glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["source_run"] == "2026-04-19_new"
    assert payload["metrics"]["soak_12v_median"] == 12.10


def test_baseline_drift_warns_when_reference_differs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import psfu.baselines as baselines

    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    monkeypatch.setattr(baselines, "BASELINES_DIR", baseline_dir)

    db = make_run_db(tmp_path, overrides={"bench_name": "bench-a"})
    row = db.fetch_run()
    db.update_run(
        row["id"],
        {
            "hardware_json": json.dumps({
                "commands": {
                    "nvidia_smi_L": {
                        "stdout": "GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-test)"
                    }
                }
            })
        },
    )
    for _ in range(5):
        db.add_sample(_base_sample(phase="soak", rail_12v=11.80, cpu_avg_mhz=4300.0))
    for _ in range(5):
        db.add_sample(_base_sample(phase="steady", rail_12v=11.40, cpu_avg_mhz=4300.0))

    baseline_path = baseline_dir / "bench-a-nvidia-geforce-rtx-5090.json"
    baseline_path.write_text(json.dumps({
        "metrics": {
            "soak_12v_median": 12.10,
            "load_12v_min": 12.00,
            "load_cpu_avg_mhz_p90": 5000.0,
        }
    }), encoding="utf-8")

    verdict, reasons, _, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["power"]["status"] == "WARN"
    assert any("Baseline drift:" in reason for reason in reasons)


def test_baseline_absent_does_not_change_clean_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import psfu.baselines as baselines

    monkeypatch.setattr(baselines, "BASELINES_DIR", tmp_path / "baselines")
    db = make_run_db(tmp_path)
    for _ in range(5):
        db.add_sample(_base_sample(phase="soak", rail_12v=12.10))
    for _ in range(5):
        db.add_sample(_base_sample(phase="steady", rail_12v=12.10))
    verdict, reasons, _, _ = VerdictEngine.evaluate(db)
    assert verdict == "PASS"
    assert not any("Baseline drift:" in reason for reason in reasons)
