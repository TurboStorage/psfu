from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from psfu.runtime import create_run_dir, load_bench_baseline
from psfu.tests.conftest import _base_sample, make_run_db
from psfu.tools.bench_gage_rr import compute_bench_gage_rr


def _seed_gage_rr_run(
    run_dir: Path,
    *,
    verdict: str = "PASS",
    soak_12v: float = 12.10,
    load_12v: float = 12.00,
    cpu_mhz: float = 5000.0,
    sample_count: int = 20,
    board_name: str = "TRX50 AI TOP",
    gpu_name: str = "NVIDIA RTX 5090",
    profile_name: str = "smoke-v1",
) -> None:
    db = make_run_db(
        run_dir,
        overrides={
            "bench_name": "bench-a",
            "profile_name": profile_name,
            "verdict": verdict,
            "hardware_json": json.dumps(
                {
                    "board": {"vendor": "Gigabyte", "name": board_name},
                    "commands": {
                        "nvidia_smi_L": {
                            "stdout": "GPU 0: {} (UUID: GPU-test)".format(gpu_name)
                        }
                    },
                }
            ),
        },
    )
    soak_samples = max(1, sample_count // 2)
    load_samples = max(1, sample_count - soak_samples)
    for _ in range(soak_samples):
        db.add_sample(_base_sample(phase="soak", rail_12v=soak_12v, cpu_avg_mhz=cpu_mhz))
    for _ in range(load_samples):
        db.add_sample(_base_sample(phase="steady", rail_12v=load_12v, cpu_avg_mhz=cpu_mhz))
    db.close()


def test_bench_gage_rr_computes_metrics(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    bench = "trx50-ai-top-nvidia-rtx-5090"
    bench_root = runs_dir / "gage-rr" / bench
    bench_root.mkdir(parents=True)
    soak_values = [12.10, 12.11, 12.09, 12.10, 12.12, 12.08, 12.11, 12.10]
    load_values = [11.99, 12.00, 11.98, 11.99, 12.01, 11.97, 12.00, 11.99]
    cpu_values = [5000.0, 5010.0, 4990.0, 5005.0, 5020.0, 4985.0, 5015.0, 5000.0]
    for idx, (soak, load, cpu) in enumerate(zip(soak_values, load_values, cpu_values), start=1):
        run_dir = bench_root / "run-{:02d}".format(idx)
        run_dir.mkdir()
        _seed_gage_rr_run(run_dir, soak_12v=soak, load_12v=load, cpu_mhz=cpu, sample_count=24)

    result = compute_bench_gage_rr(
        bench,
        runs_dir=runs_dir,
        output_dir=tmp_path / "baselines",
        min_runs=8,
        min_samples_per_run=20,
        noise_cap_v=0.15,
    )

    payload = result["payload"]
    assert result["output_path"].exists()
    assert payload["board"] == "TRX50 AI TOP"
    assert payload["gpu"] == "NVIDIA RTX 5090"
    assert payload["n_runs"] == 8
    assert payload["metrics"]["soak_12v_mean"] == pytest.approx(statistics.mean(soak_values))
    assert payload["metrics"]["soak_12v_pstdev"] == pytest.approx(statistics.pstdev(soak_values))
    assert payload["metrics"]["load_12v_min_mean"] == pytest.approx(statistics.mean(load_values))
    assert payload["metrics"]["load_12v_min_pstdev"] == pytest.approx(statistics.pstdev(load_values))
    assert payload["metrics"]["soak_cpu_mhz_p90"] == pytest.approx(statistics.mean(cpu_values))
    assert payload["metrics"]["bench_noise_floor_v"] == pytest.approx(statistics.pstdev(soak_values))


def test_bench_gage_rr_refuses_if_run_failed(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    bench = "trx50-ai-top-nvidia-rtx-5090"
    bench_root = runs_dir / "gage-rr" / bench
    bench_root.mkdir(parents=True)
    for idx in range(8):
        run_dir = bench_root / "run-{:02d}".format(idx)
        run_dir.mkdir()
        _seed_gage_rr_run(run_dir, verdict="FAIL" if idx == 3 else "PASS", sample_count=24)

    with pytest.raises(RuntimeError, match="not PASS"):
        compute_bench_gage_rr(
            bench,
            runs_dir=runs_dir,
            output_dir=tmp_path / "baselines",
            min_runs=8,
            min_samples_per_run=20,
        )
    assert not list((tmp_path / "baselines").glob("*.json"))


def test_bench_gage_rr_refuses_if_pstdev_exceeds_cap(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    bench = "trx50-ai-top-nvidia-rtx-5090"
    bench_root = runs_dir / "gage-rr" / bench
    bench_root.mkdir(parents=True)
    for idx, soak in enumerate([12.10, 12.60, 11.50, 12.55, 11.40, 12.45, 11.35, 12.50], start=1):
        run_dir = bench_root / "run-{:02d}".format(idx)
        run_dir.mkdir()
        _seed_gage_rr_run(run_dir, soak_12v=soak, sample_count=24)

    with pytest.raises(RuntimeError, match="noise floor"):
        compute_bench_gage_rr(
            bench,
            runs_dir=runs_dir,
            output_dir=tmp_path / "baselines",
            min_runs=8,
            min_samples_per_run=20,
            noise_cap_v=0.15,
        )


def test_bench_gage_rr_refuses_if_n_runs_below_minimum(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    bench = "trx50-ai-top-nvidia-rtx-5090"
    bench_root = runs_dir / "gage-rr" / bench
    bench_root.mkdir(parents=True)
    for idx in range(7):
        run_dir = bench_root / "run-{:02d}".format(idx)
        run_dir.mkdir()
        _seed_gage_rr_run(run_dir, sample_count=24)

    with pytest.raises(RuntimeError, match="need at least 8"):
        compute_bench_gage_rr(
            bench,
            runs_dir=runs_dir,
            output_dir=tmp_path / "baselines",
            min_runs=8,
            min_samples_per_run=20,
        )


def test_load_bench_baseline_missing_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psfu.runtime.BASELINES_DIR", tmp_path / "baselines")
    assert load_bench_baseline("TRX50 AI TOP", "NVIDIA RTX 5090") is None


def test_load_bench_baseline_present_returns_dict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    payload = {"bench_id": "trx50-ai-top-nvidia-rtx-5090", "metrics": {"bench_noise_floor_v": 0.018}}
    (baseline_dir / "trx50-ai-top-nvidia-rtx-5090.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    monkeypatch.setattr("psfu.runtime.BASELINES_DIR", baseline_dir)
    assert load_bench_baseline("TRX50 AI TOP", "NVIDIA RTX 5090") == payload


def test_create_run_dir_category_routes_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psfu.runtime.RUNS_DIR", tmp_path / "runs")
    run_dir = create_run_dir("HX1500-03", category=Path("gage-rr") / "bench-a")
    assert run_dir.parent == tmp_path / "runs" / "gage-rr" / "bench-a"
    assert run_dir.name.endswith("hx1500-03")
