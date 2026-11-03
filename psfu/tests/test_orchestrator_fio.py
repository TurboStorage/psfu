"""S4 orchestrator tests: FioSpec routing vs sha256 fallback.

Covers ``psfu.runners.orchestrator.build_phase_commands`` storage-axis
dispatch. Follows PLAN.md §Resume checkpoint DoD:

* test_storage_stress_uses_fio_runner_by_default
* test_storage_stress_falls_back_when_fio_unbuilt
"""
from __future__ import annotations

from pathlib import Path

from psfu.models import Phase
from psfu.runners.orchestrator import (
    FIO_JOBS_DIR,
    FioSpec,
    _select_fio_job_for_phase,
    _select_fio_job_for_phase_destructive,
    build_phase_commands,
)


def _stress_phase(duration_sec: int = 120) -> Phase:
    return Phase(
        name="steady",
        duration_sec=duration_sec,
        cpu_mode="idle",
        gpu_mode="idle",
        mem_mode="idle",
        storage_mode="stress",
    )


def test_select_fio_job_long_phase_uses_seq_read() -> None:
    phase = _stress_phase(duration_sec=600)
    job = _select_fio_job_for_phase(phase)
    assert job.name == "seq_read_full_capacity.fio"
    assert job.parent == FIO_JOBS_DIR


def test_select_fio_job_short_phase_uses_qd1_latency() -> None:
    phase = _stress_phase(duration_sec=10)
    job = _select_fio_job_for_phase(phase)
    assert job.name == "rand_read_qd1_latency.fio"


def test_select_destructive_speed_verify_phase_uses_2m_nj16_qd8_job() -> None:
    phase = Phase(
        name="speed-verify",
        duration_sec=1800,
        cpu_mode="idle",
        gpu_mode="idle",
        mem_mode="idle",
        storage_mode="stress",
    )
    job = _select_fio_job_for_phase_destructive(phase)
    assert job.name == "seq_write_verify_2m_nj16_qd8.fio"
    assert job.parent == FIO_JOBS_DIR


def test_storage_stress_uses_fio_runner_by_default(tmp_path, monkeypatch) -> None:
    """fio binary + operator-selected NVMe DUT + run_dir → FioSpec."""
    phase = _stress_phase(duration_sec=120)
    monkeypatch.setattr("psfu.runners.orchestrator.stress_ng_binary_path", lambda: None)
    monkeypatch.setattr("psfu.runners.orchestrator.gpu_burn_usable_binary", lambda: None)
    monkeypatch.setattr("psfu.runners.orchestrator.fio_binary_path", lambda: "/usr/bin/fio")
    cpu_cmd, gpu_cmd, gpu_cwd, mem_cmd, storage_cmd, fio_spec = build_phase_commands(
        phase,
        False,
        storage_dut_paths=["/dev/nvme0n1"],
        run_dir=tmp_path,
    )
    assert fio_spec is not None
    assert isinstance(fio_spec, FioSpec)
    assert fio_spec.device == "/dev/nvme0n1"
    assert fio_spec.phase_name == "steady"
    assert fio_spec.destructive is False
    assert fio_spec.run_dir == tmp_path
    assert fio_spec.job_file.name == "seq_read_full_capacity.fio"
    # Exactly one of (storage_cmd, fio_spec) populated — never both.
    assert storage_cmd is None


def test_storage_stress_falls_back_when_fio_unbuilt(tmp_path, monkeypatch) -> None:
    """fio binary absent → legacy sha256 filesystem worker path picked."""
    phase = _stress_phase(duration_sec=120)
    monkeypatch.setattr("psfu.runners.orchestrator.stress_ng_binary_path", lambda: None)
    monkeypatch.setattr("psfu.runners.orchestrator.gpu_burn_usable_binary", lambda: None)
    monkeypatch.setattr("psfu.runners.orchestrator.fio_binary_path", lambda: None)
    monkeypatch.setattr(
        "psfu.runners.orchestrator.resolve_storage_targets",
        lambda paths: [Path("/tmp")],
    )
    cpu_cmd, gpu_cmd, gpu_cwd, mem_cmd, storage_cmd, fio_spec = build_phase_commands(
        phase,
        False,
        storage_dut_paths=["/dev/nvme0n1"],
        run_dir=tmp_path,
    )
    assert fio_spec is None
    assert storage_cmd is not None
    assert any("storage_read.py" in arg for arg in storage_cmd)


def test_storage_stress_falls_back_when_no_nvme_dut_selected(tmp_path, monkeypatch) -> None:
    """fio available but operator selected no NVMe DUT in the lobby → fallback."""
    phase = _stress_phase(duration_sec=120)
    monkeypatch.setattr("psfu.runners.orchestrator.stress_ng_binary_path", lambda: None)
    monkeypatch.setattr("psfu.runners.orchestrator.gpu_burn_usable_binary", lambda: None)
    monkeypatch.setattr("psfu.runners.orchestrator.fio_binary_path", lambda: "/usr/bin/fio")
    monkeypatch.setattr(
        "psfu.runners.orchestrator.resolve_storage_targets",
        lambda paths: [Path("/tmp")],
    )
    cpu_cmd, gpu_cmd, gpu_cwd, mem_cmd, storage_cmd, fio_spec = build_phase_commands(
        phase,
        False,
        storage_dut_paths=[],
        run_dir=tmp_path,
    )
    assert fio_spec is None
    assert storage_cmd is not None


def test_storage_stress_falls_back_when_run_dir_missing(tmp_path, monkeypatch) -> None:
    """run_dir is required for fio output layout. Without it, fallback path."""
    phase = _stress_phase(duration_sec=120)
    monkeypatch.setattr("psfu.runners.orchestrator.stress_ng_binary_path", lambda: None)
    monkeypatch.setattr("psfu.runners.orchestrator.gpu_burn_usable_binary", lambda: None)
    monkeypatch.setattr("psfu.runners.orchestrator.fio_binary_path", lambda: "/usr/bin/fio")
    monkeypatch.setattr(
        "psfu.runners.orchestrator.resolve_storage_targets",
        lambda paths: [Path("/tmp")],
    )
    cpu_cmd, gpu_cmd, gpu_cwd, mem_cmd, storage_cmd, fio_spec = build_phase_commands(
        phase,
        False,
        storage_dut_paths=["/dev/nvme0n1"],
        run_dir=None,
    )
    assert fio_spec is None
    assert storage_cmd is not None


def test_storage_idle_never_builds_fio_spec(tmp_path, monkeypatch) -> None:
    phase = Phase(
        name="baseline",
        duration_sec=120,
        cpu_mode="idle",
        gpu_mode="idle",
        mem_mode="idle",
        storage_mode="idle",
    )
    monkeypatch.setattr("psfu.runners.orchestrator.stress_ng_binary_path", lambda: None)
    monkeypatch.setattr("psfu.runners.orchestrator.gpu_burn_usable_binary", lambda: None)
    monkeypatch.setattr("psfu.runners.orchestrator.fio_binary_path", lambda: "/usr/bin/fio")
    cpu_cmd, gpu_cmd, gpu_cwd, mem_cmd, storage_cmd, fio_spec = build_phase_commands(
        phase,
        False,
        storage_dut_paths=["/dev/nvme0n1"],
        run_dir=tmp_path,
    )
    assert fio_spec is None
    assert storage_cmd is None


def test_build_phase_commands_dry_run_returns_all_none(tmp_path) -> None:
    phase = _stress_phase(duration_sec=120)
    result = build_phase_commands(
        phase,
        True,
        storage_dut_paths=["/dev/nvme0n1"],
        run_dir=tmp_path,
    )
    assert result == (None, None, None, None, None, None)
