import argparse
from pathlib import Path

from psfu.models import Phase, Profile
from psfu.runners.execute import dut_enabled_from_args, effective_phase_modes
from psfu.runners.orchestrator import build_phase_commands


def test_dut_enabled_from_args_explicit_true():
    """Explicit --dut-* flags override profile defaults."""
    args = argparse.Namespace(dut_cpu=True, dut_gpu=True, dut_mem=True, dut_storage=True)
    assert dut_enabled_from_args(args) == {
        "cpu": True,
        "gpu": True,
        "mem": True,
        "storage": True,
    }


def test_dut_enabled_from_args_none_defaults_to_all_true():
    """When CLI args are None (not passed), no profile → default all True."""
    args = argparse.Namespace(dut_cpu=None, dut_gpu=None, dut_mem=None, dut_storage=None)
    assert dut_enabled_from_args(args) == {
        "cpu": True,
        "gpu": True,
        "mem": True,
        "storage": True,
    }


def test_dut_enabled_defaults_off_when_profile_has_no_phases():
    """If a profile has no mem/storage phases, those DUTs default to OFF."""
    profile = Profile({
        "name": "cpu-gpu-only",
        "phases": [
            {"name": "soak", "duration_sec": 60, "cpu_mode": "idle", "gpu_mode": "idle"},
            {"name": "steady", "duration_sec": 300, "cpu_mode": "stress", "gpu_mode": "stress"},
        ],
    })
    args = argparse.Namespace()  # no dut_* attrs → falls through to profile defaults
    result = dut_enabled_from_args(args, profile=profile)
    assert result["cpu"] is True
    assert result["gpu"] is True
    assert result["mem"] is False
    assert result["storage"] is False


def test_effective_phase_modes_force_disabled_targets_idle():
    phase = Phase(
        name="stress",
        duration_sec=30,
        cpu_mode="pulse",
        gpu_mode="stress",
        mem_mode="ramp",
        storage_mode="stress",
    )
    modes = effective_phase_modes(
        phase,
        {"cpu": False, "gpu": True, "mem": False, "storage": False},
    )
    assert modes == {
        "cpu": "idle",
        "gpu": "stress",
        "mem": "idle",
        "storage": "idle",
    }


def test_build_phase_commands_respects_disabled_duts(monkeypatch):
    phase = Phase(
        name="stress",
        duration_sec=20,
        cpu_mode="stress",
        gpu_mode="stress",
        mem_mode="stress",
        storage_mode="stress",
        mem_target_pct=60,
    )
    monkeypatch.setattr("psfu.runners.orchestrator.stress_ng_binary_path", lambda: "/bin/stress-ng")
    monkeypatch.setattr("psfu.runners.orchestrator.gpu_burn_usable_binary", lambda: "/tmp/gpu-burn")
    monkeypatch.setattr("psfu.runners.orchestrator.resolve_storage_targets", lambda paths: [])

    cpu_cmd, gpu_cmd, gpu_cwd, mem_cmd, storage_cmd, fio_spec = build_phase_commands(
        phase,
        False,
        dut_enabled={"cpu": False, "gpu": True, "mem": False, "storage": False},
    )

    assert cpu_cmd is None
    assert mem_cmd is None
    assert storage_cmd is None
    assert fio_spec is None
    assert gpu_cmd == ["/tmp/gpu-burn", "-tc", "20"]
    assert gpu_cwd == str(Path("/tmp/gpu-burn").parent)
