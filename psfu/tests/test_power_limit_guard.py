"""Regression tests for PowerLimitGuard clamping to hardware min power limit.

Consumer NVIDIA cards reject ``nvidia-smi -pl`` below ``power.min_limit``
(RTX 5090 ~300W, RTX 4090 ~150W). Before the clamp, ``gpu_load_fraction=0.2``
on a 575W 5090 requested 115W, ``nvidia-smi -pl 115`` returned non-zero, the
guard logged the failure but silently ran the stress phase at full TDP.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
from unittest import mock

import pytest

from psfu.hardware import PowerLimitGuard


@pytest.fixture
def stub_hardware(monkeypatch: pytest.MonkeyPatch):
    """Stub out nvidia-smi presence + sudo so PowerLimitGuard enters fully."""
    monkeypatch.setattr("psfu.hardware.command_exists", lambda name: True)
    monkeypatch.setattr("psfu.hardware.can_run_sudo_noninteractive", lambda: True)


def _stub_apply(calls: List[Tuple[int, int]]):
    """Return a fake ``nvidia_set_power_limit`` that records (idx, watts)."""
    def _apply(watts: int, gpu_indices=None):
        idx = (gpu_indices or [0])[0]
        calls.append((idx, int(watts)))
        return True, "nvidia-smi -pl {} ok".format(watts)
    return _apply


def test_clamps_low_fraction_to_min_limit(monkeypatch, stub_hardware):
    """5090 @ 20% fraction: raw 115W → clamped to 300W min."""
    monkeypatch.setattr("psfu.hardware.nvidia_gpu_current_power_limits", lambda: [(0, 575)])
    monkeypatch.setattr("psfu.hardware.nvidia_gpu_power_limit_ranges", lambda: {0: (300, 600)})
    calls: List[Tuple[int, int]] = []
    monkeypatch.setattr("psfu.hardware.nvidia_set_power_limit", _stub_apply(calls))

    actions: List[str] = []
    with PowerLimitGuard(0.2, actions):
        pass

    assert calls[0] == (0, 300), "expected clamp to min 300W, got {}".format(calls[0])
    assert any("clamped to hw min 300" in a for a in actions), actions


def test_applies_fraction_above_min_limit(monkeypatch, stub_hardware):
    """5090 @ 60% fraction: raw 345W (> 300W min) → applied as-is."""
    monkeypatch.setattr("psfu.hardware.nvidia_gpu_current_power_limits", lambda: [(0, 575)])
    monkeypatch.setattr("psfu.hardware.nvidia_gpu_power_limit_ranges", lambda: {0: (300, 600)})
    calls: List[Tuple[int, int]] = []
    monkeypatch.setattr("psfu.hardware.nvidia_set_power_limit", _stub_apply(calls))

    actions: List[str] = []
    with PowerLimitGuard(0.6, actions):
        pass

    assert calls[0] == (0, 345)
    # Non-clamp path logs the plain "-> 345 W (nvidia-smi -pl 345 ok)" line
    assert not any("clamped" in a for a in actions)


def test_restores_original_on_exit(monkeypatch, stub_hardware):
    """Guard must restore pre-cap limit when the with block exits."""
    monkeypatch.setattr("psfu.hardware.nvidia_gpu_current_power_limits", lambda: [(0, 575)])
    monkeypatch.setattr("psfu.hardware.nvidia_gpu_power_limit_ranges", lambda: {0: (300, 600)})
    calls: List[Tuple[int, int]] = []
    monkeypatch.setattr("psfu.hardware.nvidia_set_power_limit", _stub_apply(calls))

    actions: List[str] = []
    with PowerLimitGuard(0.4, actions):
        pass

    assert calls[0][1] == 300  # enter clamp
    assert calls[-1] == (0, 575)  # exit restore


def test_no_op_at_full_scale(monkeypatch, stub_hardware):
    """scale >= 0.999 short-circuits before querying nvidia-smi."""
    monkeypatch.setattr("psfu.hardware.nvidia_gpu_current_power_limits", lambda: [(0, 575)])
    monkeypatch.setattr("psfu.hardware.nvidia_gpu_power_limit_ranges", lambda: {0: (300, 600)})
    calls: List[Tuple[int, int]] = []
    monkeypatch.setattr("psfu.hardware.nvidia_set_power_limit", _stub_apply(calls))

    with PowerLimitGuard(1.0, []):
        pass

    assert calls == []


def test_unknown_min_limit_falls_back_to_100w_floor(monkeypatch, stub_hardware):
    """If range query returns empty, fall back to legacy 100W floor."""
    monkeypatch.setattr("psfu.hardware.nvidia_gpu_current_power_limits", lambda: [(0, 575)])
    monkeypatch.setattr("psfu.hardware.nvidia_gpu_power_limit_ranges", lambda: {})
    calls: List[Tuple[int, int]] = []
    monkeypatch.setattr("psfu.hardware.nvidia_set_power_limit", _stub_apply(calls))

    actions: List[str] = []
    with PowerLimitGuard(0.1, actions):
        pass

    # 575 * 0.1 = 57, below 100W floor → clamped up to 100W
    assert calls[0] == (0, 100)


def test_multi_gpu_independent_ranges(monkeypatch, stub_hardware):
    """Each GPU clamps to its own min limit — 5090 + 4090 mixed rig."""
    monkeypatch.setattr(
        "psfu.hardware.nvidia_gpu_current_power_limits",
        lambda: [(0, 575), (1, 450)],
    )
    monkeypatch.setattr(
        "psfu.hardware.nvidia_gpu_power_limit_ranges",
        lambda: {0: (300, 600), 1: (150, 500)},
    )
    calls: List[Tuple[int, int]] = []
    monkeypatch.setattr("psfu.hardware.nvidia_set_power_limit", _stub_apply(calls))

    with PowerLimitGuard(0.3, []):
        pass

    # GPU 0 (5090): 575 * 0.3 = 172, clamped to 300
    # GPU 1 (4090): 450 * 0.3 = 135, clamped to 150
    assert (0, 300) in calls
    assert (1, 150) in calls
