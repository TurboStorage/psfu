"""Tests for the root-free sysfs CPU fallback collector."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import pytest

from psfu.collectors import cpufreq_sysfs as mod
from psfu.collectors.cpufreq_sysfs import CpuFreqSysfsCollector


def _write_cpufreq_tree(root: Path, khz_per_cpu: List[int]) -> List[Path]:
    """Create /sys/devices/system/cpu/cpu{N}/cpufreq/scaling_cur_freq files."""
    paths: List[Path] = []
    for i, khz in enumerate(khz_per_cpu):
        d = root / "sys/devices/system/cpu/cpu{}/cpufreq".format(i)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "scaling_cur_freq"
        p.write_text(str(khz))
        paths.append(p)
    return paths


def _write_proc_stat(root: Path, idle: int, non_idle: int) -> Path:
    """Create /proc/stat with a synthetic cpu aggregate line."""
    stat = root / "proc/stat"
    stat.parent.mkdir(parents=True, exist_ok=True)
    # cpu  user nice system idle iowait irq softirq steal
    # split non_idle across user/system so the parser sees a realistic shape
    user = non_idle // 2
    system = non_idle - user
    stat.write_text(
        "cpu  {} 0 {} {} 0 0 0 0\n".format(user, system, idle)
        + "cpu0 0 0 0 0 0 0 0 0\n"
    )
    return stat


def _patch_paths(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(
        mod,
        "_CPUFREQ_GLOB",
        str(root / "sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"),
    )
    monkeypatch.setattr(mod, "_PROC_STAT", str(root / "proc/stat"))


def test_unavailable_when_no_sysfs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_paths(monkeypatch, tmp_path)
    c = CpuFreqSysfsCollector()
    assert c.available is False
    assert c.start() is not None


def test_avg_and_max_from_cpufreq(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_cpufreq_tree(tmp_path, [2_000_000, 3_000_000, 4_000_000, 5_000_000])
    _write_proc_stat(tmp_path, idle=1000, non_idle=0)
    _patch_paths(monkeypatch, tmp_path)

    c = CpuFreqSysfsCollector()
    assert c.available is True
    assert c.start() is None

    snap = c.snapshot()
    # (2+3+4+5)/4 = 3500 MHz avg, 5000 MHz max
    assert snap["cpu_avg_mhz"] == pytest.approx(3500.0)
    assert snap["cpu_max_mhz"] == pytest.approx(5000.0)
    # MSR-only fields stay None
    assert snap["cpu_core_watt"] is None
    assert snap["cpu_pkg_watt_turbo"] is None


def test_busy_pct_delta(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_cpufreq_tree(tmp_path, [2_000_000])
    _write_proc_stat(tmp_path, idle=1000, non_idle=0)
    _patch_paths(monkeypatch, tmp_path)

    c = CpuFreqSysfsCollector()
    c.start()
    # First snapshot AFTER start() still has a baseline — but since start()
    # seeds it, the first snapshot computes against that seed.
    _write_proc_stat(tmp_path, idle=1100, non_idle=900)
    snap = c.snapshot()
    # delta: idle +100, total +1000 → busy% = 90%
    assert snap["cpu_busy_pct_turbo"] == pytest.approx(90.0)


def test_busy_pct_none_without_baseline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If the collector never starts (seeds baseline), first snapshot returns None for busy."""
    _write_cpufreq_tree(tmp_path, [2_000_000])
    _write_proc_stat(tmp_path, idle=1000, non_idle=500)
    _patch_paths(monkeypatch, tmp_path)

    c = CpuFreqSysfsCollector()
    # Do NOT call start() — no baseline seeded
    snap = c.snapshot()
    assert snap["cpu_busy_pct_turbo"] is None
    # MHz still works without a baseline
    assert snap["cpu_avg_mhz"] == pytest.approx(2000.0)


def test_snapshot_keys_match_turbostat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Fallback must expose the same keys as TurbostatCollector.snapshot()."""
    _write_cpufreq_tree(tmp_path, [3_000_000])
    _write_proc_stat(tmp_path, idle=1000, non_idle=0)
    _patch_paths(monkeypatch, tmp_path)

    c = CpuFreqSysfsCollector()
    c.start()
    expected = {
        "cpu_avg_mhz",
        "cpu_max_mhz",
        "cpu_busy_pct_turbo",
        "cpu_core_watt",
        "cpu_pkg_watt_turbo",
    }
    assert set(c.snapshot().keys()) == expected


def test_stop_is_noop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_cpufreq_tree(tmp_path, [2_000_000])
    _write_proc_stat(tmp_path, idle=1000, non_idle=0)
    _patch_paths(monkeypatch, tmp_path)

    c = CpuFreqSysfsCollector()
    c.start()
    c.stop()  # must not raise
    # snapshot still works after stop
    assert c.snapshot()["cpu_avg_mhz"] == pytest.approx(2000.0)
