"""Unit tests for TelemetryCollector.merge_turbostat.

Covers the 2026-04-14 fix for AMD systems where /sys/class/powercap does
not expose package RAPL — cpu_pkg_watt was NULL across the whole run
(observed 0/1815 samples on TRX50 AI TOP). Turbostat's MSR-based PkgWatt
is now mirrored into cpu_pkg_watt when RAPL is absent.
"""

from __future__ import annotations

from typing import Any, Dict

from psfu.collectors.telemetry import TelemetryCollector


def _payload() -> Dict[str, Any]:
    return {
        "cpu_pkg_watt": None,
        "cpu_pkg_watt_turbo": None,
        "cpu_avg_mhz": None,
        "cpu_max_mhz": None,
        "cpu_busy_pct_turbo": None,
        "cpu_core_watt": None,
    }


def test_turbostat_seeds_cpu_pkg_watt_when_rapl_missing():
    """AMD Zen 5: RAPL powercap not exposed → cpu_pkg_watt would stay
    NULL. Turbostat has the value; merge must mirror it over."""
    col = TelemetryCollector()
    payload = _payload()
    snap = {"cpu_pkg_watt_turbo": 275.5, "cpu_avg_mhz": 4800.0}
    col.merge_turbostat(payload, snap)
    assert payload["cpu_pkg_watt"] == 275.5
    assert payload["cpu_pkg_watt_turbo"] == 275.5


def test_rapl_wins_when_both_present():
    """When RAPL already populated cpu_pkg_watt before the merge, the
    turbostat mirror must not overwrite it. RAPL is the more direct
    reading on Intel systems."""
    col = TelemetryCollector()
    payload = _payload()
    payload["cpu_pkg_watt"] = 150.0  # set by _read_rapl_pkg_watt earlier
    snap = {"cpu_pkg_watt_turbo": 275.5}
    col.merge_turbostat(payload, snap)
    assert payload["cpu_pkg_watt"] == 150.0
    assert payload["cpu_pkg_watt_turbo"] == 275.5


def test_both_none_stays_none():
    """No RAPL, no turbostat → cpu_pkg_watt remains None (no fabrication)."""
    col = TelemetryCollector()
    payload = _payload()
    col.merge_turbostat(payload, {})
    assert payload["cpu_pkg_watt"] is None
    assert payload["cpu_pkg_watt_turbo"] is None
