"""v7 schema extensions (2026-04-16): rail ripple/dropouts, AER
uncorrectable, GPU throttle reason split, NVMe SMART temp-time pages,
NIC link speed/state, /proc/stat extended deltas, /proc/vmstat
pgmajfault rate, /proc/meminfo Dirty+Writeback, /sys/block/nvme*/stat
throughput + queue depth, chipset temp, thermal zone margin walker.

Every extractor tolerates missing sources cleanly (returns None or
skips silently) — benches without AER, without nvme-cli, without /sys
thermal zones, etc. must never raise.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from psfu.collectors.telemetry import (
    TelemetryCollector,
    _RAIL_LCRIT,
    _RAIL_RIPPLE_WINDOW,
)


def _fresh_collector() -> TelemetryCollector:
    """Instantiate without running __init__ probes (dmidecode, /proc/stat, …).

    We bypass __init__ and set the attributes the v7 methods actually touch.
    That's enough for pure-function tests; the extractors that depend on
    runtime state set what they need.
    """
    c = TelemetryCollector.__new__(TelemetryCollector)
    # Minimal state required by v7 methods.
    from collections import deque
    c._rail_history = deque(maxlen=_RAIL_RIPPLE_WINDOW)
    c._rail_dropouts = {"12v": 0, "5v": 0, "33v": 0}
    c._last_proc_stat_ext = None
    c._last_vmstat = None
    c._last_block_stat = None
    return c


# ── rail ripple / dropouts ────────────────────────────────────────────


def test_rail_ripple_returns_none_until_window_full() -> None:
    c = _fresh_collector()
    for _ in range(_RAIL_RIPPLE_WINDOW - 1):
        c._update_rail_history_and_dropouts(12.0, 5.0, 3.3)
    r12, r5, r33 = c._compute_rail_ripples()
    assert r12 is None and r5 is None and r33 is None


def test_rail_ripple_computes_sigma_in_millivolts() -> None:
    c = _fresh_collector()
    # 10 samples, all identical → σ = 0 mV.
    for _ in range(_RAIL_RIPPLE_WINDOW):
        c._update_rail_history_and_dropouts(12.00, 5.00, 3.30)
    r12, r5, r33 = c._compute_rail_ripples()
    assert r12 == 0.0
    assert r5 == 0.0
    assert r33 == 0.0


def test_rail_ripple_nonzero_for_noisy_rail() -> None:
    c = _fresh_collector()
    # Alternating 11.98 / 12.02 → σ = 0.02 V = 20 mV.
    for i in range(_RAIL_RIPPLE_WINDOW):
        v = 11.98 if i % 2 == 0 else 12.02
        c._update_rail_history_and_dropouts(v, 5.0, 3.3)
    r12, _r5, _r33 = c._compute_rail_ripples()
    assert r12 == pytest.approx(20.0, abs=0.5)


def test_rail_dropouts_tick_only_below_lcrit() -> None:
    c = _fresh_collector()
    # Three healthy samples; no ticks.
    for _ in range(3):
        c._update_rail_history_and_dropouts(12.0, 5.0, 3.3)
    assert c._rail_dropouts == {"12v": 0, "5v": 0, "33v": 0}
    # Now push a 12 V sample below lcrit → counter increments.
    c._update_rail_history_and_dropouts(_RAIL_LCRIT["12v"] - 0.1, 5.0, 3.3)
    assert c._rail_dropouts["12v"] == 1
    assert c._rail_dropouts["5v"] == 0


def test_rail_history_handles_none_samples() -> None:
    c = _fresh_collector()
    # Half the window has no 33 V reading; σ for 33 V must be None because
    # we require a full window of valid samples.
    for i in range(_RAIL_RIPPLE_WINDOW):
        c._update_rail_history_and_dropouts(12.0, 5.0, 3.3 if i % 2 == 0 else None)
    r12, r5, r33 = c._compute_rail_ripples()
    assert r12 == 0.0
    assert r5 == 0.0
    assert r33 is None


# ── PCIe AER uncorrectable ────────────────────────────────────────────


def test_aer_uncorr_total_sums_fatal_and_nonfatal(tmp_path, monkeypatch) -> None:
    """Builds a fake /sys/bus/pci/devices tree with two devices; the sum
    must include both fatal and nonfatal values across both devices."""
    pci = tmp_path / "pci"
    dev1 = pci / "0000_00_00.0"
    dev2 = pci / "0000_01_00.0"
    for d in (dev1, dev2):
        d.mkdir(parents=True)
    (dev1 / "aer_dev_fatal").write_text("TLP 2\nFCP 1\n")
    (dev1 / "aer_dev_nonfatal").write_text("RxErr 3\n")
    (dev2 / "aer_dev_nonfatal").write_text("BadTLP 4\n")

    import psfu.collectors.telemetry as tel
    _orig = tel.Path
    tel.Path = lambda p: pci if p == "/sys/bus/pci/devices" else _orig(p)
    try:
        c = _fresh_collector()
        total = c._read_pcie_aer_uncorr_total()
    finally:
        tel.Path = _orig

    assert total == 2 + 1 + 3 + 4


def test_aer_uncorr_total_none_when_root_missing(tmp_path) -> None:
    import psfu.collectors.telemetry as tel
    _orig = tel.Path
    tel.Path = lambda p: tmp_path / "absent" if p == "/sys/bus/pci/devices" else _orig(p)
    try:
        c = _fresh_collector()
        total = c._read_pcie_aer_uncorr_total()
    finally:
        tel.Path = _orig
    assert total is None


# ── NIC link speed + operstate ────────────────────────────────────────


def test_nic_link_state_happy_path(tmp_path) -> None:
    """Layout with enp7s0 reporting speed=10000 + operstate=up."""
    net = tmp_path / "net"
    iface = net / "enp7s0"
    iface.mkdir(parents=True)
    (iface / "speed").write_text("10000\n")
    (iface / "operstate").write_text("up\n")

    import psfu.collectors.telemetry as tel
    _orig = tel.Path
    tel.Path = lambda p: net if p == "/sys/class/net" else _orig(p)
    try:
        c = _fresh_collector()
        speed, up = c._read_nic_link_state()
    finally:
        tel.Path = _orig

    assert speed == 10000
    assert up == 1


def test_nic_link_state_skips_loopback(tmp_path) -> None:
    net = tmp_path / "net"
    lo = net / "lo"
    lo.mkdir(parents=True)
    (lo / "speed").write_text("0\n")
    (lo / "operstate").write_text("unknown\n")

    import psfu.collectors.telemetry as tel
    _orig = tel.Path
    tel.Path = lambda p: net if p == "/sys/class/net" else _orig(p)
    try:
        c = _fresh_collector()
        speed, up = c._read_nic_link_state()
    finally:
        tel.Path = _orig

    assert speed is None and up is None


def test_nic_link_state_none_when_speed_unknown(tmp_path) -> None:
    """Kernel returns -1 for interfaces that are administratively up but
    without a linked PHY. We return None for those."""
    net = tmp_path / "net"
    iface = net / "eth0"
    iface.mkdir(parents=True)
    (iface / "speed").write_text("-1\n")
    (iface / "operstate").write_text("down\n")

    import psfu.collectors.telemetry as tel
    _orig = tel.Path
    tel.Path = lambda p: net if p == "/sys/class/net" else _orig(p)
    try:
        c = _fresh_collector()
        speed, up = c._read_nic_link_state()
    finally:
        tel.Path = _orig

    assert speed is None and up is None


# ── GPU throttle reason split (unit logic) ───────────────────────────


def test_throttle_split_from_bitmask() -> None:
    """The split logic is inline in collect() but driven by
    self.last_throttle_reasons. Verify the bit masking matches the plan."""
    from psfu.collectors.telemetry import (
        _THROTTLE_HW_SLOWDOWN,
        _THROTTLE_HW_THERMAL,
        _THROTTLE_HW_POWER_BRAKE,
        _THROTTLE_SW_POWER_CAP,
    )
    assert _THROTTLE_HW_SLOWDOWN == 0x08
    assert _THROTTLE_HW_THERMAL == 0x40
    assert _THROTTLE_HW_POWER_BRAKE == 0x80
    assert _THROTTLE_SW_POWER_CAP == 0x04
    # And they collectively cover the four cases we care about — no overlap.
    bits = (
        _THROTTLE_HW_SLOWDOWN
        | _THROTTLE_HW_THERMAL
        | _THROTTLE_HW_POWER_BRAKE
        | _THROTTLE_SW_POWER_CAP
    )
    assert bits == 0xCC


# ── NVMe SMART warning/critical temp time ─────────────────────────────


def test_nvme_smart_temp_times_none_when_cli_missing(monkeypatch) -> None:
    import psfu.collectors.telemetry as tel
    monkeypatch.setattr(tel, "nvme_cli_binary_path", lambda: None)
    c = _fresh_collector()
    warn, crit = c._read_nvme_smart_temp_times()
    assert warn is None and crit is None


def test_nvme_smart_temp_times_parses_json(monkeypatch, tmp_path) -> None:
    """Stub nvme-cli JSON output; assert minutes-to-seconds conversion."""
    import psfu.collectors.telemetry as tel
    monkeypatch.setattr(tel, "nvme_cli_binary_path", lambda: "nvme")
    monkeypatch.setattr(
        tel,
        "run_capture",
        lambda cmd, timeout=5: (
            0,
            json.dumps({"warning_temp_time": 15, "critical_comp_time": 2}),
            "",
        ),
    )

    # Fake Path glob so we find one device.
    class _FakeDevRoot:
        def glob(self, pattern: str):
            class _P:
                name = "nvme0"
                def __str__(self): return "/dev/nvme0"
            return [_P()]
    _orig = tel.Path
    tel.Path = lambda p: _FakeDevRoot() if p == "/dev" else _orig(p)
    try:
        c = _fresh_collector()
        warn, crit = c._read_nvme_smart_temp_times()
    finally:
        tel.Path = _orig

    assert warn == 900    # 15 min * 60
    assert crit == 120    # 2  min * 60


# ── /proc/stat extended rates ─────────────────────────────────────────


def test_proc_stat_ext_rates_none_on_first_call(monkeypatch) -> None:
    c = _fresh_collector()
    import psfu.collectors.telemetry as tel
    monkeypatch.setattr(c, "_read_proc_stat_ext", lambda: (100, 200, 300))
    ctx, intr, sirq = c._compute_proc_stat_ext_rates()
    assert ctx is None and intr is None and sirq is None


def test_proc_stat_ext_rates_after_delta(monkeypatch) -> None:
    c = _fresh_collector()
    values = iter([(1000, 2000, 3000), (2000, 2500, 3100)])
    monkeypatch.setattr(c, "_read_proc_stat_ext", lambda: next(values))
    # Freeze time between calls.
    times = iter([10.0, 11.0])  # 1 s elapsed
    import psfu.collectors.telemetry as tel
    monkeypatch.setattr(tel.time, "monotonic", lambda: next(times))
    c._compute_proc_stat_ext_rates()       # primes
    ctx, intr, sirq = c._compute_proc_stat_ext_rates()
    assert ctx == 1000.0
    assert intr == 500.0
    assert sirq == 100.0


# ── /proc/vmstat pgmajfault rate ──────────────────────────────────────


def test_pgmajfault_rate_none_when_file_missing(monkeypatch) -> None:
    def _raise(*_a, **_kw):
        raise FileNotFoundError
    monkeypatch.setattr("builtins.open", _raise)
    c = _fresh_collector()
    assert c._compute_pgmajfault_rate() is None


def test_pgmajfault_rate_computes_delta(monkeypatch, tmp_path) -> None:
    vmstat = tmp_path / "vmstat"
    vmstat.write_text("pgmajfault 100\nother 0\n")

    real_open = open
    def fake_open(path, *a, **kw):
        if path == "/proc/vmstat":
            return real_open(vmstat, *a, **kw)
        return real_open(path, *a, **kw)
    monkeypatch.setattr("builtins.open", fake_open)

    c = _fresh_collector()
    import psfu.collectors.telemetry as tel
    times = iter([5.0, 6.0])
    monkeypatch.setattr(tel.time, "monotonic", lambda: next(times))
    assert c._compute_pgmajfault_rate() is None   # primes
    vmstat.write_text("pgmajfault 150\nother 0\n")
    rate = c._compute_pgmajfault_rate()
    assert rate == 50.0


# ── /proc/meminfo Dirty + Writeback ───────────────────────────────────


def test_meminfo_dirty_writeback_parses(monkeypatch, tmp_path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:  8000 kB\nDirty:       256 kB\nWriteback:   128 kB\n"
    )
    real_open = open
    def fake_open(path, *a, **kw):
        if path == "/proc/meminfo":
            return real_open(meminfo, *a, **kw)
        return real_open(path, *a, **kw)
    monkeypatch.setattr("builtins.open", fake_open)

    c = _fresh_collector()
    dirty, wb = c._read_meminfo_dirty_writeback()
    assert dirty == 256.0
    assert wb == 128.0


# ── /sys/block/nvme*/stat throughput + queue depth ───────────────────


def test_block_stat_first_call_returns_qdep_only(monkeypatch, tmp_path) -> None:
    blk = tmp_path / "block"
    dev = blk / "nvme0n1"
    dev.mkdir(parents=True)
    # read_ios, read_merges, read_sectors, read_ticks, write_ios,
    # write_merges, write_sectors, write_ticks, in_flight
    (dev / "stat").write_text("0 0 1000 0 0 0 2000 0 3\n")

    import psfu.collectors.telemetry as tel
    _orig = tel.Path
    tel.Path = lambda p: blk if p == "/sys/block" else _orig(p)
    try:
        c = _fresh_collector()
        r, w, q = c._compute_block_stat_metrics()
    finally:
        tel.Path = _orig
    assert r is None and w is None
    assert q == 3.0


def test_block_stat_second_call_returns_mbps(monkeypatch, tmp_path) -> None:
    blk = tmp_path / "block"
    dev = blk / "nvme0n1"
    dev.mkdir(parents=True)
    stat_file = dev / "stat"
    stat_file.write_text("0 0 1000 0 0 0 2000 0 2\n")

    import psfu.collectors.telemetry as tel
    _orig = tel.Path
    tel.Path = lambda p: blk if p == "/sys/block" else _orig(p)
    times = iter([100.0, 101.0])
    monkeypatch.setattr(tel.time, "monotonic", lambda: next(times))
    try:
        c = _fresh_collector()
        c._compute_block_stat_metrics()   # prime
        stat_file.write_text("0 0 3000 0 0 0 5000 0 5\n")
        r, w, q = c._compute_block_stat_metrics()
    finally:
        tel.Path = _orig
    # Δ read_sectors = 2000 × 512 B = 1,024,000 B over 1 s → 1.02 MB/s.
    assert r == pytest.approx(1.02, abs=0.01)
    # Δ write_sectors = 3000 × 512 B = 1,536,000 B over 1 s → 1.54 MB/s.
    assert w == pytest.approx(1.54, abs=0.01)
    assert q == 5.0


# ── thermal zone margin walker ────────────────────────────────────────


def test_thermal_zone_margin_picks_minimum(tmp_path) -> None:
    root = tmp_path / "thermal"
    # zone 0: 50 °C, trip at 90 °C → margin 40
    z0 = root / "thermal_zone0"
    z0.mkdir(parents=True)
    (z0 / "temp").write_text("50000\n")
    (z0 / "trip_point_0_temp").write_text("90000\n")
    # zone 1: 85 °C, trip at 95 °C → margin 10 (worse)
    z1 = root / "thermal_zone1"
    z1.mkdir()
    (z1 / "temp").write_text("85000\n")
    (z1 / "trip_point_0_temp").write_text("95000\n")

    import psfu.collectors.telemetry as tel
    _orig = tel.Path
    tel.Path = lambda p: root if p == "/sys/class/thermal" else _orig(p)
    try:
        c = _fresh_collector()
        margin = c._read_thermal_zone_margin()
    finally:
        tel.Path = _orig
    assert margin == 10.0


def test_thermal_zone_margin_none_when_root_missing(tmp_path) -> None:
    import psfu.collectors.telemetry as tel
    _orig = tel.Path
    tel.Path = lambda p: tmp_path / "absent" if p == "/sys/class/thermal" else _orig(p)
    try:
        c = _fresh_collector()
        margin = c._read_thermal_zone_margin()
    finally:
        tel.Path = _orig
    assert margin is None
