"""v6 schema extensions (2026-04-16): NIC thermals, PSI, NIC error
totals, GPU PCIe link degradation, CPU iowait %.

Each extractor tolerates missing sources cleanly (returns None) —
benches without /proc/pressure (pre-4.20 kernels), without integrated
NIC in /sys/class/net, or without temp/error counters must never raise.
"""

from __future__ import annotations

from psfu.collectors.telemetry import TelemetryCollector


# ── PSI (Pressure Stall Information) ──────────────────────────────────


def test_read_psi_resource_parses_some_avg10(tmp_path, monkeypatch) -> None:
    """Happy path: /proc/pressure/cpu returns the avg10 float from the
    ``some`` line. ``full`` line is ignored — we only want ``some``."""
    psi_file = tmp_path / "cpu"
    psi_file.write_text(
        "some avg10=1.23 avg60=0.50 avg300=0.10 total=12345\n"
        "full avg10=0.05 avg60=0.02 avg300=0.00 total=678\n"
    )

    import psfu.collectors.telemetry as tel
    _orig_path = tel.Path
    def fake_path(p):
        if p == "/proc/pressure/cpu":
            return psi_file
        return _orig_path(p)
    tel.Path = fake_path
    try:
        collector = TelemetryCollector.__new__(TelemetryCollector)
        val = collector._read_psi_resource("cpu")
    finally:
        tel.Path = _orig_path

    assert val == 1.23


def test_read_psi_resource_returns_none_when_file_missing(tmp_path) -> None:
    """Pre-4.20 kernel / no CONFIG_PSI — file absent, returns None."""
    import psfu.collectors.telemetry as tel
    _orig_path = tel.Path
    missing = tmp_path / "nope"
    def fake_path(p):
        if p.startswith("/proc/pressure/"):
            return missing
        return _orig_path(p)
    tel.Path = fake_path
    try:
        collector = TelemetryCollector.__new__(TelemetryCollector)
        val = collector._read_psi_resource("cpu")
    finally:
        tel.Path = _orig_path

    assert val is None


def test_read_psi_triple_returns_three_values(tmp_path) -> None:
    """Integration: all three resources present."""
    base = tmp_path
    (base / "cpu").write_text("some avg10=0.10 avg60=0 avg300=0 total=0\nfull avg10=0 avg60=0 avg300=0 total=0\n")
    (base / "memory").write_text("some avg10=2.50 avg60=0 avg300=0 total=0\nfull avg10=0 avg60=0 avg300=0 total=0\n")
    (base / "io").write_text("some avg10=0.00 avg60=0 avg300=0 total=0\nfull avg10=0 avg60=0 avg300=0 total=0\n")

    import psfu.collectors.telemetry as tel
    _orig_path = tel.Path
    def fake_path(p):
        if p == "/proc/pressure/cpu":
            return base / "cpu"
        if p == "/proc/pressure/memory":
            return base / "memory"
        if p == "/proc/pressure/io":
            return base / "io"
        return _orig_path(p)
    tel.Path = fake_path
    try:
        collector = TelemetryCollector.__new__(TelemetryCollector)
        cpu, mem, io = collector._read_psi_triple()
    finally:
        tel.Path = _orig_path

    assert cpu == 0.10
    assert mem == 2.50
    assert io == 0.00


# ── NIC error counters ────────────────────────────────────────────────


def test_read_nic_error_totals_sums_across_interfaces(tmp_path) -> None:
    """Two real interfaces + loopback. Sum only the real ones. Per-
    counter None propagates correctly (eth1 has no rx_crc_errors so
    the sum is just eth0's value)."""
    net = tmp_path / "net"
    eth0 = net / "eth0"
    eth1 = net / "eth1"
    lo = net / "lo"
    for d in (eth0, eth1, lo):
        (d / "statistics").mkdir(parents=True)
    (eth0 / "operstate").write_text("up\n")
    (eth1 / "operstate").write_text("up\n")
    (lo / "operstate").write_text("unknown\n")

    (eth0 / "statistics" / "rx_crc_errors").write_text("5\n")
    (eth0 / "statistics" / "rx_errors").write_text("7\n")
    (eth0 / "statistics" / "rx_dropped").write_text("2\n")
    # eth1: no rx_crc_errors file at all (counter not exposed)
    (eth1 / "statistics" / "rx_errors").write_text("3\n")
    (eth1 / "statistics" / "rx_dropped").write_text("1\n")
    # lo: should be ignored entirely even if counters present
    (lo / "statistics" / "rx_crc_errors").write_text("999\n")
    (lo / "statistics" / "rx_errors").write_text("999\n")
    (lo / "statistics" / "rx_dropped").write_text("999\n")

    import psfu.collectors.telemetry as tel
    _orig_path = tel.Path
    def fake_path(p):
        if p == "/sys/class/net":
            return net
        return _orig_path(p)
    tel.Path = fake_path
    try:
        collector = TelemetryCollector.__new__(TelemetryCollector)
        crc, err, drop = collector._read_nic_error_totals()
    finally:
        tel.Path = _orig_path

    assert crc == 5       # eth0 only; eth1 has no crc file
    assert err == 10      # 7 + 3
    assert drop == 3      # 2 + 1


def test_read_nic_error_totals_skips_down_interfaces(tmp_path) -> None:
    """An interface whose operstate is 'down' is excluded — its counters
    are stale and including them creates phantom error growth."""
    net = tmp_path / "net"
    up_if = net / "eth0"
    down_if = net / "eth1"
    (up_if / "statistics").mkdir(parents=True)
    (down_if / "statistics").mkdir(parents=True)
    (up_if / "operstate").write_text("up\n")
    (down_if / "operstate").write_text("down\n")
    (up_if / "statistics" / "rx_errors").write_text("4\n")
    (down_if / "statistics" / "rx_errors").write_text("1000\n")

    import psfu.collectors.telemetry as tel
    _orig_path = tel.Path
    def fake_path(p):
        if p == "/sys/class/net":
            return net
        return _orig_path(p)
    tel.Path = fake_path
    try:
        collector = TelemetryCollector.__new__(TelemetryCollector)
        _crc, err, _drop = collector._read_nic_error_totals()
    finally:
        tel.Path = _orig_path

    assert err == 4


def test_read_nic_error_totals_returns_none_when_no_sys_class_net(tmp_path) -> None:
    """Pathological env — /sys/class/net missing. Clean (None, None, None)."""
    import psfu.collectors.telemetry as tel
    _orig_path = tel.Path
    missing = tmp_path / "nope"
    def fake_path(p):
        if p == "/sys/class/net":
            return missing
        return _orig_path(p)
    tel.Path = fake_path
    try:
        collector = TelemetryCollector.__new__(TelemetryCollector)
        result = collector._read_nic_error_totals()
    finally:
        tel.Path = _orig_path

    assert result == (None, None, None)


# ── CPU iowait % ──────────────────────────────────────────────────────


def test_cpu_iowait_pct_returns_none_on_first_call() -> None:
    """No previous snapshot → None. Second call after update computes
    the delta."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    collector._last_iowait_snapshot = None
    collector._last_cpu_stat = (1000, 500, 50)  # total, idle, iowait
    val = collector._cpu_iowait_pct()
    assert val is None
    # Now a second tick with +100 total, +5 iowait → 5 / 100 = 5%.
    collector._last_cpu_stat = (1100, 595, 55)
    val2 = collector._cpu_iowait_pct()
    assert val2 == 5.0


def test_cpu_iowait_pct_zero_delta_returns_none() -> None:
    """Sampled twice at the same jiffy count — can't compute. None."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    collector._last_iowait_snapshot = (1000, 50)
    collector._last_cpu_stat = (1000, 500, 50)
    assert collector._cpu_iowait_pct() is None


def test_cpu_iowait_pct_clamps_to_100() -> None:
    """Malformed jiffies where iowait delta exceeds total → clamp, no crash."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    collector._last_iowait_snapshot = (1000, 0)
    collector._last_cpu_stat = (1100, 0, 200)  # iowait jumped 200 in 100 total
    val = collector._cpu_iowait_pct()
    assert val == 100.0
