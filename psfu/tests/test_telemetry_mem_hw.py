"""Static RAM hardware probes — total (GB) + DIMM speed (MT/s).

Added 2026-04-16 so the TUI RAM panel can display installed amount and
configured DIMM speed alongside the per-sample mem_used_pct. Both are
probed ONCE at collector init and mirrored into every sample.

Fields are nice-to-have context, not verdict inputs, so both must degrade
gracefully to None: no dmidecode, no sudo, unparseable output, etc.
"""

from __future__ import annotations

from psfu.collectors.telemetry import TelemetryCollector


def test_parse_mts_value_accepts_units() -> None:
    """dmidecode emits ``6400 MT/s`` and ``4800 MHz`` depending on version."""
    assert TelemetryCollector._parse_mts_value("6400 MT/s") == 6400
    assert TelemetryCollector._parse_mts_value("4800 MHz") == 4800
    assert TelemetryCollector._parse_mts_value("  3200 MT/s  ") == 3200


def test_parse_mts_value_rejects_unknown() -> None:
    """Empty slots and unpopulated DIMM records must return None so the max
    across slots doesn't get contaminated by zero / placeholder values."""
    assert TelemetryCollector._parse_mts_value("Unknown") is None
    assert TelemetryCollector._parse_mts_value("Not Specified") is None
    assert TelemetryCollector._parse_mts_value("") is None
    # Sole numeric token with no unit — reject rather than guess
    assert TelemetryCollector._parse_mts_value("6400") is None
    # Zero-speed values (dmidecode's "empty slot" sentinel on some BIOSes)
    assert TelemetryCollector._parse_mts_value("0 MT/s") is None


def test_dmidecode_mem_speed_picks_max_configured() -> None:
    """On a kit where slot speeds vary (e.g. 2 of 4 DIMMs at DDR5-6000, 2 at
    DDR5-4800), the panel should show the actual configured bus speed — which
    is the max of the ``Configured Memory Speed`` values, since all populated
    slots run the same effective rate."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    collector.available = {"dmidecode": True}

    def fake_run(cmd, timeout):  # noqa: ARG001
        out = (
            "Memory Device\n"
            "\tSpeed: 6400 MT/s\n"
            "\tConfigured Memory Speed: 6000 MT/s\n"
            "\nMemory Device\n"
            "\tSpeed: 6400 MT/s\n"
            "\tConfigured Memory Speed: 6000 MT/s\n"
            "\nMemory Device\n"
            "\tSpeed: Unknown\n"
            "\tConfigured Memory Speed: Unknown\n"
        )
        return 0, out, ""

    import psfu.collectors.telemetry as tel
    orig = tel.run_capture
    tel.run_capture = fake_run
    try:
        speed = collector._dmidecode_mem_speed(["dmidecode"])
    finally:
        tel.run_capture = orig
    assert speed == 6000


def test_dmidecode_mem_speed_falls_back_to_speed_field() -> None:
    """Older dmidecode releases omit ``Configured Memory Speed``. The parser
    must still produce a value from the plain ``Speed:`` field in that case —
    otherwise an RMA bench running an older distro shows no DIMM speed at all."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    collector.available = {"dmidecode": True}

    def fake_run(cmd, timeout):  # noqa: ARG001
        return 0, "Memory Device\n\tSpeed: 3200 MT/s\n", ""

    import psfu.collectors.telemetry as tel
    orig = tel.run_capture
    tel.run_capture = fake_run
    try:
        speed = collector._dmidecode_mem_speed(["dmidecode"])
    finally:
        tel.run_capture = orig
    assert speed == 3200


def test_dmidecode_mem_speed_returns_none_on_nonzero_rc() -> None:
    """Permission denied / dmidecode crash / missing /sys entries all manifest
    as non-zero rc. The collector must not raise — speed column stays NULL."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    collector.available = {"dmidecode": True}

    def fake_run(cmd, timeout):  # noqa: ARG001
        return 1, "", "Permission denied"

    import psfu.collectors.telemetry as tel
    orig = tel.run_capture
    tel.run_capture = fake_run
    try:
        speed = collector._dmidecode_mem_speed(["dmidecode"])
    finally:
        tel.run_capture = orig
    assert speed is None


def test_probe_mem_speed_returns_none_without_dmidecode() -> None:
    """A bench without dmidecode installed returns None cleanly — no exception,
    no retries, no 10 s timeout wait."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    collector.available = {"dmidecode": False}
    assert collector._probe_mem_speed_mts() is None
