"""Unit tests for the corsair-psu hwmon parser.

Validates the scaling + field layout the ``corsair-psu`` kernel driver
exposes on HXi/RMi/AXi-series PSUs (HX1500i, HX1200i, RM750i, etc.) when
connected via the iCue USB-header → USB-C cable.

Scaling reference (drivers/hwmon/corsair-psu.c):
  inN_input     millivolts
  currN_input   milliamps
  powerN_input  microwatts
  tempN_input   millidegrees C
  fanN_input    RPM directly
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pytest

from psfu.collectors.telemetry import TelemetryCollector


def _write_hwmon_device(
    root: Path,
    readings: Dict[str, str],
    name: str = "corsairpsu",
) -> Path:
    """Materialise a fake hwmon device directory.

    ``readings`` maps filenames (``in1_input``, ``fan1_input``, ...) to the
    raw string the kernel driver would expose. A ``name`` file with the
    driver identifier is always written so ``_read_corsair_psu`` matches it.
    """
    device = root / "hwmon0"
    device.mkdir(parents=True)
    (device / "name").write_text(name)
    for filename, value in readings.items():
        (device / filename).write_text(value)
    return device


@pytest.fixture
def hx1500i_full(tmp_path: Path) -> Path:
    """Full HX1500i-like readings at ~400W load, 90% efficiency."""
    return _write_hwmon_device(
        tmp_path,
        {
            # Voltages (millivolts)
            "in0_input":   "118200",   # 118.2V AC input
            "in1_input":   "12080",    # 12.08V
            "in2_input":   "5020",     #  5.02V
            "in3_input":   "3305",     #  3.305V
            # Currents (milliamps)
            "curr1_input": "3750",     # 3.75A AC input
            "curr2_input": "32500",    # 32.5A on +12V
            "curr3_input": "2200",     #  2.2A on +5V
            "curr4_input": "1100",     #  1.1A on +3.3V
            # Powers (microwatts)
            "power1_input": "443000000",  # 443W total input
            "power2_input": "392000000",  # 392W on 12V
            "power3_input": "11000000",   #  11W on 5V
            "power4_input": "3600000",    # 3.6W on 3.3V
            # Temps (millidegrees C)
            "temp1_input": "52500",    # 52.5C VRM
            "temp2_input": "38200",    # 38.2C ambient
            # Fan (RPM)
            "fan1_input":  "1450",
        },
    )


def test_parse_full_hx1500i_reading(hx1500i_full: Path):
    """Confirms every field the driver exposes is surfaced with correct
    unit scaling. Pins the HX1500i unit scaling; if this test
    fails after a kernel update, the scaling factors in the driver changed
    and the parser needs a matching fix."""
    col = TelemetryCollector()
    res = col._parse_corsair_psu_device(hx1500i_full)

    # Voltages in volts
    assert res["vin_v"] == 118.2
    assert res["12v"]   == 12.08
    assert res["5v"]    == 5.02
    assert res["33v"]   == 3.305

    # Currents in amps
    assert res["i_in_a"]  == 3.75
    assert res["i_12v_a"] == 32.5
    assert res["i_5v_a"]  == 2.2
    assert res["i_33v_a"] == 1.1

    # Powers in watts
    assert res["p_in_w"]   == 443.0
    assert res["p_12v_w"]  == 392.0
    assert res["p_5v_w"]   == 11.0
    assert res["p_33v_w"]  == 3.6
    assert res["p_out_w"]  == pytest.approx(406.6, abs=0.1)

    # Efficiency = p_out / p_in * 100 = 406.6 / 443 * 100 ≈ 91.8%
    assert res["efficiency_pct"] == pytest.approx(91.8, abs=0.2)

    # Temps in degrees C
    assert res["temp1_c"] == 52.5
    assert res["temp2_c"] == 38.2

    # Fan in RPM
    assert res["fan_rpm"] == 1450.0


def test_parse_missing_optional_files_does_not_crash(tmp_path: Path):
    """Older kernel drivers and non-HXi PSUs omit some fields (e.g. the
    RMi line has no input current sensor). Missing files must degrade to
    None rather than raising."""
    device = _write_hwmon_device(
        tmp_path,
        {
            "in1_input": "12010",
            "in2_input": "5010",
            # No 3.3V, no currents, no powers, no temps, no fan.
        },
    )
    col = TelemetryCollector()
    res = col._parse_corsair_psu_device(device)

    assert res["12v"]  == 12.01
    assert res["5v"]   == 5.01
    assert res["33v"]  is None
    assert res["i_in_a"]  is None
    assert res["p_in_w"]  is None
    assert res["p_out_w"] is None
    assert res["efficiency_pct"] is None
    assert res["temp1_c"] is None
    assert res["fan_rpm"] is None


def test_efficiency_none_at_idle(tmp_path: Path):
    """At near-idle (~5W input) the divisor is noisy enough that the
    efficiency calc would produce misleading values. Parser returns None
    for efficiency below a 10W input-power floor to avoid dashboarding
    garbage."""
    device = _write_hwmon_device(
        tmp_path,
        {
            "power1_input": "5000000",   # 5W input
            "power2_input": "3000000",   # 3W on 12V
        },
    )
    col = TelemetryCollector()
    res = col._parse_corsair_psu_device(device)

    assert res["p_in_w"]  == 5.0
    assert res["p_out_w"] == 3.0
    assert res["efficiency_pct"] is None


def test_malformed_value_returns_none(tmp_path: Path):
    """A kernel driver hiccup that writes a non-numeric string must not
    propagate as a parse error — the field degrades to None just like
    a missing file."""
    device = _write_hwmon_device(
        tmp_path,
        {
            "in1_input": "not-a-number",
            "in2_input": "5000",
        },
    )
    col = TelemetryCollector()
    res = col._parse_corsair_psu_device(device)

    assert res["12v"] is None
    assert res["5v"]  == 5.0


def test_parse_ignores_name_file(tmp_path: Path):
    """``_parse_corsair_psu_device`` trusts the caller to have filtered on
    ``name`` already, so the parser itself works regardless of which
    modalias variant (``corsairpsu``/``corsair-psu``) the kernel reports.
    Regression guard: the name-matching logic lives in the wrapper
    ``_read_corsair_psu`` and should NOT be duplicated in the parser."""
    col = TelemetryCollector()
    device = _write_hwmon_device(
        tmp_path,
        {"in1_input": "12000"},
        name="corsair-psu",  # hyphenated variant
    )
    res = col._parse_corsair_psu_device(device)
    assert res["12v"] == 12.0
