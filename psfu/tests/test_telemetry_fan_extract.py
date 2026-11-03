"""Fan-RPM extraction regression tests.

The ``corsair-psu`` hwmon driver surfaces the Corsair PSU fan in
``sensors -u`` output (e.g. ``fan1_input: 500``). On a bench whose
motherboard Super I/O isn't reporting any CPU/SYS fan headers — observed
on TRX50 AI TOP — the PSU fan was the only fan in sensors output and
``_extract_fan_rpms`` was labeling it as "cpu fan" in the TUI header strip.

The collector now reads ``corsair-psu`` first and passes the PSU fan RPM
as an exclusion value so the sensor-side extract skips it.
"""

from __future__ import annotations

from psfu.collectors.telemetry import TelemetryCollector


def test_fan_extract_skips_psu_fan_when_excluded() -> None:
    """When only the PSU fan is visible in sensors and we pass it as
    excluded, both cpu and sys come back None — not mislabeled PSU."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    sensors = {"fan1_input": 500.0, "psu fan": 500.0}
    cpu, sys_ = collector._extract_fan_rpms(sensors, exclude_values={500.0})
    assert cpu is None
    assert sys_ is None


def test_fan_extract_picks_mobo_fans_when_psu_excluded() -> None:
    """When the motherboard also reports fans, those still come through.
    PSU fan value is filtered; cpu + sys get the real mobo readings."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    sensors = {
        "fan1_input": 500.0,      # PSU fan (excluded)
        "fan2_input": 1800.0,     # mobo CPU fan header
        "fan3_input": 900.0,      # mobo SYS fan header
        "psu fan": 500.0,         # label-mirror of PSU fan (also excluded)
    }
    cpu, sys_ = collector._extract_fan_rpms(sensors, exclude_values={500.0})
    assert cpu == 1800.0
    assert sys_ == 900.0


def test_fan_extract_no_exclusion_preserves_legacy_behavior() -> None:
    """Benches without a Corsair PSU pass exclude_values=None and behavior
    matches the original first-two-fans-by-value heuristic."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    sensors = {"fan1_input": 1500.0, "fan2_input": 900.0}
    cpu, sys_ = collector._extract_fan_rpms(sensors)
    # Keys sort alphabetically, so fan1_input (1500) is taken as cpu and
    # fan2_input (900) as sys. This is the prior heuristic preserved.
    assert cpu == 1500.0
    assert sys_ == 900.0


def test_fan_extract_tolerates_sub_rpm_drift() -> None:
    """corsair-psu fan telemetry can jitter by ±1 RPM between the sensor
    read (at collector init) and the PSU read (mid-collect). The tolerant
    comparison in the exclusion match absorbs that drift so the PSU fan
    isn't re-surfaced as cpu fan when RPMs are 1-2 apart."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    sensors = {"fan1_input": 501.0}
    cpu, sys_ = collector._extract_fan_rpms(sensors, exclude_values={500.0})
    assert cpu is None
    assert sys_ is None
