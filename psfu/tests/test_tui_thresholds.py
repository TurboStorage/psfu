"""Color-threshold logic for the curses TUI (``_metric_level``).

These tests exercise the per-metric ``cool``/``warm``/``hot`` classification
without spinning up a real curses session. We construct a bare object and
attach the minimal attributes the function touches, so no display is needed.
"""

from __future__ import annotations

from typing import Any

from psfu.ui.tui import TuiDashboard


def _bare_dashboard() -> Any:
    """Return a ``TuiDashboard`` instance with the per-instance threshold
    dicts shadowed from the class (mirrors what ``_auto_calibrate`` does
    on its first tick). We bypass ``__init__`` so curses is never invoked.
    """
    dash = TuiDashboard.__new__(TuiDashboard)
    dash._DOMAIN = dict(TuiDashboard._DOMAIN)
    dash._DOMAIN_FLOOR = dict(TuiDashboard._DOMAIN_FLOOR)
    dash._DOMAIN_ADAPTIVE = dict(TuiDashboard._DOMAIN_ADAPTIVE)
    dash._SPARK_RANGE = dict(TuiDashboard._SPARK_RANGE)
    dash._RAIL_SPEC = {k: dict(v) for k, v in TuiDashboard._RAIL_SPEC.items()}
    dash._AUTO_CALIBRATE_SEEDS = TuiDashboard._AUTO_CALIBRATE_SEEDS
    dash._max_seen = {}
    dash._tick_count = 0
    return dash


def test_fan_adaptive_inactive_before_min_observed() -> None:
    """Below the 600-RPM observed floor, adaptive coloring is suppressed
    so an idle fan doesn't flag warm just because it's at its own max."""
    dash = _bare_dashboard()
    dash._max_seen["psu_fan_rpm"] = 400.0  # below min_observed=600
    dash._tick_count = 200                 # well past min_ticks
    assert dash._metric_level("psu_fan_rpm", 400.0) == "cool"
    # Even querying at the same max — no coloring yet.
    assert dash._metric_level("psu_fan_rpm", 400.0) == "cool"


def test_fan_adaptive_inactive_before_min_ticks() -> None:
    """Before enough ticks have elapsed, adaptive coloring is suppressed
    (early peaks can be spurious). Falls through to spark-range fallback."""
    dash = _bare_dashboard()
    dash._max_seen["psu_fan_rpm"] = 1800.0
    dash._tick_count = 30  # below min_ticks=60
    # 1800 is well under the 2500 RPM seed ceiling → cool via fallback.
    assert dash._metric_level("psu_fan_rpm", 1800.0) == "cool"


def test_fan_adaptive_active_scales_from_observed_peak() -> None:
    """Once both gates clear, warm/hot are computed as ratios of the
    observed peak, not absolute RPM. A fan peaking at 1800 flags amber
    at 1350 (75%) and red at 1620 (90%)."""
    dash = _bare_dashboard()
    dash._max_seen["psu_fan_rpm"] = 1800.0
    dash._tick_count = 120
    assert dash._metric_level("psu_fan_rpm", 400.0) == "cool"    # idle
    assert dash._metric_level("psu_fan_rpm", 1200.0) == "cool"   # below 75%
    assert dash._metric_level("psu_fan_rpm", 1350.0) == "warm"   # 75%
    assert dash._metric_level("psu_fan_rpm", 1500.0) == "warm"   # between
    assert dash._metric_level("psu_fan_rpm", 1620.0) == "hot"    # 90%
    assert dash._metric_level("psu_fan_rpm", 1800.0) == "hot"    # at peak


def test_fan_adaptive_rescales_as_peak_ratchets() -> None:
    """If the peak climbs (e.g. ramp phase pushes fan harder), the
    warm/hot cutoffs track upward with it. 1350 RPM was amber when the
    peak was 1800; once the peak reaches 2400, 1350 is cool again."""
    dash = _bare_dashboard()
    dash._tick_count = 120
    dash._max_seen["psu_fan_rpm"] = 1800.0
    assert dash._metric_level("psu_fan_rpm", 1350.0) == "warm"
    dash._max_seen["psu_fan_rpm"] = 2400.0
    # 1350 / 2400 = 0.5625 — well below 0.75 threshold
    assert dash._metric_level("psu_fan_rpm", 1350.0) == "cool"
    # New 75% mark: 2400 * 0.75 = 1800
    assert dash._metric_level("psu_fan_rpm", 1800.0) == "warm"


def test_efficiency_floor_threshold() -> None:
    """Efficiency uses _DOMAIN_FLOOR (lower = worse). 90% is healthy;
    84% is amber; 79% is red — thresholds at 85 / 80."""
    dash = _bare_dashboard()
    assert dash._metric_level("psu_efficiency_pct", 92.0) == "cool"
    assert dash._metric_level("psu_efficiency_pct", 85.1) == "cool"
    assert dash._metric_level("psu_efficiency_pct", 85.0) == "warm"  # inclusive
    assert dash._metric_level("psu_efficiency_pct", 80.0) == "hot"   # inclusive
    assert dash._metric_level("psu_efficiency_pct", 70.0) == "hot"


def test_psu_temps_ceiling_thresholds() -> None:
    """PSU VRM temp: warm 60 °C, hot 75 °C. Ambient-side: warm 45, hot 55."""
    dash = _bare_dashboard()
    assert dash._metric_level("psu_temp_c", 50.0) == "cool"
    assert dash._metric_level("psu_temp_c", 60.0) == "warm"
    assert dash._metric_level("psu_temp_c", 75.0) == "hot"
    assert dash._metric_level("psu_temp2_c", 40.0) == "cool"
    assert dash._metric_level("psu_temp2_c", 45.0) == "warm"
    assert dash._metric_level("psu_temp2_c", 55.0) == "hot"


def test_nvme_pcie_link_ranges_exist_and_render_cool() -> None:
    """NVMe PCIe link state is operator-visible context, not thresholded
    alerting in the TUI. The storage verdict rule handles downshift claims."""
    dash = _bare_dashboard()
    assert dash._SPARK_RANGE["nvme_pcie_link_gen"] == (1.0, 5.0)
    assert dash._SPARK_RANGE["nvme_pcie_link_width"] == (1.0, 4.0)
    assert dash._metric_level("nvme_pcie_link_gen", 5.0) == "cool"
    assert dash._metric_level("nvme_pcie_link_width", 4.0) == "cool"


def test_ssd_mixed_throughput_range_exists_and_renders_cool() -> None:
    dash = _bare_dashboard()
    assert dash._SPARK_RANGE["ssd_mixed_mbps"] == (0.0, 12000.0)
    assert dash._metric_level("ssd_mixed_mbps", 8000.0) == "cool"


def test_fio_latency_ranges_exist_and_render_warm_hot() -> None:
    dash = _bare_dashboard()
    assert dash._SPARK_RANGE["fio_read_lat_p99_us"] == (0.0, 50000.0)
    assert dash._metric_level("fio_read_lat_p99_us", 1500.0) == "cool"
    assert dash._metric_level("fio_read_lat_p99_us", 2000.0) == "warm"
    assert dash._metric_level("fio_read_lat_p99_999_us", 50000.0) == "hot"
