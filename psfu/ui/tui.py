from __future__ import annotations

import curses
import datetime as dt
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

from psfu.config import APP_FULL_NAME, APP_TAGLINE
from psfu.storage.devices import discover_storage_devices, parse_storage_dut_paths
from psfu.ui.sparkline import (
    BAR_EMPTY,
    BAR_FULL,
    SPARK_BLOCKS,
    fmt_mmss,
    render_progress_bar,
    render_sparkline,
    resample as _resample,
)


class TuiDashboard:
    """Curses live dashboard for Platform Stress & Failure Utility (PS-FU).

    Color philosophy: cool cyan base that shifts warm (amber -> red) only
    when metric values cross domain-specific thresholds.  The "cyberpunk"
    feel comes from electric cyan accent + steel-blue structure on a dark
    terminal, not from decorative rainbow gradients.

    Every color pair encodes *meaning*, not decoration:
      TEAL/CYAN  = cool / nominal / healthy
      AMBER      = elevated / marginal / watch
      RED        = hot / critical / out-of-spec
      GREEN      = completed / pass
      PINK       = app identity accent (header only)
      STEEL      = structural lines and borders
      SLATE      = dim labels and secondary text
    """

    HISTORY_WINDOW = 60

    # ── Color pair IDs (8 intentional pairs) ──────────────────────────
    _C_GREEN = 1
    _C_AMBER = 2
    _C_RED = 3
    _C_CYAN = 4
    _C_SLATE = 5
    _C_STEEL = 6
    _C_TEAL = 7
    _C_PINK = 8

    # ── Domain thresholds: (warm, hot) ────────────────────────────────
    # Below warm = cool.  warm..hot = amber.  Above hot = red.
    _DOMAIN: Dict[str, Tuple[float, float]] = {
        # Temperatures (\u00b0C)
        "cpu_temp_c":      (75.0, 85.0),
        "gpu_temp_c":      (75.0, 85.0),
        "gpu_vram_temp_c": (90.0, 100.0),
        "ssd_temp_c":      (55.0, 70.0),
        "mem_temp_c":      (55.0, 70.0),
        # Utilisation (%)
        "cpu_load_pct":    (85.0, 95.0),
        "gpu_util_pct":    (85.0, 95.0),
        "mem_used_pct":    (80.0, 95.0),
        "disk_busy_pct":   (75.0, 90.0),
        # Power (W) — CPU: amber at 300W, red at 345W (Threadripper TDP ~350W).
        # GPU: amber at 450W, red at 550W (5090 TDP ~575W).
        "cpu_pkg_watt":    (300.0, 345.0),
        "gpu_power_w":     (450.0, 550.0),
        # CPU MHz — thresholds set high so normal state stays cool (teal).
        # The health check handles the actual throttle alert.
        "cpu_avg_mhz":     (9000.0, 9999.0),
        "cpu_max_mhz":     (9000.0, 9999.0),
        # GPU clocks (MHz) — thresholds invert: low = bad (amber/red when throttled)
        # These are set high so the normal state is always "cool" (teal/cyan).
        # The clock-throttle health check handles the actual alerting.
        "gpu_clock_sm_mhz":  (9000.0, 9999.0),
        # VRAM (MiB) — high usage is expected during stress; not a warning
        "gpu_vram_used_mib": (20000.0, 24000.0),
        "mem_power_w":     (20.0, 40.0),
        "ssd_power_w":     (5.0, 10.0),
        # CPU turbostat power — same thresholds as RAPL
        "cpu_pkg_watt_turbo": (300.0, 345.0),
        # CPU busy% — high is expected during stress
        "cpu_busy_pct_turbo": (85.0, 95.0),
        # GPU fan — ramp is normal, 100% = thermal limit
        "gpu_fan_pct":     (80.0, 95.0),
        # PCIe gen — set unreachably high; link downgrade health check handles alerts
        "pcie_link_gen":   (9000.0, 9999.0),
        # ECC errors — any CE is abnormal, any UE is FAIL-level
        "gpu_ecc_ce":      (1.0, 5.0),
        "gpu_ecc_ue":      (0.5, 1.0),   # 0.5 unreachable (int); red on first UE
        # Throttle active — binary 0/1; amber the moment it fires
        "gpu_throttle_active": (0.5, 0.99),
        # ── PSU (corsair-psu hwmon) ─────────────────────────────────
        # Power and per-rail current thresholds are rescaled at dashboard
        # init from the rated wattage (lobby field). These defaults target
        # the HX1500i on the DUT bench (1500 W rated, 125 A max on +12 V).
        # _apply_psu_thresholds() overwrites with scaled values once the
        # wattage is known — a different-sized PSU (HX850i, AX1600i, etc.)
        # will get proportionally adjusted thresholds.
        "psu_p_out_w":       (1200.0, 1425.0),   # 80%, 95% of 1500 W
        "psu_p_in_w":        (1320.0, 1575.0),   # ~88%, ~105% (input > output)
        "psu_i_12v_a":       (100.0,  118.0),    # 80%, ~95% of 125 A
        "psu_i_5v_a":        (18.0,   22.0),     # HX1500i 5 V rail ~25 A max
        "psu_i_33v_a":       (18.0,   22.0),     # HX1500i 3.3 V rail ~25 A max
        # PSU temperatures (internal VRM vs ambient). Corsair HXi rated
        # for 50 °C ambient operation; internal VRM runs ~15-25 °C hotter.
        "psu_temp_c":        (60.0, 75.0),   # primary (VRM / hot spot)
        "psu_temp2_c":       (45.0, 55.0),   # secondary (ambient-side)
        # PSU fan coloring is adaptive — see _DOMAIN_ADAPTIVE below.
        # ── Platform thermal + link health (v5 schema) ──────────────
        # VRM: Threadripper motherboard VRMs rated ~90 C junction; above
        # that throttles. Values below assume a board-level (not die-level)
        # VRM sensor — those read cooler than junction so the band shifts.
        "mobo_vrm_temp_c":     (70.0, 85.0),
        # Case/board ambient — chassis airflow canary. >40 C indicates
        # poor intake / failing fan on most open-bench setups.
        "mobo_case_temp_c":    (40.0, 50.0),
        # Per-CCD max — mirrors cpu_temp_c band. Spread stays cool as long
        # as cooling is even; >15 C spread is a real red flag.
        "cpu_ccd_max_c":       (75.0, 85.0),
        "cpu_ccd_spread_c":    (10.0, 20.0),
        # NVMe controller + NAND — same band as composite.
        "ssd_temp_ctrl_c":     (55.0, 70.0),
        "ssd_temp_nand_c":     (55.0, 70.0),
        # AER correctable total — absolute count is context-dependent
        # (diff vs phase-start is the real signal). Floor both thresholds
        # high so baseline doesn't false-color; a verdict rule will handle
        # growth-based alerting separately.
        "pcie_aer_corr_total": (1000.0, 10000.0),
        # ── v6 schema additions ─────────────────────────────────────
        # Integrated NIC thermals. MAC (controller) and PHY (transceiver)
        # run cooler than CPU — 70 C is getting warm, 85 C is the
        # spec-sheet ceiling for most Gbit/10G silicon.
        "nic_mac_temp_c":          (70.0, 85.0),
        "nic_phy_temp_c":          (70.0, 85.0),
        # PSI — % of last 10s that at least one task stalled. Baseline
        # is near zero on a healthy idle bench. 10 % means the system
        # is intermittently stuck; 30 % means something is wrong.
        "psi_cpu_some_avg10":      (10.0, 30.0),
        "psi_mem_some_avg10":      (10.0, 30.0),
        "psi_io_some_avg10":       (10.0, 30.0),
        # IOwait — same philosophy as PSI IO: normally near 0, >20 %
        # sustained is a stuck-on-storage signal.
        "cpu_iowait_pct":          (20.0, 50.0),
        # NIC error counters — absolute totals, tolerate baseline
        # contamination from counters that never reset. Consumer
        # diffs vs phase start for the real signal.
        "nic_rx_crc_errors_total": (10.0, 100.0),
        "nic_rx_errors_total":     (100.0, 1000.0),
        "nic_rx_dropped_total":    (1000.0, 10000.0),
        # ── v7 schema additions ─────────────────────────────────────
        # Rail ripple σ (mV) over rolling 10 s window. Healthy ATX PSUs
        # under load sit at ~10-25 mV on +12 V; >50 mV is capacitor-aging
        # territory, >100 mV is failing. Tighter band on 5V and 3.3V —
        # they carry less current so the same absolute ripple is a bigger
        # fraction of the rail.
        "rail_12v_ripple_mv":  (50.0, 100.0),
        "rail_5v_ripple_mv":   (25.0, 50.0),
        "rail_33v_ripple_mv":  (20.0, 40.0),
        # Rail dropouts — any sub-second brownout below lcrit is abnormal.
        # warm at 1 (one dip = cable handshake / transient), crit at 5
        # (sustained OCP/PSU misbehavior).
        "rail_12v_dropouts":   (1.0, 5.0),
        "rail_5v_dropouts":    (1.0, 5.0),
        "rail_33v_dropouts":   (1.0, 5.0),
        # PCIe AER uncorrectable — fatal-class events. Any count > 0 is
        # a hardware fault signal. warm=1 / crit=1 → goes red immediately
        # on the first event.
        "pcie_aer_uncorr_total": (0.5, 0.9),
        # GPU throttle split. HW_SLOWDOWN / HW_THERMAL / HW_POWER_BRAKE
        # are real hardware protection events — amber the moment they
        # fire. SW_POWER_CAP is the normal "at TDP" state under gpu-burn
        # so its thresholds are set above binary (never triggers a
        # color flip) — it's kept visible as data, not an alert.
        "gpu_throttle_hw_slowdown":    (0.5, 0.99),
        "gpu_throttle_hw_thermal":     (0.5, 0.99),
        "gpu_throttle_hw_power_brake": (0.5, 0.99),
        "gpu_throttle_sw_power_cap":   (9000.0, 9999.0),
        # NVMe SMART temp-time counters (cumulative seconds the drive
        # spent above warning / critical temps). Warning should never
        # exceed low-single-digit seconds on a healthy drive; critical
        # must stay at 0.
        "ssd_warn_temp_time_s":  (60.0, 600.0),
        "ssd_crit_temp_time_s":  (1.0, 10.0),
        # NIC link state — link_up is binary; 0 = link dropped = red.
        # link_speed_mbps intentionally omits a _DOMAIN threshold; a
        # _DOMAIN_FLOOR entry below handles downgrade detection once
        # the advertised max is known.
        "nic_link_up":          (0.5, 0.01),  # both below 1 → amber/red on 0
        # CPU per-core MHz spread — well-balanced Ryzen/Threadripper
        # under steady stress sits around 50-100 MHz σ; >200 means
        # cores are landing at wildly different clocks (throttling or
        # scheduling imbalance).
        "cpu_freq_spread_mhz":  (200.0, 400.0),
        # CPU C6% — healthy idle sits high (>80%); under stress should
        # drop near 0. High C6 mid-stress means cores are parked when
        # they shouldn't be. Thresholds reversed handled by the stress
        # phase itself — keep these permissive so idle doesn't false-flag.
        "cpu_c6_pct":           (9000.0, 9999.0),
        # Context switches / interrupts / softirqs per second —
        # baseline varies massively by workload, so thresholds are set
        # far above typical stress runs to avoid false-coloring.
        # The data is what matters; coloring is a nicety.
        "cpu_ctx_switches_per_s": (100000.0, 500000.0),
        "cpu_interrupts_per_s":   (100000.0, 500000.0),
        "cpu_softirq_per_s":      (100000.0, 500000.0),
        # Major page faults per second — healthy systems under stress
        # see near-zero majfaults. A burst means the system is swapping
        # or hitting memory pressure.
        "mem_pgmajfault_per_s":   (10.0, 100.0),
        # Dirty / writeback KB — transient buffering is normal; growth
        # to hundreds of MB means I/O is stalling.
        "mem_dirty_kb":           (500000.0, 2000000.0),    # 500 MB, 2 GB
        "mem_writeback_kb":       (100000.0, 500000.0),     # 100 MB, 500 MB
        # SSD throughput — context-dependent (reading from an NVMe
        # during PS-FU storage phase vs. idle). Thresholds set high to
        # avoid false-coloring under normal stress traffic.
        "ssd_read_mbps":          (7000.0, 9000.0),
        "ssd_write_mbps":         (5000.0, 7000.0),
        "ssd_mixed_mbps":         (9000.0, 12000.0),
        # fio tail latency rendered in microseconds for operator readability.
        "fio_read_lat_p99_us":        (2000.0, 10000.0),
        "fio_write_lat_p99_us":       (5000.0, 20000.0),
        "fio_read_lat_p99_999_us":    (10000.0, 50000.0),
        "fio_write_lat_p99_999_us":   (20000.0, 100000.0),
        # SSD queue depth — 1-2 in-flight is normal; 32+ is sustained
        # backpressure.
        "ssd_queue_depth":        (16.0, 32.0),
        # Chipset temp — TRX50 chipset runs 50-70 C under load; 85+ is
        # the silicon's spec-sheet ceiling.
        "mobo_chipset_temp_c":    (75.0, 85.0),
        # Thermal zone margin — kernel thermal framework reports
        # "distance to trip" per zone. We track the minimum across
        # zones (worst case). Margin < 10 C = warm, < 5 C = about to
        # trip. Floor semantics handled by _DOMAIN_FLOOR below.
        # Memory bandwidth (MB/s) — purely informational. Thresholds
        # set unreachably high so sparklines render cool/teal under
        # normal operation.
        "mem_bw_read_mbps":       (900000.0, 999999.0),
        "mem_bw_write_mbps":      (900000.0, 999999.0),
    }

    # ── Adaptive thresholds: derived from observed max ────────────────
    # Absolute RPM thresholds don't work across PSU models. Consumer HXi
    # units run zero-RPM mode and may peak ~1800 RPM; server-class PSUs
    # idle near 3000 and peak 8000+. A hardcoded 1500/2000 either never
    # flags on a server PSU or constantly flags on a high-wattage unit
    # that idles loud. Adaptive coloring takes warm/hot as a ratio of
    # the highest observation so far, gated on minimum observed value
    # and tick count so a fan sitting at idle doesn't false-flag before
    # we have a real peak to calibrate against.
    #
    # Format: key -> (warm_ratio, hot_ratio, min_observed, min_ticks)
    _DOMAIN_ADAPTIVE: Dict[str, Tuple[float, float, float, int]] = {
        # Fan: warm at 75 % of observed peak, red at 90 %. Only active
        # once we've seen 600+ RPM (rules out zero-RPM idle) and 60+
        # ticks (≈1 min @ 1 Hz — enough to see a ramp). Below those
        # gates, falls through to the sparkline-ceiling fallback in
        # _metric_level which stays quiet at low RPM.
        "psu_fan_rpm": (0.75, 0.90, 600.0, 60),
    }

    # ── Floor-based thresholds (LOWER = worse) ────────────────────────
    # Mirror of _DOMAIN for metrics where the danger direction is
    # inverted. ``(warm_floor, hot_floor)`` means: value <= warm_floor →
    # amber; value <= hot_floor → red. ``warm_floor > hot_floor``.
    _DOMAIN_FLOOR: Dict[str, Tuple[float, float]] = {
        # PSU efficiency — HX1500i is 80+ Platinum (~92% at 50% load).
        # Sub-85% is elevated; sub-80% means the PSU is losing serious
        # energy as heat — a failing unit trips here before the rails drop.
        "psu_efficiency_pct": (85.0, 80.0),
        # ── v7 ──────────────────────────────────────────────────────
        # Thermal zone margin — distance (°C) to nearest trip point
        # across all kernel thermal zones. <10 C = warm, <5 C = red.
        # Zero or negative means a zone is past its trip and something
        # should already be throttling.
        "thermal_zone_margin_c": (10.0, 5.0),
    }

    # ── Sparkline absolute ranges: (floor, ceiling) ──────────────────
    # Seed values used before auto-calibration kicks in. Once real data
    # arrives, _auto_calibrate() overwrites these on the instance with
    # ranges derived from actual hardware (110% of max observed value).
    # Percentage and rail metrics keep fixed ranges — those have natural
    # bounds that don't depend on hardware.
    _SPARK_RANGE: Dict[str, Tuple[float, float]] = {
        # Fixed: percentage metrics (0-100 is always the range)
        "cpu_load_pct":       (0.0, 100.0),
        "cpu_busy_pct_turbo": (0.0, 100.0),
        "gpu_util_pct":       (0.0, 100.0),
        "gpu_fan_pct":        (0.0, 100.0),
        "mem_used_pct":       (0.0, 100.0),
        "disk_busy_pct":      (0.0, 100.0),
        # Fixed: rails (tight range around nominal)
        "rail_12v":           (11.0, 13.0),
        "rail_5v":            (4.5, 5.5),
        "rail_33v":           (3.0, 3.6),
        # Fixed: PCIe gen (1-5)
        "pcie_link_gen":      (1.0, 5.0),
        # NVMe PCIe link state — same semantics as GPU PCIe, but on the
        # storage path. Width seeds at x4 because that is the common NVMe
        # endpoint width; a wider value still renders fine if observed.
        "nvme_pcie_link_gen":   (1.0, 5.0),
        "nvme_pcie_link_width": (1.0, 4.0),
        # ECC counters — normally 0; fixed range so the sparkline isn't flat-line noise
        "gpu_ecc_ce":         (0.0, 10.0),
        "gpu_ecc_ue":         (0.0, 5.0),
        # Throttle active — binary 0/1
        "gpu_throttle_active": (0.0, 1.0),
        # PSU voltages — tight bands like DC rails. Vin auto-detects
        # US (120 V) vs EU (230 V) bench from the first observed reading
        # in _auto_calibrate().
        "psu_vin_v":          (100.0, 130.0),
        # PSU efficiency has a natural floor; 70-100 gives good resolution
        # in the healthy range without wasting bar height on impossible
        # sub-50% readings.
        "psu_efficiency_pct": (70.0, 100.0),
        # Per-rail currents, power — seeded from the 1500 W default;
        # _apply_psu_thresholds() rescales once rated wattage is known.
        "psu_p_out_w":        (0.0, 1500.0),
        "psu_p_in_w":         (0.0, 1650.0),
        "psu_i_12v_a":        (0.0, 125.0),
        "psu_i_5v_a":         (0.0, 25.0),
        "psu_i_33v_a":        (0.0, 25.0),
        # Per-rail output power — most of the lobby wattage lives on 12V,
        # so scale that to the full rated band; 5V/3.3V rarely exceed 60-80 W.
        "psu_p_12v_w":        (0.0, 1500.0),
        "psu_p_5v_w":         (0.0, 100.0),
        "psu_p_33v_w":        (0.0, 100.0),
        # Static RAM hardware context — fixed ranges cover common kits
        # (8 GB mini up to 1 TB workstation; DDR4-3200 up to DDR5-8000).
        # Sparklines will be flat-line since these don't change mid-run.
        "mem_total_gb":       (0.0, 1024.0),
        "mem_speed_mts":      (0.0, 8400.0),
        # CCD spread — 0..25 C covers well-cooled to obvious imbalance.
        "cpu_ccd_spread_c":   (0.0, 25.0),
        # AER correctable total — log-ish spread; 0..10k absolute floor.
        # Auto-ratchet via _auto_calibrate once real growth is observed.
        "pcie_aer_corr_total": (0.0, 10000.0),
        # PSI some avg10 — % of last 10s with at least one stalled task.
        # Healthy idle/steady sits near 0; >30% on any resource is a stall signal.
        "psi_cpu_some_avg10": (0.0, 50.0),
        "psi_mem_some_avg10": (0.0, 50.0),
        "psi_io_some_avg10":  (0.0, 50.0),
        # CPU iowait % — healthy is <5%; >20% means CPU is stalling on I/O.
        "cpu_iowait_pct":     (0.0, 100.0),
        # NIC link-error totals — normally 0. Small fixed ceiling keeps the
        # sparkline informative when counters do start ticking; auto-ratchet
        # via _auto_calibrate if real growth arrives.
        "nic_rx_crc_errors_total": (0.0, 100.0),
        "nic_rx_errors_total":     (0.0, 100.0),
        "nic_rx_dropped_total":    (0.0, 1000.0),
        # GPU PCIe max link gen — static per GPU; fixed 1..5 range.
        "gpu_pcie_link_gen_max":   (1.0, 5.0),
        # ── v7 schema additions ─────────────────────────────────────
        # Rail ripple (mV) — normally 10-25 mV on a healthy PSU.
        # 0..200 gives headroom above the 100 mV crit threshold so the
        # bar stays readable as units age.
        "rail_12v_ripple_mv":  (0.0, 200.0),
        "rail_5v_ripple_mv":   (0.0, 100.0),
        "rail_33v_ripple_mv":  (0.0, 100.0),
        # Rail dropout counters — normally 0. Fixed 0..10 floor;
        # auto-ratchet via _auto_calibrate if something really goes
        # wrong and the counter blows past it.
        "rail_12v_dropouts":   (0.0, 10.0),
        "rail_5v_dropouts":    (0.0, 10.0),
        "rail_33v_dropouts":   (0.0, 10.0),
        # PCIe AER uncorrectable — normally 0; any growth is serious.
        "pcie_aer_uncorr_total": (0.0, 10.0),
        # GPU throttle split — binary 0/1 flags.
        "gpu_throttle_hw_slowdown":    (0.0, 1.0),
        "gpu_throttle_hw_thermal":     (0.0, 1.0),
        "gpu_throttle_hw_power_brake": (0.0, 1.0),
        "gpu_throttle_sw_power_cap":   (0.0, 1.0),
        # NVMe SMART temp-time counters (cumulative seconds).
        # Floor 0; warn threshold at 60 s so 0..600 is a useful band.
        "ssd_warn_temp_time_s":  (0.0, 600.0),
        "ssd_crit_temp_time_s":  (0.0, 60.0),
        # NIC link state — binary up/down. Speed auto-calibrates from
        # observed max so 10G, 2.5G, and 1G benches all render right.
        "nic_link_up":          (0.0, 1.0),
        # CPU C6% — percent of time in deep C-state.
        "cpu_c6_pct":           (0.0, 100.0),
        # Thermal zone margin — widen bounds beyond current
        # temperatures so the sparkline shows "getting hotter =
        # smaller margin" direction.
        "thermal_zone_margin_c": (0.0, 50.0),
        # SSD queue depth — typical sustained load sits 1-4; 32 is a
        # good ceiling that shows backpressure clearly.
        "ssd_queue_depth":       (0.0, 32.0),
        "ssd_mixed_mbps":        (0.0, 12000.0),
        "fio_read_lat_p99_us":      (0.0, 50000.0),
        "fio_write_lat_p99_us":     (0.0, 50000.0),
        "fio_read_lat_p99_999_us":  (0.0, 100000.0),
        "fio_write_lat_p99_999_us": (0.0, 100000.0),
        # Temps + fan auto-calibrate from observed peaks (see seeds below).
    }

    # Metrics whose sparkline ceilings auto-calibrate from observed data.
    # Maps metric key → initial seed ceiling (used until first real value).
    _AUTO_CALIBRATE_SEEDS: Dict[str, float] = {
        "cpu_temp_c":         105.0,
        "cpu_pkg_watt_turbo": 400.0,
        "cpu_avg_mhz":        6000.0,
        "cpu_max_mhz":        6000.0,
        "gpu_temp_c":         95.0,
        "gpu_vram_temp_c":    110.0,
        "gpu_power_w":        600.0,
        "gpu_clock_sm_mhz":   3000.0,
        "gpu_vram_used_mib":  24576.0,
        "ssd_temp_c":         80.0,
        "mem_temp_c":         80.0,
        "mem_power_w":        50.0,
        "ssd_power_w":        15.0,
        "cpu_pkg_watt":       400.0,
        # PCIe throughput — seed at 10 GB/s (10 000 000 KB/s) for PCIe 5 x16;
        # auto-calibrates quickly from real traffic
        "pcie_tx_kbps":       10_000_000.0,
        "pcie_rx_kbps":       10_000_000.0,
        # Load average — seed at Threadripper thread count (96 threads worst case)
        "loadavg_1m":         96.0,
        # PSU temps + fan — seeded here so _auto_calibrate picks them up
        # and ratchets the sparkline ceiling from real peaks.
        "psu_temp_c":         80.0,
        "psu_temp2_c":        60.0,
        "psu_fan_rpm":        2500.0,
        # Platform thermal extensions — pick seeds above expected idle
        # so the sparkline renders cleanly out of the gate.
        "mobo_vrm_temp_c":    90.0,
        "mobo_case_temp_c":   55.0,
        "cpu_ccd_max_c":      95.0,
        "ssd_temp_ctrl_c":    80.0,
        "ssd_temp_nand_c":    80.0,
        # Integrated NIC MAC/PHY sensors run ~40-60 C idle on TRX50;
        # seed 90 C so the sparkline has headroom before auto-ratchet.
        "nic_mac_temp_c":     90.0,
        "nic_phy_temp_c":     90.0,
        # ── v7 auto-calibrated ──────────────────────────────────────
        # CPU fabric / interrupt rates — seed at 200k/s for typical
        # stress runs; ratchets up if a workload pushes harder.
        "cpu_ctx_switches_per_s": 200000.0,
        "cpu_interrupts_per_s":   200000.0,
        "cpu_softirq_per_s":      200000.0,
        # CPU per-core MHz spread — well-balanced parts sit ~100 MHz,
        # unbalanced/throttled can hit 500+ MHz σ.
        "cpu_freq_spread_mhz":    500.0,
        # Memory major page faults/s — near zero healthy; seed 100/s.
        "mem_pgmajfault_per_s":   100.0,
        # Dirty / writeback — seed at 100 MB and 10 MB respectively.
        "mem_dirty_kb":           1000000.0,
        "mem_writeback_kb":        500000.0,
        # SSD throughput — seed at PCIe 4 x4 realistic peak (~7 GB/s).
        "ssd_read_mbps":          7000.0,
        "ssd_write_mbps":         5000.0,
        "ssd_mixed_mbps":         12000.0,
        "fio_read_lat_p99_us":      50000.0,
        "fio_write_lat_p99_us":     50000.0,
        "fio_read_lat_p99_999_us":  100000.0,
        "fio_write_lat_p99_999_us": 100000.0,
        # NVMe SMART cumulative counters — usually flat at zero, but when
        # they move the sparkline should widen enough to show the step.
        "nvme_media_errors":          10.0,
        "nvme_num_err_log_entries":   100.0,
        "nvme_unsafe_shutdowns":      10.0,
        # Chipset temp — TRX50 chipset runs 50-70 C idle.
        "mobo_chipset_temp_c":     90.0,
        # NIC link speed (Mbps) — seeds at 10 GbE; auto-calibrates
        # to whatever port is up (1G, 2.5G, 10G, etc.).
        "nic_link_speed_mbps":    10000.0,
        # Memory bandwidth (MB/s) — DDR5 dual-channel peaks around
        # ~80 GB/s (80 000 MB/s); TRX50 8-channel DDR5-5600 pushes
        # 250+ GB/s. Seed generously.
        "mem_bw_read_mbps":       400000.0,
        "mem_bw_write_mbps":      400000.0,
    }

    # Rails are range-based (deviation from nominal), handled separately.
    # psu_vin_v nominal auto-selects between 120 V (US) and 230 V (EU) in
    # _auto_calibrate based on the first observed reading — the HX1500i is
    # universal-input so the same bench tool needs to work in either region.
    _RAIL_SPEC: Dict[str, Dict[str, float]] = {
        "rail_12v": {"nominal": 12.0, "warn": 0.3, "crit": 0.6},
        "rail_5v":  {"nominal": 5.0,  "warn": 0.15, "crit": 0.25},
        "rail_33v": {"nominal": 3.3,  "warn": 0.1,  "crit": 0.2},
        "psu_vin_v": {"nominal": 120.0, "warn": 10.0, "crit": 15.0},
    }

    # Display-only transform applied in ``_fmt_val`` right before format().
    # The DB value stays canonical; only the rendered text is scaled.
    # Thresholds in ``_val_attr`` use the pre-transform value.
    # Currently empty — we show MT/s as-is for RAM speed so the value on
    # screen matches what's printed on the DIMM box / dmidecode / vendor
    # spec sheets (e.g. DDR5-6400 shows "6400"). A tech doesn't need to
    # mentally double-convert clock → data rate during an RMA.
    _DISPLAY_TRANSFORM: Dict[str, Callable[[float], float]] = {}

    # ── Lobby editable fields ─────────────────────────────────────────
    # (key, display_label, type, required, cycle_options)
    # type "text"  → inline text editor  (Enter to edit, Enter/Tab to confirm)
    # type "cycle" → cycles options on Enter; "profile" uses self._profile_names
    _LOBBY_FIELDS_DEF: List[Tuple] = [
        ("profile",     "Profile ", "cycle", True,  []),   # options = self._profile_names
        ("dur_scale",   "Scale   ", "text",  False, []),   # duration multiplier, default 1.0
        ("psu_serial",  "Serial  ", "text",  False, []),
        ("storage_duts","StorDUT ", "text",  False, []),
        ("psu_wattage", "Wattage ", "text",  False, []),
        ("psu_make",    "Make    ", "text",  False, []),
        ("psu_model",   "Model   ", "text",  False, []),
        ("operator",    "Operator", "text",  False, []),
        ("meter",       "Meter   ", "cycle", False, ["yes", "no", "unknown"]),
        ("paperclip",   "Papercl.", "cycle", False, ["not-run", "pass", "fail"]),
        ("post_result", "POST    ", "cycle", False, ["not-tested", "pass", "fail"]),
    ]

    # Panel order is also grid-row order within each column. DUT_GRID_COLUMNS
    # below assigns each key to a column, so the visual layout is:
    #
    #   ┌ CPU ┐  ┌ PSU ┐
    #   └─────┘  │     │
    #   ┌ GPU ┐  └─────┘
    #   │     │  ┌ RAM ┐
    #   │     │  └─────┘
    #   │     │  ┌ SSD ┐
    #   └─────┘  └─────┘
    #
    # Metrics inside each panel are priority-ordered (top rows survive
    # compact-mode clipping) following the "temp → power → load → clocks"
    # pattern established for CPU: heat is the first thing you want to see,
    # then how hard the component is being driven, then fine-grained stats.
    DUT_PANELS = [
        {
            "key": "cpu", "title": "CPU", "worker_label": "CPU",
            "primary": "cpu_temp_c",
            # CPU-die-only metrics. Motherboard Super I/O sensors, NIC
            # thermals, NIC link counters, and kernel PSI live on the
            # MOBO panel — keeping this panel focused on CPU silicon
            # means temp/power/clocks don't get clipped at short heights.
            # CCDmax + CCD\u0394 come from Zen 4/5 per-die sensors (Tccd*).
            # Storm Peak (TR 7000) on current stable kernels: k10temp does
            # NOT expose per-CCD labels, so CCDmax + CCD\u0394 both stay
            # dashed. cpu_temp_c (Tctl) still populates the Temp row;
            # echoing Tctl as CCDmax would just duplicate the Temp row.
            "metrics": [
                ("Temp",    "cpu_temp_c",          "{:.0f}\u00b0"),
                ("CCDmax",  "cpu_ccd_max_c",       "{:.0f}\u00b0"),
                ("CCD\u0394", "cpu_ccd_spread_c",  "{:.1f}\u00b0"),
                ("Power",   "cpu_pkg_watt_turbo",  "{:.0f}W"),
                ("Load",    "cpu_load_pct",        "{:.0f}%"),
                ("IOwait",  "cpu_iowait_pct",      "{:.0f}%"),
                ("Avg",     "cpu_avg_mhz",         "{:.0f}M"),
                ("Max",     "cpu_max_mhz",         "{:.0f}M"),
                # Freq\u0394 = σ of per-core Avg_MHz from turbostat. Balanced
                # stress sits near 50-100 MHz; >200 means cores are
                # landing at different clocks (imbalance or throttle).
                ("Freq\u0394", "cpu_freq_spread_mhz", "{:.0f}M"),
                # C6 = % of time in deep C-state. Near-zero under stress
                # is expected; high C6 mid-stress = cores parked wrong.
                ("C6",      "cpu_c6_pct",          "{:.0f}%"),
                # Ctx/Intr/Sirq = /proc/stat deltas (ops/s). High rates
                # during stress show kernel activity; runaway means
                # a misbehaving driver or interrupt storm.
                ("Ctx",     "cpu_ctx_switches_per_s", "{:.0f}"),
                ("Intr",    "cpu_interrupts_per_s",   "{:.0f}"),
                ("Sirq",    "cpu_softirq_per_s",      "{:.0f}"),
                # Memory bandwidth (MB/s) — sampled every 30 s via
                # `perf stat` uncore IMC counters, held between samples.
                # Lives on CPU panel because DIMM bandwidth is a
                # CPU-fabric property, not a platform thermal one.
                ("MBWr",    "mem_bw_read_mbps",    "{:.0f}MB"),
                ("MBWw",    "mem_bw_write_mbps",   "{:.0f}MB"),
                ("LdAvg",   "loadavg_1m",          "{:.2f}"),
            ],
        },
        # MOBO panel — motherboard Super I/O thermals (VRM, case/system),
        # integrated NIC thermals (MAC, PHY), NIC link-layer error
        # counters (CRC / general / dropped), and kernel PSI (CPU / mem /
        # io). Everything here degrades to "\u2500\u2500" cleanly on hardware
        # that doesn't expose the source — pre-4.20 kernels without PSI,
        # boards without integrated NIC, Intel benches without the exact
        # Super I/O label names we key on, etc. Primary is mobo_vrm_temp_c
        # so the panel's sparkline tracks VRM thermals.
        #
        # Why split from CPU: the CPU panel was hitting 16 rows and the
        # priority-clipper drops bottom rows first on short terminals —
        # PSI + NIC thermals vanished exactly when they were needed. NIC
        # error counters and gpu_pcie_link_gen_max were CSV-only with no
        # live visibility, so a silent link-drop or CRC storm mid-run had
        # no indicator. A dedicated panel surfaces all platform-health
        # signals in one eye-scan.
        {
            "key": "mobo", "title": "MOBO", "worker_label": "PLT",
            "primary": "mobo_vrm_temp_c",
            "metrics": [
                ("VRM",     "mobo_vrm_temp_c",        "{:.0f}\u00b0"),
                ("Chipset", "mobo_chipset_temp_c",    "{:.0f}\u00b0"),
                ("Case",    "mobo_case_temp_c",       "{:.0f}\u00b0"),
                # Tmargin = smallest distance (°C) to any kernel thermal
                # trip across all zones. <10 C warm, <5 C red.
                ("Tmargin", "thermal_zone_margin_c",  "{:.0f}\u00b0"),
                ("NICmac",  "nic_mac_temp_c",         "{:.0f}\u00b0"),
                ("NICphy",  "nic_phy_temp_c",         "{:.0f}\u00b0"),
                # NIC link state — speed in Mbps + up/down flag.
                # A mid-run downgrade (10G→1G) or link drop lights up
                # here instantly instead of hiding in the CSV.
                ("NICspd",  "nic_link_speed_mbps",    "{:.0f}Mb"),
                ("NIClink", "nic_link_up",            "{:.0f}"),
                # NIC link-layer errors. Normally zero; any sustained
                # growth during a run points at cable / port / switch
                # issues that can pollute a verdict via kernel noise.
                ("NICerr",  "nic_rx_errors_total",       "{:.0f}"),
                ("NICcrc",  "nic_rx_crc_errors_total",   "{:.0f}"),
                ("NICdrop", "nic_rx_dropped_total",      "{:.0f}"),
                # PSI — % of last 10s that at least one task stalled
                # waiting on the resource. Zero on healthy benches;
                # non-zero during stress isolates where the bottleneck
                # actually is (CPU starvation vs memory vs storage).
                ("PSIcpu",  "psi_cpu_some_avg10",     "{:.1f}"),
                ("PSImem",  "psi_mem_some_avg10",     "{:.1f}"),
                ("PSIio",   "psi_io_some_avg10",      "{:.1f}"),
                # MajFlt = major page faults/s (from /proc/vmstat).
                # Healthy stress is near zero; bursts = swap / pressure.
                ("MajFlt",  "mem_pgmajfault_per_s",   "{:.0f}"),
                # Dirty / Writeback (KB) — transient buffering, watch
                # for sustained growth into hundreds of MB.
                ("Dirty",   "mem_dirty_kb",           "{:.0f}K"),
                ("WrBk",    "mem_writeback_kb",       "{:.0f}K"),
            ],
        },
        # PSU panel — populated by the corsair-psu hwmon driver when an
        # HXi/RMi/AXi PSU is connected via the iCue USB-header cable. Every
        # metric degrades to a dash when the driver isn't loaded or the PSU
        # doesn't report that particular field (e.g. RMi lacks 3.3V current).
        # Primary is p_out_w so the panel's sparkline tracks PSU load.
        {
            "key": "psu", "title": "PSU", "worker_label": "PSU",
            "primary": "psu_p_out_w",
            # Panel order: primary VRM temp on
            # top, then Pin/Pout, then each rail as a (V, I, W) trio so
            # per-rail load is visible in one eye-scan. Fan, then the
            # less-critical tail (Tamb / Eff / Vin) so clipping drops
            # those first. ``psu_temp_c`` is the primary sensor in the
            # corsair-psu driver — VRM / transformer / primary-side
            # switch FETs. ``psu_temp2_c`` is the secondary / case sensor.
            # Rail voltages read from rail_* columns because on TRX50
            # (Super I/O-blocked) that's already the Corsair reading.
            "metrics": [
                ("Temp",  "psu_temp_c",         "{:.0f}\u00b0"),
                ("Pin",   "psu_p_in_w",         "{:.0f}W"),
                ("Pout",  "psu_p_out_w",        "{:.0f}W"),
                ("12V",   "rail_12v",           "{:.2f}V"),
                ("I12",   "psu_i_12v_a",        "{:.1f}A"),
                ("P12",   "psu_p_12v_w",        "{:.0f}W"),
                # 12V ripple σ (mV) over rolling 10 s window. Cap
                # aging raises this before mean drifts — the single
                # highest-signal v7 metric for PSU RMA work.
                ("12Vrip", "rail_12v_ripple_mv", "{:.0f}mV"),
                # Sub-second brownout counter — any tick where the
                # rail dropped below lcrit. Increments on real OCP
                # trips, not mean-drift sag.
                ("12Vdrp", "rail_12v_dropouts",  "{:.0f}"),
                ("5V",    "rail_5v",            "{:.2f}V"),
                ("I5",    "psu_i_5v_a",         "{:.1f}A"),
                ("P5",    "psu_p_5v_w",         "{:.0f}W"),
                ("5Vrip", "rail_5v_ripple_mv",  "{:.0f}mV"),
                ("5Vdrp", "rail_5v_dropouts",   "{:.0f}"),
                ("3.3V",  "rail_33v",           "{:.2f}V"),
                ("I3",    "psu_i_33v_a",        "{:.1f}A"),
                ("P3.3",  "psu_p_33v_w",        "{:.0f}W"),
                ("3Vrip", "rail_33v_ripple_mv", "{:.0f}mV"),
                ("3Vdrp", "rail_33v_dropouts",  "{:.0f}"),
                # AERue = PCIe AER uncorrectable (fatal + nonfatal)
                # cumulative across all devices. Must stay at 0; any
                # growth is a hardware fault event — often PSU-induced
                # when correlated with 12V dropouts.
                ("AERue", "pcie_aer_uncorr_total", "{:.0f}"),
                ("Fan",   "psu_fan_rpm",        "{:.0f}"),
                ("Tamb",  "psu_temp2_c",        "{:.0f}\u00b0"),
                ("Eff",   "psu_efficiency_pct", "{:.0f}%"),
                ("Vin",   "psu_vin_v",          "{:.0f}V"),
            ],
        },
        # UPS panel (v8) — upstream mains telemetry via Network UPS Tools.
        # Sits in column 1 between PSU and RAM so mains→PSU flow reads
        # top-to-bottom in the power column. Populated when a NUT-
        # visible UPS is attached (Cyberpower CP1500PFCLCD via
        # usbhid-ups, APC Smart-UPS family via usbhid-ups or apcsmart).
        # Every metric degrades to "──" cleanly when no UPS is visible,
        # when NUT is not installed, or when the vendor omits that
        # specific variable (Cyberpower omits input.frequency AND live
        # ups.realpower — only emits ups.realpower.nominal which is the
        # rated plate, not live draw). Primary is ups_input_voltage_v so
        # the sparkline tracks mains voltage — a wall sag at the moment
        # a DUT shuts down is the single highest-value signal this
        # panel captures for cross-referencing intermittent
        # failures. Like MOBO, UPS is passive telemetry (no stress
        # worker attaches), so the worker tag renders as MON.
        {
            "key": "ups", "title": "UPS", "worker_label": "UPS",
            "primary": "ups_input_voltage_v",
            "metrics": [
                # Mains in — voltage + frequency. On Cyberpower
                # CP1500PFCLCD, InHz stays dashed (vendor omits
                # input.frequency); APC reports it.
                ("Vin",     "ups_input_voltage_v",     "{:.0f}V"),
                ("InHz",    "ups_input_freq_hz",       "{:.0f}Hz"),
                # Output voltage — diverges from Vin under AVR trim /
                # boost or inverter mode. A Vout at 120 V while Vin sags
                # to 95 V means the UPS is actively compensating.
                ("Vout",    "ups_output_voltage_v",    "{:.0f}V"),
                # Load = % of UPS VA rating. Bench runs typically land
                # at 10-30 % on a 1500 VA unit. Watch for runaway
                # growth during transient phases.
                ("Load",    "ups_load_pct",            "{:.0f}%"),
                # Pout = live real power out (W). APC emits;
                # Cyberpower does NOT — stays dashed on bench UPS.
                ("Pout",    "ups_realpower_w",         "{:.0f}W"),
                # PoutE = derived live watts, computed from the rated
                # plate (ups.realpower.nominal) × load%. This is what
                # PowerPanel for Windows shows on Cyberpower units that
                # omit ups.realpower — same math, populated on both
                # vendors so the CyberPower tick has a wattage number.
                # On APC, Pout is authoritative and PoutE is a cross-
                # check; on CyberPower, PoutE is the only wattage.
                ("PoutE",   "ups_realpower_est_w",     "{:.0f}W"),
                # VAout = derived live apparent power (VA), using the
                # nominal VA parsed from the model-name string
                # (CP1500PFCRM2U → 1500, Smart-UPS X 1500 → 1500).
                # Apparent-power is what the VA rating actually bounds —
                # for PSU work, VA headroom matters because low PF loads
                # eat VA budget faster than watts suggest.
                ("VAout",   "ups_apparent_est_va",     "{:.0f}VA"),
                # Battery state — charge % + runtime estimate (s).
                # Runtime is the UPS's own estimate based on current
                # load, not a ground-truth number. 1500 s = 25 min at
                # current load. Sudden collapse = battery going bad.
                ("Batt",    "ups_battery_charge_pct",  "{:.0f}%"),
                ("Runtime", "ups_battery_runtime_s",   "{:.0f}s"),
                # OnBat = derived 0/1. 1 = UPS transferred to battery
                # (mains lost or out of tolerance). Single highest-
                # signal row for cross-referencing a DUT shutdown
                # against actual line state.
                ("OnBat",   "ups_on_battery",          "{:.0f}"),
                # Xfers = cumulative OL→OB edge count. Monotonic,
                # starts at 0. First-ever poll on battery does NOT
                # increment (no prior OL state to transition from —
                # important for operators who plug PS-FU in during
                # an outage). Growth mid-run = real transfer event.
                ("Xfers",   "ups_transfer_count",      "{:.0f}"),
            ],
        },
        {
            "key": "gpu", "title": "GPU", "worker_label": "GPU",
            "primary": "gpu_temp_c",
            "metrics": [
                ("Temp",  "gpu_temp_c",          "{:.0f}\u00b0"),
                ("Power", "gpu_power_w",         "{:.0f}W"),
                ("Load",  "gpu_util_pct",        "{:.0f}%"),
                ("Clock", "gpu_clock_sm_mhz",    "{:.0f}M"),
                ("VRAM",  "gpu_vram_used_mib",   "{:.0f}Mi"),
                ("Fan",   "gpu_fan_pct",         "{:.0f}%"),
                # PCIe row shows current gen. Cards drop to Gen1 at
                # idle for power saving — that's normal. PCIemax is
                # the advertised cap from NVML (static per GPU, min
                # across GPUs on multi-GPU rigs). Comparing the two
                # live makes a mid-run link downgrade visible at a
                # glance; CSV still retains both for post-run analysis.
                ("PCIe",    "pcie_link_gen",          "Gen{:.0f}"),
                ("PCIemax", "gpu_pcie_link_gen_max",  "Gen{:.0f}"),
                # HWThr = HW-class throttle bits only (HW_SLOWDOWN |
                # HW_THERMAL | HW_POWER_BRAKE). SW_POWER_CAP is excluded
                # because it's the normal "at TDP" state under gpu_burn
                # — it was pinning the old Thrtl metric to 1 constantly.
                # HWThr=1 is a real event: thermal trip or PSU brownout.
                ("HWThr",   "gpu_throttle_active",         "{:.0f}"),
                # v7: split the NVML throttle-reason bitmask so the
                # specific cause is visible. HWslow/HWtherm/HWbrake
                # are the three real-fault bits (0x08 / 0x40 / 0x80);
                # HWbrake specifically fires on PSU OCP events so it's
                # the highest-signal throttle bit for PSU work.
                ("HWslow",  "gpu_throttle_hw_slowdown",    "{:.0f}"),
                ("HWtherm", "gpu_throttle_hw_thermal",     "{:.0f}"),
                ("HWbrake", "gpu_throttle_hw_power_brake", "{:.0f}"),
                # SW_POWER_CAP (0x04) — kept visible but never colors
                # since pegging TDP under gpu-burn is normal state.
                ("SWcap",   "gpu_throttle_sw_power_cap",   "{:.0f}"),
                ("TX",    "pcie_tx_kbps",        "{:.0f}K"),
                ("RX",    "pcie_rx_kbps",        "{:.0f}K"),
                # Trailing three are usually N/A on consumer (RTX 5090):
                # GDDR7 memory temp + ECC counters are datacenter-GPU
                # features (H100/A100). Kept here so they light up on
                # workstations with pro GPUs; compact mode drops them
                # first on short windows.
                ("VRAM\u00b0", "gpu_vram_temp_c", "{:.0f}\u00b0"),
                ("ECC-C", "gpu_ecc_ce", "{:.0f}"),
                ("ECC-U", "gpu_ecc_ue", "{:.0f}"),
            ],
        },
        {
            "key": "mem", "title": "RAM", "worker_label": "MEM",
            "primary": "mem_used_pct",
            # Total + Speed are static hardware context (probed once at
            # collector init, mirrored into every sample). Speed is None
            # when dmidecode isn't available or lacks root — degrades to
            # "──" in that case. mem_temp_c / mem_power_w stay unpopulated
            # on consumer boards without on-DIMM thermal diodes exposed.
            "metrics": [
                ("Temp",  "mem_temp_c",    "{:.0f}\u00b0"),
                ("Load",  "mem_used_pct",  "{:.0f}%"),
                ("Total", "mem_total_gb",  "{:.0f}G"),
                # Shown in MT/s to match DIMM box labels and vendor spec
                # sheets (DDR5-6400 kit → "6400"). Simpler for techs than
                # converting to actual memory clock (3200 MHz).
                ("Speed", "mem_speed_mts", "{:.0f}MT/s"),
                ("Power", "mem_power_w",   "{:.0f}W"),
            ],
        },
        {
            "key": "storage", "title": "SSD", "worker_label": "SSD",
            "primary": "disk_busy_pct",
            # Temp  = composite (what the drive reports as the worst-case
            #         weighted average across its sensors).
            # Ctrl  = controller die (NVMe Temperature Sensor 1).
            # NAND  = NAND package (NVMe Temperature Sensor 2).
            # A Ctrl-NAND delta that widens under PSU stress points at a
            # 3.3V/5V sag — the controller throttles to stay inside power,
            # and the delta collapses or inverts.
            # AER   = running sum of correctable AER counters across every
            # PCI device. Absolute value is context-dependent; a verdict
            # rule will alert on growth during stress phases.
            "metrics": [
                ("Temp",  "ssd_temp_c",          "{:.0f}\u00b0"),
                ("Ctrl",  "ssd_temp_ctrl_c",     "{:.0f}\u00b0"),
                ("NAND",  "ssd_temp_nand_c",     "{:.0f}\u00b0"),
                ("Power", "ssd_power_w",         "{:.0f}W"),
                ("Load",  "disk_busy_pct",       "{:.0f}%"),
                # Throughput + queue depth from /sys/block/nvme*/stat
                # deltas. RdMB/WrMB surface the real bandwidth the
                # drive is pushing; QDep shows backpressure from a
                # stuck controller or saturated PCIe lane.
                ("RdMB",  "ssd_read_mbps",       "{:.0f}MB"),
                ("WrMB",  "ssd_write_mbps",      "{:.0f}MB"),
                ("MixMB", "ssd_mixed_mbps",      "{:.0f}MB"),
                ("R99",   "fio_read_lat_p99_us", "{:.0f}u"),
                ("W99",   "fio_write_lat_p99_us", "{:.0f}u"),
                ("R5N",   "fio_read_lat_p99_999_us", "{:.0f}u"),
                ("W5N",   "fio_write_lat_p99_999_us", "{:.0f}u"),
                ("QDep",  "ssd_queue_depth",     "{:.0f}"),
                ("NvmT",  "nvme_composite_temp_c", "{:.0f}\u00b0"),
                ("Used",  "nvme_percentage_used", "{:.0f}%"),
                ("Spare", "nvme_available_spare", "{:.0f}%"),
                ("MErr",  "nvme_media_errors",   "{:.0f}"),
                ("ErrLg", "nvme_num_err_log_entries", "{:.0f}"),
                ("Unsf",  "nvme_unsafe_shutdowns", "{:.0f}"),
                # Live PCIe link from the per-device lspci watcher. Kept
                # adjacent to the SMART summary so the operator can see
                # thermal / endurance / bus state together at a glance.
                ("NGen",  "nvme_pcie_link_gen", "{:.0f}"),
                ("NWid",  "nvme_pcie_link_width", "x{:.0f}"),
                ("Busy",  "nvme_controller_busy_time_min", "{:.0f}m"),
                # NVMe SMART cumulative time-above-threshold counters.
                # WarnT > 0 = drive spent time above its warning temp;
                # CritT > 0 = drive hit the thermal-shutdown ceiling.
                # Any growth during a PS-FU run correlates directly
                # with thermal throttling events.
                ("WarnT", "ssd_warn_temp_time_s", "{:.0f}s"),
                ("CritT", "ssd_crit_temp_time_s", "{:.0f}s"),
                ("NVMeW", "nvme_warning_temp_time_min", "{:.0f}m"),
                ("NVMeC", "nvme_critical_temp_time_min", "{:.0f}m"),
                ("AER",   "pcie_aer_corr_total", "{:.0f}"),
            ],
        },
    ]

    # Column assignment for the 2-col grid layout. Each column renders its
    # panels stacked top-to-bottom in the order listed. Keeping this
    # declarative (rather than deriving from DUT_PANELS index parity) lets
    # the left and right columns hold unequal numbers of panels — the old
    # pair-based layout couldn't place SSD under RAM without leaving SSD
    # stranded on the left alone.
    DUT_GRID_COLUMNS: List[List[str]] = [
        ["cpu", "mobo", "gpu"],
        ["psu", "ups", "mem", "storage"],
    ]

    # ── Constructor ───────────────────────────────────────────────────

    def __init__(
        self,
        stdscr: "curses._CursesWindow",
        profile: object,
        psu_serial: str,
        app_name: str,
        dut_enabled: Optional[Dict[str, bool]] = None,
        initial_values: Optional[Dict[str, str]] = None,
    ) -> None:
        self.stdscr = stdscr
        self.profile = profile
        self.psu_serial = psu_serial or "?"
        self.app_name = app_name
        self.run_total_seconds = max(1, sum(p.duration_sec for p in profile.phases))
        # Header timer uses a dynamic total that absorbs inter-phase gaps
        # (worker teardown + startup adds a few seconds between phases) and
        # operator skips. Recomputed on every start_phase() as
        # ``elapsed_at_phase_start + current_phase + future_phases``. Without
        # this, the header bar hits 100% before the last phase finishes —
        # wall clock ≈ 30:15 target but actual run ≈ 30:29 on a psu-v1
        # profile with 5 phase transitions (~3-6 s gap each). The lobby
        # still shows ``run_total_seconds`` (profile-declared total).
        self._run_total_display = self.run_total_seconds
        self._run_start_mono = time.monotonic()

        self.phase_idx = 0
        self.current_phase: Optional[object] = None
        self.phase_elapsed = 0.0
        self.phase_total = 0
        self._phase_extra = ""
        self._tick_count = 0

        history_keys = {mk for p in self.DUT_PANELS for _, mk, _ in p["metrics"]}
        self.history: Dict[str, Deque[Optional[float]]] = {
            k: deque(maxlen=self.HISTORY_WINDOW) for k in history_keys
        }
        self.latest_sample: Dict[str, object] = {}
        # Auto-calibration: track max observed value per metric so sparkline
        # ceilings adapt to actual hardware instead of hardcoded guesses.
        self._max_seen: Dict[str, float] = {}
        self._ranges_initialized: bool = False
        self.cpu_worker_state = "-"
        self.gpu_worker_state = "-"
        self.mem_worker_state = "-"
        self.storage_worker_state = "-"
        # Which DUTs the profile actually exercises (has non-idle phases for)
        self.dut_available: Dict[str, bool] = {"cpu": True, "gpu": True, "mem": True, "storage": True}
        if hasattr(profile, "available_duts"):
            self.dut_available.update(profile.available_duts())
        self.dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
        if dut_enabled:
            self.dut_enabled.update(
                {k: bool(v) for k, v in dut_enabled.items() if k in self.dut_enabled}
            )
        self.log_lines: Deque[Tuple[str, str, str, str]] = deque(maxlen=32)
        self.event_counts = {"crit": 0, "warn": 0, "info": 0}
        self.user_paused = False
        self.user_abort = False
        self.user_skip_phase = False
        self.awaiting_start = True
        self.start_hint: Optional[str] = None
        self._storage_inventory = discover_storage_devices()

        # ── Lobby state ───────────────────────────────────────────────
        iv = initial_values or {}
        _cur_profile = str(iv.get("profile") or getattr(profile, "name", "") or "")
        self._profile_names: List[str] = self._gather_profile_names(_cur_profile)
        self._lobby_values: Dict[str, str] = {
            "profile":     _cur_profile,
            "dur_scale":   str(iv.get("dur_scale")   or "1.0"),
            "psu_serial":  str(iv.get("psu_serial")  or psu_serial or ""),
            "storage_duts": str(iv.get("storage_duts") or ""),
            "psu_wattage": str(iv.get("psu_wattage") or ""),
            "psu_make":    str(iv.get("psu_make")    or ""),
            "psu_model":   str(iv.get("psu_model")   or ""),
            "operator":    str(iv.get("operator")    or ""),
            "meter":       str(iv.get("meter_available")   or "no"),
            "paperclip":   str(iv.get("paperclip_result")  or "not-run"),
            "post_result": str(iv.get("post_result") or "not-tested"),
        }
        self._field_cursor = 0        # index into _LOBBY_FIELDS_DEF
        self._field_editing = False   # currently editing a text field
        self._field_edit_buf = ""     # live edit buffer
        self._field_edit_orig = ""    # saved value for ESC-cancel

        # Modal popup picker state (cycle fields with multiple options)
        self._picker_open = False
        self._picker_options: List[str] = []
        self._picker_idx = 0
        self._picker_field_key = ""
        self._picker_label = ""
        self._storage_picker_open = False
        self._storage_picker_idx = 0
        self._storage_picker_selected: List[str] = []

        # Scrollable pad state. All live + lobby rendering happens into an
        # off-screen curses.newpad so metrics are never clipped when the
        # terminal is shorter than the content. stdscr only hosts the viewport
        # scroll + picker overlay + the terminal-too-narrow banner.
        # Scroll position is clamped in _render after content height is known.
        self._pad: Optional["curses._CursesWindow"] = None
        self._pad_rows: int = 0
        self._pad_cols: int = 0
        self._render_target: "curses._CursesWindow" = stdscr
        self._scroll_y: int = 0
        self._content_height: int = 0
        # Flipped to True by key handlers that change what's on screen
        # (scroll, resize, picker toggle). _poll_input renders once after
        # draining the key queue so scroll keys respond within one poll
        # interval (~100 ms) instead of waiting up to 1 s for the next
        # tick's update_sample() render.
        self._needs_render: bool = False

        try:
            curses.curs_set(0)
        except curses.error:
            pass
        try:
            curses.cbreak()
            curses.noecho()
            self.stdscr.keypad(True)
        except curses.error:
            pass
        self.stdscr.nodelay(True)
        self._has_color = False
        try:
            curses.start_color()
            curses.use_default_colors()
            if getattr(curses, "COLORS", 0) >= 256:
                curses.init_pair(self._C_GREEN, 34, -1)   # muted forest green
                curses.init_pair(self._C_AMBER, 214, -1)  # warm amber
                curses.init_pair(self._C_RED, 196, -1)     # bright red
                curses.init_pair(self._C_CYAN, 51, -1)     # electric cyan
                curses.init_pair(self._C_SLATE, 245, -1)   # medium gray
                curses.init_pair(self._C_STEEL, 67, -1)    # blue-gray
                curses.init_pair(self._C_TEAL, 43, -1)     # cool blue-green
                curses.init_pair(self._C_PINK, 197, -1)    # hot pink accent
            else:
                curses.init_pair(self._C_GREEN, curses.COLOR_GREEN, -1)
                curses.init_pair(self._C_AMBER, curses.COLOR_YELLOW, -1)
                curses.init_pair(self._C_RED, curses.COLOR_RED, -1)
                curses.init_pair(self._C_CYAN, curses.COLOR_CYAN, -1)
                curses.init_pair(self._C_SLATE, curses.COLOR_WHITE, -1)
                curses.init_pair(self._C_STEEL, curses.COLOR_BLUE, -1)
                curses.init_pair(self._C_TEAL, curses.COLOR_CYAN, -1)
                curses.init_pair(self._C_PINK, curses.COLOR_MAGENTA, -1)
            self._has_color = True
        except curses.error:
            pass

    # ── Public API (signatures unchanged) ─────────────────────────────

    def start_phase(
        self,
        phase: object,
        phase_idx: int,
        phase_modes: Optional[Dict[str, str]] = None,
    ) -> None:
        self.current_phase = phase
        self.phase_idx = phase_idx
        self.phase_total = max(1, int(phase.duration_sec))
        self.phase_elapsed = 0.0
        self._phase_extra = ""
        # Recompute the header's "total" to absorb accumulated inter-phase
        # gaps and any operator-skipped phases before this one. At the
        # instant this phase starts, the true remaining runtime is this
        # phase's target plus all future phases' targets; add wall-clock
        # elapsed to get the new total. For the very first phase this is
        # essentially the static sum; for later phases the total drifts
        # outward by the gap + skip amount so the bar stays accurate
        # through the final phase.
        run_el = time.monotonic() - self._run_start_mono
        remaining = sum(int(p.duration_sec)
                        for p in self.profile.phases[phase_idx:])
        self._run_total_display = max(1, int(run_el) + remaining)
        modes = phase_modes or {
            "cpu": getattr(phase, "cpu_mode", "idle"),
            "gpu": getattr(phase, "gpu_mode", "idle"),
            "mem": getattr(phase, "mem_mode", "idle"),
            "storage": getattr(phase, "storage_mode", "idle"),
        }
        self.log_event(
            "info",
            phase.name,
            "phase start ({}s  CPU={}  GPU={}  MEM={}  SSD={})".format(
                phase.duration_sec,
                modes["cpu"], modes["gpu"], modes["mem"], modes["storage"],
            ),
        )
        self._render()

    def update_sample(
        self,
        sample: Dict[str, object],
        elapsed_phase: float,
        cpu_state: str,
        gpu_state: str,
        mem_state: str = "-",
        storage_state: str = "-",
    ) -> None:
        self.phase_elapsed = elapsed_phase
        try:
            read_mbps = sample.get("ssd_read_mbps")
            write_mbps = sample.get("ssd_write_mbps")
            if isinstance(read_mbps, (int, float)) or isinstance(write_mbps, (int, float)):
                sample["ssd_mixed_mbps"] = float(read_mbps or 0.0) + float(write_mbps or 0.0)
        except (TypeError, ValueError):
            pass
        self.latest_sample = sample
        self.cpu_worker_state = cpu_state
        self.gpu_worker_state = gpu_state
        self.mem_worker_state = mem_state
        self.storage_worker_state = storage_state
        self._tick_count += 1
        # Auto-calibrate all sparkline ranges from observed data every tick.
        self._auto_calibrate(sample)
        for key in self.history:
            value = sample.get(key)
            if value is None:
                self.history[key].append(None)
            else:
                try:
                    self.history[key].append(float(value))
                except (TypeError, ValueError):
                    self.history[key].append(None)
        self._poll_input()
        self._render()

    def _auto_calibrate(self, sample: Dict[str, object]) -> None:
        """Auto-scale sparkline ceilings and domain thresholds from live data.

        Called every tick. For every metric listed in ``_AUTO_CALIBRATE_SEEDS``,
        tracks the highest observed value and sets the sparkline ceiling to
        110% of that (so bars fill proportionally at peak). On the first tick,
        seeds from class defaults. As the run progresses and load ramps, the
        ceiling ratchets upward — never downward — so the scale stays stable.

        Special cases:
        - ``gpu_vram_used_mib``: ceiling is the card's actual total VRAM (from
          ``gpu_vram_total_mib``), not 110% of max used.
        - Domain thresholds for VRAM set to 85%/95% of actual total.
        """
        # Shadow class-level dicts onto this instance on first call.
        if not self._ranges_initialized:
            self._SPARK_RANGE = dict(self.__class__._SPARK_RANGE)
            self._DOMAIN = dict(self.__class__._DOMAIN)
            self._DOMAIN_FLOOR = dict(self.__class__._DOMAIN_FLOOR)
            self._DOMAIN_ADAPTIVE = dict(self.__class__._DOMAIN_ADAPTIVE)
            self._RAIL_SPEC = {k: dict(v) for k, v in self.__class__._RAIL_SPEC.items()}
            # Seed auto-calibrate metrics into the instance range dict.
            for key, seed_ceil in self._AUTO_CALIBRATE_SEEDS.items():
                if key not in self._SPARK_RANGE:
                    self._SPARK_RANGE[key] = (0.0, seed_ceil)
            # Scale PSU power/current thresholds to the rated wattage the
            # operator entered in the lobby. Class defaults target 1500 W;
            # an 850 W PSU gets thresholds shifted down proportionally so
            # the coloring is calibrated to what's actually on the bench.
            self._apply_psu_thresholds()
            self._ranges_initialized = True

        # AC mains nominal: detect on first positive reading. HX1500i is
        # universal-input; a bench in the EU sees ~230 V where a US shop
        # sees ~120 V. Pin the RAIL_SPEC nominal on the first reading so
        # the coloring isn't false-flagging every sample as out-of-spec
        # in the wrong region.
        if not getattr(self, "_vin_nominal_locked", False):
            raw_vin = sample.get("psu_vin_v")
            if raw_vin is not None:
                try:
                    vin_val = float(raw_vin)
                    if vin_val > 50.0:  # ignore driver-startup zeros
                        spec = self._RAIL_SPEC["psu_vin_v"]
                        spec["nominal"] = 230.0 if vin_val >= 180.0 else 120.0
                        # Widen sparkline range to match — EU benches need
                        # headroom above 230 V, not a 100-130 V band.
                        if vin_val >= 180.0:
                            self._SPARK_RANGE["psu_vin_v"] = (200.0, 250.0)
                        self._vin_nominal_locked = True
                except (TypeError, ValueError):
                    pass

        # VRAM total: use actual card capacity, not 110% of usage.
        vram_total = sample.get("gpu_vram_total_mib")
        if vram_total is not None:
            try:
                total_mib = float(vram_total)
                if total_mib > 0:
                    self._SPARK_RANGE["gpu_vram_used_mib"] = (0.0, total_mib)
                    self._DOMAIN["gpu_vram_used_mib"] = (
                        total_mib * 0.85, total_mib * 0.95,
                    )
            except (TypeError, ValueError):
                pass

        # Auto-scale all other calibratable metrics.
        for key in self._AUTO_CALIBRATE_SEEDS:
            if key == "gpu_vram_used_mib":
                continue  # handled above
            raw = sample.get(key)
            if raw is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if val <= 0:
                continue

            prev_max = self._max_seen.get(key, 0.0)
            if val > prev_max:
                self._max_seen[key] = val
                new_ceiling = val * 1.1
                current_ceil = self._SPARK_RANGE.get(key, (0.0, 0.0))[1]
                if new_ceiling > current_ceil:
                    self._SPARK_RANGE[key] = (0.0, new_ceiling)

    def _apply_psu_thresholds(self) -> None:
        """Scale PSU power/current thresholds to the rated wattage the
        operator entered in the lobby ``Wattage`` field. Called once, from
        the first tick of ``_auto_calibrate``. Unparseable/blank wattage
        falls back to 1500 W (HX1500i default on the DUT).

        Ratios:
          * p_out_w:  warm at 80 %, hot at 95 % of rated.
          * p_in_w:   warm at 88 %, hot at 105 % of rated (input is always
                      higher than output; Platinum ≈ 92 % efficient, so a
                      fully-loaded 1500 W PSU draws ~1630 W at the wall).
          * i_12v_a:  80 % / 95 % of (rated / 12) — 125 A ceiling at
                      1500 W, scales linearly for smaller PSUs.

        5 V / 3.3 V rails are left at their class defaults — those rails
        rarely scale with total wattage (an HX1500i and HX850i both ship
        with roughly the same 25 A cap on 5 V / 3.3 V).
        """
        rated_w = self._parse_rated_wattage()
        if rated_w is None:
            rated_w = 1500.0
        self._DOMAIN["psu_p_out_w"] = (rated_w * 0.80, rated_w * 0.95)
        self._DOMAIN["psu_p_in_w"]  = (rated_w * 0.88, rated_w * 1.05)
        max_12v = rated_w / 12.0
        self._DOMAIN["psu_i_12v_a"] = (max_12v * 0.80, max_12v * 0.95)
        self._SPARK_RANGE["psu_p_out_w"]  = (0.0, rated_w)
        self._SPARK_RANGE["psu_p_in_w"]   = (0.0, rated_w * 1.1)
        self._SPARK_RANGE["psu_i_12v_a"]  = (0.0, max_12v)
        # 12V rail carries ~95 % of total load — scale its sparkline band
        # with the lobby wattage so per-rail power readout is legible.
        self._SPARK_RANGE["psu_p_12v_w"]  = (0.0, rated_w)

    def _parse_rated_wattage(self) -> Optional[float]:
        """Parse the lobby ``Wattage`` field, tolerating operator variants
        like ``1500``, ``1500W``, ``1500 w``, ``1.5kW``. Returns None on
        anything unparseable so the caller can fall back to a default."""
        raw = str(self._lobby_values.get("psu_wattage", "") or "").strip().lower()
        if not raw:
            return None
        raw = raw.replace(" ", "").replace("watts", "w")
        try:
            if raw.endswith("kw"):
                return float(raw[:-2]) * 1000.0
            if raw.endswith("w"):
                return float(raw[:-1])
            return float(raw)
        except (ValueError, TypeError):
            return None

    def log_event(self, severity: str, phase_name: str, message: str) -> None:
        if severity in self.event_counts:
            self.event_counts[severity] += 1
        ts = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
        trimmed = message.replace("\n", " ").strip()
        if len(trimmed) > 100:
            trimmed = trimmed[:97] + "..."
        self.log_lines.append((ts, severity, phase_name, trimmed))
        self._render()

    def set_phase_extra(self, text: str) -> None:
        self._phase_extra = (text or "")[:28]
        self._render()

    def selected_duts(self) -> Dict[str, bool]:
        return dict(self.dut_enabled)

    def _storage_profile_requires_dut(self) -> bool:
        return bool(getattr(self.profile, "name", "").lower().startswith("storage"))

    def _storage_picker_paths(self) -> List[str]:
        return [device.path for device in self._storage_inventory]

    def _any_modal_open(self) -> bool:
        return bool(self._picker_open or self._storage_picker_open)

    def lobby_config(self) -> Dict[str, object]:
        """Return config collected in the lobby screen.  Call after wait_for_start."""
        return {
            "profile_name":    self._lobby_values["profile"],
            "dur_scale":       self._lobby_values["dur_scale"],
            "psu_serial":      self._lobby_values["psu_serial"],
            "storage_duts":    self._lobby_values["storage_duts"],
            "psu_wattage":     self._lobby_values["psu_wattage"],
            "psu_make":        self._lobby_values["psu_make"],
            "psu_model":       self._lobby_values["psu_model"],
            "operator":        self._lobby_values["operator"],
            "meter_available": self._lobby_values["meter"],
            "paperclip_result": self._lobby_values["paperclip"],
            "post_result":     self._lobby_values["post_result"],
            "dut_enabled":     dict(self.dut_enabled),
            "aborted":         self.user_abort,
        }

    @staticmethod
    def _gather_profile_names(current: str = "") -> List[str]:
        """Scan profiles/ for available profile names; current profile first."""
        try:
            from psfu.config import PROFILES_DIR  # local import avoids circularity
            names: List[str] = []
            for p in sorted(PROFILES_DIR.glob("*.json")):
                names.append(p.stem)
            dev_dir = PROFILES_DIR / "dev"
            if dev_dir.exists():
                for p in sorted(dev_dir.glob("*.json")):
                    names.append(p.stem)
        except Exception:
            return [current] if current else []
        if not names:
            return [current] if current else []
        if current and current in names:
            names = [current] + [n for n in names if n != current]
        elif current:
            names = [current] + names
        return names

    def _try_reload_profile(self) -> bool:
        """Load the profile in _lobby_values['profile'] and update dashboard state.
        Returns True on success; on failure sets start_hint and returns False."""
        name = self._lobby_values.get("profile", "").strip()
        if not name:
            return False
        try:
            from psfu.runtime import load_profile  # local import
            new_profile = load_profile(name)
        except Exception:
            self.start_hint = "Profile not found: {}".format(name)
            return False
        self.profile = new_profile
        self.run_total_seconds = max(1, sum(p.duration_sec for p in new_profile.phases))
        self._run_total_display = self.run_total_seconds
        self.dut_available = {"cpu": True, "gpu": True, "mem": True, "storage": True}
        if hasattr(new_profile, "available_duts"):
            self.dut_available.update(new_profile.available_duts())
        for key in list(self.dut_enabled):
            if not self.dut_available.get(key, True):
                self.dut_enabled[key] = False
        self.start_hint = None
        return True

    def wait_for_start(self) -> None:
        while self.awaiting_start and not self.user_abort:
            self._poll_input()
            self._render()
            time.sleep(0.05)

    # ── Color helpers ─────────────────────────────────────────────────

    def _cp(self, pair_id: int, extra: int = 0) -> int:
        if not self._has_color:
            return extra
        try:
            return curses.color_pair(pair_id) | extra
        except curses.error:
            return extra

    def _metric_level(self, key: str, value: float) -> str:
        """Return 'cool', 'warm', or 'hot' based on domain thresholds.

        Auto-calibrated metrics also check against their sparkline ceiling:
        >=95% of ceiling = hot, >=85% = warm.  This ensures that any metric
        pinned at the top of its observed range shows red, even if no
        hardcoded domain threshold exists for it.
        """
        spec = self._RAIL_SPEC.get(key)
        if spec is not None:
            dev = abs(value - spec["nominal"])
            if dev >= spec["crit"]:
                return "hot"
            if dev >= spec["warn"]:
                return "warm"
            return "cool"
        thresholds = self._DOMAIN.get(key)
        if thresholds is not None:
            warm_t, hot_t = thresholds
            if value >= hot_t:
                return "hot"
            if value >= warm_t:
                return "warm"

        # Floor-based thresholds for metrics where LOWER = worse
        # (efficiency being the canonical example). warm_floor > hot_floor
        # — cross below warm_floor for amber, below hot_floor for red.
        floor = getattr(self, "_DOMAIN_FLOOR", {}).get(key)
        if floor is not None:
            warm_f, hot_f = floor
            if value <= hot_f:
                return "hot"
            if value <= warm_f:
                return "warm"

        # Adaptive thresholds: ratio of observed max. Activates once
        # min_value and min_ticks conditions are met (so an idle fan
        # doesn't false-flag against its own low peak). When not yet
        # active, falls through to the sparkline-ceiling fallback.
        adaptive = getattr(self, "_DOMAIN_ADAPTIVE", {}).get(key)
        if adaptive is not None:
            warm_r, hot_r, min_val, min_ticks = adaptive
            max_seen = self._max_seen.get(key, 0.0)
            if max_seen >= min_val and self._tick_count >= min_ticks:
                if value >= max_seen * hot_r:
                    return "hot"
                if value >= max_seen * warm_r:
                    return "warm"
                return "cool"

        # For auto-calibrated metrics: if value is near the observed ceiling,
        # flag it.  This catches "maxed out at scale limit" visually.
        spark_range = self._SPARK_RANGE.get(key)
        if spark_range is not None and key in self._AUTO_CALIBRATE_SEEDS:
            _, ceiling = spark_range
            if ceiling > 0:
                ratio = value / ceiling
                if ratio >= 0.95:
                    return "hot"
                if ratio >= 0.85:
                    return "warm"

        return "cool"

    def _level_attr(self, level: str) -> int:
        if level == "hot":
            return self._cp(self._C_RED, curses.A_BOLD)
        if level == "warm":
            return self._cp(self._C_AMBER, curses.A_BOLD)
        return self._cp(self._C_CYAN)

    def _val_attr(self, key: str, value: object) -> int:
        if value is None:
            return 0
        try:
            return self._level_attr(self._metric_level(key, float(value)))
        except (TypeError, ValueError):
            return 0

    def _status_badge(self) -> Tuple[str, int]:
        c, w = self.event_counts["crit"], self.event_counts["warn"]
        if c > 0:
            return "CRITICAL", self._cp(self._C_RED, curses.A_BOLD)
        if w >= 3:
            return "WARN", self._cp(self._C_AMBER, curses.A_BOLD)
        if w > 0:
            return "WATCH", self._cp(self._C_AMBER)
        return "NOMINAL", self._cp(self._C_GREEN)

    # ── Drawing primitives ────────────────────────────────────────────

    def _s(self, y: int, x: int, text: str, attr: int = 0) -> None:
        """Safe addstr to the current render target — silently clips if out of bounds.

        Target is the scrollable pad during _render_live / _render_prerun and
        stdscr during the picker overlay / size banner.
        """
        if y < 0 or not text:
            return
        try:
            self._render_target.addstr(y, x, text, attr)
        except curses.error:
            pass

    def _hline(self, y: int, w: int, heavy: bool = False) -> None:
        self._s(y, 0, ("\u2501" if heavy else "\u2500") * w, self._cp(self._C_STEEL))

    def _trunc(self, text: str, width: int) -> str:
        if width <= 0:
            return ""
        if len(text) <= width:
            return text
        return text[: max(1, width - 1)] + "\u2026" if width > 1 else text[:1]

    def _box(self, y: int, x: int, w: int, h: int) -> None:
        if w < 4 or h < 3:
            return
        a = self._cp(self._C_STEEL)
        self._s(y, x, "\u250c" + "\u2500" * (w - 2) + "\u2510", a)
        for r in range(y + 1, y + h - 1):
            self._s(r, x, "\u2502", a)
            self._s(r, x + w - 1, "\u2502", a)
        self._s(y + h - 1, x, "\u2514" + "\u2500" * (w - 2) + "\u2518", a)

    def _fmt_val(self, key: str, fmt: str) -> Tuple[str, int]:
        value = self.latest_sample.get(key) if self.latest_sample else None
        if value is None:
            return "\u2500\u2500", self._cp(self._C_SLATE)
        try:
            fv = float(value)
            # Colour still driven by the canonical stored value — thresholds
            # live on the pre-transform scale. Only the rendered text scales.
            attr = self._val_attr(key, fv) or self._cp(self._C_CYAN)
            transform = self._DISPLAY_TRANSFORM.get(key)
            display_v = transform(fv) if transform is not None else fv
            return fmt.format(display_v), attr
        except (TypeError, ValueError):
            return "\u2500\u2500", self._cp(self._C_SLATE)

    # ── Sparkline with domain-aware per-character coloring ────────────

    def _draw_spark(
        self, row: int, x: int, metric_key: str,
        values: List[Optional[float]], width: int,
    ) -> None:
        """Draw a sparkline where each bar is scaled and colored by absolute
        domain thresholds, not by relative position in the data window."""
        spark = render_sparkline(values, width, abs_range=self._SPARK_RANGE.get(metric_key))
        sampled = _resample(values, width)
        for i, ch in enumerate(spark):
            sv = sampled[i] if i < len(sampled) else None
            if not ch.strip() or sv is None:
                self._s(row, x + i, ch, self._cp(self._C_SLATE))
                continue
            level = self._metric_level(metric_key, float(sv))
            if level == "hot":
                attr = self._cp(self._C_RED, curses.A_BOLD)
            elif level == "warm":
                attr = self._cp(self._C_AMBER)
            else:
                attr = self._cp(self._C_TEAL)
            self._s(row, x + i, ch, attr)

    # ── Worker state badge ────────────────────────────────────────────

    def _worker_tag(self, state: str) -> Tuple[str, int]:
        """Tag summarizes the PS-FU *stress worker* state for this DUT —
        NOT the physical hardware state. A GPU pulling 16 W with fans at
        100 % during a cpu-only phase is "REST" from the worker's POV
        because no stress is being driven against it; the panel metrics
        still show real hardware telemetry.

        "IDLE" was the old label for the no-worker-attached case. Renamed
        to "REST" after it kept being read as "GPU is dead" when really
        it meant "profile doesn't stress GPU this phase."
        """
        s = (state or "-").strip().lower()
        if "running" in s:
            return "RUN", self._cp(self._C_GREEN, curses.A_BOLD)
        if "paus" in s:
            return "PAUSE", self._cp(self._C_AMBER, curses.A_BOLD)
        if "rc=" in s:
            return "EXIT", self._cp(self._C_RED, curses.A_BOLD)
        if "monitor" in s:
            # MOBO panel is pure passive telemetry — no stress worker is
            # ever attached. "MON" signals "always monitoring" so it
            # doesn't read as REST (worker resting) or OFF (disabled).
            return "MON", self._cp(self._C_TEAL)
        if s == "-" or "idle" in s:
            return "REST", self._cp(self._C_SLATE)
        if "disabled" in s:
            return "OFF", self._cp(self._C_SLATE)
        return self._trunc(s.upper(), 5), self._cp(self._C_SLATE)

    def _dut_state(self, key: str) -> str:
        if key in ("mobo", "ups"):
            # Both panels are passive telemetry — no stress worker ever
            # attaches. Worker tag renders as MON (teal) via the
            # "monitor" branch in _worker_tag.
            return "monitor"
        if not self.dut_enabled.get(key, True):
            return "disabled"
        return {
            "cpu": self.cpu_worker_state, "gpu": self.gpu_worker_state,
            "mem": self.mem_worker_state, "storage": self.storage_worker_state,
        }.get(key, "-")

    # ── DUT panel card ────────────────────────────────────────────────

    def _draw_panel(
        self,
        row: int,
        x: int,
        w: int,
        panel: dict,
        maxy: int,
        max_metrics: Optional[int] = None,
    ) -> int:
        """Draw one DUT card.  Each metric gets its own row with value +
        inline sparkline so you can watch Load, Power, and Temp independently.

        ``max_metrics`` clips the metric list to its first N entries so a
        panel with a long metric tail (GPU has 13) can fit into a short
        window alongside its sibling panels. The DUT_PANELS metric order is
        priority-ordered — first entry is most important — so clipping from
        the end drops the least-critical rows first.

        Layout (N metric rows + 3):
          ┌ CPU ─ RUN ────────────────────────────────┐
          │  Load  98%  ▁▂▃▅▆▇██████████████████████  │
          │  Power 285W ▁▁▁▃▅▇██████████████████████  │
          │  Temp  78°  ▁▁▂▃▃▃▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇▇▇▇▇  │
          └───────────────────────────────────────────┘
        """
        metrics = panel["metrics"]
        if max_metrics is not None and max_metrics < len(metrics):
            metrics = metrics[:max(1, max_metrics)]
        H = len(metrics) + 3  # top border + metric rows + blank + bottom border
        # maxy here is the pad row cap (over-sized), not the terminal height —
        # scroll handles anything taller than the viewport.
        if row + H > maxy or w < 28:
            return row
        key = str(panel["key"])
        title = str(panel["title"])
        enabled = self.dut_enabled.get(key, True)
        available = self.dut_available.get(key, True)

        self._box(row, x, w, H)

        if not available:
            self._s(row, x + 2, " {} ".format(title), self._cp(self._C_SLATE))
            self._s(row, x + len(title) + 5, "N/A", self._cp(self._C_SLATE))
            self._s(row + 2, x + 2, "no phases in profile", self._cp(self._C_SLATE))
            return row + H

        # Title line
        t_color = self._cp(self._C_CYAN, curses.A_BOLD) if enabled else self._cp(self._C_SLATE)
        self._s(row, x + 2, " {} ".format(title), t_color)
        tag, tag_attr = self._worker_tag(self._dut_state(key))
        self._s(row, x + len(title) + 5, tag, tag_attr)

        # Per-metric rows: label  value  sparkline
        val_col = 9       # column offset for value (after "  Load " etc.)
        spark_start = 17  # column offset for sparkline start
        spark_w = max(8, w - spark_start - 2)

        for i, (label, mk, fmt) in enumerate(metrics):
            mrow = row + 1 + i
            mk_str = str(mk)
            # Label
            self._s(mrow, x + 2, "{:<5}".format(label), self._cp(self._C_SLATE))
            # Value
            v_str, v_attr = self._fmt_val(mk_str, str(fmt))
            self._s(mrow, x + val_col, "{:<7}".format(v_str), v_attr)
            # Sparkline
            vals = list(self.history.get(mk_str, []))
            self._draw_spark(mrow, x + spark_start, mk_str, vals, spark_w)

        return row + H

    # ── Input handling ────────────────────────────────────────────────

    def _toggle_dut(self, key: str) -> None:
        if key in self.dut_enabled and self.dut_available.get(key, True):
            self.dut_enabled[key] = not self.dut_enabled[key]
            self.start_hint = None

    def _poll_input(self) -> None:
        # Drain all available keys — a single getch() only returns one char,
        # so queued resize events or escape sequences can starve real keypresses.
        # If any key flips ``_needs_render`` (scroll / resize / picker toggle /
        # field edit), render once after the drain so the operator sees an
        # immediate update instead of waiting up to 1 s for the next tick's
        # update_sample() to refresh.
        drained = False
        while True:
            try:
                ch = self.stdscr.getch()
            except curses.error:
                break
            if ch == -1:
                break
            self._handle_key(ch)
            drained = True
        if drained and self._needs_render:
            self._needs_render = False
            self._render()

    def _open_picker(self, key: str, label: str, options: List[str]) -> None:
        """Open the modal popup picker for a cycle field."""
        if not options:
            return
        self._picker_options = list(options)
        cur = self._lobby_values.get(key, options[0])
        self._picker_idx = options.index(cur) if cur in options else 0
        self._picker_field_key = key
        self._picker_label = label.strip()
        self._picker_open = True

    def _close_picker(self, *, commit: bool) -> None:
        if commit and self._picker_options:
            key = self._picker_field_key
            new_val = self._picker_options[self._picker_idx]
            old_val = self._lobby_values.get(key)
            self._lobby_values[key] = new_val
            if key == "profile" and new_val != old_val:
                self._try_reload_profile()
        self._picker_open = False
        self._picker_options = []
        self._picker_field_key = ""
        self._picker_label = ""

    def _handle_picker_key(self, ch: int) -> None:
        n = len(self._picker_options)
        if n == 0:
            self._close_picker(commit=False)
        elif ch in (27,):  # ESC
            self._close_picker(commit=False)
        elif ch in (10, 13, curses.KEY_ENTER):
            self._close_picker(commit=True)
        elif ch in (curses.KEY_UP, ord("k")):
            self._picker_idx = (self._picker_idx - 1) % n
        elif ch in (curses.KEY_DOWN, ord("j")):
            self._picker_idx = (self._picker_idx + 1) % n
        elif ch == curses.KEY_HOME:
            self._picker_idx = 0
        elif ch == curses.KEY_END:
            self._picker_idx = n - 1
        elif ch == curses.KEY_PPAGE:
            self._picker_idx = max(0, self._picker_idx - 5)
        elif ch == curses.KEY_NPAGE:
            self._picker_idx = min(n - 1, self._picker_idx + 5)
        elif 32 <= ch <= 126:
            # First-letter jump: find next option starting with this letter
            letter = chr(ch).lower()
            for offset in range(1, n + 1):
                cand = (self._picker_idx + offset) % n
                if self._picker_options[cand].lower().startswith(letter):
                    self._picker_idx = cand
                    break

    def _handle_key(self, ch: int) -> None:
        if ch == curses.KEY_RESIZE:
            # On terminal resize curses' internal buffer can retain cells
            # outside the new dimensions — they show up as "ghost" box
            # borders or stale text at the top of the screen until something
            # overwrites them. clearok(True) forces the whole physical
            # screen to be redrawn from scratch on the next refresh so the
            # ghost characters are wiped.
            try:
                self.stdscr.clear()
                self.stdscr.clearok(True)
            except curses.error:
                pass
            self._needs_render = True
            return
        # ── Scroll keys (global, bypass all mode-specific handling) ──
        # PgUp/PgDn/Home/End scroll the viewport in both lobby and live modes.
        # UP/DOWN scroll only in live mode — they're claimed by field cursor
        # navigation in the lobby. Scroll is clamped against content height
        # in _render so out-of-range values auto-correct on the next redraw.
        if not self._any_modal_open() and not self._field_editing:
            if ch == curses.KEY_PPAGE:
                self._scroll_y = max(0, self._scroll_y - 10)
                self._needs_render = True
                return
            if ch == curses.KEY_NPAGE:
                self._scroll_y += 10
                self._needs_render = True
                return
            if ch == curses.KEY_HOME:
                self._scroll_y = 0
                self._needs_render = True
                return
            if ch == curses.KEY_END:
                self._scroll_y = max(0, self._content_height)
                self._needs_render = True
                return
            if not self.awaiting_start and ch == curses.KEY_UP:
                self._scroll_y = max(0, self._scroll_y - 1)
                self._needs_render = True
                return
            if not self.awaiting_start and ch == curses.KEY_DOWN:
                self._scroll_y += 1
                self._needs_render = True
                return
        if self._picker_open:
            self._handle_picker_key(ch)
        elif self._storage_picker_open:
            self._handle_storage_picker_key(ch)
        elif self.awaiting_start:
            n_fields = len(self._LOBBY_FIELDS_DEF)
            if self._field_editing:
                # ── Text editing mode ──
                if ch in (27,):  # ESC — cancel
                    key = self._LOBBY_FIELDS_DEF[self._field_cursor][0]
                    self._lobby_values[key] = self._field_edit_orig
                    self._field_editing = False
                elif ch in (9, curses.KEY_BTAB):  # Tab — confirm + advance
                    key = self._LOBBY_FIELDS_DEF[self._field_cursor][0]
                    self._lobby_values[key] = self._field_edit_buf
                    self._field_editing = False
                    self._field_cursor = (self._field_cursor + (1 if ch == 9 else -1)) % n_fields
                elif ch in (10, 13, curses.KEY_ENTER):  # Enter — confirm
                    key = self._LOBBY_FIELDS_DEF[self._field_cursor][0]
                    self._lobby_values[key] = self._field_edit_buf
                    self._field_editing = False
                elif ch in (curses.KEY_BACKSPACE, 127, 8):
                    self._field_edit_buf = self._field_edit_buf[:-1]
                elif 32 <= ch <= 126:
                    self._field_edit_buf += chr(ch)
            else:
                # ── Navigation / action mode ──
                # DUT hotkeys (always available)
                _dut_map = {
                    ord("1"): "cpu",  ord("c"): "cpu",  ord("C"): "cpu",
                    ord("2"): "gpu",  ord("g"): "gpu",  ord("G"): "gpu",
                    ord("3"): "mem",  ord("m"): "mem",  ord("M"): "mem",
                    ord("4"): "storage", ord("d"): "storage", ord("D"): "storage",
                }
                if ch in _dut_map:
                    self._toggle_dut(_dut_map[ch])
                elif ch in (curses.KEY_DOWN, ord("\t"), 9):  # ↓ / Tab
                    self._field_cursor = (self._field_cursor + 1) % n_fields
                elif ch in (curses.KEY_UP, curses.KEY_BTAB):  # ↑ / Shift-Tab
                    self._field_cursor = (self._field_cursor - 1) % n_fields
                elif ch in (10, 13, curses.KEY_ENTER):
                    # Enter on text field → start editing
                    # Enter on cycle field → open popup picker (or cycle if 1 option)
                    fi = self._field_cursor
                    key, label, ftype, _, options = self._LOBBY_FIELDS_DEF[fi]
                    if ftype == "text":
                        if key == "storage_duts" and self._storage_inventory:
                            self._open_storage_picker()
                        else:
                            self._field_edit_orig = self._lobby_values.get(key, "")
                            self._field_edit_buf  = self._field_edit_orig
                            self._field_editing   = True
                    elif ftype == "cycle":
                        opts = self._profile_names if key == "profile" else options
                        if len(opts) >= 2:
                            self._open_picker(key, label, opts)
                        elif opts:
                            # Single option — nothing to pick, just confirm it
                            old_val = self._lobby_values.get(key)
                            self._lobby_values[key] = opts[0]
                            if key == "profile" and opts[0] != old_val:
                                self._try_reload_profile()
                elif ch == ord(" "):
                    # Space → START (requires at least one DUT)
                    serial_val = (self._lobby_values.get("psu_serial", "") or "").strip()
                    if not serial_val:
                        serial_val = time.strftime("run-%Y%m%d-%H%M%S")
                        self._lobby_values["psu_serial"] = serial_val
                    # Block only when DUTs exist but none are enabled.
                    # All-idle profiles (e.g. smoke-v1) have no available DUTs
                    # by design — don't block them.
                    any_available = any(self.dut_available.values())
                    any_enabled   = any(self.dut_enabled.values())
                    if any_available and not any_enabled:
                        self.start_hint = "Enable at least one DUT target."
                    elif self._storage_profile_requires_dut() and not self._lobby_values.get("storage_duts", "").strip():
                        self.start_hint = "Select at least one storage DUT."
                    else:
                        # Sync psu_serial to header display
                        self.psu_serial = self._lobby_values.get("psu_serial", "?") or "?"
                        self.awaiting_start = False
                        self.start_hint = None
                elif ch in (ord("q"), ord("Q")):
                    self.user_abort = True
        elif ch in (ord("p"), ord("P"), ord(" ")):
            self.user_paused = not self.user_paused
        elif ch in (ord("n"), ord("N")):
            self.user_skip_phase = True
        elif ch in (ord("q"), ord("Q")):
            self.user_abort = True

    # ── Render dispatch ───────────────────────────────────────────────

    def _render(self) -> None:
        try:
            self.stdscr.erase()
            maxy, maxx = self.stdscr.getmaxyx()
            w = maxx
            if w < 50 or maxy < 3:
                self._render_target = self.stdscr
                self._s(0, 0, "terminal too narrow ({}); need 50+".format(w))
                self.stdscr.refresh()
                return

            self._ensure_pad(maxy, w)
            self._pad.erase()
            self._render_target = self._pad

            if self.awaiting_start:
                self._render_prerun(self._pad_rows, w)
            else:
                self._render_live(self._pad_rows, w)

            # Clamp scroll position against the actual content we just drew.
            # _content_height is set to the last row written + 1 by the
            # sub-renderers; keeps scroll bounds honest when the view shrinks.
            max_scroll = max(0, self._content_height - maxy)
            if self._scroll_y < 0:
                self._scroll_y = 0
            elif self._scroll_y > max_scroll:
                self._scroll_y = max_scroll

            self._render_target = self.stdscr
            self.stdscr.noutrefresh()
            self._pad.noutrefresh(
                self._scroll_y, 0,
                0, 0,
                maxy - 1, w - 1,
            )

            # Scroll hint in top-right — shown only when content exceeds viewport
            # so the operator knows arrow keys will reveal more.
            if max_scroll > 0:
                self._render_target = self.stdscr
                hint = "\u2195 {}/{}".format(self._scroll_y + 1, self._content_height)
                try:
                    self.stdscr.addstr(
                        maxy - 1, max(0, w - len(hint) - 1),
                        hint, self._cp(self._C_SLATE, curses.A_REVERSE),
                    )
                except curses.error:
                    pass
                self.stdscr.noutrefresh()

            if self._picker_open:
                self._render_target = self.stdscr
                self._render_picker(maxy, w)
                self.stdscr.noutrefresh()
            if self._storage_picker_open:
                self._render_target = self.stdscr
                self._render_storage_picker(maxy, w)
                self.stdscr.noutrefresh()

            curses.doupdate()
        except curses.error:
            pass

    def _ensure_pad(self, viewport_h: int, viewport_w: int) -> None:
        """Allocate (or resize) the off-screen render pad.

        The pad needs to fit the tallest possible content. Live view tops out
        at roughly: 6 header rows + (max panels per column × ~16 rows each) +
        event feed (32 lines). 200 rows is a generous ceiling; curses pads are
        cheap so oversize is fine. Re-allocate when the terminal width changes
        so horizontal geometry matches the viewport exactly.
        """
        rows = 200
        cols = max(50, viewport_w)
        if self._pad is None or self._pad_rows != rows or self._pad_cols != cols:
            self._pad = curses.newpad(rows, cols)
            self._pad_rows = rows
            self._pad_cols = cols

    def _render_picker(self, maxy: int, maxx: int) -> None:
        """Modal popup centered on screen, listing options with current selection."""
        opts = self._picker_options
        n = len(opts)
        if n == 0:
            return
        # Sizing: at least 24 wide, fit the longest option + padding, max 60 wide
        max_label = max((len(o) for o in opts), default=0)
        title = "Select {}".format(self._picker_label or "")
        inner_w = max(24, len(title) + 4, max_label + 6)
        inner_w = min(inner_w, max(20, maxx - 4))
        # Visible rows: title + separator + N options + separator + footer
        max_list_rows = max(3, maxy - 8)
        list_rows = min(n, max_list_rows)
        H = list_rows + 5
        W = inner_w + 2  # borders
        y0 = max(1, (maxy - H) // 2)
        x0 = max(1, (maxx - W) // 2)

        # Window backdrop
        self._box(y0, x0, W, H)
        # Title
        title_attr = self._cp(self._C_CYAN, curses.A_BOLD)
        self._s(y0, x0 + 2, " {} ".format(self._trunc(title, inner_w - 2)), title_attr)

        # Scroll window so selected index stays visible
        if self._picker_idx < 0:
            self._picker_idx = 0
        if self._picker_idx >= n:
            self._picker_idx = n - 1
        if list_rows >= n:
            top = 0
        else:
            half = list_rows // 2
            top = max(0, min(n - list_rows, self._picker_idx - half))

        for i in range(list_rows):
            opt_idx = top + i
            if opt_idx >= n:
                break
            row = y0 + 1 + i
            label = self._trunc(opts[opt_idx], inner_w - 4)
            is_sel = (opt_idx == self._picker_idx)
            if is_sel:
                attr = self._cp(self._C_PINK, curses.A_BOLD | curses.A_REVERSE)
                marker = "\u25b8 "
            else:
                attr = self._cp(self._C_TEAL)
                marker = "  "
            line = (marker + label).ljust(inner_w - 2)
            self._s(row, x0 + 2, line, attr)

        # Scroll indicator + position
        pos_row = y0 + list_rows + 1
        if pos_row < y0 + H - 2:
            indicator = "{}/{}".format(self._picker_idx + 1, n)
            if top > 0:
                indicator = "\u2191 " + indicator
            if top + list_rows < n:
                indicator = indicator + " \u2193"
            self._s(pos_row, x0 + 2, indicator, self._cp(self._C_SLATE))

        # Footer hints
        foot_row = y0 + H - 2
        hints = [("\u2191\u2193", "navigate", self._C_TEAL),
                 ("Enter", "select", self._C_GREEN),
                 ("a-z", "jump", self._C_TEAL),
                 ("ESC", "cancel", self._C_AMBER)]
        cx = x0 + 2
        right_lim = x0 + W - 2
        for kh, desc, col in hints:
            if cx + len(kh) + len(desc) + 4 > right_lim:
                break
            self._s(foot_row, cx, kh, self._cp(col, curses.A_BOLD))
            cx += len(kh) + 1
            self._s(foot_row, cx, desc, self._cp(self._C_SLATE))
            cx += len(desc) + 3

    def _open_storage_picker(self) -> None:
        paths = self._storage_picker_paths()
        if not paths:
            return
        current = set(parse_storage_dut_paths(self._lobby_values.get("storage_duts", "")))
        self._storage_picker_selected = [path for path in paths if path in current]
        self._storage_picker_idx = 0
        self._storage_picker_open = True

    def _close_storage_picker(self, *, commit: bool) -> None:
        if commit:
            ordered = [path for path in self._storage_picker_paths() if path in set(self._storage_picker_selected)]
            self._lobby_values["storage_duts"] = ",".join(ordered)
        self._storage_picker_open = False
        self._storage_picker_idx = 0
        self._storage_picker_selected = []

    def _handle_storage_picker_key(self, ch: int) -> None:
        paths = self._storage_picker_paths()
        n = len(paths)
        if n == 0:
            self._close_storage_picker(commit=False)
            return
        if ch in (27,):
            self._close_storage_picker(commit=False)
        elif ch in (10, 13, curses.KEY_ENTER):
            self._close_storage_picker(commit=True)
        elif ch == ord(" "):
            current = paths[self._storage_picker_idx]
            if current in self._storage_picker_selected:
                self._storage_picker_selected = [path for path in self._storage_picker_selected if path != current]
            else:
                self._storage_picker_selected.append(current)
        elif ch in (curses.KEY_UP, ord("k")):
            self._storage_picker_idx = (self._storage_picker_idx - 1) % n
        elif ch in (curses.KEY_DOWN, ord("j")):
            self._storage_picker_idx = (self._storage_picker_idx + 1) % n
        elif ch == curses.KEY_HOME:
            self._storage_picker_idx = 0
        elif ch == curses.KEY_END:
            self._storage_picker_idx = n - 1

    def _render_storage_picker(self, maxy: int, maxx: int) -> None:
        paths = self._storage_picker_paths()
        n = len(paths)
        if n == 0:
            return
        title = "Select Storage DUTs"
        inner_w = max(42, len(title) + 4)
        inner_w = min(inner_w, max(30, maxx - 4))
        max_list_rows = max(3, maxy - 8)
        list_rows = min(n, max_list_rows)
        H = list_rows + 5
        W = inner_w + 2
        y0 = max(1, (maxy - H) // 2)
        x0 = max(1, (maxx - W) // 2)
        self._box(y0, x0, W, H)
        self._s(y0, x0 + 2, " {} ".format(title), self._cp(self._C_CYAN, curses.A_BOLD))

        top = 0 if list_rows >= n else max(0, min(n - list_rows, self._storage_picker_idx - (list_rows // 2)))
        selected = set(self._storage_picker_selected)
        for i in range(list_rows):
            opt_idx = top + i
            if opt_idx >= n:
                break
            row = y0 + 1 + i
            device = self._storage_inventory[opt_idx]
            mark = "[x]" if device.path in selected else "[ ]"
            label = "{} {} {} [{}]".format(mark, device.path, device.model or "?", device.role)
            is_sel = opt_idx == self._storage_picker_idx
            attr = self._cp(self._C_PINK, curses.A_BOLD | curses.A_REVERSE) if is_sel else self._cp(self._C_TEAL)
            self._s(row, x0 + 2, self._trunc(label, inner_w - 2).ljust(inner_w - 2), attr)

        foot_row = y0 + H - 2
        hints = [("\u2191\u2193", "navigate", self._C_TEAL),
                 ("Space", "toggle", self._C_GREEN),
                 ("Enter", "commit", self._C_GREEN),
                 ("ESC", "cancel", self._C_AMBER)]
        cx = x0 + 2
        right_lim = x0 + W - 2
        for kh, desc, col in hints:
            if cx + len(kh) + len(desc) + 4 > right_lim:
                break
            self._s(foot_row, cx, kh, self._cp(col, curses.A_BOLD))
            cx += len(kh) + 1
            self._s(foot_row, cx, desc, self._cp(self._C_SLATE))
            cx += len(desc) + 3

    # ── Pre-run lobby / config screen ────────────────────────────────

    def _render_prerun(self, maxy: int, w: int) -> None:  # noqa: C901
        # ── Header ──
        sn = self._lobby_values.get("psu_serial") or "?"
        left  = " {} \u2014 {} \u25b6 LOBBY ".format(self.app_name, APP_FULL_NAME)
        right = " SN:{} ".format(sn)
        fill  = max(0, w - len(left) - len(right))
        self._s(0, 0, (left + " " * fill + right)[:w],
                self._cp(self._C_PINK, curses.A_BOLD | curses.A_REVERSE))
        self._hline(1, w, heavy=True)
        self._s(2, 2, APP_TAGLINE, self._cp(self._C_SLATE))
        self._hline(3, w)

        # ── Profile info (updates live when profile field changes) ──
        row = 4
        # Compute scaled runtime for display
        try:
            _sc = float(self._lobby_values.get("dur_scale") or "1.0")
        except ValueError:
            _sc = 1.0
        scaled_total = max(1, int(self.run_total_seconds * _sc))
        runtime_str = fmt_mmss(scaled_total)
        if abs(_sc - 1.0) > 0.001:
            runtime_str += "  (\u00d7{:.2f})".format(_sc)

        self._s(row, 2, "Profile", self._cp(self._C_SLATE))
        self._s(row, 12, self._trunc(self.profile.name, 26), self._cp(self._C_CYAN, curses.A_BOLD))
        self._s(row, 40, "{}ph  {}".format(len(self.profile.phases), runtime_str),
                self._cp(self._C_TEAL))
        row += 1
        self._s(row, 2, "Sequence", self._cp(self._C_SLATE))
        seq = " \u2192 ".join(p.name for p in self.profile.phases)
        self._s(row, 12, self._trunc(seq, w - 14), self._cp(self._C_TEAL))
        row += 1
        self._hline(row, w)
        row += 1

        # ── Two-column: fields left, DUT toggles right ──
        half = max(42, w // 2)
        BOX_W = min(16, half - 14)

        # Section headers
        self._s(row, 2, "RUN CONFIG", self._cp(self._C_SLATE, curses.A_BOLD))
        self._s(row, half + 2, "DUT TARGETS", self._cp(self._C_SLATE, curses.A_BOLD))
        row += 1
        fields_start_row = row

        # ── All fields, left column ──
        for fi, (key, label, ftype, required, options) in enumerate(self._LOBBY_FIELDS_DEF):
            frow = fields_start_row + fi
            is_sel  = (self._field_cursor == fi)
            is_edit = is_sel and self._field_editing

            bullet_a = self._cp(self._C_CYAN, curses.A_BOLD) if is_sel else self._cp(self._C_SLATE)
            self._s(frow, 1, "\u25b8" if is_sel else " ", bullet_a)
            label_a  = self._cp(self._C_CYAN) if is_sel else self._cp(self._C_SLATE)
            self._s(frow, 3, label, label_a)

            if ftype == "text":
                if is_edit:
                    disp = (self._field_edit_buf + "\u2588")[-BOX_W:]
                    disp = disp.ljust(BOX_W)
                    self._s(frow, 12, "[{}]".format(disp), self._cp(self._C_PINK, curses.A_BOLD))
                else:
                    val  = self._lobby_values.get(key, "")
                    disp = val[:BOX_W].ljust(BOX_W)
                    if required and not val:
                        box_a = self._cp(self._C_AMBER, curses.A_BOLD) if is_sel else self._cp(self._C_AMBER)
                    else:
                        box_a = self._cp(self._C_CYAN, curses.A_BOLD) if is_sel else self._cp(self._C_TEAL)
                    self._s(frow, 12, "[{}]".format(disp), box_a)
                    # For Scale field: show scaled total inline
                    if key == "dur_scale":
                        try:
                            sc2 = float(val) if val else 1.0
                            hint = "\u2192 {}".format(fmt_mmss(max(1, int(self.run_total_seconds * sc2))))
                        except ValueError:
                            hint = "\u2192 ?"
                        self._s(frow, 12 + BOX_W + 3, hint, self._cp(self._C_SLATE))
            else:  # cycle
                opts = self._profile_names if key == "profile" else options
                val  = self._lobby_values.get(key, opts[0] if opts else "")
                val_a = self._cp(self._C_TEAL, curses.A_BOLD) if is_sel else self._cp(self._C_TEAL)
                self._s(frow, 12, "< {} >".format(self._trunc(val, BOX_W - 2)), val_a)

        # ── DUT toggles, right column ──
        # Only the four stress-worker DUTs are toggleable; PSU, MOBO,
        # and UPS panels are pure telemetry surfaces, always on.
        # Hotkey numbers here are pinned to the _dut_map in _poll_input
        # (1=cpu, 2=gpu, 3=mem, 4=storage), not the DUT_PANELS
        # iteration order — an earlier bug printed "[2] PSU" while
        # hotkey 2 actually toggled GPU. The static tuple below keeps
        # display and hotkeys aligned as new passive panels land.
        _toggleable = (("cpu", "CPU"), ("gpu", "GPU"), ("mem", "RAM"), ("storage", "SSD"))
        for idx, (dkey, title) in enumerate(_toggleable):
            r = fields_start_row + idx
            available = self.dut_available.get(dkey, True)
            on        = self.dut_enabled.get(dkey, True)
            col = half + 2
            if not available:
                self._s(r, col, "[{}] {}  \u2500  N/A".format(idx + 1, title),
                        self._cp(self._C_SLATE))
            elif on:
                self._s(r, col, "[{}] {}  \u25cf  ON".format(idx + 1, title),
                        self._cp(self._C_CYAN, curses.A_BOLD))
            else:
                self._s(r, col, "[{}] {}  \u25cb  OFF".format(idx + 1, title),
                        self._cp(self._C_SLATE))

        inv_row = fields_start_row + len(_toggleable) + 1
        self._s(inv_row, half + 2, "DISCOVERED DRIVES", self._cp(self._C_SLATE, curses.A_BOLD))
        for idx, device in enumerate(self._storage_inventory[:4], start=1):
            mounts = ",".join(device.mountpoints) if device.mountpoints else "-"
            line = "{} {} {} [{}:{}] mnt={}".format(
                device.path,
                device.vendor or "?",
                device.model or "?",
                device.kind,
                device.role,
                mounts,
            )
            self._s(inv_row + idx, half + 2, self._trunc(line, max(10, w - half - 4)), self._cp(self._C_TEAL))

        # ── Footer ──
        footer_row = max(
            fields_start_row + len(self._LOBBY_FIELDS_DEF) + 1,
            inv_row + min(len(self._storage_inventory), 4) + 2,
        )
        self._hline(footer_row, w)
        cr = footer_row + 1
        if self._field_editing:
            hints = [("type", "input", self._C_TEAL),
                     ("Enter/Tab", "confirm", self._C_GREEN),
                     ("ESC", "cancel", self._C_AMBER)]
        else:
            hints = [("\u2191\u2193/Tab", "navigate", self._C_TEAL),
                     ("Enter", "edit/cycle", self._C_GREEN),
                     ("1-4", "DUT", self._C_TEAL),
                     ("Space", "START", self._C_GREEN),
                     ("PgUp/PgDn", "scroll", self._C_TEAL),
                     ("Q", "quit", self._C_AMBER)]
        cx = 2
        for kh, desc, col in hints:
            if cx + len(kh) + len(desc) + 4 > w:
                break
            self._s(cr, cx, kh, self._cp(col, curses.A_BOLD))
            cx += len(kh) + 1
            self._s(cr, cx, desc, self._cp(self._C_SLATE))
            cx += len(desc) + 3

        last_row = cr
        if self.start_hint and not self._field_editing:
            hr = footer_row + 2
            self._s(hr, 2, self._trunc(self.start_hint, w - 4),
                    self._cp(self._C_RED, curses.A_BOLD))
            last_row = hr

        self._content_height = last_row + 1

    # ── Live dashboard ────────────────────────────────────────────────

    def _render_live(self, maxy: int, w: int) -> None:  # noqa: C901
        row = 0

        # ── Header bar ──
        hb = "\u23f8" if self.user_paused else (
            "\u25b6" if self._tick_count % 2 == 0 else "\u25b7")
        badge, badge_attr = self._status_badge()
        if self.user_paused:
            badge, badge_attr = "PAUSED", self._cp(self._C_AMBER, curses.A_BOLD)
        clock = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S") + "Z"

        left = " {} {} ".format(hb, self.app_name)
        right = " {}  SN:{}  {} ".format(badge, self.psu_serial, clock)
        fill = max(0, w - len(left) - len(right))
        self._s(row, 0, (left + " " * fill + right)[:w], curses.A_REVERSE)
        # Overlay badge with severity color
        bx = len(left) + fill + 1
        if bx + len(badge) < w:
            self._s(row, bx, badge, badge_attr | curses.A_REVERSE)
        row += 1

        # ── Progress ──
        run_el = time.monotonic() - self._run_start_mono
        # Use the dynamic total so the bar and label don't hit 100% before
        # the last phase actually finishes (gaps + skips absorbed).
        run_total = self._run_total_display
        run_pct = min(100, int(run_el / run_total * 100))
        ph_name = self.current_phase.name if self.current_phase else "waiting"
        ph_pct = min(100, int(self.phase_elapsed / self.phase_total * 100)) if self.phase_total else 0
        ph_rem = max(0, self.phase_total - int(self.phase_elapsed))

        # Run bar
        rlabel = "RUN {:>3}%  {}/{}".format(
            run_pct, fmt_mmss(run_el), fmt_mmss(run_total))

        # Phase label — built before bar_x so the bar column can be sized to
        # fit the longer of the two labels. Prevents T- time / pulse extra
        # from being truncated on phases like "transient" that carry a
        # pulse-state indicator.
        extra = "  {}".format(self._phase_extra) if self._phase_extra else ""
        plabel = "{} [{}/{}] T-{}{}".format(
            ph_name, self.phase_idx + 1, len(self.profile.phases),
            fmt_mmss(ph_rem), extra)

        bar_x = max(len(rlabel), len(plabel)) + 3
        bar_x = min(bar_x, max(12, w - 12))  # leave at least 10 cols for the bar
        bar_w = max(10, w - bar_x - 1)
        self._s(row, 1, rlabel, self._cp(self._C_CYAN, curses.A_BOLD))
        self._s(row, bar_x, render_progress_bar(run_el, run_total, bar_w),
                self._cp(self._C_CYAN))
        row += 1

        # Phase bar
        self._s(row, 1, self._trunc(plabel, bar_x - 2), self._cp(self._C_TEAL, curses.A_BOLD))
        self._s(row, bar_x,
                render_progress_bar(self.phase_elapsed, self.phase_total or 1, bar_w),
                self._cp(self._C_TEAL))
        row += 1

        # Phase dots
        x = 1
        for idx, phase in enumerate(self.profile.phases):
            if x >= w - 6:
                break
            if idx < self.phase_idx:
                tok = "\u2713 {}".format(phase.name)
                a = self._cp(self._C_GREEN)
            elif idx == self.phase_idx:
                tok = "\u25cf {}".format(phase.name)
                a = self._cp(self._C_CYAN, curses.A_BOLD)
            else:
                tok = "\u25cb {}".format(phase.name)
                a = self._cp(self._C_SLATE)
            tok = self._trunc(tok, 16)
            self._s(row, x, tok, a)
            x += len(tok) + 2
        row += 1

        # ── System strip ──
        # One-line status bar: motherboard fans + event counters. All PSU
        # telemetry (Vin, per-rail V/I/W, Pin/Pout/Eff, temps, fan) lives
        # in the PSU panel below — this strip used to duplicate it.
        self._hline(row, w, heavy=True)
        row += 1

        # Three separate fan readings — cpu (mobo CPU header), sys (mobo
        # SYS header), psu (corsair-psu hwmon, sourced independently from
        # the motherboard Super I/O). Previously "cpu" was silently picking
        # up the PSU fan on benches where the motherboard wasn't reporting
        # any headers; telemetry.py now excludes the PSU fan value from
        # the cpu/sys extraction, so this strip reflects physical reality.
        fc, fca = self._fmt_val("fan_cpu_rpm", "{:.0f}")
        fs, fsa = self._fmt_val("fan_sys_rpm", "{:.0f}")
        fp, fpa = self._fmt_val("psu_fan_rpm", "{:.0f}")
        cx = 1
        self._s(row, cx, "Fan", self._cp(self._C_SLATE)); cx += 4
        self._s(row, cx, "cpu", self._cp(self._C_SLATE)); cx += 4
        self._s(row, cx, fc, fca); cx += len(fc) + 2
        self._s(row, cx, "sys", self._cp(self._C_SLATE)); cx += 4
        self._s(row, cx, fs, fsa); cx += len(fs) + 2
        self._s(row, cx, "psu", self._cp(self._C_SLATE)); cx += 4
        self._s(row, cx, fp, fpa); cx += len(fp) + 1
        self._s(row, cx, "rpm", self._cp(self._C_SLATE))
        # Events summary flush right
        ev = "{}C {}W {}I".format(
            self.event_counts["crit"], self.event_counts["warn"],
            self.event_counts["info"])
        self._s(row, max(0, w - len(ev) - 1), ev, badge_attr)
        row += 1

        self._hline(row, w, heavy=True)
        row += 1

        # ── DUT grid ──
        # 2-column layout above w=100, single-column below. The column
        # layout is declared in DUT_GRID_COLUMNS so the two columns can
        # hold unequal numbers of panels (CPU+GPU on the left; PSU+RAM+SSD
        # on the right). All metrics always render at full height — the
        # scrollable pad absorbs overflow, so operators see every row by
        # scrolling rather than losing metrics to a window-size fit.
        two_col = w >= 100
        gap = 2
        pw = ((w - gap) // 2) if two_col else w
        per_panel_overhead = 3  # top border + blank + bottom border

        panels_by_key: Dict[str, dict] = {
            str(p["key"]): p for p in self.DUT_PANELS
        }
        if two_col:
            columns_keys: List[List[str]] = [list(col) for col in self.DUT_GRID_COLUMNS]
        else:
            # Single-column view: flatten left column then right column so
            # the reading order stays CPU → GPU → PSU → RAM → SSD.
            flat: List[str] = []
            for col in self.DUT_GRID_COLUMNS:
                flat.extend(col)
            columns_keys = [flat]

        def _column_height(keys: List[str]) -> int:
            total = 0
            for k in keys:
                panel = panels_by_key.get(k)
                if panel is None:
                    continue
                total += len(panel["metrics"]) + per_panel_overhead
            return total

        tallest_column = max((_column_height(keys) for keys in columns_keys), default=0)

        for col_idx, keys in enumerate(columns_keys):
            px = col_idx * (pw + gap) if two_col else 0
            cur = row
            for k in keys:
                panel = panels_by_key.get(k)
                if panel is None:
                    continue
                metric_cap = len(panel["metrics"])
                self._draw_panel(cur, px, pw, panel, self._pad_rows,
                                 max_metrics=metric_cap)
                cur += metric_cap + per_panel_overhead

        # Advance the outer row cursor by the tallest column so the footer
        # lands below all rendered panels.
        row += tallest_column

        # ── Footer ──
        self._hline(row, w)
        row += 1
        self._s(row, 1, "P/Space", self._cp(self._C_TEAL, curses.A_BOLD))
        self._s(row, 9, "pause", self._cp(self._C_SLATE))
        self._s(row, 16, "N", self._cp(self._C_TEAL, curses.A_BOLD))
        self._s(row, 18, "skip phase", self._cp(self._C_SLATE))
        self._s(row, 30, "Q", self._cp(self._C_AMBER, curses.A_BOLD))
        self._s(row, 32, "abort", self._cp(self._C_SLATE))
        self._s(row, 40, "\u2191\u2193 PgUp PgDn", self._cp(self._C_TEAL, curses.A_BOLD))
        self._s(row, 55, "scroll", self._cp(self._C_SLATE))
        row += 1

        # ── Event feed ──
        # Always render the full tail — scrollable pad shows everything,
        # operator reveals older lines by scrolling down.
        recent = list(self.log_lines)
        if recent:
            label = " EVENT FEED "
            self._hline(row, w)
            self._s(row, 1, label, self._cp(self._C_STEEL, curses.A_BOLD))
            row += 1
        for ts, sev, pn, msg in recent:
            if row >= self._pad_rows:
                break
            if sev == "crit":
                a = self._cp(self._C_RED, curses.A_BOLD)
            elif sev == "warn":
                a = self._cp(self._C_AMBER, curses.A_BOLD)
            elif sev == "info":
                a = self._cp(self._C_STEEL)
            else:
                a = 0
            line = "{} {:<4} {:<10} {}".format(ts, sev.upper(), self._trunc(pn, 10), msg)
            self._s(row, 0, self._trunc(line, w), a)
            row += 1

        self._content_height = row
