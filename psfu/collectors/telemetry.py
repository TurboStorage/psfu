import json
import math
import os
import platform
import re
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
from psfu.common import command_exists, run_capture
from psfu.hardware import nvme_cli_binary_path

# Rolling window size (samples) for rail ripple σ. At 1 Hz sampling this
# is a 10 s window — long enough to smooth out legitimate single-sample
# VRM transients but short enough that cap-aging ripple growth is
# visible within a phase. First ``_RAIL_RIPPLE_WINDOW - 1`` ticks
# return None because a σ over fewer than 2 samples is meaningless.
_RAIL_RIPPLE_WINDOW = 10

# Lower critical thresholds per rail for dropout counting. A dropout is
# a single sample below lcrit — the cumulative count is the load-bearing
# number (sub-second brownouts too fast for the 1 Hz rail mean to show).
# Tightened slightly vs ATX lcrit ranges so a healthy PSU under load
# never ticks the counter; a unit that ticks even once mid-run is a
# suspect for the multimeter loop.
_RAIL_LCRIT = {
    "12v": 11.40,  # ATX spec floor; HX-series typically holds >11.9 V
    "5v":  4.75,   # ATX spec floor
    "33v": 3.135,  # ATX spec floor (3.3 V -5%)
}

# NVMe SMART temp-time probe cadence — every N ticks. ``nvme smart-log``
# forks one subprocess per drive; the counters are cumulative so probing
# every tick wastes ~100-200 ms of per-tick budget for no extra signal.
# 30 ticks ≈ 30 s at 1 Hz, well inside a typical phase duration so the
# value refreshes at least once per step phase.
_NVME_SMART_TICK_INTERVAL = 30


# NVML throttle reason bits — split into separate telemetry columns so
# the cause is eye-scannable in the TUI and analyzable post-run. The
# aggregate mask logic for ``gpu_throttle_active`` remains in collect().
_THROTTLE_HW_SLOWDOWN = 0x08
_THROTTLE_SW_POWER_CAP = 0x04
_THROTTLE_HW_THERMAL = 0x40
_THROTTLE_HW_POWER_BRAKE = 0x80

# Optional pynvml — falls back to nvidia-smi subprocess if unavailable.
try:
    import pynvml
    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False


def _max_or_none(values: List[Optional[float]]) -> Optional[float]:
    cleaned = [v for v in values if v is not None]
    return max(cleaned) if cleaned else None


def _sum_or_none(values: List[Optional[float]]) -> Optional[float]:
    cleaned = [v for v in values if v is not None]
    return round(sum(cleaned), 2) if cleaned else None


def _min_int_or_none(values: List) -> Optional[int]:
    cleaned = [int(v) for v in values if v is not None]
    return min(cleaned) if cleaned else None


class TelemetryCollector:
    def __init__(self) -> None:
        self.available = {
            "sensors":    command_exists("sensors"),
            "nvidia-smi": command_exists("nvidia-smi"),
            "ipmitool":   command_exists("ipmitool"),
            "dmidecode":  command_exists("dmidecode"),
        }
        self._last_cpu_stat = self._read_proc_stat()
        # Parallel snapshot for iowait %: (total_jiffies, iowait_jiffies).
        # Updated inside _cpu_iowait_pct after each read. Separate from
        # _last_cpu_stat because _cpu_load_pct consumes that state first.
        self._last_iowait_snapshot: Optional[Tuple[int, int]] = None
        self._last_diskstats = self._read_diskstats()
        self._topology_counts = self._read_bus_topology_counts()

        # Static RAM hardware context — probed once at init, mirrored into
        # every sample. Total comes from /proc/meminfo (always available);
        # DIMM speed needs dmidecode (root). Values may be None on benches
        # without dmidecode or without sudo.
        self._mem_total_gb: Optional[float] = self._probe_mem_total_gb()
        self._mem_speed_mts: Optional[int] = self._probe_mem_speed_mts()

        self._rapl_paths: List[Path] = self._find_rapl_package_paths()
        self._rapl_lasts: Dict[int, Tuple[float, float]] = {}

        self.last_gpu_visible: bool = False
        self.last_hw_throttle: bool = False
        self.last_pcie_states: List[Dict[str, Optional[int]]] = []
        self.last_gpu_smi_err: bool = False

        # pynvml state — initialized lazily on first collect.
        self._nvml_initialized: bool = False
        self._nvml_handles: List[object] = []
        self._use_nvml: bool = _HAS_PYNVML
        self._nvml_failed: bool = False  # True after init failure; stops retrying
        self.last_throttle_reasons: int = 0

        # --- v7 rolling / cumulative state ---
        # Rail history for ripple (σ) computation. Bounded deque of the
        # last N samples per rail — oldest entries drop off when new ones
        # arrive, so memory stays O(window).
        self._rail_history: Deque[Dict[str, Optional[float]]] = deque(
            maxlen=_RAIL_RIPPLE_WINDOW,
        )
        # Cumulative dropout counts per rail (sub-second brownouts — each
        # sample below lcrit ticks the counter by 1). Starts at 0 and only
        # ever grows; the consumer diffs against a phase-start baseline
        # the same way pcie_aer_corr_total works.
        self._rail_dropouts: Dict[str, int] = {"12v": 0, "5v": 0, "33v": 0}

        # /proc/stat extended snapshot for ctx/intr/softirq rates. Parallel
        # to _last_cpu_stat (used by _cpu_load_pct) but holds the raw
        # cumulative counters + monotonic timestamp so rates can be
        # computed without another /proc/stat read.
        self._last_proc_stat_ext: Optional[Tuple[float, int, int, int]] = None

        # /proc/vmstat pgmajfault snapshot for rate computation.
        self._last_vmstat: Optional[Tuple[float, int]] = None

        # /sys/block/nvme*/stat snapshot — per-device tuple of
        # (monotonic_ts, sectors_read, sectors_written, inflight). We sum
        # across devices to get bench-wide read/write MB/s and queue depth.
        self._last_block_stat: Optional[Dict[str, Tuple[float, int, int, int]]] = None

        # NVMe SMART temp-time cache — ``nvme smart-log -o json`` forks one
        # subprocess per drive (~100-200 ms each on WSL). The values are
        # cumulative drive-lifetime counters, so probing every tick costs
        # render latency for no extra signal. Probe every
        # ``_NVME_SMART_TICK_INTERVAL`` ticks and replay the cached tuple
        # between probes; on the first tick after init the cache is empty
        # so the probe runs immediately.
        self._nvme_smart_tick: int = 0
        self._nvme_smart_cached: Tuple[Optional[int], Optional[int]] = (None, None)

    def close(self) -> None:
        """Release NVML if we initialized it. Paired with ``_ensure_nvml``;
        NVML spec requires every nvmlInit to be matched with nvmlShutdown.
        Safe to call multiple times and safe to call when NVML was never
        initialized."""
        if self._nvml_initialized and _HAS_PYNVML:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False
            self._nvml_handles = []

    def collect(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "cpu_load_pct":      self._cpu_load_pct(),
            "loadavg_1m":        self._loadavg(),
            "mem_used_pct":      self._mem_used_pct(),
            "mem_total_gb":      self._mem_total_gb,
            "mem_speed_mts":     self._mem_speed_mts,
            "disk_busy_pct":     self._disk_busy_pct(),
            "ssd_temp_c":        None,  # populated below from _read_nvme_temps()
            "hba_count":         self._topology_counts.get("hba_count"),
            "i2c_adapter_count": self._topology_counts.get("i2c_adapter_count"),
            "i3c_bus_count":     self._topology_counts.get("i3c_bus_count"),
            "cpu_temp_c":        None,
            "cpu_pkg_watt":      None,
            "cpu_avg_mhz":       None,
            "cpu_busy_pct_turbo": None,
            "cpu_core_watt":     None,
            "cpu_pkg_watt_turbo": None,
            "gpu_temp_c":        None,
            "gpu_vram_temp_c":   None,
            "gpu_power_w":       None,
            "gpu_power_total_w": None,
            "gpu_util_pct":      None,
            "gpu_clock_sm_mhz":  None,
            "gpu_clock_mem_mhz": None,
            "gpu_vram_used_mib": None,
            "gpu_vram_total_mib": None,
            "gpu_fan_pct":       None,
            "gpu_ecc_ce":        None,
            "gpu_ecc_ue":        None,
            "gpu_throttle_active": None,
            "gpu_retired_pages": None,
            "gpu_power_viol_us": None,
            "gpu_thermal_viol_us": None,
            "pcie_link_gen":     None,
            "pcie_link_width":   None,
            "pcie_tx_kbps":      None,
            "pcie_rx_kbps":      None,
            "rail_12v":          None,
            "rail_5v":           None,
            "rail_33v":          None,
            "fan_cpu_rpm":       None,
            "fan_sys_rpm":       None,
            # --- Corsair PSU (corsair-psu hwmon driver) ---
            # Populated when an RMi/HXi/AXi-series PSU is connected via the
            # iCue USB-header → USB-C cable and corsair_psu kernel module is
            # loaded. All fields stay None on benches without a compatible
            # Corsair PSU or without the driver loaded.
            "psu_vin_v":         None,  # AC input voltage
            "psu_i_in_a":        None,  # input current (AC side)
            "psu_i_12v_a":       None,  # +12V rail current
            "psu_i_5v_a":        None,  # +5V rail current
            "psu_i_33v_a":       None,  # +3.3V rail current
            "psu_p_in_w":        None,  # total input power (AC side)
            "psu_p_12v_w":       None,  # +12V rail output power
            "psu_p_5v_w":        None,  # +5V rail output power
            "psu_p_33v_w":       None,  # +3.3V rail output power
            "psu_p_out_w":       None,  # sum of rail output powers
            "psu_efficiency_pct": None, # p_out / p_in * 100
            "psu_temp_c":        None,  # primary temp (VRM)
            "psu_temp2_c":       None,  # secondary temp (ambient)
            "psu_fan_rpm":       None,
            # Platform thermal + link-health context (v5 schema).
            "mobo_vrm_temp_c":   None,
            "mobo_case_temp_c":  None,
            "cpu_ccd_max_c":     None,
            "cpu_ccd_spread_c":  None,
            "ssd_temp_ctrl_c":   None,
            "ssd_temp_nand_c":   None,
            "pcie_aer_corr_total": None,
            # NIC + PSI + iowait + GPU link degradation (v6 schema).
            "nic_mac_temp_c":          None,
            "nic_phy_temp_c":          None,
            "psi_cpu_some_avg10":      None,
            "psi_mem_some_avg10":      None,
            "psi_io_some_avg10":       None,
            "nic_rx_crc_errors_total": None,
            "nic_rx_errors_total":     None,
            "nic_rx_dropped_total":    None,
            "gpu_pcie_link_gen_max":   None,
            "cpu_iowait_pct":          None,
            # --- v7 Tier 1 (PSU-diagnostic value) ---
            "rail_12v_ripple_mv":        None,
            "rail_5v_ripple_mv":         None,
            "rail_33v_ripple_mv":        None,
            "rail_12v_dropouts":         None,
            "rail_5v_dropouts":          None,
            "rail_33v_dropouts":         None,
            "pcie_aer_uncorr_total":     None,
            "gpu_throttle_hw_slowdown":  None,
            "gpu_throttle_hw_thermal":   None,
            "gpu_throttle_hw_power_brake": None,
            "gpu_throttle_sw_power_cap": None,
            "ssd_warn_temp_time_s":      None,
            "ssd_crit_temp_time_s":      None,
            "nic_link_speed_mbps":       None,
            "nic_link_up":               None,
            # --- v7 Tier 2 (platform-health coverage) ---
            "cpu_freq_spread_mhz":       None,
            "cpu_c6_pct":                None,
            "cpu_ctx_switches_per_s":    None,
            "cpu_interrupts_per_s":      None,
            "cpu_softirq_per_s":         None,
            "mem_pgmajfault_per_s":      None,
            "mem_dirty_kb":              None,
            "mem_writeback_kb":          None,
            "ssd_read_mbps":             None,
            "ssd_write_mbps":            None,
            "ssd_queue_depth":           None,
            "mobo_chipset_temp_c":       None,
            "thermal_zone_margin_c":     None,
            # --- v7 memory bandwidth (lower cadence, subprocess sampler) ---
            "mem_bw_read_mbps":          None,
            "mem_bw_write_mbps":         None,
            "source_json":       json.dumps({}, sort_keys=True),
        }

        source: Dict[str, object] = {}
        source["topology"] = dict(self._topology_counts)

        payload["cpu_pkg_watt"] = self._read_rapl_pkg_watt()

        # Read Corsair FIRST so the sensor-fan extraction below can exclude
        # the PSU fan value. Without this exclusion the PSU fan shows up as
        # the first/only fan on benches where the motherboard Super I/O
        # isn't reporting any headers (observed on TRX50 AI TOP — only fan
        # in ``sensors -u`` is the corsair-psu hwmon fan), which then
        # renders as "CPU fan" in the TUI.
        corsair = self._read_corsair_psu()

        if self.available["sensors"]:
            sensors_data = self._read_sensors()
            source["sensors"] = sensors_data
            payload["cpu_temp_c"] = self._extract_first_number(
                sensors_data, ["package", "tdie", "tctl", "cpu_temp", "temp1"],
            )
            payload["rail_12v"] = self._extract_first_number(sensors_data, ["12v"])
            payload["rail_5v"]  = self._extract_first_number(sensors_data, ["5v"])
            payload["rail_33v"] = self._extract_first_number(sensors_data, ["3.3", "33v", "3v"])
            psu_fan = corsair.get("fan_rpm") if corsair else None
            payload["fan_cpu_rpm"], payload["fan_sys_rpm"] = self._extract_fan_rpms(
                sensors_data,
                exclude_values={psu_fan} if psu_fan is not None else None,
            )
            # Board-level thermal context. ``vrm temp`` and ``case temp``
            # come from the motherboard Super I/O on boards that expose
            # them (TRX50 AI TOP does). Both stay None on boards that
            # don't — no verdict rule depends on either yet.
            payload["mobo_vrm_temp_c"]  = self._extract_first_number(
                sensors_data, ["vrm temp", "vrm_temp"],
            )
            payload["mobo_case_temp_c"] = self._extract_first_number(
                sensors_data, ["case temp", "case_temp"],
            )
            # Per-CCD CPU temps — aggregate to max + spread so one column
            # pair works for any CPU die count (Zen 4/5 Threadripper has
            # up to 8 CCDs). Spread > 20 C under uniform load is a real
            # signal; raw per-CCD values are over-granular for a 1 Hz TUI.
            ccd_max, ccd_spread = self._extract_ccd_stats(sensors_data)
            payload["cpu_ccd_max_c"]    = ccd_max
            payload["cpu_ccd_spread_c"] = ccd_spread
            # Integrated NIC thermal context. Some boards (TRX50 AI TOP)
            # expose MAC + PHY die temps via the Super I/O. Cooking these
            # under sustained network load is a quiet RMA path — link
            # stays up but CRC errors climb and eventually the port dies.
            payload["nic_mac_temp_c"] = self._extract_first_number(
                sensors_data, ["mac temperature", "mac_temp", "mac temp"],
            )
            payload["nic_phy_temp_c"] = self._extract_first_number(
                sensors_data, ["phy temperature", "phy_temp", "phy temp"],
            )
            # Chipset temp — TRX50 AI TOP's nct6796 exposes this under
            # ``chipset`` or similar Super I/O labels. Dashed out on
            # boards without a chipset sensor. Not a verdict input yet;
            # growth during soak suggests airflow degradation in the
            # lower chamber.
            payload["mobo_chipset_temp_c"] = self._extract_first_number(
                sensors_data, ["chipset", "pch temp", "pch_temp", "tsi"],
            )

        # NVMe split sensor temps. Falls back to None cleanly on boards
        # or kernels that don't expose hwmon nvme devices.
        composite, ctrl, nand = self._read_nvme_temps()
        payload["ssd_temp_c"]      = composite
        payload["ssd_temp_ctrl_c"] = ctrl
        payload["ssd_temp_nand_c"] = nand

        # PCIe correctable-AER counter snapshot. Absolute running total
        # across all PCI devices — the consumer diffs against a baseline
        # (future verdict rule) to detect growth during stress phases.
        payload["pcie_aer_corr_total"] = self._read_pcie_aer_corr_total()

        # Kernel PSI (Pressure Stall Information). The ``some avg10``
        # line is "% of the last 10 s that at least one task stalled on
        # this resource." Zero on healthy benches; non-zero during
        # legitimate stress is fine — the signal is non-zero memory PSI
        # during a PSU stress test where memtester alone tested clean,
        # which points at a failing rail feeding the DIMMs.
        psi_cpu, psi_mem, psi_io = self._read_psi_triple()
        payload["psi_cpu_some_avg10"] = psi_cpu
        payload["psi_mem_some_avg10"] = psi_mem
        payload["psi_io_some_avg10"]  = psi_io

        # NIC link-error totals summed across real interfaces (loopback
        # excluded). Absolute counters — consumer diffs vs baseline.
        # CRC error growth during soak = cable / SFP / port issue.
        nic_crc, nic_err, nic_drop = self._read_nic_error_totals()
        payload["nic_rx_crc_errors_total"] = nic_crc
        payload["nic_rx_errors_total"]     = nic_err
        payload["nic_rx_dropped_total"]    = nic_drop

        # CPU iowait % from /proc/stat. Complements disk_busy_pct:
        # iowait-high + busy-low = failing storage controller (CPUs
        # waiting but drive barely moving); iowait-high + busy-high =
        # saturated-but-healthy. On first collect we have no previous
        # snapshot so this reads None, then ticks valid from t+1.
        payload["cpu_iowait_pct"] = self._cpu_iowait_pct()

        # === v7: rail ripple + dropouts (Tier 1, PSU diagnostic) ===
        # Feed the current rail reading into the rolling window AFTER
        # all rail fallbacks (sensors → ipmitool → corsair) have had
        # their chance to populate. Ripple is computed over whatever
        # reading actually landed in the column. Dropouts are counted
        # against lcrit using the same post-fallback value.
        # We populate these AFTER the corsair block below; leave a
        # placeholder here and move the call to just before the NVML
        # section so it sees the finalized rail values. Actual insertion
        # happens at the bottom of collect() — see the v7-finalization
        # block below.

        # === v7: PCIe AER uncorrectable (Tier 1, PSU-induced link corruption) ===
        payload["pcie_aer_uncorr_total"] = self._read_pcie_aer_uncorr_total()

        # === v7: NIC link state (Tier 1, PHY/port health) ===
        link_speed, link_up = self._read_nic_link_state()
        payload["nic_link_speed_mbps"] = link_speed
        payload["nic_link_up"]         = link_up

        # === v7: NVMe SMART temp-time pages (Tier 1, thermal throttle dwell) ===
        # Cumulative drive-lifetime counters — probe every N ticks and
        # replay the cached tuple between probes to cut per-tick fork cost.
        if self._nvme_smart_tick <= 0:
            self._nvme_smart_cached = self._read_nvme_smart_temp_times()
            self._nvme_smart_tick = _NVME_SMART_TICK_INTERVAL
        else:
            self._nvme_smart_tick -= 1
        warn_s, crit_s = self._nvme_smart_cached
        payload["ssd_warn_temp_time_s"] = warn_s
        payload["ssd_crit_temp_time_s"] = crit_s

        # === v7: /proc/stat extended deltas (Tier 2) ===
        ctx_rate, intr_rate, softirq_rate = self._compute_proc_stat_ext_rates()
        payload["cpu_ctx_switches_per_s"] = ctx_rate
        payload["cpu_interrupts_per_s"]   = intr_rate
        payload["cpu_softirq_per_s"]      = softirq_rate

        # === v7: /proc/vmstat pgmajfault rate (Tier 2) ===
        payload["mem_pgmajfault_per_s"] = self._compute_pgmajfault_rate()

        # === v7: /proc/meminfo Dirty + Writeback (Tier 2) ===
        dirty_kb, wb_kb = self._read_meminfo_dirty_writeback()
        payload["mem_dirty_kb"]     = dirty_kb
        payload["mem_writeback_kb"] = wb_kb

        # === v7: /sys/block/nvme*/stat throughput + queue depth (Tier 2) ===
        read_mbps, write_mbps, qdep = self._compute_block_stat_metrics()
        payload["ssd_read_mbps"]    = read_mbps
        payload["ssd_write_mbps"]   = write_mbps
        payload["ssd_queue_depth"]  = qdep

        # === v7: thermal zone margin walker (Tier 2) ===
        payload["thermal_zone_margin_c"] = self._read_thermal_zone_margin()

        if self.available["ipmitool"]:
            ipmi = self._read_ipmitool_rails()
            source["ipmitool"] = ipmi
            if payload["rail_12v"] is None:
                payload["rail_12v"] = ipmi.get("12v")
            if payload["rail_5v"] is None:
                payload["rail_5v"] = ipmi.get("5v")
            if payload["fan_cpu_rpm"] is None:
                payload["fan_cpu_rpm"] = ipmi.get("fan_cpu_rpm")
            if payload["fan_sys_rpm"] is None:
                payload["fan_sys_rpm"] = ipmi.get("fan_sys_rpm")
            if payload["mobo_vrm_temp_c"] is None:
                payload["mobo_vrm_temp_c"] = ipmi.get("vrm_temp")
            if payload["mobo_case_temp_c"] is None:
                payload["mobo_case_temp_c"] = ipmi.get("case_temp")
            if payload["nic_mac_temp_c"] is None:
                payload["nic_mac_temp_c"] = ipmi.get("lan_temp")
            # On dual-socket boards sensors -u setdefault drops
            # the second CPU's Tctl. Use IPMI to surface both and store the max
            # so cpu_temp_c always reflects the hottest package.
            ipmi_cpu_temps = [
                t for t in (ipmi.get("cpu0_temp"), ipmi.get("cpu1_temp"))
                if t is not None
            ]
            if ipmi_cpu_temps:
                ipmi_max = max(ipmi_cpu_temps)
                cur = payload.get("cpu_temp_c")
                payload["cpu_temp_c"] = ipmi_max if cur is None else max(cur, ipmi_max)

        if corsair:
            source["corsair-psu"] = corsair
            # Rails: prefer on-board sensors (already populated above) but fall
            # back to the Corsair driver. The Corsair telemetry is the more
            # authoritative reading when available — same PSU that's being
            # tested is reporting its own rails — but we don't overwrite
            # on-board sensors to keep the rail_*V columns comparable across
            # historical runs that only had motherboard readings.
            if payload["rail_12v"] is None:
                payload["rail_12v"] = corsair.get("12v")
            if payload["rail_5v"] is None:
                payload["rail_5v"] = corsair.get("5v")
            if payload["rail_33v"] is None:
                payload["rail_33v"] = corsair.get("33v")
            # PSU-specific fields — always sourced from the Corsair driver.
            payload["psu_vin_v"]           = corsair.get("vin_v")
            payload["psu_i_in_a"]          = corsair.get("i_in_a")
            payload["psu_i_12v_a"]         = corsair.get("i_12v_a")
            payload["psu_i_5v_a"]          = corsair.get("i_5v_a")
            payload["psu_i_33v_a"]         = corsair.get("i_33v_a")
            payload["psu_p_in_w"]          = corsair.get("p_in_w")
            payload["psu_p_12v_w"]         = corsair.get("p_12v_w")
            payload["psu_p_5v_w"]          = corsair.get("p_5v_w")
            payload["psu_p_33v_w"]         = corsair.get("p_33v_w")
            payload["psu_p_out_w"]         = corsair.get("p_out_w")
            payload["psu_efficiency_pct"]  = corsair.get("efficiency_pct")
            payload["psu_temp_c"]          = corsair.get("temp1_c")
            payload["psu_temp2_c"]         = corsair.get("temp2_c")
            payload["psu_fan_rpm"]         = corsair.get("fan_rpm")

        if self.available["nvidia-smi"]:
            # Try pynvml first (richer data, no subprocess overhead). Fall
            # back to nvidia-smi subprocess if NVML init fails or pynvml is
            # not installed. The _nvml_failed flag prevents retrying after a
            # permanent init failure.
            used_nvml = False
            if self._use_nvml and self._init_nvml():
                try:
                    gpus, gpu_smi_err = self._read_gpu_nvml()
                    used_nvml = True
                except Exception:
                    gpus, gpu_smi_err = self._read_gpu()
            else:
                gpus, gpu_smi_err = self._read_gpu()
            self.last_gpu_smi_err = gpu_smi_err
            source["nvidia-smi"] = gpus
            if used_nvml:
                source["gpu_backend"] = "pynvml"
            if gpus:
                payload["gpu_temp_c"]       = _max_or_none([g.get("temperature.gpu") for g in gpus])
                payload["gpu_vram_temp_c"]  = _max_or_none([g.get("temperature.memory") for g in gpus])
                payload["gpu_power_w"]      = gpus[0].get("power.draw")
                payload["gpu_power_total_w"] = _sum_or_none([g.get("power.draw") for g in gpus])
                payload["gpu_util_pct"]     = _max_or_none([g.get("utilization.gpu") for g in gpus])
                payload["gpu_clock_sm_mhz"] = _min_int_or_none([g.get("clocks_sm_mhz") for g in gpus])
                payload["gpu_clock_mem_mhz"] = _min_int_or_none([g.get("clocks_mem_mhz") for g in gpus])
                payload["gpu_vram_used_mib"] = _sum_or_none([g.get("vram_used_mib") for g in gpus])
                payload["gpu_vram_total_mib"] = _sum_or_none([g.get("vram_total_mib") for g in gpus])
                payload["gpu_fan_pct"]      = _max_or_none([g.get("fan_pct") for g in gpus])
                payload["gpu_ecc_ce"]       = _sum_or_none([g.get("ecc_ce") for g in gpus])
                payload["gpu_ecc_ue"]       = _sum_or_none([g.get("ecc_ue") for g in gpus])
                payload["pcie_link_gen"]    = _min_int_or_none([g.get("pcie_link_gen") for g in gpus])
                payload["pcie_link_width"]  = _min_int_or_none([g.get("pcie_link_width") for g in gpus])
                # Use min advertised max across multi-GPU systems (rare).
                # If one GPU can do Gen5 and one only Gen4, the bench is
                # Gen4 and we should compare against Gen4.
                payload["gpu_pcie_link_gen_max"] = _min_int_or_none([g.get("pcie_link_gen_max") for g in gpus])
                # pynvml-only fields — None when using subprocess fallback.
                payload["pcie_tx_kbps"]      = _sum_or_none([g.get("pcie_tx_kbps") for g in gpus])
                payload["pcie_rx_kbps"]      = _sum_or_none([g.get("pcie_rx_kbps") for g in gpus])
                payload["gpu_retired_pages"] = _sum_or_none([g.get("retired_pages") for g in gpus])
                payload["gpu_power_viol_us"] = _max_or_none([g.get("power_viol_us") for g in gpus])
                payload["gpu_thermal_viol_us"] = _max_or_none([g.get("thermal_viol_us") for g in gpus])
            self.last_gpu_visible = bool(gpus)
            # Parse throttle reasons bitmask. NVML encoding (from
            # nvml.h ``nvmlClocksThrottleReasons``):
            #   0x01  GPU_IDLE              — idle, no load (benign)
            #   0x02  APPLICATIONS_CLOCKS   — nvidia-smi -ac set (benign)
            #   0x04  SW_POWER_CAP          — at TDP limit (EXPECTED under
            #         gpu_burn — does NOT indicate a problem)
            #   0x08  HW_SLOWDOWN           — umbrella HW throttle
            #   0x20  SW_THERMAL_SLOWDOWN   — driver-level thermal limit
            #   0x40  HW_THERMAL_SLOWDOWN   — hardware thermal trip
            #   0x80  HW_POWER_BRAKE        — external power brake
            #         (voltage-driven — prime PSU brownout signal)
            #
            # ``gpu_throttle_active`` (the sample column surfaced in the
            # TUI + feeding any future verdict rule) flags HW-class throttles
            # only (HW_SLOWDOWN | HW_THERMAL | HW_POWER_BRAKE = 0xC8).
            # SW_POWER_CAP is intentionally excluded so a GPU running at its
            # rated TDP under stress does NOT show as throttling — that was
            # the "Thrtl 1" always-on noise on the TRX50+5090 bench.
            self.last_throttle_reasons = 0
            for g in gpus:
                raw = g.get("throttle_reasons_active", "")
                if isinstance(raw, str) and raw.startswith("0x"):
                    try:
                        self.last_throttle_reasons |= int(raw, 16)
                    except ValueError:
                        pass
            _HW_THROTTLE_MASK = 0x08 | 0x40 | 0x80
            self.last_hw_throttle = bool(self.last_throttle_reasons & 0x80)
            payload["gpu_throttle_active"] = (
                1 if (self.last_throttle_reasons & _HW_THROTTLE_MASK) else 0
            )
            # v7: split the bitmask into eye-scannable per-cause columns.
            # SW_POWER_CAP is expected and benign under gpu-burn (5090 hits
            # 575 W TDP instantly); the other three are real HW protection
            # events. Keeping them as 0/1 columns lets the TUI render
            # distinct rows and keeps post-run grep/analysis simple.
            payload["gpu_throttle_hw_slowdown"]    = 1 if (self.last_throttle_reasons & _THROTTLE_HW_SLOWDOWN) else 0
            payload["gpu_throttle_hw_thermal"]     = 1 if (self.last_throttle_reasons & _THROTTLE_HW_THERMAL) else 0
            payload["gpu_throttle_hw_power_brake"] = 1 if (self.last_throttle_reasons & _THROTTLE_HW_POWER_BRAKE) else 0
            payload["gpu_throttle_sw_power_cap"]   = 1 if (self.last_throttle_reasons & _THROTTLE_SW_POWER_CAP) else 0
            self.last_pcie_states = [
                {"gen": g.get("pcie_link_gen"), "width": g.get("pcie_link_width")}
                for g in gpus
            ]

        # Fallback: seed turbostat-keyed fields from RAPL/proc so TUI panels
        # display something useful when turbostat isn't running (no sudo).
        # merge_turbostat() overwrites these if turbostat is active.
        if payload["cpu_pkg_watt_turbo"] is None and payload["cpu_pkg_watt"] is not None:
            payload["cpu_pkg_watt_turbo"] = payload["cpu_pkg_watt"]

        # === v7: rail ripple + dropouts — run LAST so all rail fallbacks
        # (sensors → ipmitool → corsair) have populated the values being
        # fed into the window and dropout-counter.
        self._update_rail_history_and_dropouts(
            payload.get("rail_12v"),
            payload.get("rail_5v"),
            payload.get("rail_33v"),
        )
        rip12, rip5, rip33 = self._compute_rail_ripples()
        payload["rail_12v_ripple_mv"] = rip12
        payload["rail_5v_ripple_mv"]  = rip5
        payload["rail_33v_ripple_mv"] = rip33
        payload["rail_12v_dropouts"] = self._rail_dropouts["12v"]
        payload["rail_5v_dropouts"]  = self._rail_dropouts["5v"]
        payload["rail_33v_dropouts"] = self._rail_dropouts["33v"]

        payload["source_json"] = json.dumps(source, sort_keys=True, default=str)
        return payload

    def _cpu_load_pct(self) -> Optional[float]:
        try:
            current = self._read_proc_stat()
            previous = self._last_cpu_stat
            self._last_cpu_stat = current
            if current is None or previous is None:
                return None
            total_delta = current[0] - previous[0]
            # Historical "idle" semantics include iowait — treat a CPU
            # waiting for disk as not-busy. Keeps cpu_load_pct stable.
            idle_delta = (current[1] - previous[1]) + (current[2] - previous[2])
            if total_delta <= 0:
                return None
            busy_pct = (1.0 - (idle_delta / float(total_delta))) * 100.0
            return round(max(0.0, min(100.0, busy_pct)), 2)
        except Exception:
            return None

    def _cpu_iowait_pct(self) -> Optional[float]:
        """Percent of CPU time in iowait over the interval since the
        previous /proc/stat snapshot. Complements ``disk_busy_pct``:

          iowait-high + busy-low  → failing storage controller (CPU
              stalled but drive barely moving).
          iowait-high + busy-high → saturated-but-healthy.
          iowait-low  + busy-high → drive pinned but nothing waiting
              on it (background compaction, e.g.).

        Uses the same stat snapshot cached by ``_cpu_load_pct`` —
        which is read FIRST each tick — so this method does not need
        its own state. Returns None if the previous snapshot was
        missing (first tick) or the interval was zero.
        """
        try:
            # _last_cpu_stat was just updated by _cpu_load_pct; we need
            # the previous sample which is no longer cached. Work
            # around by reading fresh and diffing: keep a parallel
            # prev-iowait snapshot.
            prev_total, prev_iow = self._last_iowait_snapshot or (None, None)
            current = self._last_cpu_stat  # already read this tick
            if current is None:
                self._last_iowait_snapshot = None
                return None
            cur_total, _cur_idle, cur_iow = current
            self._last_iowait_snapshot = (cur_total, cur_iow)
            if prev_total is None:
                return None
            total_delta = cur_total - prev_total
            iow_delta = cur_iow - prev_iow
            if total_delta <= 0:
                return None
            pct = (iow_delta / float(total_delta)) * 100.0
            return round(max(0.0, min(100.0, pct)), 2)
        except Exception:
            return None

    def _read_proc_stat(self) -> Optional[Tuple[int, int, int]]:
        """Return ``(total_jiffies, idle_jiffies, iowait_jiffies)``.

        Column order (from ``man 5 proc``, ``/proc/stat``):
        user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice.
        We keep idle and iowait separate — legacy callers (``_cpu_load_pct``)
        re-combine them, new callers (``_cpu_iowait_pct``) want iowait alone.
        """
        try:
            with open("/proc/stat", "r", encoding="utf-8") as handle:
                first = handle.readline().strip().split()
            if not first or first[0] != "cpu":
                return None
            values = [int(value) for value in first[1:]]
            total = sum(values)
            idle = values[3]
            iowait = values[4] if len(values) > 4 else 0
            return total, idle, iowait
        except Exception:
            return None

    def _loadavg(self) -> Optional[float]:
        try:
            return round(os.getloadavg()[0], 2)
        except OSError:
            return None

    def _mem_used_pct(self) -> Optional[float]:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                meminfo = {}
                for line in handle:
                    key, value = line.split(":", 1)
                    meminfo[key] = int(value.strip().split()[0])
            total = meminfo["MemTotal"]
            available = meminfo["MemAvailable"]
            used_pct = ((total - available) / float(total)) * 100.0
            return round(used_pct, 2)
        except Exception:
            return None

    def _probe_mem_total_gb(self) -> Optional[float]:
        """Installed RAM — sum of populated DIMM sizes from dmidecode, NOT
        /proc/meminfo. MemTotal is always short of the actual kit by the
        kernel image + EFI reserved regions + BIOS-locked CMA pools; a
        256 GB kit shows as 251.x through /proc/meminfo, which reads as a
        bug on a TUI that displays "installed amount". The DIMM-sum path
        gives the real sticker value. Falls back to /proc/meminfo when
        dmidecode is unavailable."""
        dimm_total = self._dmidecode_total_gb()
        if dimm_total is not None:
            return dimm_total
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split(":", 1)[1].strip().split()[0])
                        return round(kb / 1024.0 / 1024.0, 1)
        except Exception:
            return None
        return None

    def _dmidecode_total_gb(self) -> Optional[float]:
        """Sum `Size:` fields in dmidecode type-17 records — the actual
        installed DIMM capacity (e.g. 256 for 8x32GB). Unpopulated slots
        report `No Module Installed` and are skipped.

        Tries bare dmidecode first (some kernels read
        /sys/firmware/dmi/tables unprivileged), then sudo -n. Returns None
        if neither path works — caller falls back to /proc/meminfo."""
        if not self.available.get("dmidecode"):
            return None
        total = self._dmidecode_dimm_sum(["dmidecode", "-t", "memory"])
        if total is not None:
            return total
        try:
            from psfu.hardware import (
                can_run_sudo_noninteractive,
                sudo_prefix,
            )
        except ImportError:
            return None
        if not can_run_sudo_noninteractive():
            return None
        return self._dmidecode_dimm_sum(
            sudo_prefix() + ["dmidecode", "-t", "memory"]
        )

    def _dmidecode_dimm_sum(self, cmd: List[str]) -> Optional[float]:
        code, out, _ = run_capture(cmd, timeout=10)
        if code != 0 or not out:
            return None
        total_mb = 0
        saw_populated = False
        for raw in out.splitlines():
            line = raw.strip()
            if not line.startswith("Size:"):
                continue
            val = line.split(":", 1)[1].strip()
            if "No Module" in val or val.lower() in ("unknown", "not specified", ""):
                continue
            tokens = val.split()
            if len(tokens) < 2:
                continue
            try:
                n = int(tokens[0])
            except ValueError:
                continue
            unit = tokens[1].lower()
            if unit.startswith("gb") or unit.startswith("gi"):
                total_mb += n * 1024
            elif unit.startswith("mb") or unit.startswith("mi"):
                total_mb += n
            elif unit.startswith("tb") or unit.startswith("ti"):
                total_mb += n * 1024 * 1024
            else:
                continue
            saw_populated = True
        if not saw_populated:
            return None
        return round(total_mb / 1024.0, 0)

    def _probe_mem_speed_mts(self) -> Optional[int]:
        """Probe configured DIMM speed via dmidecode type-17 records.

        Returns the max non-zero "Configured Memory Speed" across populated
        slots (all DIMMs in a working kit run at the same speed, so max is
        functionally equivalent to any-populated-slot). Falls back to
        "Speed:" if "Configured Memory Speed:" is missing (older dmidecode).

        dmidecode needs root. We try bare first (some kernels allow reading
        /sys/firmware/dmi/tables unprivileged), then try sudo -n if
        available, then give up. Returns None on any failure path — mem
        speed is nice-to-have context, not a verdict input."""
        if not self.available.get("dmidecode"):
            return None
        speed = self._dmidecode_mem_speed(["dmidecode", "-t", "memory"])
        if speed is not None:
            return speed
        # Lazy import to avoid circular: hardware.py imports common, which
        # is also imported here, but can_run_sudo_noninteractive lives in
        # hardware.py. Import at call site keeps module-load clean.
        try:
            from psfu.hardware import (
                can_run_sudo_noninteractive,
                sudo_prefix,
            )
        except ImportError:
            return None
        if not can_run_sudo_noninteractive():
            return None
        return self._dmidecode_mem_speed(
            sudo_prefix() + ["dmidecode", "-t", "memory"]
        )

    def _dmidecode_mem_speed(self, cmd: List[str]) -> Optional[int]:
        code, out, _ = run_capture(cmd, timeout=10)
        if code != 0 or not out:
            return None
        configured: List[int] = []
        reported: List[int] = []
        for raw in out.splitlines():
            line = raw.strip()
            if line.startswith("Configured Memory Speed:"):
                v = self._parse_mts_value(line.split(":", 1)[1])
                if v:
                    configured.append(v)
            elif line.startswith("Speed:"):
                v = self._parse_mts_value(line.split(":", 1)[1])
                if v:
                    reported.append(v)
        if configured:
            return max(configured)
        if reported:
            return max(reported)
        return None

    @staticmethod
    def _parse_mts_value(raw: str) -> Optional[int]:
        """Extract MT/s from a dmidecode value like '6400 MT/s' or '4800 MHz'.
        Returns None for 'Unknown' / 'Not Specified' / missing units."""
        token = raw.strip().split()
        if len(token) < 2:
            return None
        try:
            val = int(token[0])
        except ValueError:
            return None
        if val <= 0:
            return None
        return val

    def _read_diskstats(self) -> Optional[Tuple[float, int]]:
        try:
            total_io_ms = 0
            with open("/proc/diskstats", "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) < 14:
                        continue
                    name = parts[2]
                    if (
                        name.startswith(("loop", "ram", "zram", "dm-", "md"))
                        or re.match(r"^nvme\d+n\d+p\d+$", name)
                        or re.match(r"^mmcblk\d+p\d+$", name)
                        or re.match(r"^sd[a-z]+\d+$", name)
                        or re.match(r"^vd[a-z]+\d+$", name)
                        or re.match(r"^xvd[a-z]+\d+$", name)
                    ):
                        continue
                    total_io_ms += int(parts[12])
            return time.monotonic(), total_io_ms
        except Exception:
            return None

    def _disk_busy_pct(self) -> Optional[float]:
        current = self._read_diskstats()
        previous = self._last_diskstats
        self._last_diskstats = current
        if current is None or previous is None:
            return None
        cur_t, cur_ms = current
        prev_t, prev_ms = previous
        dt = cur_t - prev_t
        if dt <= 0:
            return None
        busy_ms = max(0, cur_ms - prev_ms)
        busy_pct = (busy_ms / (dt * 1000.0)) * 100.0
        return round(max(0.0, min(100.0, busy_pct)), 2)

    def _read_nvme_temps(
        self,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return ``(composite, controller, nand)`` temps in °C.

        Under Linux hwmon, NVMe drives expose:
          * ``temp1_input`` — composite temperature (weighted across sensors)
          * ``temp2_input`` — Temperature Sensor 1 (typically controller)
          * ``temp3_input`` — Temperature Sensor 2 (typically NAND)

        Vendors vary on the sensor-to-physical mapping, but temp2/temp3
        as "controller / NAND" is the convention NVMe spec consumers use.
        We take the max across all detected NVMe devices per slot so a
        multi-drive bench surfaces the hottest reading for each role —
        same shape as the prior single-value ssd_temp_c.

        ``drivetemp`` (SATA) only exposes temp1_input and contributes
        only to composite.
        """
        composite: List[float] = []
        ctrl: List[float] = []
        nand: List[float] = []
        hwmon_root = Path("/sys/class/hwmon")
        if not hwmon_root.exists():
            return None, None, None
        try:
            for device in hwmon_root.iterdir():
                name_file = device / "name"
                if not name_file.exists():
                    continue
                try:
                    name = name_file.read_text().strip().lower()
                except Exception:
                    continue
                if name not in ("nvme", "drivetemp"):
                    continue
                for temp_file in sorted(device.glob("temp*_input")):
                    try:
                        milli_c = float(temp_file.read_text().strip())
                    except Exception:
                        continue
                    if milli_c <= 0:
                        continue
                    c = round(milli_c / 1000.0, 1)
                    basename = temp_file.name  # temp1_input, temp2_input, ...
                    if basename == "temp1_input":
                        composite.append(c)
                    elif basename == "temp2_input" and name == "nvme":
                        ctrl.append(c)
                    elif basename == "temp3_input" and name == "nvme":
                        nand.append(c)
                    else:
                        # temp4+ (unusual) still folds into composite so it's
                        # not silently dropped; extremely rare in practice.
                        composite.append(c)
        except Exception:
            return None, None, None
        return (
            max(composite) if composite else None,
            max(ctrl)      if ctrl      else None,
            max(nand)      if nand      else None,
        )

    def _read_pcie_aer_corr_total(self) -> Optional[int]:
        """Sum of correctable AER (Advanced Error Reporting) counters
        across every PCI device that exposes one.

        The file lives at ``/sys/bus/pci/devices/<BDF>/aer_dev_correctable``
        when the kernel was built with PCIEAER_ROOTPORT and the device
        has the AER capability. Format is key-value pairs:

            RxErr 0
            BadTLP 0
            BadDLLP 0
            ...

        We sum every numeric value on every device. A growing total during
        a stress phase is an early link-degradation signal (before the
        kernel watcher catches a fatal AER). Absolute value is meaningless
        — the consumer diffs against a phase-start baseline.

        Returns ``None`` if no readable AER file exists so the column
        cleanly stays NULL on systems without AER instead of pretending
        the count is zero.
        """
        pci_root = Path("/sys/bus/pci/devices")
        if not pci_root.exists():
            return None
        total: Optional[int] = None
        try:
            for dev in pci_root.iterdir():
                aer_file = dev / "aer_dev_correctable"
                if not aer_file.is_file():
                    continue
                try:
                    text = aer_file.read_text()
                except (PermissionError, OSError):
                    continue
                for line in text.splitlines():
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    try:
                        n = int(parts[1])
                    except ValueError:
                        continue
                    total = (total or 0) + n
        except Exception:
            return None
        return total

    def _read_psi_triple(
        self,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return ``(cpu, memory, io) some-avg10`` PSI percentages.

        /proc/pressure/{cpu,memory,io} each expose:

            some avg10=0.00 avg60=0.00 avg300=0.00 total=0
            full avg10=0.00 avg60=0.00 avg300=0.00 total=0

        We read the ``some avg10`` field — percent of the last 10 s
        that at least one task was stalled waiting on that resource.
        ``full`` (ALL non-idle tasks stalled) is a stronger signal but
        much rarer and dominated by ``some`` in almost all failure
        modes we care about.

        Absent on kernels older than 4.20 or kernels built without
        CONFIG_PSI — returns (None, None, None) cleanly.
        """
        return (
            self._read_psi_resource("cpu"),
            self._read_psi_resource("memory"),
            self._read_psi_resource("io"),
        )

    def _read_psi_resource(self, resource: str) -> Optional[float]:
        path = Path("/proc/pressure/{}".format(resource))
        if not path.exists():
            return None
        try:
            text = path.read_text()
        except (PermissionError, OSError):
            return None
        # First line starts with "some ".
        for line in text.splitlines():
            if not line.startswith("some "):
                continue
            for part in line.split()[1:]:
                if part.startswith("avg10="):
                    try:
                        return round(float(part.split("=", 1)[1]), 2)
                    except ValueError:
                        return None
            break
        return None

    def _read_nic_error_totals(
        self,
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Return ``(rx_crc_errors, rx_errors, rx_dropped)`` summed across
        every real interface in ``/sys/class/net``. ``lo`` is excluded
        — loopback errors are zero on healthy kernels and non-zero
        values there point at kernel memory pressure, not link health.

        Absolute counters — the consumer diffs vs a phase-start
        baseline to detect growth during stress.

        Tuple element is None when no interfaces contributed the
        corresponding counter (some NICs don't expose rx_crc_errors
        separately); never raises.
        """
        net_root = Path("/sys/class/net")
        if not net_root.exists():
            return None, None, None
        totals = {"rx_crc_errors": None, "rx_errors": None, "rx_dropped": None}
        try:
            for iface in net_root.iterdir():
                if iface.name == "lo":
                    continue
                # Skip down / virtual / loopback-type interfaces where
                # error counters are meaningless. operstate "down" is
                # fine — we still count "up" and "unknown" (unknown is
                # the typical state for PHYs without link detection).
                operstate = iface / "operstate"
                try:
                    if operstate.read_text().strip() == "down":
                        continue
                except OSError:
                    pass
                stats_dir = iface / "statistics"
                if not stats_dir.is_dir():
                    continue
                for counter in totals.keys():
                    f = stats_dir / counter
                    if not f.is_file():
                        continue
                    try:
                        n = int(f.read_text().strip())
                    except (ValueError, OSError):
                        continue
                    totals[counter] = (totals[counter] or 0) + n
        except Exception:
            return None, None, None
        return (
            totals["rx_crc_errors"],
            totals["rx_errors"],
            totals["rx_dropped"],
        )

    def _extract_ccd_stats(
        self,
        sensors_data: Dict[str, float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return ``(max_ccd_temp, ccd_spread)`` from per-CCD sensor values.

        AMD's k10temp driver exposes per-CCD temps as ``Tccd1``..``Tccd8``
        on Zen 4/5 when the kernel module supports it. Some CPU/kernel
        combinations (observed: a Threadripper 7000-series system on
        an older kernel) expose only ``Tctl``.

        Match order:
          1. ``tccd1``..``tccdN`` (ideal)
          2. ``ccd1``..``ccdN``   (no-``t``-prefix variant on some drivers)
          3. nothing              → (None, None)

        Deliberately NO ``Tctl`` fallback: ``cpu_temp_c`` already comes
        from ``Tctl`` on this platform, so echoing the same number as
        CCDmax would just duplicate the Temp row and imply we have
        per-die data we don't. A dashed CCDmax is more honest.

        Spread > 20 C under uniform load suggests a bad thermal
        interface on one die or a failing VRM phase feeding that CCD.
        """
        values: List[float] = []
        for key, value in sensors_data.items():
            k = key.lower()
            if not (k.startswith("tccd") or k.startswith("ccd")):
                continue
            if not isinstance(value, float):
                continue
            if value <= 0:
                continue
            suffix = k[4:] if k.startswith("tccd") else k[3:]
            if not suffix or not suffix[0].isdigit():
                continue
            values.append(value)
        if not values:
            return None, None
        return (
            round(max(values), 1),
            round(max(values) - min(values), 1),
        )

    def _read_bus_topology_counts(self) -> Dict[str, int]:
        return {
            "hba_count": self._count_hbas(),
            "i2c_adapter_count": self._count_i2c_adapters(),
            "i3c_bus_count": self._count_i3c_buses(),
        }

    def _count_hbas(self) -> int:
        pci_root = Path("/sys/bus/pci/devices")
        if not pci_root.exists():
            return 0
        count = 0
        try:
            for device in pci_root.iterdir():
                class_file = device / "class"
                if not class_file.exists():
                    continue
                try:
                    class_code = class_file.read_text().strip().lower()
                except Exception:
                    continue
                if class_code.startswith(("0x0100", "0x0104", "0x0106", "0x0107")):
                    count += 1
        except Exception:
            return 0
        return count

    def _count_i2c_adapters(self) -> int:
        root = Path("/sys/class/i2c-dev")
        if not root.exists():
            return 0
        try:
            return sum(1 for _ in root.iterdir())
        except Exception:
            return 0

    def _count_i3c_buses(self) -> int:
        root = Path("/sys/bus/i3c/devices")
        if not root.exists():
            return 0
        try:
            return sum(1 for item in root.iterdir() if item.name.startswith("i3c-"))
        except Exception:
            return 0

    def _read_sensors(self) -> Dict[str, float]:
        """Parse ``sensors -u`` output while preserving the feature-label
        hierarchy.

        ``sensors -u`` produces a structure like::

            it8689-isa-0290
            Adapter: ISA adapter
            +12V:
              in0_input: 11.916
              in0_min: 10.000
            fan1:
              fan1_input: 1200.000

        Previously this parser flattened everything onto the raw subfeature
        keys (``in0_input``, ``fan1_input``), which made downstream pattern
        matchers like ``_extract_first_number(["12v"])`` permanently miss —
        there is no ``12v`` substring in ``in0_input``. The fix is to track
        the most recent feature label block so each ``*_input`` value is
        *also* indexed under its human-readable label (``+12v``, ``fan1``).

        The raw key is still stored for unlabeled subfeatures and as a
        fallback, so lookups that already work (e.g. ``temp1``) do not
        regress.
        """
        code, stdout, _ = run_capture(["sensors", "-u"], timeout=5)
        if code != 0:
            return {}
        data: Dict[str, float] = {}
        current_label: Optional[str] = None
        for raw_line in stdout.splitlines():
            if not raw_line.strip():
                current_label = None
                continue
            is_indented = raw_line[:1] in (" ", "\t")
            line = raw_line.strip()
            if ":" not in line:
                # Chip header (e.g. "k10temp-pci-00c3") — resets label scope.
                current_label = None
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if not value:
                # Label line introducing a feature block (e.g. "+12V:", "Tctl:").
                # Only unindented labels anchor a block; indented empty-value
                # lines are malformed and ignored.
                if not is_indented:
                    current_label = key
                continue
            try:
                number = float(value.split()[0])
            except (ValueError, IndexError):
                continue
            # Always record the raw subfeature key so existing raw-name lookups
            # still work (e.g. pattern "temp1" on boards without Tctl/Tdie labels).
            data.setdefault(key, number)
            # Also index by the enclosing label — this is what lets the verdict
            # engine find 12V drift on a board whose labels are "+12V" / "+5V".
            if is_indented and current_label and key.endswith("_input"):
                data.setdefault(current_label.lower(), number)
        return data

    def _read_gpu(self) -> Tuple[List[Dict[str, object]], bool]:
        """Returns (gpu_list, has_err).

        has_err is True if nvidia-smi output contained "ERR!" in any field for
        any GPU. Numeric fields with "ERR!" are stored as None so downstream
        float/int consumers never see a string — the ERR! signal is propagated
        via the boolean return value, which the caller uses to set
        last_gpu_smi_err and fire a verdict event."""
        query = (
            "temperature.gpu,"
            "temperature.memory,"
            "power.draw,"
            "utilization.gpu,"
            "fan.speed,"
            "clocks.current.sm,"
            "clocks.current.memory,"
            "memory.used,"
            "memory.total,"
            "ecc.errors.corrected.volatile.total,"
            "ecc.errors.uncorrected.volatile.total,"
            "pcie.link.gen.current,"
            "pcie.link.width.current,"
            "clocks_throttle_reasons.active"
        )
        code, stdout, _ = run_capture(
            ["nvidia-smi", "--query-gpu={}".format(query), "--format=csv,noheader,nounits"],
            timeout=5,
        )
        if code != 0 or not stdout:
            return [], False

        float_keys = ["temperature.gpu", "temperature.memory", "power.draw", "utilization.gpu"]
        int_keys   = [
            "fan_pct",
            "clocks_sm_mhz", "clocks_mem_mhz",
            "vram_used_mib", "vram_total_mib",
            "ecc_ce", "ecc_ue",
            "pcie_link_gen", "pcie_link_width",
        ]
        str_keys   = ["throttle_reasons_active"]
        all_keys   = float_keys + int_keys + str_keys

        results: List[Dict[str, object]] = []
        has_err = False
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            # Detect ERR! in any field before type conversion.
            if any(p.upper() == "ERR!" for p in parts):
                has_err = True
            gpu: Dict[str, object] = {}
            for key, value in zip(all_keys, parts):
                if key in str_keys:
                    gpu[key] = value
                elif key in int_keys:
                    try:
                        gpu[key] = int(value)
                    except (ValueError, TypeError):
                        gpu[key] = None
                else:
                    try:
                        gpu[key] = float(value)
                    except (ValueError, TypeError):
                        gpu[key] = None
            results.append(gpu)
        return results, has_err

    def _init_nvml(self) -> bool:
        """Lazily initialize NVML and cache device handles.

        Returns True if NVML is ready to use. On failure, sets
        ``_nvml_failed`` so we never retry — fall back to subprocess
        for the rest of the run.
        """
        if self._nvml_initialized:
            return True
        if self._nvml_failed or not _HAS_PYNVML:
            return False
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._nvml_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)
            ]
            self._nvml_initialized = True
            return True
        except Exception:
            self._nvml_failed = True
            self._use_nvml = False
            return False

    def _read_gpu_nvml(self) -> Tuple[List[Dict[str, object]], bool]:
        """Read GPU telemetry via pynvml (direct NVML bindings).

        Returns the same (gpu_list, has_err) shape as ``_read_gpu()`` so
        the merge logic in ``collect()`` is unchanged. Each GPU dict includes
        extra keys that the nvidia-smi subprocess path cannot provide:
        ``retired_pages``, ``power_viol_us``, ``thermal_viol_us``,
        ``pcie_tx_kbps``, ``pcie_rx_kbps``.
        """
        results: List[Dict[str, object]] = []
        has_err = False

        for handle in self._nvml_handles:
            gpu: Dict[str, object] = {}

            # --- Temperature ---
            try:
                gpu["temperature.gpu"] = float(
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                )
            except Exception:
                gpu["temperature.gpu"] = None

            # VRAM temp via field value (NVML_FI_DEV_MEMORY_TEMP = 82).
            try:
                fv = pynvml.nvmlDeviceGetFieldValues(handle, [82])[0]
                if fv.nvmlReturn == 0:  # NVML_SUCCESS
                    gpu["temperature.memory"] = float(fv.value.uiVal)
                else:
                    gpu["temperature.memory"] = None
            except Exception:
                gpu["temperature.memory"] = None

            # --- Power (milliwatts → watts) ---
            try:
                gpu["power.draw"] = round(
                    pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 1
                )
            except Exception:
                gpu["power.draw"] = None

            # --- Utilization ---
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu["utilization.gpu"] = float(util.gpu)
            except Exception:
                gpu["utilization.gpu"] = None

            # --- Clocks ---
            try:
                gpu["clocks_sm_mhz"] = int(
                    pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                )
            except Exception:
                gpu["clocks_sm_mhz"] = None
            try:
                gpu["clocks_mem_mhz"] = int(
                    pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                )
            except Exception:
                gpu["clocks_mem_mhz"] = None

            # --- VRAM ---
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu["vram_used_mib"] = round(mem.used / (1024 * 1024), 1)
                gpu["vram_total_mib"] = round(mem.total / (1024 * 1024), 1)
            except Exception:
                gpu["vram_used_mib"] = None
                gpu["vram_total_mib"] = None

            # --- ECC (volatile) ---
            try:
                gpu["ecc_ce"] = int(pynvml.nvmlDeviceGetTotalEccErrors(
                    handle,
                    pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    pynvml.NVML_VOLATILE_ECC,
                ))
            except Exception:
                gpu["ecc_ce"] = None
            try:
                gpu["ecc_ue"] = int(pynvml.nvmlDeviceGetTotalEccErrors(
                    handle,
                    pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    pynvml.NVML_VOLATILE_ECC,
                ))
            except Exception:
                gpu["ecc_ue"] = None

            # --- PCIe link ---
            try:
                gpu["pcie_link_gen"] = int(
                    pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
                )
            except Exception:
                gpu["pcie_link_gen"] = None
            try:
                gpu["pcie_link_width"] = int(
                    pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
                )
            except Exception:
                gpu["pcie_link_width"] = None
            # Advertised max link gen for degradation detection. Cards
            # drop to Gen1 at idle by design (power savings), so the
            # signal is current < max *during stress*. Consumer compares
            # pcie_link_gen vs gpu_pcie_link_gen_max when utilization is
            # non-trivial. Static per GPU, but poll each sample so we
            # don't need special one-shot wiring in execute.py.
            try:
                gpu["pcie_link_gen_max"] = int(
                    pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
                )
            except Exception:
                gpu["pcie_link_gen_max"] = None

            # --- Throttle reasons bitmask ---
            try:
                gpu["throttle_reasons_active"] = hex(
                    pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                )
            except Exception:
                gpu["throttle_reasons_active"] = ""

            # --- Fan speed (%) ---
            try:
                gpu["fan_pct"] = int(pynvml.nvmlDeviceGetFanSpeed(handle))
            except Exception:
                gpu["fan_pct"] = None

            # --- pynvml-only: PCIe throughput (KB/s) ---
            try:
                gpu["pcie_tx_kbps"] = int(pynvml.nvmlDeviceGetPcieThroughput(
                    handle, pynvml.NVML_PCIE_UTIL_TX_BYTES,
                ))
            except Exception:
                gpu["pcie_tx_kbps"] = None
            try:
                gpu["pcie_rx_kbps"] = int(pynvml.nvmlDeviceGetPcieThroughput(
                    handle, pynvml.NVML_PCIE_UTIL_RX_BYTES,
                ))
            except Exception:
                gpu["pcie_rx_kbps"] = None

            # --- pynvml-only: retired pages ---
            try:
                sbit = pynvml.nvmlDeviceGetRetiredPages(
                    handle,
                    pynvml.NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS,
                )
            except Exception:
                sbit = []
            try:
                dbit = pynvml.nvmlDeviceGetRetiredPages(
                    handle,
                    pynvml.NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR,
                )
            except Exception:
                dbit = []
            gpu["retired_pages"] = len(sbit) + len(dbit)

            # --- pynvml-only: violation status (cumulative ns → μs) ---
            try:
                pv = pynvml.nvmlDeviceGetViolationStatus(
                    handle, pynvml.NVML_PERF_POLICY_POWER,
                )
                gpu["power_viol_us"] = int(pv.violationTime / 1000)
            except Exception:
                gpu["power_viol_us"] = None
            try:
                tv = pynvml.nvmlDeviceGetViolationStatus(
                    handle, pynvml.NVML_PERF_POLICY_THERMAL,
                )
                gpu["thermal_viol_us"] = int(tv.violationTime / 1000)
            except Exception:
                gpu["thermal_viol_us"] = None

            # Silent-None guard: if the three safety-critical fields are all
            # None simultaneously for a handle, NVML is effectively dead for
            # that GPU (GPU reset, driver error, handle invalidated mid-run).
            # Without this, every downstream verdict field silently reads None
            # forever while ``last_gpu_visible`` stays True — no FAIL fires.
            # Flagging has_err=True triggers the caller's nvidia-smi subprocess
            # fallback and restores the real-visibility signal.
            if (
                gpu.get("temperature.gpu") is None
                and gpu.get("power.draw") is None
                and gpu.get("utilization.gpu") is None
            ):
                has_err = True

            results.append(gpu)

        return results, has_err

    def _find_rapl_package_paths(self) -> List[Path]:
        """Return energy_uj paths for all RAPL package domains.

        Multi-socket boards (e.g. dual-EPYC) expose package-0
        and package-1 as separate zones; power is summed across all of them.
        """
        powercap = Path("/sys/class/powercap")
        if not powercap.exists():
            return []
        try:
            zones = sorted(powercap.iterdir())
        except Exception:
            return []
        paths: List[Path] = []
        for zone in zones:
            name_file  = zone / "name"
            energy_file = zone / "energy_uj"
            if not name_file.exists() or not energy_file.exists():
                continue
            try:
                name = name_file.read_text().strip().lower()
            except Exception:
                continue
            if "package" in name or name == "pkg":
                try:
                    energy_file.read_text()
                    paths.append(energy_file)
                except Exception:
                    continue
        return paths

    def _read_rapl_pkg_watt(self) -> Optional[float]:
        if not self._rapl_paths:
            return None
        now = time.monotonic()
        total_watts: List[float] = []
        for idx, path in enumerate(self._rapl_paths):
            try:
                energy_uj = float(path.read_text().strip())
                energy_j  = energy_uj / 1_000_000.0
                if idx in self._rapl_lasts:
                    prev_j, prev_t = self._rapl_lasts[idx]
                    dt = now - prev_t
                    if dt > 0:
                        watts = (energy_j - prev_j) / dt
                        if 0.0 <= watts <= 1500.0:
                            total_watts.append(watts)
                self._rapl_lasts[idx] = (energy_j, now)
            except Exception:
                continue
        if not total_watts:
            return None
        return round(sum(total_watts), 1)

    def _read_corsair_psu(self) -> Dict[str, Optional[float]]:
        """Read Corsair PSU telemetry via the ``corsair-psu`` kernel driver.

        The driver surfaces the full HID report from RMi/HXi/AXi-series PSUs
        (HX1500i, HX1200i, HX1000i, HX850i, HX750i, RM650i, etc.) through
        /sys/class/hwmon. Tested against an HX1500i over the iCue USB-header
        → USB-C cable — the driver claims the hidraw device automatically once
        the cable is connected and ``corsair_psu`` is loaded.

        Fields (kernel driver scaling):
          in0..in3_input  — millivolts  (AC input, 12V, 5V, 3.3V)
          curr1..curr4_input — milliamps (input, 12V, 5V, 3.3V rail currents)
          power1..power4_input — microwatts (total input, 12V, 5V, 3.3V)
          temp1/temp2_input — millidegrees C (VRM/primary, secondary/ambient)
          fan1_input — RPM directly

        Every read is guarded with ``.exists()`` — older kernel drivers or
        non-HXi models omit subsets of this layout and returning None for a
        missing file is the expected shape.
        """
        hwmon = Path("/sys/class/hwmon")
        if not hwmon.exists():
            return {}
        try:
            for device in hwmon.iterdir():
                name_file = device / "name"
                if not name_file.exists():
                    continue
                try:
                    name = name_file.read_text().strip()
                except Exception:
                    continue
                if name not in ("corsairpsu", "corsair-psu"):
                    continue
                return self._parse_corsair_psu_device(device)
        except Exception:
            pass
        return {}

    def _parse_corsair_psu_device(
        self, device: Path,
    ) -> Dict[str, Optional[float]]:
        """Parse a single corsair-psu hwmon device directory.

        Split out so tests can drive the parser against a mocked hwmon
        layout under ``tmp_path`` without touching /sys.
        """

        def _read_scaled(
            filename: str, divisor: float, ndigits: int = 3,
        ) -> Optional[float]:
            path = device / filename
            if not path.exists():
                return None
            try:
                raw = path.read_text().strip()
            except Exception:
                return None
            try:
                return round(float(raw) / divisor, ndigits)
            except (ValueError, TypeError):
                return None

        res: Dict[str, Optional[float]] = {}

        # --- Voltages: millivolts → volts ---
        res["vin_v"]  = _read_scaled("in0_input", 1000.0, 2)
        res["12v"]    = _read_scaled("in1_input", 1000.0, 3)
        res["5v"]     = _read_scaled("in2_input", 1000.0, 3)
        res["33v"]    = _read_scaled("in3_input", 1000.0, 3)

        # --- Currents: milliamps → amps ---
        res["i_in_a"]   = _read_scaled("curr1_input", 1000.0, 2)
        res["i_12v_a"]  = _read_scaled("curr2_input", 1000.0, 2)
        res["i_5v_a"]   = _read_scaled("curr3_input", 1000.0, 2)
        res["i_33v_a"]  = _read_scaled("curr4_input", 1000.0, 2)

        # --- Powers: microwatts → watts ---
        res["p_in_w"]    = _read_scaled("power1_input", 1_000_000.0, 1)
        res["p_12v_w"]   = _read_scaled("power2_input", 1_000_000.0, 1)
        res["p_5v_w"]    = _read_scaled("power3_input", 1_000_000.0, 1)
        res["p_33v_w"]   = _read_scaled("power4_input", 1_000_000.0, 1)

        # --- Temperatures: millidegrees → degrees C ---
        res["temp1_c"] = _read_scaled("temp1_input", 1000.0, 1)
        res["temp2_c"] = _read_scaled("temp2_input", 1000.0, 1)

        # --- Fan: RPM directly ---
        fan_raw = _read_scaled("fan1_input", 1.0, 0)
        res["fan_rpm"] = fan_raw

        # --- Derived: total output, efficiency ---
        out_parts = [res["p_12v_w"], res["p_5v_w"], res["p_33v_w"]]
        out_sum = [v for v in out_parts if v is not None]
        res["p_out_w"] = round(sum(out_sum), 1) if out_sum else None
        if (
            res["p_in_w"] is not None
            and res["p_out_w"] is not None
            and res["p_in_w"] > 10.0  # avoid divide-by-noise at idle
        ):
            res["efficiency_pct"] = round(
                (res["p_out_w"] / res["p_in_w"]) * 100.0, 1,
            )
        else:
            res["efficiency_pct"] = None

        return res

    def _read_ipmitool_rails(self) -> Dict[str, Optional[float]]:
        """Parse ``ipmitool sensor`` for voltage rails, fan RPMs, and temps.

        Returns a flat dict keyed by canonical names:
          ``12v``, ``5v``      — voltage rail volts
          ``fan_cpu_rpm``      — max CPU fan RPM across all CPU fan sensors
          ``fan_sys_rpm``      — max system fan RPM across non-CPU fan sensors
          ``vrm_temp``         — max VRM temp across sockets (VR_P*_TEMP)
          ``case_temp``        — board ambient / M.2 ambient
          ``lan_temp``         — onboard LAN temperature

        Sensors with no-reading status (``ns``, ``na``, ``nr``) are skipped
        so boards with unpopulated headers don't inject 0.0 or error strings.
        """
        code, stdout, _ = run_capture(["ipmitool", "sensor"], timeout=15)
        if code != 0 or not stdout:
            return {}
        result: Dict[str, Optional[float]] = {}
        cpu_fans: List[float] = []
        sys_fans: List[float] = []
        for line in stdout.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            name   = parts[0].lower().strip()
            raw    = parts[1].strip()
            units  = parts[2].lower() if len(parts) > 2 else ""
            status = parts[4].strip().lower() if len(parts) > 4 else ""
            if status in ("ns", "na", "nr", "no reading"):
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if "volt" in units:
                if "12" in name:
                    result.setdefault("12v", value)
                elif "5" in name and "3.3" not in name:
                    result.setdefault("5v", value)
            elif "rpm" in units:
                if "cpu" in name:
                    cpu_fans.append(value)
                else:
                    sys_fans.append(value)
            elif "degree" in units:
                if name == "cpu0_temp":
                    result.setdefault("cpu0_temp", value)
                elif name == "cpu1_temp":
                    result.setdefault("cpu1_temp", value)
                elif "vr_p" in name and "_temp" in name:
                    cur = result.get("vrm_temp")
                    result["vrm_temp"] = value if cur is None else max(cur, value)
                elif "m2_amb" in name or ("m2" in name and "amb" in name):
                    result.setdefault("case_temp", value)
                elif "ob_lan" in name or "lan_temp" in name:
                    result.setdefault("lan_temp", value)
        if cpu_fans:
            result["fan_cpu_rpm"] = max(cpu_fans)
        if sys_fans:
            result["fan_sys_rpm"] = max(sys_fans)
        return result

    def merge_turbostat(
        self,
        payload: Dict[str, object],
        turbostat_snap: Dict[str, object],
    ) -> None:
        """Merge turbostat snapshot into the telemetry payload.

        Called from the tick loop in execute.py after collect().  The turbostat
        collector runs on its own thread; this just copies its latest readings
        into the sample dict before it's written to the DB.
        """
        for key in (
            "cpu_avg_mhz", "cpu_max_mhz", "cpu_busy_pct_turbo",
            "cpu_core_watt", "cpu_pkg_watt_turbo",
            # v7 additions
            "cpu_freq_spread_mhz", "cpu_c6_pct",
        ):
            val = turbostat_snap.get(key)
            if val is not None:
                payload[key] = val

        # AMD Zen 5 (Threadripper 7000) does not expose package RAPL via
        # /sys/class/powercap by default, so _read_rapl_pkg_watt returns
        # None and cpu_pkg_watt stays NULL across the whole run. Turbostat
        # reads the same MSRs directly and populates cpu_pkg_watt_turbo —
        # mirror that back into cpu_pkg_watt so downstream consumers
        # (summary charts, future power rules, DB queries) see a value.
        if payload.get("cpu_pkg_watt") is None and payload.get("cpu_pkg_watt_turbo") is not None:
            payload["cpu_pkg_watt"] = payload["cpu_pkg_watt_turbo"]

    def _extract_fan_rpms(
        self,
        sensors_data: Dict[str, float],
        exclude_values: Optional[set] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Extract CPU/system fan RPMs from the parsed sensors dict.

        After the _read_sensors label-aware parse, each physical fan can appear
        twice: once under the label (``fan1``) and once under the raw
        subfeature (``fan1_input``). Dedupe on value so the same fan is not
        counted as both CPU and SYS. This is still a heuristic — on many boards
        the order is unpredictable — and should be replaced by a per-bench
        profile override when the fan-spike verdict rule gets tightened.

        ``exclude_values`` lets the caller filter out fans already attributed
        to another subsystem. The PSU fan (sourced from the ``corsair-psu``
        hwmon driver) otherwise surfaces here as the first fan on benches
        where the motherboard Super I/O isn't reporting any headers, and
        gets mislabeled "cpu fan".
        """
        unique: List[Tuple[str, float]] = []
        # Seed the seen set with the exclusion values so they're treated as
        # already-claimed. Float equality is load-bearing here — corsair
        # and sensors both emit the same integer-ish RPM so exact match is
        # reliable, but hedge with a tolerant comparison anyway.
        seen_values: set = set()
        excluded: List[float] = [
            float(v) for v in (exclude_values or set()) if v is not None
        ]
        for key, value in sorted(sensors_data.items()):
            if "fan" not in key.lower():
                continue
            if not isinstance(value, float) or value <= 0:
                continue
            if value in seen_values:
                continue
            if any(abs(value - ex) < 2.0 for ex in excluded):
                # Likely the PSU fan echoing through sensors; skip.
                # ±2 RPM covers drift between the sensor poll and the
                # corsair-psu poll without risking collision with a real
                # second fan (mobo headers typically differ by >100 RPM).
                continue
            seen_values.add(value)
            unique.append((key, value))
        cpu_rpm = unique[0][1] if len(unique) > 0 else None
        sys_rpm = unique[1][1] if len(unique) > 1 else None
        return cpu_rpm, sys_rpm

    # =========================================================
    # v7 — rail ripple + dropout counters
    # =========================================================

    def _update_rail_history_and_dropouts(
        self,
        v12: Optional[float],
        v5: Optional[float],
        v33: Optional[float],
    ) -> None:
        """Push one sample per rail into the rolling ripple window AND
        tick the dropout counter for any rail below its lcrit.

        None readings are pushed as None so the window stays aligned
        with the sample index — the ripple computation skips Nones when
        computing σ. This matters on benches where rail_33v is never
        populated: the 12V/5V windows should still slide normally.
        """
        self._rail_history.append({"12v": v12, "5v": v5, "33v": v33})
        for rail, value in (("12v", v12), ("5v", v5), ("33v", v33)):
            if value is None:
                continue
            try:
                f_value = float(value)
            except (TypeError, ValueError):
                continue
            if f_value < _RAIL_LCRIT[rail]:
                self._rail_dropouts[rail] += 1

    def _compute_rail_ripples(
        self,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return ``(12v_mv, 5v_mv, 33v_mv)`` — σ of each rail over the
        current rolling window, reported in millivolts.

        σ < 2 samples is meaningless, so we require at least
        ``_RAIL_RIPPLE_WINDOW`` samples of actual (non-None) data for
        that rail. This means the first N-1 ticks return None for every
        rail — no-op until the window fills — which is the correct
        calibration behavior.
        """
        return (
            self._std_mv([s["12v"] for s in self._rail_history]),
            self._std_mv([s["5v"]  for s in self._rail_history]),
            self._std_mv([s["33v"] for s in self._rail_history]),
        )

    @staticmethod
    def _std_mv(samples: List[Optional[float]]) -> Optional[float]:
        """Standard deviation in millivolts over the valid samples. Returns
        None until we have a full window of valid data — a partial window
        gives a misleadingly small σ that would false-signal "clean rails"
        on the first 9 ticks of every run."""
        clean = [float(s) for s in samples if s is not None]
        if len(clean) < _RAIL_RIPPLE_WINDOW:
            return None
        mean = sum(clean) / len(clean)
        var = sum((v - mean) ** 2 for v in clean) / len(clean)
        # σ in volts → millivolts, 2 decimals for TUI readability.
        return round(math.sqrt(var) * 1000.0, 2)

    # =========================================================
    # v7 — PCIe AER uncorrectable (mirror of corr variant)
    # =========================================================

    def _read_pcie_aer_uncorr_total(self) -> Optional[int]:
        """Sum of uncorrectable AER counters across every PCI device.

        Parses both ``aer_dev_fatal`` and ``aer_dev_nonfatal`` — an UE
        of either flavor must stay at zero for the whole run. Growth
        mid-run is an instant FAIL signal (PSU-induced link corruption,
        CPU lane flakiness, or a physically failing device).

        Returns ``None`` when no readable AER-UE file exists so the
        column stays NULL on systems without AER instead of pretending
        the count is zero.
        """
        pci_root = Path("/sys/bus/pci/devices")
        if not pci_root.exists():
            return None
        total: Optional[int] = None
        try:
            for dev in pci_root.iterdir():
                for fname in ("aer_dev_fatal", "aer_dev_nonfatal"):
                    aer_file = dev / fname
                    if not aer_file.is_file():
                        continue
                    try:
                        text = aer_file.read_text()
                    except (PermissionError, OSError):
                        continue
                    for line in text.splitlines():
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        try:
                            n = int(parts[1])
                        except ValueError:
                            continue
                        total = (total or 0) + n
        except Exception:
            return None
        return total

    # =========================================================
    # v7 — NIC link speed + operstate
    # =========================================================

    def _read_nic_link_state(self) -> Tuple[Optional[int], Optional[int]]:
        """Return ``(speed_mbps, up_flag)`` for the primary physical NIC.

        Picks the first non-loopback, non-virtual interface that reports
        a numeric speed. Multi-NIC benches are rare — and when they do
        exist we care about the uplink being used for iperf / file I/O,
        which is the one reporting a real speed. ``up_flag`` is 1 when
        operstate is "up" (or "unknown" — common for PHYs without link
        detection that are still carrying traffic), else 0.

        Returns ``(None, None)`` when no real interface exposes speed
        (all virtual, all loopback, all down with operstate=down).
        """
        net_root = Path("/sys/class/net")
        if not net_root.exists():
            return None, None
        try:
            for iface in sorted(net_root.iterdir()):
                if iface.name == "lo":
                    continue
                # Skip virtual / tunnel interfaces — /sys/class/net entries
                # that are symlinks into /sys/devices/virtual/ don't have
                # PHY-level link semantics.
                try:
                    if "virtual" in os.readlink(iface):
                        continue
                except OSError:
                    pass
                speed_f = iface / "speed"
                oper_f  = iface / "operstate"
                if not speed_f.exists():
                    continue
                try:
                    speed_raw = speed_f.read_text().strip()
                    speed = int(speed_raw)
                except (ValueError, OSError):
                    continue
                # Kernel returns -1 for "unknown" (interface down or PHY
                # autoneg not complete). Skip those — caller sees None
                # instead of a nonsense negative speed.
                if speed <= 0:
                    continue
                up_flag = 0
                try:
                    state = oper_f.read_text().strip().lower()
                    if state in ("up", "unknown"):
                        up_flag = 1
                except OSError:
                    pass
                return speed, up_flag
        except Exception:
            return None, None
        return None, None

    # =========================================================
    # v7 — NVMe SMART warning/critical temp time
    # =========================================================

    def _read_nvme_smart_temp_times(
        self,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Return ``(warn_time_s, crit_time_s)`` — cumulative seconds the
        hottest NVMe drive has spent in warning / critical composite-temp
        states. From NVMe SMART log page 02h, 64-bit fields at offsets
        200 (warning) and 208 (critical). Unit is minutes in the spec;
        we convert to seconds so the column stays in the same unit as
        PS-FU's other time counters.

        Requires ``nvme-cli`` (``nvme smart-log``). Fallback path: parse
        whatever ``storage_health.py`` already captured at run start.
        When neither works, returns (None, None) cleanly.

        The value is cumulative since the drive's manufacture — the
        consumer diffs vs a phase-start baseline to detect growth
        during the run.
        """
        nvme_bin = nvme_cli_binary_path()
        if not nvme_bin:
            return None, None
        dev_root = Path("/dev")
        warn_max: Optional[int] = None
        crit_max: Optional[int] = None
        try:
            candidates = sorted(
                p for p in dev_root.glob("nvme*")
                if re.match(r"^nvme\d+$", p.name)
            )
        except Exception:
            candidates = []
        for dev in candidates:
            code, stdout, _ = run_capture(
                [nvme_bin, "smart-log", str(dev), "-o", "json"], timeout=5,
            )
            if code != 0 or not stdout:
                continue
            try:
                data = json.loads(stdout)
            except (ValueError, TypeError):
                continue
            # nvme-cli JSON keys: ``warning_temp_time`` and
            # ``critical_comp_time``. Unit is minutes per NVMe spec.
            wt = data.get("warning_temp_time")
            ct = data.get("critical_comp_time")
            if isinstance(wt, (int, float)):
                wt_s = int(round(float(wt) * 60.0))
                warn_max = wt_s if warn_max is None else max(warn_max, wt_s)
            if isinstance(ct, (int, float)):
                ct_s = int(round(float(ct) * 60.0))
                crit_max = ct_s if crit_max is None else max(crit_max, ct_s)
        return warn_max, crit_max

    # =========================================================
    # v7 — /proc/stat extended (ctx, intr, softirq rates)
    # =========================================================

    def _read_proc_stat_ext(self) -> Optional[Tuple[int, int, int]]:
        """Return ``(ctxt, intr, softirq)`` cumulative counters from
        /proc/stat. All are monotonic since boot — callers compute rates
        by diffing against a previous snapshot."""
        try:
            ctxt: Optional[int] = None
            intr: Optional[int] = None
            softirq: Optional[int] = None
            with open("/proc/stat", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("ctxt "):
                        ctxt = int(line.split()[1])
                    elif line.startswith("intr "):
                        intr = int(line.split()[1])
                    elif line.startswith("softirq "):
                        softirq = int(line.split()[1])
            if ctxt is None or intr is None or softirq is None:
                return None
            return ctxt, intr, softirq
        except Exception:
            return None

    def _compute_proc_stat_ext_rates(
        self,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return ``(ctx_rate, intr_rate, softirq_rate)`` in per-second
        units over the interval since the previous snapshot. First call
        returns (None, None, None) because there's no baseline yet."""
        current = self._read_proc_stat_ext()
        now = time.monotonic()
        prev = self._last_proc_stat_ext
        if current is None:
            self._last_proc_stat_ext = None
            return None, None, None
        ctxt, intr, softirq = current
        self._last_proc_stat_ext = (now, ctxt, intr, softirq)
        if prev is None:
            return None, None, None
        prev_t, prev_ctxt, prev_intr, prev_softirq = prev
        dt = now - prev_t
        if dt <= 0:
            return None, None, None
        return (
            round(max(0.0, (ctxt - prev_ctxt) / dt), 1),
            round(max(0.0, (intr - prev_intr) / dt), 1),
            round(max(0.0, (softirq - prev_softirq) / dt), 1),
        )

    # =========================================================
    # v7 — /proc/vmstat pgmajfault rate
    # =========================================================

    def _compute_pgmajfault_rate(self) -> Optional[float]:
        """Major page fault rate (faults/sec). Rising mid-run usually
        means the bench is swapping or the page cache is being shredded
        by a bad process. Zero on healthy stress runs (memory is pinned,
        working set fits). None on the first tick and on kernels that
        don't expose vmstat."""
        try:
            cur_val: Optional[int] = None
            with open("/proc/vmstat", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("pgmajfault "):
                        cur_val = int(line.split()[1])
                        break
            if cur_val is None:
                return None
        except Exception:
            return None
        now = time.monotonic()
        prev = self._last_vmstat
        self._last_vmstat = (now, cur_val)
        if prev is None:
            return None
        prev_t, prev_val = prev
        dt = now - prev_t
        if dt <= 0:
            return None
        return round(max(0.0, (cur_val - prev_val) / dt), 2)

    # =========================================================
    # v7 — /proc/meminfo Dirty + Writeback
    # =========================================================

    def _read_meminfo_dirty_writeback(
        self,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return ``(dirty_kb, writeback_kb)`` from /proc/meminfo.

        ``Dirty`` = modified pages not yet written back. ``Writeback`` =
        actively being flushed. A growing Writeback column while
        disk_busy_pct is pinned + ssd_write_mbps is low = a failing
        drive controller blocking syncs."""
        dirty: Optional[float] = None
        wb: Optional[float] = None
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("Dirty:"):
                        dirty = float(line.split()[1])
                    elif line.startswith("Writeback:"):
                        wb = float(line.split()[1])
                    if dirty is not None and wb is not None:
                        break
        except Exception:
            return None, None
        return dirty, wb

    # =========================================================
    # v7 — /sys/block/nvme*/stat throughput + queue depth
    # =========================================================

    def _read_block_stats(self) -> Optional[Dict[str, Tuple[int, int, int]]]:
        """Return ``{device_name: (sectors_read, sectors_written, inflight)}``
        for every whole-disk NVMe (partitions excluded).

        ``/sys/block/nvme0n1/stat`` columns (per kernel block/stat
        docs): read_ios, read_merges, read_sectors, read_ticks, write_ios,
        write_merges, write_sectors, write_ticks, in_flight, io_ticks,
        time_in_queue, discard_ios, ...

        We care about indices 2 (read_sectors), 6 (write_sectors), and
        8 (in_flight = currently queued ops). Sectors are always 512 B.
        """
        sys_block = Path("/sys/block")
        if not sys_block.exists():
            return None
        result: Dict[str, Tuple[int, int, int]] = {}
        try:
            for dev in sys_block.iterdir():
                name = dev.name
                if not re.match(r"^nvme\d+n\d+$", name):
                    continue
                stat_file = dev / "stat"
                if not stat_file.is_file():
                    continue
                try:
                    parts = stat_file.read_text().split()
                except OSError:
                    continue
                if len(parts) < 9:
                    continue
                try:
                    read_sectors  = int(parts[2])
                    write_sectors = int(parts[6])
                    inflight      = int(parts[8])
                except ValueError:
                    continue
                result[name] = (read_sectors, write_sectors, inflight)
        except Exception:
            return None
        return result if result else None

    def _compute_block_stat_metrics(
        self,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return ``(read_mbps, write_mbps, queue_depth)`` summed across
        all NVMe whole-disk devices. MB/s is decimal MB (×10⁶ B), NOT
        MiB, to match the convention used by ``nvme-cli`` and vendor
        datasheets. Queue depth is the instantaneous in_flight sum,
        which is a point-in-time value (no delta needed)."""
        cur = self._read_block_stats()
        now = time.monotonic()
        prev = self._last_block_stat
        if cur is None:
            self._last_block_stat = None
            return None, None, None
        # Store monotonic ts + per-device tuple for next call.
        snapshot: Dict[str, Tuple[float, int, int, int]] = {
            name: (now, r, w, q) for name, (r, w, q) in cur.items()
        }
        self._last_block_stat = snapshot
        inflight_sum = sum(q for _r, _w, q in cur.values())
        if prev is None:
            # Queue depth doesn't need a delta, but throughput does.
            return None, None, float(inflight_sum)
        read_mbps_total  = 0.0
        write_mbps_total = 0.0
        saw_delta = False
        for name, (_r, _w, _q) in cur.items():
            if name not in prev:
                continue
            prev_t, prev_r, prev_w, _prev_q = prev[name]
            dt = now - prev_t
            if dt <= 0:
                continue
            cur_r, cur_w, _cur_q = cur[name]
            # Sectors are 512 B. MB = bytes / 1e6.
            read_mbps_total  += max(0, cur_r - prev_r)  * 512.0 / 1e6 / dt
            write_mbps_total += max(0, cur_w - prev_w) * 512.0 / 1e6 / dt
            saw_delta = True
        if not saw_delta:
            return None, None, float(inflight_sum)
        return (
            round(read_mbps_total, 2),
            round(write_mbps_total, 2),
            float(inflight_sum),
        )

    # =========================================================
    # v7 — thermal zone margin walker
    # =========================================================

    def _read_thermal_zone_margin(self) -> Optional[float]:
        """Minimum margin (°C) to any active trip point across every
        thermal zone. Negative means at least one zone has already
        crossed a trip — growth in the negative direction is the
        load-bearing signal.

        Per-zone layout (from kernel docs):
          /sys/class/thermal/thermal_zone*/temp              — current milliC
          /sys/class/thermal/thermal_zone*/trip_point_N_temp — trip milliC
          /sys/class/thermal/thermal_zone*/trip_point_N_type — passive/active/hot/critical

        We consider all trip types — a passive trip firing is still a
        sign the zone is under duress. The minimum margin across all
        zones / trips surfaces the worst-case zone without needing per-
        zone columns.
        """
        root = Path("/sys/class/thermal")
        if not root.exists():
            return None
        min_margin: Optional[float] = None
        try:
            for zone in root.iterdir():
                if not zone.name.startswith("thermal_zone"):
                    continue
                temp_f = zone / "temp"
                if not temp_f.is_file():
                    continue
                try:
                    cur_milli = int(temp_f.read_text().strip())
                except (ValueError, OSError):
                    continue
                cur_c = cur_milli / 1000.0
                # Invalid / disabled zones sometimes report large
                # sentinel values; skip anything implausible.
                if cur_c < -50.0 or cur_c > 150.0:
                    continue
                for trip_f in sorted(zone.glob("trip_point_*_temp")):
                    try:
                        trip_milli = int(trip_f.read_text().strip())
                    except (ValueError, OSError):
                        continue
                    if trip_milli <= 0:
                        continue  # disabled trip
                    trip_c = trip_milli / 1000.0
                    margin = round(trip_c - cur_c, 1)
                    if min_margin is None or margin < min_margin:
                        min_margin = margin
        except Exception:
            return None
        return min_margin

    def _extract_first_number(
        self,
        mapping: Dict[str, float],
        needles: List[str],
    ) -> Optional[float]:
        lowered = [needle.lower() for needle in needles]
        for key, value in mapping.items():
            key_lower = key.lower()
            if any(needle in key_lower for needle in lowered):
                return float(value)
        return None
