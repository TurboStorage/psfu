import json
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from psfu.common import utc_now


# Bump this + append a new _MIGRATIONS tuple whenever the schema changes.
# This is distinct from PROTOCOL_VERSION in config.py: PROTOCOL_VERSION
# tracks verdict rule semantics, CURRENT_SCHEMA_VERSION tracks table shape.
CURRENT_SCHEMA_VERSION = 10


def _quote_ident(name: str) -> str:
    """Double-quote a SQLite identifier for safe interpolation.

    Values are always parameterized (`?` binding), so data injection is
    blocked. But column names come from `payload.keys()` which is built
    by internal callers — if a future caller ever derives a key from an
    external string, an un-quoted identifier becomes malformed SQL or worse.
    SQLite's identifier quoting is `"..."` with `""` as the escape for a
    literal double-quote. We also reject identifiers containing a NUL byte,
    which SQLite treats as a terminator and would silently truncate.
    """
    if "\x00" in name:
        raise ValueError("SQL identifier contains NUL byte: {!r}".format(name))
    return '"' + name.replace('"', '""') + '"'

# Each entry: (from_version, to_version, [sql_statements]).
# ALTER TABLE statements are tolerated if the column already exists — this
# makes migrations idempotent on DBs that were built by the pre-schema_version
# code path (which ran the same ALTER TABLE list directly at __init__).
_MIGRATIONS: List[Tuple[int, int, List[str]]] = [
    (1, 2, [
        "ALTER TABLE sample ADD COLUMN cpu_pkg_watt REAL",
        "ALTER TABLE sample ADD COLUMN cpu_avg_mhz REAL",
        "ALTER TABLE sample ADD COLUMN cpu_max_mhz REAL",
        "ALTER TABLE sample ADD COLUMN cpu_busy_pct_turbo REAL",
        "ALTER TABLE sample ADD COLUMN cpu_core_watt REAL",
        "ALTER TABLE sample ADD COLUMN cpu_pkg_watt_turbo REAL",
        "ALTER TABLE sample ADD COLUMN gpu_vram_temp_c REAL",
        "ALTER TABLE sample ADD COLUMN gpu_power_total_w REAL",
        "ALTER TABLE sample ADD COLUMN gpu_clock_sm_mhz INTEGER",
        "ALTER TABLE sample ADD COLUMN gpu_clock_mem_mhz INTEGER",
        "ALTER TABLE sample ADD COLUMN gpu_vram_used_mib REAL",
        "ALTER TABLE sample ADD COLUMN gpu_vram_total_mib REAL",
        "ALTER TABLE sample ADD COLUMN gpu_fan_pct INTEGER",
        "ALTER TABLE sample ADD COLUMN gpu_ecc_ce REAL",
        "ALTER TABLE sample ADD COLUMN gpu_ecc_ue REAL",
        "ALTER TABLE sample ADD COLUMN gpu_throttle_active INTEGER",
        "ALTER TABLE sample ADD COLUMN gpu_retired_pages REAL",
        "ALTER TABLE sample ADD COLUMN gpu_power_viol_us REAL",
        "ALTER TABLE sample ADD COLUMN gpu_thermal_viol_us REAL",
        "ALTER TABLE sample ADD COLUMN pcie_link_gen INTEGER",
        "ALTER TABLE sample ADD COLUMN pcie_link_width INTEGER",
        "ALTER TABLE sample ADD COLUMN pcie_tx_kbps REAL",
        "ALTER TABLE sample ADD COLUMN pcie_rx_kbps REAL",
        "ALTER TABLE sample ADD COLUMN fan_cpu_rpm REAL",
        "ALTER TABLE sample ADD COLUMN fan_sys_rpm REAL",
        "ALTER TABLE sample ADD COLUMN disk_busy_pct REAL",
        "ALTER TABLE sample ADD COLUMN ssd_temp_c REAL",
        "ALTER TABLE sample ADD COLUMN hba_count INTEGER",
        "ALTER TABLE sample ADD COLUMN i2c_adapter_count INTEGER",
        "ALTER TABLE sample ADD COLUMN i3c_bus_count INTEGER",
        "ALTER TABLE run ADD COLUMN load_scale REAL",
        "ALTER TABLE run ADD COLUMN subsystem_results_json TEXT",
    ]),
    # v2 → v3: Corsair PSU telemetry via corsair-psu hwmon driver.
    # Adds AC input voltage/current/power, per-rail currents and powers,
    # two temperature sensors, fan RPM, derived total output power and
    # efficiency. Populated only when an HXi/RMi/AXi PSU is connected via
    # the iCue USB-header cable; otherwise all columns stay NULL — no
    # verdict rule consumes these yet, so PROTOCOL_VERSION does not bump.
    (2, 3, [
        "ALTER TABLE sample ADD COLUMN psu_vin_v REAL",
        "ALTER TABLE sample ADD COLUMN psu_i_in_a REAL",
        "ALTER TABLE sample ADD COLUMN psu_i_12v_a REAL",
        "ALTER TABLE sample ADD COLUMN psu_i_5v_a REAL",
        "ALTER TABLE sample ADD COLUMN psu_i_33v_a REAL",
        "ALTER TABLE sample ADD COLUMN psu_p_in_w REAL",
        "ALTER TABLE sample ADD COLUMN psu_p_12v_w REAL",
        "ALTER TABLE sample ADD COLUMN psu_p_5v_w REAL",
        "ALTER TABLE sample ADD COLUMN psu_p_33v_w REAL",
        "ALTER TABLE sample ADD COLUMN psu_p_out_w REAL",
        "ALTER TABLE sample ADD COLUMN psu_efficiency_pct REAL",
        "ALTER TABLE sample ADD COLUMN psu_temp_c REAL",
        "ALTER TABLE sample ADD COLUMN psu_temp2_c REAL",
        "ALTER TABLE sample ADD COLUMN psu_fan_rpm REAL",
    ]),
    # v3 → v4: RAM hardware context (total installed + configured DIMM speed).
    # Static for the run — probed once at collector init from /proc/meminfo
    # and dmidecode, then mirrored into every sample so the TUI RAM panel
    # can render them alongside per-sample mem_used_pct. dmidecode requires
    # root; mem_speed_mts stays NULL when sudo isn't available. No verdict
    # rule consumes either field yet.
    (3, 4, [
        "ALTER TABLE sample ADD COLUMN mem_total_gb REAL",
        "ALTER TABLE sample ADD COLUMN mem_speed_mts INTEGER",
    ]),
    # v4 → v5: Platform thermal + link-health context.
    # - mobo_vrm_temp_c / mobo_case_temp_c: board-level thermal data from
    #   Super I/O. VRM especially matters for Threadripper PSU RMA — a
    #   sagging 12V forces the VRM to push more current at lower voltage
    #   and temp rises directly.
    # - cpu_ccd_max_c / cpu_ccd_spread_c: per-CCD (Tccd*) distribution on
    #   Zen 4/5. Spread > 20 C under uniform load suggests uneven cooling
    #   or VRM phase imbalance.
    # - ssd_temp_ctrl_c / ssd_temp_nand_c: split NVMe temps (sensor 1/2
    #   from nvme smart-log) rather than only the composite. Controller
    #   vs NAND delta separates thermal choke from bulk heat.
    # - pcie_aer_corr_total: running sum of correctable AER counters
    #   across all PCI devices. Non-zero growth during PSU stress is an
    #   early sign of link instability from rail noise — fatal AER is
    #   already caught by the kernel watcher, but correctable fires first.
    # No verdict rule consumes any of these yet.
    (4, 5, [
        "ALTER TABLE sample ADD COLUMN mobo_vrm_temp_c REAL",
        "ALTER TABLE sample ADD COLUMN mobo_case_temp_c REAL",
        "ALTER TABLE sample ADD COLUMN cpu_ccd_max_c REAL",
        "ALTER TABLE sample ADD COLUMN cpu_ccd_spread_c REAL",
        "ALTER TABLE sample ADD COLUMN ssd_temp_ctrl_c REAL",
        "ALTER TABLE sample ADD COLUMN ssd_temp_nand_c REAL",
        "ALTER TABLE sample ADD COLUMN pcie_aer_corr_total INTEGER",
    ]),
    # v5 → v6: NIC thermals + link health + kernel stall + GPU link
    # degradation + CPU iowait. All five added 2026-04-16 after mining
    # the live TRX50 AI TOP sensor dump and finding unclaimed signal.
    # - nic_mac_temp_c / nic_phy_temp_c: integrated NIC MAC + PHY temps
    #   (Super I/O exposes them on this board as "mac temperature" /
    #   "phy temperature"). Sustained traffic + bad airflow cooks the
    #   controller silently — this surfaces it.
    # - psi_cpu_some_avg10 / psi_mem_some_avg10 / psi_io_some_avg10:
    #   kernel PSI (Pressure Stall Information) averages. Directly
    #   measure "system is stalled waiting on X" in a way that
    #   separates healthy-under-load from actually-sick.
    # - nic_rx_crc_errors_total / nic_rx_errors_total / nic_rx_dropped_total:
    #   summed across all real interfaces (excludes lo). CRC growth
    #   during a soak = cable / SFP / port issue.
    # - gpu_pcie_link_gen_max: NVML advertised max link gen for this
    #   GPU. Paired with pcie_link_gen to detect degradation (a Gen5
    #   card stuck at Gen3 under load = bad riser / slot / power).
    # - cpu_iowait_pct: /proc/stat cpu column 5 derivative. Complements
    #   disk_busy_pct — iowait-high + busy-low = failing storage
    #   controller, iowait-high + busy-high = saturated but healthy.
    # No verdict rule consumes any of these yet.
    (5, 6, [
        "ALTER TABLE sample ADD COLUMN nic_mac_temp_c REAL",
        "ALTER TABLE sample ADD COLUMN nic_phy_temp_c REAL",
        "ALTER TABLE sample ADD COLUMN psi_cpu_some_avg10 REAL",
        "ALTER TABLE sample ADD COLUMN psi_mem_some_avg10 REAL",
        "ALTER TABLE sample ADD COLUMN psi_io_some_avg10 REAL",
        "ALTER TABLE sample ADD COLUMN nic_rx_crc_errors_total INTEGER",
        "ALTER TABLE sample ADD COLUMN nic_rx_errors_total INTEGER",
        "ALTER TABLE sample ADD COLUMN nic_rx_dropped_total INTEGER",
        "ALTER TABLE sample ADD COLUMN gpu_pcie_link_gen_max INTEGER",
        "ALTER TABLE sample ADD COLUMN cpu_iowait_pct REAL",
    ]),
    # v6 → v7: Telemetry Upgrade Plan Tier 1 + Tier 2 + memory bandwidth.
    # Full rationale is in the module docs for these collectors. 30 new columns:
    #
    # ── Tier 1 — PSU-diagnostic signals (15) ─────────────────────────
    # Rail ripple mV: σ of rail voltage over rolling 10 s. Failing bulk
    # caps raise ripple before mean drifts — multimeter can't see this.
    # Rail dropouts: cumulative count where voltage crossed lcrit even
    # briefly. Catches sub-second brownouts too fast for 1 Hz rail mean.
    # pcie_aer_uncorr_total: UE counter. Must stay zero; any growth =
    # instant FAIL-level signal (PSU brownout corrupts PCIe).
    # GPU throttle reasons split: the single gpu_throttle_active bit is
    # a HW-mask summary; these four columns isolate *which* reason fired.
    # hw_power_brake is the high-value one — it fires on PSU OCP events.
    # SSD warn/crit temp time: NVMe SMART log page 02h cumulative seconds.
    # Growth during run = thermal throttle (classic 3.3V/5V sag symptom).
    # NIC link speed/up: downgrade (10G → 1G) or hard link-loss often
    # correlates with PSU brownouts on the NIC's power plane.
    #
    # ── Tier 2 — Platform health (13) ────────────────────────────────
    # cpu_freq_spread_mhz: σ of per-core MHz — CCX-scoped throttle detector.
    # cpu_c6_pct: deep-idle residency from turbostat — stuck low = power
    # plane issue.
    # cpu_{ctx_switches,interrupts,softirq}_per_s: Δ from /proc/stat.
    # Runaway IRQ storm = bad device (PCIe link flapping, USB controller).
    # mem_pgmajfault_per_s: Δ from /proc/vmstat. Major faults = swap hits.
    # mem_dirty_kb / mem_writeback_kb: /proc/meminfo pending-write pressure.
    # ssd_read_mbps / ssd_write_mbps / ssd_queue_depth: /sys/block/nvme*/stat
    # delta — surfaces controller throttle (throughput collapses while
    # queue builds) during PSU sag.
    # mobo_chipset_temp_c: TRX50 chipset sensor (separate from VRM/case).
    # thermal_zone_margin_c: min degrees-to-trip across all ACPI zones.
    #
    # ── Memory bandwidth (2) ─────────────────────────────────────────
    # mem_bw_read_mbps / mem_bw_write_mbps: perf-stat uncore IMC events
    # sampled every 30 s on a separate thread (1 Hz is too expensive).
    # Columns hold at last reading between samples so sparkline renders
    # flat rather than gappy. None when perf/PMU not available.
    #
    # No verdict rule consumes any of these yet — pure telemetry coverage
    # add. PROTOCOL_VERSION is NOT bumped.
    (6, 7, [
        # Tier 1
        "ALTER TABLE sample ADD COLUMN rail_12v_ripple_mv REAL",
        "ALTER TABLE sample ADD COLUMN rail_5v_ripple_mv REAL",
        "ALTER TABLE sample ADD COLUMN rail_33v_ripple_mv REAL",
        "ALTER TABLE sample ADD COLUMN rail_12v_dropouts INTEGER",
        "ALTER TABLE sample ADD COLUMN rail_5v_dropouts INTEGER",
        "ALTER TABLE sample ADD COLUMN rail_33v_dropouts INTEGER",
        "ALTER TABLE sample ADD COLUMN pcie_aer_uncorr_total INTEGER",
        "ALTER TABLE sample ADD COLUMN gpu_throttle_hw_slowdown INTEGER",
        "ALTER TABLE sample ADD COLUMN gpu_throttle_hw_thermal INTEGER",
        "ALTER TABLE sample ADD COLUMN gpu_throttle_hw_power_brake INTEGER",
        "ALTER TABLE sample ADD COLUMN gpu_throttle_sw_power_cap INTEGER",
        "ALTER TABLE sample ADD COLUMN ssd_warn_temp_time_s INTEGER",
        "ALTER TABLE sample ADD COLUMN ssd_crit_temp_time_s INTEGER",
        "ALTER TABLE sample ADD COLUMN nic_link_speed_mbps INTEGER",
        "ALTER TABLE sample ADD COLUMN nic_link_up INTEGER",
        # Tier 2
        "ALTER TABLE sample ADD COLUMN cpu_freq_spread_mhz REAL",
        "ALTER TABLE sample ADD COLUMN cpu_c6_pct REAL",
        "ALTER TABLE sample ADD COLUMN cpu_ctx_switches_per_s REAL",
        "ALTER TABLE sample ADD COLUMN cpu_interrupts_per_s REAL",
        "ALTER TABLE sample ADD COLUMN cpu_softirq_per_s REAL",
        "ALTER TABLE sample ADD COLUMN mem_pgmajfault_per_s REAL",
        "ALTER TABLE sample ADD COLUMN mem_dirty_kb REAL",
        "ALTER TABLE sample ADD COLUMN mem_writeback_kb REAL",
        "ALTER TABLE sample ADD COLUMN ssd_read_mbps REAL",
        "ALTER TABLE sample ADD COLUMN ssd_write_mbps REAL",
        "ALTER TABLE sample ADD COLUMN ssd_queue_depth REAL",
        "ALTER TABLE sample ADD COLUMN mobo_chipset_temp_c REAL",
        "ALTER TABLE sample ADD COLUMN thermal_zone_margin_c REAL",
        # Memory bandwidth
        "ALTER TABLE sample ADD COLUMN mem_bw_read_mbps REAL",
        "ALTER TABLE sample ADD COLUMN mem_bw_write_mbps REAL",
    ]),
    # v7 → v8: Upstream UPS telemetry via Network UPS Tools (NUT).
    # See collectors/ups.py.
    # Covers a diagnostic gap seen on a PSU that shut down at <1% load
    # 11–13 s into a run, where mains-side conditions at the shutdown
    # timestamp were not captured. A bench UPS already sees wall
    # sags and OL→OB transfers; piping those into the per-tick sample
    # means we can cross-reference a shutdown against actual line state.
    #
    # Seven live-numeric columns map one-to-one from NUT variables
    # (input.voltage, input.frequency, output.voltage, ups.load,
    # ups.realpower, battery.charge, battery.runtime). ups_status_raw is
    # the whitespace-delimited token string NUT emits (e.g. "OL", "OB
    # DISCHRG"). ups_on_battery is the derived 0/1 computed in the
    # collector from whole-token "OB" match (substring would false-hit
    # "BOOST"). ups_transfer_count is a cumulative OL→OB edge count —
    # monotonic, starts at 0, first-ever poll on battery does NOT
    # increment (no prior OL state to transition from).
    #
    # Cross-vendor caveats baked into the column contract: Cyberpower
    # CP1500PFCLCD omits both `input.frequency` AND `ups.realpower` (only
    # emits `ups.realpower.nominal` which is the rated plate, not live
    # draw), so ups_input_freq_hz + ups_realpower_w will be NULL on
    # Cyberpower runs and populated on APC. NULL-handling is the
    # collector's responsibility — db.py just stores whatever comes in.
    #
    # No verdict rule consumes any of these yet. Per Rule 1 + Rule 5,
    # PROTOCOL_VERSION stays at 0.12; a future `mains_sag_during_run` /
    # `ups_transfer_during_run` rule ships as its own commit with the
    # rule + test + profile that exercises it. This commit is pure
    # telemetry coverage.
    (7, 8, [
        "ALTER TABLE sample ADD COLUMN ups_input_voltage_v REAL",
        "ALTER TABLE sample ADD COLUMN ups_input_freq_hz REAL",
        "ALTER TABLE sample ADD COLUMN ups_output_voltage_v REAL",
        "ALTER TABLE sample ADD COLUMN ups_load_pct REAL",
        "ALTER TABLE sample ADD COLUMN ups_realpower_w REAL",
        "ALTER TABLE sample ADD COLUMN ups_battery_charge_pct REAL",
        "ALTER TABLE sample ADD COLUMN ups_battery_runtime_s REAL",
        "ALTER TABLE sample ADD COLUMN ups_status_raw TEXT",
        "ALTER TABLE sample ADD COLUMN ups_on_battery INTEGER",
        "ALTER TABLE sample ADD COLUMN ups_transfer_count INTEGER",
    ]),
    # v9 — live watts + apparent VA for vendors that omit ups.realpower.
    # CyberPower CP1500 family firmware never emits live watts — only
    # ``ups.realpower.nominal`` (the plate) and ``ups.load`` (percent).
    # PowerPanel on Windows computes live watts as ``plate × load% / 100``
    # and that's what ``ups_realpower_est_w`` does here, so the live TUI
    # tick shows a wattage number on CyberPower the same way APC's
    # firmware-provided ``ups_realpower_w`` does.
    #
    # ``ups_apparent_nominal_va`` is parsed from the model-name string
    # (CP1500PFCRM2U → 1500, Smart-UPS X 1500 → 1500). It's the one
    # signal NUT doesn't expose as a variable at all — models print
    # their VA in the name and nowhere else. ``ups_apparent_est_va``
    # uses the same load% × nominal derivation.
    #
    # No verdict rule consumes any of these (Rule 1). PROTOCOL_VERSION
    # stays at 0.12 — pure telemetry coverage.
    (8, 9, [
        "ALTER TABLE sample ADD COLUMN ups_realpower_nominal_w REAL",
        "ALTER TABLE sample ADD COLUMN ups_realpower_est_w REAL",
        "ALTER TABLE sample ADD COLUMN ups_apparent_nominal_va REAL",
        "ALTER TABLE sample ADD COLUMN ups_apparent_est_va REAL",
    ]),
    # v9 → v10: 1 Hz extended NVMe SMART poll for selected NVMe DUTs.
    # Pure telemetry coverage in the SSD-validation track (S1). These
    # columns mirror the pinned first selected NVMe device into sample rows;
    # per-device raw JSONL lives under runs/.../storage/nvme/<dev>/.
    # No verdict rule consumes these yet, so PROTOCOL_VERSION does not bump.
    (9, 10, [
        "ALTER TABLE sample ADD COLUMN nvme_composite_temp_c REAL",
        "ALTER TABLE sample ADD COLUMN nvme_warning_temp_time_min REAL",
        "ALTER TABLE sample ADD COLUMN nvme_critical_temp_time_min REAL",
        "ALTER TABLE sample ADD COLUMN nvme_percentage_used REAL",
        "ALTER TABLE sample ADD COLUMN nvme_available_spare REAL",
        "ALTER TABLE sample ADD COLUMN nvme_available_spare_threshold REAL",
        "ALTER TABLE sample ADD COLUMN nvme_data_units_read INTEGER",
        "ALTER TABLE sample ADD COLUMN nvme_data_units_written INTEGER",
        "ALTER TABLE sample ADD COLUMN nvme_host_read_cmds INTEGER",
        "ALTER TABLE sample ADD COLUMN nvme_host_write_cmds INTEGER",
        "ALTER TABLE sample ADD COLUMN nvme_controller_busy_time_min REAL",
        "ALTER TABLE sample ADD COLUMN nvme_power_cycles INTEGER",
        "ALTER TABLE sample ADD COLUMN nvme_unsafe_shutdowns INTEGER",
        "ALTER TABLE sample ADD COLUMN nvme_media_errors INTEGER",
        "ALTER TABLE sample ADD COLUMN nvme_num_err_log_entries INTEGER",
        "ALTER TABLE sample ADD COLUMN nvme_pcie_link_gen INTEGER",
        "ALTER TABLE sample ADD COLUMN nvme_pcie_link_width INTEGER",
    ]),
]


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        # check_same_thread=False so the kernel watcher thread can write events
        # too. All writes go through self._lock, so this is safe even though
        # sqlite3 connection objects are not thread-safe by default.
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        # Harden SQLite for weak filesystems (WSL 9p DrvFs, network shares).
        # Default rollback journal + synchronous=FULL hits "disk I/O error"
        # on 9p under sustained commit pressure. These PRAGMAs are safe and
        # near-no-ops on ext4, and each run has its own DB so EXCLUSIVE
        # locking never contends with anything.
        self.conn.execute("PRAGMA busy_timeout=10000")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA locking_mode=EXCLUSIVE")
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._schema_version: int = 0
        try:
            self._create()
        except BaseException:
            # Close the connection on schema/migration failure so the EXCLUSIVE
            # lock is released. Without this, a botched open would leave the
            # DB file locked until GC collected the instance — which under
            # locking_mode=EXCLUSIVE means a subsequent sqlite3.connect()
            # (e.g. recovery / rescore path) sees "database is locked".
            try:
                self.conn.close()
            except sqlite3.Error:
                pass
            raise

    def _create(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS run (
              id INTEGER PRIMARY KEY,
              started_at TEXT NOT NULL,
              ended_at TEXT,
              profile_name TEXT NOT NULL,
              profile_version TEXT NOT NULL,
              protocol_version TEXT NOT NULL,
              psu_make TEXT,
              psu_model TEXT,
              psu_wattage INTEGER,
              psu_serial TEXT,
              psu_lot TEXT,
              operator TEXT,
              reason TEXT,
              ambient_c REAL,
              manual_idle_12v REAL,
              manual_load_12v REAL,
              manual_idle_5v REAL,
              manual_load_5v REAL,
              meter_available INTEGER NOT NULL DEFAULT 0,
              paperclip_result TEXT,
              post_result TEXT,
              known_good_swap_performed INTEGER NOT NULL DEFAULT 0,
              known_good_swap_result TEXT,
              original_system_retest_result TEXT,
              bench_name TEXT,
              bench_last_verified TEXT,
              os_name TEXT,
              preflight_json TEXT NOT NULL,
              hardware_json TEXT NOT NULL,
              telemetry_status TEXT NOT NULL,
              load_scale REAL,
              verdict TEXT,
              verdict_reasons_json TEXT,
              subsystem_results_json TEXT,
              notes TEXT
            );

            CREATE TABLE IF NOT EXISTS sample (
              id INTEGER PRIMARY KEY,
              ts TEXT NOT NULL,
              phase TEXT NOT NULL,
              cpu_load_pct REAL,
              loadavg_1m REAL,
              mem_used_pct REAL,
              disk_busy_pct REAL,
              ssd_temp_c REAL,
              hba_count INTEGER,
              i2c_adapter_count INTEGER,
              i3c_bus_count INTEGER,
              cpu_temp_c REAL,
              cpu_pkg_watt REAL,
              cpu_avg_mhz REAL,
              cpu_max_mhz REAL,
              cpu_busy_pct_turbo REAL,
              cpu_core_watt REAL,
              cpu_pkg_watt_turbo REAL,
              gpu_temp_c REAL,
              gpu_vram_temp_c REAL,
              gpu_power_w REAL,
              gpu_power_total_w REAL,
              gpu_util_pct REAL,
              gpu_clock_sm_mhz INTEGER,
              gpu_clock_mem_mhz INTEGER,
              gpu_vram_used_mib REAL,
              gpu_vram_total_mib REAL,
              gpu_fan_pct INTEGER,
              gpu_ecc_ce REAL,
              gpu_ecc_ue REAL,
              gpu_throttle_active INTEGER,
              gpu_retired_pages REAL,
              gpu_power_viol_us REAL,
              gpu_thermal_viol_us REAL,
              pcie_link_gen INTEGER,
              pcie_link_width INTEGER,
              pcie_tx_kbps REAL,
              pcie_rx_kbps REAL,
              rail_12v REAL,
              rail_5v REAL,
              rail_33v REAL,
              fan_cpu_rpm REAL,
              fan_sys_rpm REAL,
              psu_vin_v REAL,
              psu_i_in_a REAL,
              psu_i_12v_a REAL,
              psu_i_5v_a REAL,
              psu_i_33v_a REAL,
              psu_p_in_w REAL,
              psu_p_12v_w REAL,
              psu_p_5v_w REAL,
              psu_p_33v_w REAL,
              psu_p_out_w REAL,
              psu_efficiency_pct REAL,
              psu_temp_c REAL,
              psu_temp2_c REAL,
              psu_fan_rpm REAL,
              mem_total_gb REAL,
              mem_speed_mts INTEGER,
              mobo_vrm_temp_c REAL,
              mobo_case_temp_c REAL,
              cpu_ccd_max_c REAL,
              cpu_ccd_spread_c REAL,
              ssd_temp_ctrl_c REAL,
              ssd_temp_nand_c REAL,
              pcie_aer_corr_total INTEGER,
              nic_mac_temp_c REAL,
              nic_phy_temp_c REAL,
              psi_cpu_some_avg10 REAL,
              psi_mem_some_avg10 REAL,
              psi_io_some_avg10 REAL,
              nic_rx_crc_errors_total INTEGER,
              nic_rx_errors_total INTEGER,
              nic_rx_dropped_total INTEGER,
              gpu_pcie_link_gen_max INTEGER,
              cpu_iowait_pct REAL,
              rail_12v_ripple_mv REAL,
              rail_5v_ripple_mv REAL,
              rail_33v_ripple_mv REAL,
              rail_12v_dropouts INTEGER,
              rail_5v_dropouts INTEGER,
              rail_33v_dropouts INTEGER,
              pcie_aer_uncorr_total INTEGER,
              gpu_throttle_hw_slowdown INTEGER,
              gpu_throttle_hw_thermal INTEGER,
              gpu_throttle_hw_power_brake INTEGER,
              gpu_throttle_sw_power_cap INTEGER,
              ssd_warn_temp_time_s INTEGER,
              ssd_crit_temp_time_s INTEGER,
              nic_link_speed_mbps INTEGER,
              nic_link_up INTEGER,
              cpu_freq_spread_mhz REAL,
              cpu_c6_pct REAL,
              cpu_ctx_switches_per_s REAL,
              cpu_interrupts_per_s REAL,
              cpu_softirq_per_s REAL,
              mem_pgmajfault_per_s REAL,
              mem_dirty_kb REAL,
              mem_writeback_kb REAL,
              ssd_read_mbps REAL,
              ssd_write_mbps REAL,
              ssd_queue_depth REAL,
              mobo_chipset_temp_c REAL,
              thermal_zone_margin_c REAL,
              mem_bw_read_mbps REAL,
              mem_bw_write_mbps REAL,
              ups_input_voltage_v REAL,
              ups_input_freq_hz REAL,
              ups_output_voltage_v REAL,
              ups_load_pct REAL,
              ups_realpower_w REAL,
              ups_battery_charge_pct REAL,
              ups_battery_runtime_s REAL,
              ups_status_raw TEXT,
              ups_on_battery INTEGER,
              ups_transfer_count INTEGER,
              ups_realpower_nominal_w REAL,
              ups_realpower_est_w REAL,
              ups_apparent_nominal_va REAL,
              ups_apparent_est_va REAL,
              nvme_composite_temp_c REAL,
              nvme_warning_temp_time_min REAL,
              nvme_critical_temp_time_min REAL,
              nvme_percentage_used REAL,
              nvme_available_spare REAL,
              nvme_available_spare_threshold REAL,
              nvme_data_units_read INTEGER,
              nvme_data_units_written INTEGER,
              nvme_host_read_cmds INTEGER,
              nvme_host_write_cmds INTEGER,
              nvme_controller_busy_time_min REAL,
              nvme_power_cycles INTEGER,
              nvme_unsafe_shutdowns INTEGER,
              nvme_media_errors INTEGER,
              nvme_num_err_log_entries INTEGER,
              nvme_pcie_link_gen INTEGER,
              nvme_pcie_link_width INTEGER,
              source_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sample_ts ON sample(ts);

            CREATE TABLE IF NOT EXISTS event (
              id INTEGER PRIMARY KEY,
              ts TEXT NOT NULL,
              phase TEXT NOT NULL,
              severity TEXT NOT NULL,
              kind TEXT NOT NULL,
              message TEXT NOT NULL
            );
            """
        )
        self.conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Apply pending migrations from _MIGRATIONS, tracking state in the
        schema_version table.

        The schema_version table is introduced in this code path — legacy DBs
        that pre-date it get bootstrapped at version 1 and then migrated
        forward. New DBs come out of _create() at the current schema shape;
        they also start at v1 and run the migrations, but the ALTER TABLE
        idempotency (OperationalError catch) keeps that a no-op.

        Adding a new column:
          1. Add the column to the CREATE TABLE in _create() so fresh DBs
             get it directly.
          2. Bump CURRENT_SCHEMA_VERSION.
          3. Append a new (from, to, [sql]) tuple to _MIGRATIONS with the
             ALTER TABLE statement.
          4. Update any rescore/fetch code that reads the new column.
        """
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version ("
            "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        row = self.conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
        current = row[0] if row and row[0] is not None else None
        if current is None:
            # Legacy DB or fresh DB: stamp v1 baseline so the migration loop
            # below has something to advance from. Commit immediately so that
            # if a later per-migration rollback fires, the v1 baseline is not
            # rolled back with it — that would leave schema_version empty and
            # re-trigger the bootstrap on next open.
            current = 1
            self.conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (1, utc_now()),
            )
            self.conn.commit()

        for from_v, to_v, stmts in _MIGRATIONS:
            if to_v <= current:
                continue
            if from_v != current:
                raise RuntimeError(
                    "Migration gap: DB at v{}, next migration expects v{}. "
                    "Check _MIGRATIONS ordering.".format(current, from_v)
                )
            # Wrap each migration step's statements + the version row insert
            # in a single transaction. Without this, a process kill between
            # the last ALTER TABLE and the schema_version insert would leave
            # the DB at the new shape but stamped at the old version — the
            # next open would replay the migration. Idempotent today (ALTERs
            # swallow duplicate-column); dangerous when a future migration
            # adds a multi-step data transform.
            #
            # We use conn.commit()/conn.rollback() rather than explicit
            # BEGIN/COMMIT SQL so we don't fight Python's sqlite3 implicit
            # transaction management.
            try:
                for sql in stmts:
                    try:
                        self.conn.execute(sql)
                    except sqlite3.OperationalError as exc:
                        # Only swallow the idempotent "column already exists"
                        # case (legacy DB ran the old _migrate ALTER TABLE list,
                        # or fresh DB where the column was already in _create).
                        # Anything else — disk full, read-only FS, corrupt page
                        # — must propagate so we don't half-migrate and produce
                        # a broken evidence DB.
                        if "duplicate column name" not in str(exc).lower():
                            raise
                self.conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (to_v, utc_now()),
                )
                self.conn.commit()
            except BaseException:
                # Any error (OperationalError, KeyboardInterrupt, etc.) must
                # roll back so the schema_version row does not land without
                # its matching shape changes, and vice versa.
                try:
                    self.conn.rollback()
                except sqlite3.Error:
                    pass
                raise
            current = to_v

        if current > CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                "DB schema v{} is newer than this binary supports (v{}). "
                "Downgrade-reads are unsafe — upgrade PS-FU.".format(
                    current, CURRENT_SCHEMA_VERSION
                )
            )

        self._schema_version = current
        self.conn.commit()

    def insert_run(self, payload: Dict[str, object]) -> int:
        with self._lock:
            columns = ", ".join(_quote_ident(col) for col in payload.keys())
            placeholders = ", ".join(["?"] * len(payload))
            cur = self.conn.execute(
                "INSERT INTO run ({}) VALUES ({})".format(columns, placeholders),
                list(payload.values()),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def update_run(self, run_id: int, payload: Dict[str, object]) -> None:
        with self._lock:
            sets = ", ".join("{} = ?".format(_quote_ident(col)) for col in payload)
            values = list(payload.values())
            values.append(run_id)
            self.conn.execute("UPDATE run SET {} WHERE id = ?".format(sets), values)
            self.conn.commit()

    def add_sample(self, payload: Dict[str, object]) -> None:
        with self._lock:
            columns = ", ".join(_quote_ident(col) for col in payload.keys())
            placeholders = ", ".join(["?"] * len(payload))
            self.conn.execute(
                "INSERT INTO sample ({}) VALUES ({})".format(columns, placeholders),
                list(payload.values()),
            )
            self.conn.commit()

    def add_event(self, phase: str, severity: str, kind: str, message: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO event (ts, phase, severity, kind, message) VALUES (?, ?, ?, ?, ?)",
                [utc_now(), phase, severity, kind, message],
            )
            self.conn.commit()

    def fetch_run(self) -> sqlite3.Row:
        # Lock held across execute() + fetchone() — sqlite3 Connection is not
        # re-entrant at the C layer, and the kernel watcher thread may be
        # writing events concurrently via add_event().
        with self._lock:
            # Belt-and-braces: _migrate() ran at __init__, so on healthy
            # construction this is always satisfied. Guards code paths that
            # hypothetically bypass __init__ (e.g. connection handoff from
            # another process) from reading columns a stale schema doesn't
            # have yet — silent NULLs were the exact failure mode this guards.
            if self._schema_version < CURRENT_SCHEMA_VERSION:
                raise RuntimeError(
                    "DB schema v{} below CURRENT_SCHEMA_VERSION v{} — "
                    "migration did not run or schema_version table missing.".format(
                        self._schema_version, CURRENT_SCHEMA_VERSION
                    )
                )
            return self.conn.execute("SELECT * FROM run ORDER BY id DESC LIMIT 1").fetchone()

    def fetch_samples(self) -> List[sqlite3.Row]:
        with self._lock:
            return self.conn.execute("SELECT * FROM sample ORDER BY id").fetchall()

    def fetch_events(self) -> List[sqlite3.Row]:
        with self._lock:
            return self.conn.execute("SELECT * FROM event ORDER BY id").fetchall()

    def close(self) -> None:
        """Close the SQLite connection and release any file-level locks.

        Refcount-based GC is unreliable here because we enable ``locking_mode=
        EXCLUSIVE`` — the OS-level lock persists until the connection is
        explicitly closed. On Windows/synced-folder paths this has caused the next
        ``psfu report`` / ``psfu verdict`` command to fail with "database is
        locked" when the previous run's connection was still dangling. Safe
        to call repeatedly; subsequent calls are a no-op.
        """
        conn = getattr(self, "conn", None)
        if conn is None:
            return
        with self._lock:
            try:
                conn.close()
            except Exception:
                pass
            self.conn = None  # type: ignore[assignment]

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
