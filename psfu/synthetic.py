"""Synthetic data generator: deterministic fake runs for demos and tests.

Nothing here touches hardware. ``build_synthetic_run`` writes a complete run
directory (SQLite DB, scored verdict, exported evidence bundle) from a seeded
random model of a 1 Hz sensor log, in one of three scenarios:

* ``pass`` -- stable rails, no hardware events
* ``warn`` -- 12V rail sags into the warning band under load
* ``fail`` -- deeper 12V sag plus repeated PCIe AER events and an NVMe I/O timeout

The same generator backs ``psfu demo`` and the golden tests, so the sample
output in the README is exactly what the verdict engine produces.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from psfu.common import utc_now
from psfu.config import PROTOCOL_VERSION
from psfu.persistence.db import Database
from psfu.reports.exporter import export_run_artifacts
from psfu.storage.cliff import analyze_write_cliff, write_cliff_analysis
from psfu.verdict.rules import VerdictEngine

SCENARIOS = ("pass", "warn", "fail")

# (phase name, duration in seconds)
_PHASES: List[Tuple[str, int]] = [
    ("soak", 30),
    ("full-load", 60),
    ("steady", 60),
    ("transient", 30),
    ("cooldown", 30),
]
_EPOCH = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)


def base_run_metadata() -> Dict[str, Any]:
    return {
        "started_at": utc_now(),
        "profile_name": "test-profile",
        "profile_version": "0.1",
        "protocol_version": PROTOCOL_VERSION,
        "psu_make": "Testco",
        "psu_model": "T-1000",
        "psu_wattage": 1000,
        "psu_serial": "T1000-ABC",
        "psu_lot": "LOT-X",
        "operator": "pytest",
        "reason": "unit test",
        "ambient_c": 22.0,
        "manual_idle_12v": None,
        "manual_load_12v": None,
        "manual_idle_5v": None,
        "manual_load_5v": None,
        "meter_available": 0,
        "paperclip_result": "pass",
        "post_result": "pass",
        "known_good_swap_performed": 0,
        "known_good_swap_result": "not-done",
        "original_system_retest_result": "not-run",
        "bench_name": "test-bench",
        "bench_last_verified": "2026-04-11",
        "os_name": "pytest",
        "preflight_json": json.dumps({"checks": []}),
        "hardware_json": json.dumps({}),
        "telemetry_status": "partial",
        "verdict": None,
        "verdict_reasons_json": None,
        "notes": "",
    }


def base_sample(phase: str = "soak", **overrides: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ts": utc_now(),
        "phase": phase,
        "cpu_load_pct": 5.0,
        "loadavg_1m": 0.1,
        "mem_used_pct": 20.0,
        "disk_busy_pct": 0.0,
        "ssd_temp_c": 40.0,
        "hba_count": 0,
        "i2c_adapter_count": 0,
        "i3c_bus_count": 0,
        "cpu_temp_c": 45.0,
        "cpu_pkg_watt": 30.0,
        "cpu_avg_mhz": None,
        "cpu_max_mhz": None,
        "cpu_busy_pct_turbo": None,
        "cpu_core_watt": None,
        "cpu_pkg_watt_turbo": None,
        "gpu_temp_c": 40.0,
        "gpu_vram_temp_c": 50.0,
        "gpu_power_w": 30.0,
        "gpu_power_total_w": 30.0,
        "gpu_util_pct": 0.0,
        "gpu_clock_sm_mhz": 210,
        "gpu_clock_mem_mhz": 405,
        "gpu_vram_used_mib": 500.0,
        "gpu_vram_total_mib": 24576.0,
        "gpu_fan_pct": None,
        "gpu_ecc_ce": None,
        "gpu_ecc_ue": None,
        "gpu_throttle_active": 0,
        "gpu_retired_pages": None,
        "gpu_power_viol_us": None,
        "gpu_thermal_viol_us": None,
        "pcie_link_gen": 4,
        "pcie_link_width": 16,
        "pcie_tx_kbps": None,
        "pcie_rx_kbps": None,
        "rail_12v": 12.10,
        "rail_5v": 5.02,
        "rail_33v": 3.31,
        "fan_cpu_rpm": 800.0,
        "fan_sys_rpm": 600.0,
        "psu_vin_v": None,
        "psu_i_in_a": None,
        "psu_i_12v_a": None,
        "psu_i_5v_a": None,
        "psu_i_33v_a": None,
        "psu_p_in_w": None,
        "psu_p_12v_w": None,
        "psu_p_5v_w": None,
        "psu_p_33v_w": None,
        "psu_p_out_w": None,
        "psu_efficiency_pct": None,
        "psu_temp_c": None,
        "psu_temp2_c": None,
        "psu_fan_rpm": None,
        "mem_total_gb": None,
        "mem_speed_mts": None,
        "mobo_vrm_temp_c": None,
        "mobo_case_temp_c": None,
        "cpu_ccd_max_c": None,
        "cpu_ccd_spread_c": None,
        "ssd_temp_ctrl_c": None,
        "ssd_temp_nand_c": None,
        "pcie_aer_corr_total": None,
        "nic_mac_temp_c": None,
        "nic_phy_temp_c": None,
        "psi_cpu_some_avg10": None,
        "psi_mem_some_avg10": None,
        "psi_io_some_avg10": None,
        "nic_rx_crc_errors_total": None,
        "nic_rx_errors_total": None,
        "nic_rx_dropped_total": None,
        "gpu_pcie_link_gen_max": None,
        "cpu_iowait_pct": None,
        # --- v7 Tier 1 ---
        "rail_12v_ripple_mv": None,
        "rail_5v_ripple_mv": None,
        "rail_33v_ripple_mv": None,
        "rail_12v_dropouts": None,
        "rail_5v_dropouts": None,
        "rail_33v_dropouts": None,
        "pcie_aer_uncorr_total": None,
        "gpu_throttle_hw_slowdown": None,
        "gpu_throttle_hw_thermal": None,
        "gpu_throttle_hw_power_brake": None,
        "gpu_throttle_sw_power_cap": None,
        "ssd_warn_temp_time_s": None,
        "ssd_crit_temp_time_s": None,
        "nic_link_speed_mbps": None,
        "nic_link_up": None,
        # --- v7 Tier 2 ---
        "cpu_freq_spread_mhz": None,
        "cpu_c6_pct": None,
        "cpu_ctx_switches_per_s": None,
        "cpu_interrupts_per_s": None,
        "cpu_softirq_per_s": None,
        "mem_pgmajfault_per_s": None,
        "mem_dirty_kb": None,
        "mem_writeback_kb": None,
        "ssd_read_mbps": None,
        "ssd_write_mbps": None,
        "ssd_queue_depth": None,
        "mobo_chipset_temp_c": None,
        "thermal_zone_margin_c": None,
        # --- v7 memory bandwidth ---
        "mem_bw_read_mbps": None,
        "mem_bw_write_mbps": None,
        # --- v10 NVMe telemetry ---
        "nvme_composite_temp_c": None,
        "nvme_warning_temp_time_min": None,
        "nvme_critical_temp_time_min": None,
        "nvme_percentage_used": None,
        "nvme_available_spare": None,
        "nvme_available_spare_threshold": None,
        "nvme_data_units_read": None,
        "nvme_data_units_written": None,
        "nvme_host_read_cmds": None,
        "nvme_host_write_cmds": None,
        "nvme_controller_busy_time_min": None,
        "nvme_power_cycles": None,
        "nvme_unsafe_shutdowns": None,
        "nvme_media_errors": None,
        "nvme_num_err_log_entries": None,
        "nvme_pcie_link_gen": None,
        "nvme_pcie_link_width": None,
        "source_json": json.dumps({}),
    }
    payload.update(overrides)
    return payload


def _sag_v(scenario: str, phase: str) -> float:
    """12V sag (volts) applied during a phase for the given scenario."""
    if phase in ("soak", "cooldown"):
        return 0.0
    if scenario == "warn":
        return 0.55 if phase == "full-load" else 0.20
    if scenario == "fail":
        return 1.20 if phase == "full-load" else 0.45
    return 0.06 if phase == "full-load" else 0.03


def _synth_samples(scenario: str, rng: random.Random) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    t = 0
    for phase, seconds in _PHASES:
        loaded = phase in ("full-load", "steady", "transient")
        for _ in range(seconds):
            ts = (_EPOCH + dt.timedelta(seconds=t)).isoformat()
            heat = min(1.0, t / 90.0) if loaded else 0.2
            v12 = 12.10 - _sag_v(scenario, phase) + rng.gauss(0, 0.008)
            gpu_w = round(310 + rng.gauss(0, 8), 1) if loaded else round(30 + rng.gauss(0, 1), 1)
            samples.append(
                base_sample(
                    phase,
                    ts=ts,
                    cpu_load_pct=round(96 + rng.gauss(0, 1.5), 1) if loaded else round(4 + rng.random() * 3, 1),
                    cpu_temp_c=round(45 + 38 * heat + rng.gauss(0, 0.6), 1),
                    cpu_pkg_watt=round(240 + rng.gauss(0, 6), 1) if loaded else round(28 + rng.gauss(0, 1), 1),
                    gpu_temp_c=round(40 + 41 * heat + rng.gauss(0, 0.5), 1),
                    gpu_power_w=gpu_w,
                    gpu_power_total_w=gpu_w,
                    disk_busy_pct=round(70 + rng.gauss(0, 5), 1) if phase == "transient" else round(rng.random() * 2, 1),
                    gpu_util_pct=99.0 if loaded else 0.0,
                    rail_12v=round(v12, 3),
                    rail_5v=round(5.02 + rng.gauss(0, 0.004), 3),
                    rail_33v=round(3.31 + rng.gauss(0, 0.003), 3),
                    ssd_temp_c=round(38 + 14 * heat + rng.gauss(0, 0.4), 1),
                    mem_used_pct=round(62 + rng.gauss(0, 0.4), 1) if loaded else 21.0,
                    fan_cpu_rpm=round(800 + 1500 * heat + rng.gauss(0, 20)),
                )
            )
            t += 1
    return samples


def _synth_events(scenario: str) -> List[Tuple[str, str, str, str]]:
    """(phase, severity, kind, message) rows."""
    events: List[Tuple[str, str, str, str]] = [
        ("run", "info", "dut_selection", "armed DUTs: CPU, GPU, MEM, STORAGE (synthetic)"),
        ("full-load", "info", "phase_start", "phase full-load started (synthetic)"),
    ]
    if scenario == "fail":
        events += [
            (
                "full-load",
                "warn",
                "pcie_aer",
                "pcie 0000:00:00.0: AER: Corrected error received (synthetic #{})".format(n),
            )
            for n in range(1, 6)
        ]
        events.append(
            (
                "transient",
                "crit",
                "nvme_io_timeout",
                "nvme nvme0: I/O 123 QID 4 timeout, aborting (synthetic)",
            )
        )
    return events


def _total_seconds() -> int:
    return sum(seconds for _, seconds in _PHASES)


def _add_events(db: Database, rows: List[Tuple[str, str, str, str]]) -> None:
    """Insert events with deterministic timestamps (``Database.add_event``
    stamps wall-clock time, which would make the demo output irreproducible)."""
    phase_start = {}
    t = 0
    for name, seconds in _PHASES:
        phase_start[name] = t
        t += seconds
    with db._lock:
        for i, (phase, severity, kind, message) in enumerate(rows):
            ts = (_EPOCH + dt.timedelta(seconds=phase_start.get(phase, 0) + 5 + i)).isoformat()
            db.conn.execute(
                "INSERT INTO event (ts, phase, severity, kind, message) VALUES (?, ?, ?, ?, ?)",
                [ts, phase, severity, kind, message],
            )
        db.conn.commit()


def _synth_write_cliff(run_dir: Path, scenario: str) -> None:
    """A fake fio interval trace with a pSLC-style write cliff."""
    rng = random.Random(11)
    rows: List[Dict[str, Any]] = []
    for sec in range(240):
        if scenario == "pass":
            bw = 3_000_000
        else:
            bw = 5_800_000 if sec < 90 else 1_400_000
        rows.append({"write_bw_kbps": bw * (1 + rng.gauss(0, 0.02))})
    analysis = analyze_write_cliff(rows, phase_name="cliff-write", device="/dev/nvme0n1")
    if analysis is None:
        return
    out = run_dir / "storage" / "fio" / "cliff-write" / "seq_write-nvme0n1"
    out.mkdir(parents=True, exist_ok=True)
    write_cliff_analysis(out / "write-cliff-analysis.json", analysis)


def build_synthetic_run(
    run_dir: Path,
    scenario: str = "pass",
    *,
    seed: int = 7,
    export: bool = True,
) -> Dict[str, Any]:
    """Create a fully scored synthetic run in ``run_dir``.

    Returns ``{"status", "reasons", "claim_scope", "subsystems", "run_dir"}``.
    Fully deterministic for a given ``(scenario, seed)``.
    """
    if scenario not in SCENARIOS:
        raise ValueError("scenario must be one of {}".format(", ".join(SCENARIOS)))
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    db = Database(run_dir / "run.sqlite")
    try:
        meta = base_run_metadata()
        meta.update(
            {
                "started_at": _EPOCH.isoformat(),
                "profile_name": "synthetic-demo",
                "psu_make": "ExampleCo",
                "psu_model": "EX-1000",
                "psu_serial": "SYNTH-0001",
                "psu_lot": "LOT-DEMO",
                "operator": "demo",
                "reason": "synthetic demo ({})".format(scenario),
                "bench_name": "synthetic-bench",
                "os_name": "synthetic",
                "bench_last_verified": _EPOCH.date().isoformat(),
            }
        )
        run_id = db.insert_run(meta)
        for sample in _synth_samples(scenario, rng):
            db.add_sample(sample)
        _add_events(db, _synth_events(scenario))

        status, reasons, claim_scope, subsystems = VerdictEngine.evaluate(db)
        db.update_run(
            run_id,
            {
                "ended_at": (_EPOCH + dt.timedelta(seconds=_total_seconds())).isoformat(),
                "verdict": status,
                "verdict_reasons_json": json.dumps(reasons, indent=2),
                "subsystem_results_json": json.dumps(subsystems, indent=2),
                "telemetry_status": claim_scope,
            },
        )
        _synth_write_cliff(run_dir, scenario)
        if export:
            export_run_artifacts(run_dir, db)
    finally:
        db.close()
    return {
        "status": status,
        "reasons": reasons,
        "claim_scope": claim_scope,
        "subsystems": subsystems,
        "run_dir": run_dir,
    }
