import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import sqlite3

from psfu.config import FM_LOG_PATH, batch_tracker_path
from psfu.ui.operator import defensible_conclusion
from psfu.runtime import write_csv, write_json
from psfu.persistence.db import Database
from psfu.reports.markdown import render_summary, render_fm_entry, render_rma_summary, is_placeholder_serial
from psfu.verdict.confidence import coverage_confidence
from psfu.verdict.evidence import flatten_subsystem_evidence


def _rel(run_dir: Path, path: Path) -> str:
    """Path relative to the run dir (posix separators).

    Bundles are shared with third parties, so verdict.json must not embed
    absolute local paths (which would leak usernames and directory layout).
    """
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return path.name


def _load_storage_workload_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    candidates = sorted(run_dir.glob("storage/fio/*/*/write-cliff-analysis.json"))
    if not candidates:
        return None
    path = candidates[-1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    classification = payload.get("classification") if isinstance(payload.get("classification"), dict) else {}
    workload = payload.get("workload_stats") if isinstance(payload.get("workload_stats"), dict) else {}
    peak = payload.get("peak_window") if isinstance(payload.get("peak_window"), dict) else {}
    return {
        "phase_name": payload.get("phase_name"),
        "classification": classification.get("slug"),
        "classification_label": classification.get("label"),
        "cliff_ratio": payload.get("cliff_ratio"),
        "peak_window_avg_write_bw_kbps": peak.get("avg_write_bw_kbps"),
        "tail_window_median_kbps": payload.get("tail_window_median_kbps"),
        "major_drop_count": workload.get("major_drop_count"),
        "major_rise_count": workload.get("major_rise_count"),
        "tail_window_write_bw_cv": workload.get("tail_window_write_bw_cv"),
        "baseline_loaded": bool(payload.get("baseline_loaded")),
        "artifact": _rel(run_dir, path),
    }


def build_verdict_payload(
    run_dir: Path,
    run_row: sqlite3.Row,
    events: List[sqlite3.Row],
) -> Dict[str, Any]:
    row_keys = set(run_row.keys())
    confidence = coverage_confidence(run_dir, run_row, events)
    subsystems = json.loads(
        (run_row["subsystem_results_json"] if "subsystem_results_json" in row_keys else "{}") or "{}"
    )
    overall_evidence = flatten_subsystem_evidence(subsystems)
    storage_workload = _load_storage_workload_summary(run_dir)
    payload = {
        "overall": {
            "status": run_row["verdict"],
            "claim_scope": run_row["telemetry_status"],
            "confidence": confidence["score"],
            "confidence_factors": confidence["factors"],
            "reasons": json.loads(
                (run_row["verdict_reasons_json"] if "verdict_reasons_json" in row_keys else "[]") or "[]"
            ),
            "evidence": overall_evidence,
        },
        "subsystems": subsystems,
        "run": {
            "started_at": run_row["started_at"],
            "ended_at": run_row["ended_at"],
            "profile_name": run_row["profile_name"],
            "profile_version": run_row["profile_version"],
            "protocol_version": run_row["protocol_version"],
            "psu_make": run_row["psu_make"],
            "psu_model": run_row["psu_model"],
            "psu_wattage": run_row["psu_wattage"],
            "psu_serial": run_row["psu_serial"],
            "psu_lot": run_row["psu_lot"],
            "operator": run_row["operator"],
            "reason": run_row["reason"],
            "bench_name": run_row["bench_name"],
        },
        "artifacts": {
            "summary": "summary.md",
            "verdict": "verdict.json",
            "samples": "samples.csv",
            "events": "events.csv",
            "hardware": "hardware.json",
            "preflight": "preflight.json",
            "storage_health_start": "storage-health-start.json",
            "storage_health_end": "storage-health-end.json",
        },
    }
    if storage_workload is not None:
        payload["storage_workload"] = storage_workload
        payload["artifacts"]["storage_write_cliff_analysis"] = storage_workload["artifact"]
    return payload


def export_run_artifacts(run_dir: Path, db: Database) -> None:
    run_row = db.fetch_run()
    samples = db.fetch_samples()
    events = db.fetch_events()

    write_json(run_dir / "hardware.json", json.loads(run_row["hardware_json"]))
    write_json(run_dir / "preflight.json", json.loads(run_row["preflight_json"]))
    write_json(
        run_dir / "verdict.json",
        build_verdict_payload(run_dir, run_row, events),
    )
    
    sample_cols = [
        "ts", "phase", "cpu_load_pct", "loadavg_1m", "mem_used_pct", "disk_busy_pct",
        "cpu_iowait_pct",
        "ssd_temp_c", "ssd_temp_ctrl_c", "ssd_temp_nand_c",
        "ssd_warn_temp_time_s", "ssd_crit_temp_time_s",
        "ssd_read_mbps", "ssd_write_mbps", "ssd_queue_depth",
        "hba_count", "i2c_adapter_count", "i3c_bus_count",
        "cpu_temp_c", "cpu_pkg_watt",
        "cpu_ccd_max_c", "cpu_ccd_spread_c",
        "cpu_freq_spread_mhz", "cpu_c6_pct",
        "cpu_ctx_switches_per_s", "cpu_interrupts_per_s", "cpu_softirq_per_s",
        "mobo_vrm_temp_c", "mobo_case_temp_c", "mobo_chipset_temp_c",
        "thermal_zone_margin_c",
        "nic_mac_temp_c", "nic_phy_temp_c",
        "nic_link_speed_mbps", "nic_link_up",
        "gpu_temp_c", "gpu_vram_temp_c",
        "gpu_power_w", "gpu_power_total_w", "gpu_util_pct",
        "gpu_throttle_hw_slowdown", "gpu_throttle_hw_thermal",
        "gpu_throttle_hw_power_brake", "gpu_throttle_sw_power_cap",
        "pcie_link_gen", "pcie_link_width", "gpu_pcie_link_gen_max",
        "pcie_aer_corr_total", "pcie_aer_uncorr_total",
        "psi_cpu_some_avg10", "psi_mem_some_avg10", "psi_io_some_avg10",
        "nic_rx_crc_errors_total", "nic_rx_errors_total", "nic_rx_dropped_total",
        "mem_pgmajfault_per_s", "mem_dirty_kb", "mem_writeback_kb",
        "mem_bw_read_mbps", "mem_bw_write_mbps",
        "nvme_composite_temp_c", "nvme_warning_temp_time_min", "nvme_critical_temp_time_min",
        "nvme_percentage_used", "nvme_available_spare", "nvme_available_spare_threshold",
        "nvme_data_units_read", "nvme_data_units_written",
        "nvme_host_read_cmds", "nvme_host_write_cmds",
        "nvme_controller_busy_time_min", "nvme_power_cycles", "nvme_unsafe_shutdowns",
        "nvme_media_errors", "nvme_num_err_log_entries",
        "nvme_pcie_link_gen", "nvme_pcie_link_width",
        "rail_12v", "rail_5v", "rail_33v",
        "rail_12v_ripple_mv", "rail_5v_ripple_mv", "rail_33v_ripple_mv",
        "rail_12v_dropouts", "rail_5v_dropouts", "rail_33v_dropouts",
        "fan_cpu_rpm", "fan_sys_rpm",
    ]
    write_csv(run_dir / "samples.csv", samples, sample_cols)
    write_csv(run_dir / "events.csv", events, ["ts", "phase", "severity", "kind", "message"])

    def_conclusion = defensible_conclusion

    (run_dir / "summary.md").write_text(
        render_summary(run_dir, run_row, samples, events, def_conclusion),
        encoding="utf-8",
    )
    (run_dir / "fm-entry.md").write_text(
        render_fm_entry(
            run_dir,
            run_row,
            samples,
            events,
            def_conclusion,
            batch_tracker_path().name,
        ),
        encoding="utf-8",
    )
    (run_dir / "rma-summary.md").write_text(
        render_rma_summary(run_dir, run_row, samples, events, def_conclusion),
        encoding="utf-8",
    )

    _write_bundle_hash(run_dir)


# Files the RMA claim depends on. Any post-run edit to these invalidates
# the bundle. Ordered for deterministic hash output. New files must be added
# here explicitly so adding a file doesn't silently drop evidence integrity.
_BUNDLE_FILES: List[str] = [
    "verdict.json",
    "summary.md",
    "fm-entry.md",
    "rma-summary.md",
    "samples.csv",
    "events.csv",
    "hardware.json",
    "preflight.json",
]


def _write_bundle_hash(run_dir: Path) -> None:
    """Write ``bundle.sha256`` covering the canonical evidence files.

    Format: two lines per file — ``<sha256>  <filename>`` (double-space,
    coreutils compatible). Re-running ``sha256sum -c bundle.sha256`` in the
    run dir re-validates the bundle. A missing file writes MISSING in place
    of the hash; a post-run edit changes the hash; either way tampering is
    detectable without crypto infrastructure.
    """
    lines: List[str] = []
    for name in _BUNDLE_FILES:
        path = run_dir / name
        if not path.exists():
            lines.append("MISSING  {}".format(name))
            continue
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        lines.append("{}  {}".format(h.hexdigest(), name))
    (run_dir / "bundle.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_bundle_hash_digest(run_dir: Path) -> Optional[str]:
    """Digest of the bundle.sha256 file itself — short identifier suitable
    for the FM log one-liner. Returns None if bundle.sha256 is missing."""
    path = run_dir / "bundle.sha256"
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


_FM_LOG_STUB = """\
# Failure Mode Log

<!-- newest first -->
"""


def append_to_fm_log(run_dir: Path, db: Database) -> Optional[str]:
    run_row = db.fetch_run()
    if run_row["verdict"] != "FAIL":
        return None

    fm_entry_path = run_dir / "fm-entry.md"
    if not fm_entry_path.exists():
        return None

    # Create the log file if it doesn't exist (first run on USB / standalone).
    if not FM_LOG_PATH.exists():
        try:
            FM_LOG_PATH.write_text(_FM_LOG_STUB, encoding="utf-8")
        except OSError:
            return None

    try:
        log_content = FM_LOG_PATH.read_text(encoding="utf-8")
    except OSError:
        return None

    existing = re.findall(r"FM-(\d{4})", log_content)
    next_num = max((int(n) for n in existing), default=0) + 1
    fm_number = "FM-{:04d}".format(next_num)

    try:
        entry_text = fm_entry_path.read_text(encoding="utf-8")
    except OSError:
        return None

    entry_text = entry_text.replace("FM-XXXX", fm_number)

    # Append evidence-integrity line so the FM log entry records which
    # bundle state the claim is based on. If someone edits verdict.json
    # after the fact, the bundle hash recomputed from bundle.sha256 will
    # no longer match this recorded value.
    bundle_digest = read_bundle_hash_digest(run_dir)
    if bundle_digest is not None:
        entry_text = entry_text.rstrip() + "\n\n**Bundle hash:** `{}` (see `bundle.sha256` in run dir)\n".format(
            bundle_digest
        )

    try:
        fm_entry_path.write_text(entry_text, encoding="utf-8")
    except OSError:
        pass

    marker = "<!-- newest first -->"
    if marker not in log_content:
        return None

    idx = log_content.index(marker) + len(marker)
    block = "\n\n" + entry_text.strip() + "\n\n---"
    new_log = log_content[:idx] + block + log_content[idx:]

    try:
        FM_LOG_PATH.write_text(new_log, encoding="utf-8")
    except OSError:
        return None

    # Auto-copy run bundle to cases/SN-<serial>/PS-FU-run/ for RMA package.
    # Non-fatal — a copy failure never blocks the FM log write.
    psu_serial = run_row["psu_serial"] or ""
    if psu_serial and not is_placeholder_serial(psu_serial):
        dest = FM_LOG_PATH.parent / "cases" / "SN-{}".format(psu_serial) / "PS-FU-run"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(str(run_dir), str(dest))
        except OSError:
            pass

    return fm_number


def _batch_tracker_verdict_label(verdict: str) -> str:
    if verdict == "PASS":
        return "**PASS**"
    if verdict == "FAIL":
        return "**FAIL**"
    if verdict in ("WARN", "INCONCLUSIVE"):
        return "**SUSPECT**"
    return verdict


def _batch_tracker_status_key(verdict: str) -> Optional[str]:
    if verdict == "PASS":
        return "PASS"
    if verdict == "FAIL":
        return "FAIL"
    if verdict in ("WARN", "INCONCLUSIVE"):
        return "SUSPECT (needs full 1hr)"
    return None


def update_batch_tracker(run_dir: Path, db: Database) -> bool:
    run_row = db.fetch_run()
    verdict = run_row["verdict"] or "INVALID"
    status_key = _batch_tracker_status_key(verdict)
    if status_key is None:
        return False
    tracker_path = batch_tracker_path()
    if not tracker_path.exists():
        return False

    if is_placeholder_serial(run_row["psu_serial"]):
        return False

    try:
        content = tracker_path.read_text(encoding="utf-8")
    except OSError:
        return False

    lines = content.splitlines(keepends=True)
    row_re = re.compile(r"^\|\s*(\d+)\s*\|(\s*)\|")
    filled_idx = None
    for i, line in enumerate(lines):
        m = row_re.match(line)
        if m and not m.group(2).strip():
            row_num = int(m.group(1))
            def _v(val: object, fallback: str = "n/a") -> str:
                return str(val) if val is not None else fallback

            make_model = "{} {} ({}W)".format(
                run_row["psu_make"] or "?",
                run_row["psu_model"] or "?",
                run_row["psu_wattage"] or "?",
            )
            serial = run_row["psu_serial"] or "<!-- NEEDS -->"
            lot = run_row["psu_lot"] or "<!-- NEEDS -->"
            paperclip = run_row["paperclip_result"] or "not-run"
            idle_12v = _v(run_row["manual_idle_12v"])
            load_12v = _v(run_row["manual_load_12v"])
            idle_5v  = _v(run_row["manual_idle_5v"])
            load_5v  = _v(run_row["manual_load_5v"])
            result_label = _batch_tracker_verdict_label(verdict)
            date_str = (run_row["ended_at"] or run_row["started_at"] or "")[:10]
            note = "Tested {}. Profile: {}. Run: {}.".format(
                date_str, run_row["profile_name"] or "?", run_dir.name
            )
            filled_line = "| {:<2} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                row_num, make_model, serial, lot, paperclip,
                idle_12v, load_12v, idle_5v, load_5v,
                result_label, note,
            )
            lines[i] = filled_line
            filled_idx = i
            break

    if filled_idx is None:
        return False

    summary_re = re.compile(r"^(\|\s*)(.+?)(\s*\|\s*)(\d+)(\s*\|)")
    def _inc_summary_row(keyword: str, delta: int) -> None:
        for j, line in enumerate(lines):
            m = summary_re.match(line)
            if m and keyword.lower() in m.group(2).lower():
                new_count = max(0, int(m.group(4)) + delta)
                lines[j] = m.group(1) + m.group(2) + m.group(3) + str(new_count) + m.group(5) + "\n"
                return

    _inc_summary_row(status_key, +1)
    _inc_summary_row("Not yet tested", -1)

    try:
        tracker_path.write_text("".join(lines), encoding="utf-8")
    except OSError:
        return False

    return True
