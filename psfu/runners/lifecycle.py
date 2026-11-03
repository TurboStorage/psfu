"""Run lifecycle: prepare_run / finalize_run and supporting helpers.

Extracted from execute.py so the main phase loop file stays focused on the
per-tick sample loop and orchestration. This module handles everything that
happens before phases start and after they finish.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from psfu.batch import maybe_batch_pattern_event
from psfu.collectors.edac import edac_delta, edac_events_from_delta, snapshot_edac
from psfu.common import command_exists, utc_now
from psfu.config import APP_NAME, FM_LOG_PATH, PROTOCOL_VERSION, batch_tracker_path
from psfu.env import ensure_runtime_dependencies
from psfu.hardware import (
    can_run_sudo_noninteractive,
    gpu_burn_binary_path,
    nvidia_gpu_current_power_limits,
    nvidia_gpu_power_limit_ranges,
)
from psfu.power_budget import compute_power_budget, format_budget_report
from psfu.ui.operator import (
    final_multimeter_readings,
    post_run_observations,
    pre_run_observations,
    prompt_storage_dut_paths,
    prompt_int_if_missing,
    prompt_if_missing,
)
from psfu.reports.exporter import (
    append_to_fm_log,
    export_run_artifacts,
    update_batch_tracker,
)
from psfu.runtime import (
    current_bench_slug,
    create_run_dir,
    get_hardware_snapshot,
    get_preflight,
    load_profile,
    print_subsystem_summary,
    write_json,
)
from psfu.persistence.db import Database
from psfu.collectors.storage_health import capture_storage_health, storage_health_events
from psfu.collectors.storage_ocp import capture_storage_ocp, storage_ocp_events
from psfu.collectors.storage_ocp_latency import write_ocp_latency_artifacts, storage_ocp_latency_events
from psfu.collectors.storage_ata import capture_storage_ata, storage_ata_events
from psfu.collectors.nvme_fw_check import check_nvme_fw_revisions
from psfu.verdict.rules import VerdictEngine
from psfu.storage.devices import (
    capture_nvme_artifacts,
    discover_storage_devices,
    parse_storage_dut_paths,
    validate_storage_dut_paths,
)


@dataclass
class RunPreparation:
    profile: Any
    serial: str
    run_dir: Path
    storage_health_start: Dict[str, object]
    storage_ocp_start: Dict[str, object]
    storage_ocp_latency_start: Dict[str, object]
    storage_ata_start: Dict[str, object]
    edac_start: Dict[str, object]
    bootstrap_report: Dict[str, object]
    db: Database
    run_id: int
    psu_wattage_val: int
    meter_available: bool


def _validate_gpu_power_scaling(profile: Any) -> None:
    scaled_gpu_phases = [
        phase for phase in profile.phases
        if float(getattr(phase, "gpu_load_fraction", 1.0)) < 0.999
    ]
    if not scaled_gpu_phases:
        return

    phase_bits = ", ".join(
        "{}={}{}".format(
            phase.name,
            int(round(float(phase.gpu_load_fraction) * 100)),
            "%",
        )
        for phase in scaled_gpu_phases
    )

    if not command_exists("nvidia-smi"):
        raise SystemExit(_gpu_scaling_banner(
            profile.name, phase_bits,
            reason="nvidia-smi is not on PATH",
            fix=(
                "  Install the NVIDIA user-space tools (nvidia-utils or the driver\n"
                "  package that ships nvidia-smi), or remove gpu_load_fraction from\n"
                "  the listed phases to run them at 100% TDP."
            ),
        ))

    if not can_run_sudo_noninteractive():
        raise SystemExit(_gpu_scaling_banner(
            profile.name, phase_bits,
            reason="non-interactive sudo is not cached for `nvidia-smi -pl`",
            fix=(
                "  1. In your shell, run:\n"
                "         sudo -v\n"
                "     Enter your password once; credentials cache for ~15 min.\n\n"
                "  2. Re-run PS-FU within 15 minutes of step 1.\n\n"
                "  Alternative: remove gpu_load_fraction from the listed phases\n"
                "  to run them at 100% TDP (no sudo required)."
            ),
        ))

    _warn_on_gpu_scaling_clamp(profile, scaled_gpu_phases)


def _warn_on_gpu_scaling_clamp(profile: Any, scaled_gpu_phases: List[Any]) -> None:
    """Consumer GPUs reject `nvidia-smi -pl` below power.min_limit (RTX 5090
    ~300W, RTX 4090 ~150W). Surface the clamp up front so the operator isn't
    surprised when step-20/40 phases draw the same wattage."""
    current = nvidia_gpu_current_power_limits()
    ranges = nvidia_gpu_power_limit_ranges()
    if not current or not ranges:
        return
    clamped: List[str] = []
    for phase in scaled_gpu_phases:
        scale = float(getattr(phase, "gpu_load_fraction", 1.0))
        for idx, cap in current:
            min_limit, _ = ranges.get(idx, (0, 0))
            if not min_limit:
                continue
            raw = int(cap * scale)
            if raw < min_limit:
                clamped.append(
                    "  {}: GPU {} request {}W -> clamped to min {}W ({}% effective)".format(
                        phase.name, idx, raw, min_limit,
                        int(round(min_limit / cap * 100)),
                    )
                )
    if not clamped:
        return
    bar = "-" * 72
    sys.stdout.write(
        "\n" + bar + "\n"
        "  NOTE: GPU power scaling will be clamped by hardware min limit\n"
        + bar + "\n"
        + "\n".join(clamped) + "\n"
        + bar + "\n"
    )
    sys.stdout.flush()


def _gpu_scaling_banner(profile_name: str, phase_bits: str, *, reason: str, fix: str) -> str:
    """Render a prominent operator-facing banner for GPU power-scaling preflight
    failures. Preflight is the last place a 25-minute run gets aborted before
    workers spawn — a one-line error is easy to miss; a banner is not."""
    bar = "=" * 72
    return (
        "\n" + bar + "\n"
        "  OPERATOR ACTION REQUIRED — GPU power scaling preflight failed\n"
        + bar + "\n"
        "  Profile:  {profile}\n"
        "  Phases:   {phases}\n"
        "  Tool:     nvidia-smi -pl (requires root)\n"
        "  Reason:   {reason}\n\n"
        "{fix}\n"
        + bar
    ).format(profile=profile_name, phases=phase_bits, reason=reason, fix=fix)


def _compose_notes(user_notes: Optional[str], override_used: bool, budget) -> str:
    """Preserve operator notes; append an override marker if the bench power
    budget check was bypassed so the record survives into FM log / RMA bundle.
    """
    base = (user_notes or "").strip()
    if not override_used:
        return base
    marker = (
        "[bench-power-budget] --allow-undersized-psu used: PSU {}W vs required {}W "
        "sustained (CPU {}W TDP + GPU {}W max + {}W overhead)."
    ).format(
        budget.psu_rated_watts,
        budget.sustained_required,
        budget.cpu_tdp_watts,
        budget.gpu_max_watts,
        budget.overhead_watts,
    )
    return "{}\n\n{}".format(base, marker) if base else marker


def _step(label: str) -> float:
    """Print a prep step and return monotonic timestamp for elapsed calc."""
    sys.stdout.write("\r\033[K[prep] {}...".format(label))
    sys.stdout.flush()
    return time.monotonic()


def _step_done(t0: float) -> None:
    elapsed = time.monotonic() - t0
    sys.stdout.write(" done ({:.1f}s)\n".format(elapsed))
    sys.stdout.flush()


def prepare_run(args: argparse.Namespace) -> RunPreparation:
    """Load the profile, create the run dir, snapshot storage health, run the
    bootstrap readiness pass, and insert the initial run row into SQLite.
    Returns a bundle of everything the phase loop needs."""

    prep_start = time.monotonic()
    t0 = _step("loading profile")
    profile = load_profile(args.profile)
    _validate_gpu_power_scaling(profile)
    _step_done(t0)

    storage_inventory = discover_storage_devices()
    storage_dut_paths = validate_storage_dut_paths(
        list(getattr(args, "storage_dut_path", []) or []),
        storage_inventory,
    )
    requires_storage_dut = bool(profile.name.lower().startswith("storage"))
    if requires_storage_dut and not storage_dut_paths:
        storage_dut_paths = prompt_storage_dut_paths(
            None,
            storage_inventory,
            required=requires_storage_dut,
        )
        args.storage_dut_path = storage_dut_paths

    serial = args.psu_serial or time.strftime("run-%Y%m%d-%H%M%S")
    t0 = _step("creating run dir")
    run_category = None
    if getattr(args, "bench_gage_rr", False):
        run_category = Path("gage-rr") / current_bench_slug()
    run_dir = create_run_dir(serial, category=run_category)
    _step_done(t0)

    t0 = _step("storage health snapshot")
    storage_health_start = capture_storage_health()
    write_json(run_dir / "storage-health-start.json", storage_health_start)
    capture_nvme_artifacts(run_dir, storage_dut_paths, stage="start")
    # S7 (v0.18) — OCP 0xC0 SMART extended snapshot. Pair captured here +
    # in finalize_run; delta scored via storage_ocp_events. Drives that
    # don't expose OCP emit a single info-only ssd_ocp_not_supported event
    # (Crucial, older Samsungs, most WD Blues).
    storage_ocp_start = capture_storage_ocp(storage_dut_paths or None)
    write_json(run_dir / "storage-ocp-start.json", storage_ocp_start)
    # OCP 0xC3 latency monitor log — pre-run snapshot. Diagnostic only;
    # no FAIL/WARN verdict rules. Captures per-bucket IO latency histograms
    # for forensics. Drives that don't expose 0xC3 emit ssd_ocp_latency_not_supported.
    storage_ocp_latency_start = write_ocp_latency_artifacts(run_dir, storage_dut_paths or None, "start")
    # S9 (v0.19) — ATA SMART attribute snapshot for SATA/HDD DUTs.
    # Scans all /dev/sd? by default; NVMe DUTs emit one info-only
    # ata_smart_not_supported event per device and no FAIL path. Paired
    # with finalize_run for delta scoring via storage_ata_events.
    storage_ata_start = capture_storage_ata()
    write_json(run_dir / "storage-ata-start.json", storage_ata_start)
    _step_done(t0)

    t0 = _step("EDAC snapshot")
    edac_start = snapshot_edac()
    write_json(run_dir / "edac-start.json", edac_start)
    _step_done(t0)

    sys.stdout.write("[prep] checking runtime dependencies...\n")
    sys.stdout.flush()
    t0 = time.monotonic()
    bootstrap_report = ensure_runtime_dependencies(auto_install=False)
    write_json(run_dir / "bootstrap.json", bootstrap_report)
    elapsed = time.monotonic() - t0
    sys.stdout.write("[prep] dependencies checked ({:.1f}s)\n".format(elapsed))
    sys.stdout.flush()

    t0 = _step("initializing database")
    db = Database(run_dir / "run.sqlite")
    _step_done(t0)

    psu_wattage_val = 0
    if args.psu_wattage is not None:
        try:
            psu_wattage_val = int(str(args.psu_wattage).strip() or "0")
        except ValueError:
            psu_wattage_val = 0

    # Bench power budget gate. PSU-under-test must be rated to
    # cover sustained CPU + GPU + bench overhead, otherwise PS-FU stress load
    # can push a healthy PSU past its envelope and cause bench damage that
    # masquerades as a DUT failure. Operator override flag is recorded.
    allow_undersized = bool(getattr(args, "allow_undersized_psu", False))
    budget = compute_power_budget(psu_wattage_val)
    psfu_requires_psu_watts = budget.sustained_required
    psu_budget_override = False
    print(format_budget_report(budget))
    if psu_wattage_val <= 0:
        print(
            "[WARN] Bench power budget: PSU wattage not provided. Proceeding "
            "without the power-budget preflight gate. Pass --psu-wattage (nameplate "
            "watts) on future runs to verify the PSU can safely power a "
            "{}W sustained bench load.".format(psfu_requires_psu_watts)
        )
    if not budget.is_adequate():
        print(
            "[WARN] Bench power budget: PSU rated {}W is insufficient for this "
            "bench (sustained need {}W, short {}W). Recommended "
            "PSU >= {}W. Proceeding anyway (warn-only); the run records this.".format(
                psu_wattage_val,
                psfu_requires_psu_watts,
                budget.shortfall_watts(),
                psfu_requires_psu_watts,
            )
        )
        psu_budget_override = True

    # Pre-run observations: paperclip, POST, meter availability.
    pre_obs = pre_run_observations(args)

    t0 = _step("preflight checks")
    preflight_data = get_preflight()
    _step_done(t0)

    t0 = _step("hardware snapshot (lscpu, nvidia-smi, smartctl...)")
    hardware_data = get_hardware_snapshot()
    _step_done(t0)

    metadata: Dict[str, object] = {
        "started_at": utc_now(),
        "profile_name": profile.name,
        "profile_version": profile.version,
        "protocol_version": PROTOCOL_VERSION,
        "psu_make": args.psu_make or "",
        "psu_model": args.psu_model or "",
        "psu_wattage": psu_wattage_val,
        "psu_serial": serial,
        "psu_lot": args.psu_lot or "",
        "operator": args.operator if args.operator is not None else os.getenv("USER", "tester"),
        "reason": args.reason if args.reason is not None else "RMA return",
        "ambient_c": float(args.ambient_c) if args.ambient_c is not None else None,
        "manual_idle_12v": None,
        "manual_load_12v": None,
        "manual_idle_5v": None,
        "manual_load_5v": None,
        "meter_available": pre_obs["meter_available"],
        "paperclip_result": pre_obs["paperclip_result"],
        "post_result": pre_obs["post_result"],
        "known_good_swap_performed": 0,
        "known_good_swap_result": "not-done",
        "original_system_retest_result": "not-run",
        "bench_name": args.bench_name if args.bench_name is not None else platform.node(),
        "bench_last_verified": args.bench_last_verified or "",
        "os_name": args.os_name if args.os_name is not None else platform.platform(),
        "preflight_json": json.dumps(preflight_data, sort_keys=True),
        "hardware_json": json.dumps(hardware_data, sort_keys=True),
        "telemetry_status": "partial",
        "verdict": None,
        "verdict_reasons_json": None,
        "notes": _compose_notes(args.notes, psu_budget_override, budget),
    }

    t0 = _step("inserting run metadata")
    run_id = db.insert_run(metadata)
    _step_done(t0)

    # S15 (v0.21) — NVMe FW blocklist check. Runs once at prep time using the
    # DUT paths already confirmed. Events go into the DB as "preflight" phase
    # so they appear in the event stream and score through verdict/rules.py.
    for fw_sev, fw_kind, fw_msg in check_nvme_fw_revisions(
        list(getattr(args, "storage_dut_path", []) or [])
    ):
        db.add_event("preflight", fw_sev, fw_kind, fw_msg)

    prep_elapsed = time.monotonic() - prep_start
    print("[prep] complete in {:.1f}s".format(prep_elapsed))
    print("")
    print("Run folder: {}".format(run_dir))
    print("Profile: {} ({})".format(profile.name, profile.description))
    if args.dry_run:
        print("Dry run enabled: no stress workers will be launched.")
    readiness = bootstrap_report.get("telemetry_readiness", {})
    print("Telemetry readiness: {}".format(str(readiness.get("level", "unknown")).upper()))
    for message in readiness.get("messages", []):
        print("- {}".format(message))
    if bootstrap_report.get("actions"):
        print("Bootstrap actions:")
        for action in bootstrap_report["actions"]:
            print("- {}".format(action))
    print("")

    return RunPreparation(
        profile=profile,
        serial=serial,
        run_dir=run_dir,
        storage_health_start=storage_health_start,
        storage_ocp_start=storage_ocp_start,
        storage_ocp_latency_start=storage_ocp_latency_start,
        storage_ata_start=storage_ata_start,
        edac_start=edac_start,
        bootstrap_report=bootstrap_report,
        db=db,
        run_id=run_id,
        psu_wattage_val=psu_wattage_val,
        meter_available=bool(pre_obs["meter_available"]),
    )


def finalize_run(
    args: argparse.Namespace,
    run_dir: Path,
    db: Database,
    run_id: int,
    storage_health_start: Dict[str, object],
    storage_ocp_start: Optional[Dict[str, object]] = None,
    storage_ocp_latency_start: Optional[Dict[str, object]] = None,
    storage_ata_start: Optional[Dict[str, object]] = None,
    edac_start: Optional[Dict[str, object]] = None,
) -> int:
    """Post-phase teardown: capture end-of-run storage/EDAC health, fold
    operator observations + manual readings into the DB, score the verdict,
    export artifacts, and update the FM log + batch tracker."""
    storage_health_end = capture_storage_health()
    write_json(run_dir / "storage-health-end.json", storage_health_end)
    capture_nvme_artifacts(run_dir, list(getattr(args, "storage_dut_path", []) or []), stage="end")
    for severity, kind, message in storage_health_events(storage_health_start, storage_health_end):
        db.add_event("run", severity, kind, message)

    # S7 (v0.18) — OCP 0xC0 end snapshot + delta. Growth on any scored
    # counter = FAIL; drives that never exposed OCP emit one info-only
    # event per drive. Skipped entirely if prepare_run didn't capture a
    # start snapshot (e.g. recover-only path).
    if storage_ocp_start is not None:
        storage_ocp_end = capture_storage_ocp(
            list(getattr(args, "storage_dut_path", []) or []) or None
        )
        write_json(run_dir / "storage-ocp-end.json", storage_ocp_end)
        for severity, kind, message in storage_ocp_events(storage_ocp_start, storage_ocp_end):
            db.add_event("run", severity, kind, message)

    # OCP 0xC3 latency monitor end snapshot + info events (diagnostic only).
    if storage_ocp_latency_start is not None:
        storage_ocp_latency_end = write_ocp_latency_artifacts(
            run_dir,
            list(getattr(args, "storage_dut_path", []) or []) or None,
            "end",
        )
        for severity, kind, message in storage_ocp_latency_events(storage_ocp_latency_start, storage_ocp_latency_end):
            db.add_event("run", severity, kind, message)

    # S9 (v0.19) — ATA SMART end snapshot + delta. Reallocated /
    # pending / offline-uncorrectable sector growth = FAIL (hard media
    # degradation); UDMA CRC growth = WARN (cable/connector, not drive).
    # Skipped entirely when prepare_run didn't capture a start snapshot
    # (recover-only path).
    if storage_ata_start is not None:
        storage_ata_end = capture_storage_ata()
        write_json(run_dir / "storage-ata-end.json", storage_ata_end)
        for severity, kind, message in storage_ata_events(storage_ata_start, storage_ata_end):
            db.add_event("run", severity, kind, message)

    # EDAC end-of-run snapshot + delta
    edac_end = snapshot_edac()
    write_json(run_dir / "edac-end.json", edac_end)
    if edac_start is not None:
        delta = edac_delta(edac_start, edac_end)
        write_json(run_dir / "edac-delta.json", delta)
        for ev in edac_events_from_delta(delta):
            db.add_event("run", ev["severity"], ev["kind"], ev["message"])

    # Post-run observations: known-good swap result, original system retest.
    observations = post_run_observations(args)
    db.update_run(run_id, observations)

    # Multimeter readings: captured mid-run in plain mode (after soak + after steady).
    # If they're still None here (TUI run, early abort, or operator skipped), prompt now.
    run_row = db.fetch_run()
    if run_row["meter_available"]:
        missing = final_multimeter_readings(args, run_row)
        if any(v is not None for v in missing.values()):
            db.update_run(run_id, missing)

    verdict, reasons, claim_scope, subsystems = VerdictEngine.evaluate(db)
    db.update_run(
        run_id,
        {
            "ended_at": utc_now(),
            "verdict": verdict,
            "verdict_reasons_json": json.dumps(reasons, indent=2),
            "subsystem_results_json": json.dumps(subsystems, indent=2),
            "telemetry_status": claim_scope,
        },
    )

    batch_event = maybe_batch_pattern_event(run_dir, db)
    if batch_event is not None:
        db.add_event("run", batch_event["severity"], batch_event["kind"], batch_event["message"])
        verdict, reasons, claim_scope, subsystems = VerdictEngine.evaluate(db)
        db.update_run(
            run_id,
            {
                "verdict": verdict,
                "verdict_reasons_json": json.dumps(reasons, indent=2),
                "subsystem_results_json": json.dumps(subsystems, indent=2),
                "telemetry_status": claim_scope,
            },
        )

    export_run_artifacts(run_dir, db)

    fm_number = append_to_fm_log(run_dir, db)
    if fm_number:
        print("FM log: appended {} to {}".format(fm_number, FM_LOG_PATH))

    if update_batch_tracker(run_dir, db):
        print("Batch tracker: updated {}".format(batch_tracker_path()))
    elif not batch_tracker_path().exists():
        print("[INFO] No batch tracker found at {} -- create one to enable auto-recording.".format(
            batch_tracker_path()
        ))

    print("Verdict: {}".format(verdict))
    print("Scope: {}".format(claim_scope))
    print_subsystem_summary(subsystems)
    for reason in reasons:
        print("- {}".format(reason))
    print("Summary: {}".format(run_dir / "summary.md"))
    # Release the EXCLUSIVE lock so subsequent CLI commands (report, verdict,
    # chart) can reopen the DB on Windows/synced-folder paths.
    try:
        db.close()
    except Exception:
        pass
    return 0


def args_to_lobby_values(args: argparse.Namespace) -> Dict[str, str]:
    """Extract current CLI arg values as initial lobby field values."""
    return {
        "psu_serial":      str(getattr(args, "psu_serial",  None) or ""),
        "psu_wattage":     str(getattr(args, "psu_wattage", None) or ""),
        "psu_make":        str(getattr(args, "psu_make",    None) or ""),
        "psu_model":       str(getattr(args, "psu_model",   None) or ""),
        "operator":        str(getattr(args, "operator",    None) or ""),
        "meter_available": str(getattr(args, "meter_available",  None) or ""),
        "paperclip_result": str(getattr(args, "paperclip_result", None) or ""),
        "post_result":     str(getattr(args, "post_result", None) or ""),
        "storage_duts":    ",".join(getattr(args, "storage_dut_path", []) or []),
    }


def apply_lobby_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    """Write lobby-collected values back to args so prepare_run sees them."""
    profile_name = str(cfg.get("profile_name") or "").strip()
    if profile_name:
        args.profile = profile_name
    serial = str(cfg.get("psu_serial") or "").strip()
    args.psu_serial  = serial if serial else None
    wattage = str(cfg.get("psu_wattage") or "").strip()
    args.psu_wattage = wattage if wattage else None
    make   = str(cfg.get("psu_make")  or "").strip()
    args.psu_make    = make   if make   else None
    model  = str(cfg.get("psu_model") or "").strip()
    args.psu_model   = model  if model  else None
    oper   = str(cfg.get("operator")  or "").strip()
    args.operator    = oper   if oper   else None
    args.meter_available  = cfg.get("meter_available")  or "no"
    args.paperclip_result = cfg.get("paperclip_result") or "not-run"
    args.post_result      = cfg.get("post_result")      or "not-tested"
    args.storage_dut_path = parse_storage_dut_paths(str(cfg.get("storage_duts") or ""))
    duts = cfg.get("dut_enabled") or {}
    if isinstance(duts, dict):
        args.dut_cpu     = bool(duts.get("cpu",     True))
        args.dut_gpu     = bool(duts.get("gpu",     True))
        args.dut_mem     = bool(duts.get("mem",     True))
        args.dut_storage = bool(duts.get("storage", True))
