"""Live execution engine for a single PSU run.

Contains `run_test` (the `run` CLI command handler) and the inner
`execute_phases` function that runs the 1 Hz sample loop, per-phase
worker orchestration, and live health checks.

Lifecycle (prepare_run / finalize_run) lives in lifecycle.py.
Health checks live in health_checks.py.
"""

from __future__ import annotations

import argparse
import curses
import json
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from psfu.collectors.cpufreq_sysfs import CpuFreqSysfsCollector
from psfu.collectors.kernel import KernelWatcher
from psfu.collectors.mem_bandwidth import MemoryBandwidthCollector
from psfu.collectors.nvme_telemetry import NvmeTelemetryCollector
from psfu.collectors.pcie_link import PcieLinkCollector
from psfu.collectors.telemetry import TelemetryCollector
from psfu.collectors.turbostat import TurbostatCollector
from psfu.collectors.ups import UpsCollector, list_upses
from psfu.common import utc_now
from psfu.config import APP_NAME
from psfu.hardware import (
    PowerLimitGuard,
    stress_ng_binary_path,
)
from psfu.ui.operator import (
    idle_multimeter_readings,
    load_multimeter_readings,
)
from psfu.ui.tui import TuiDashboard
from psfu.runners.health_checks import (
    CpuHealthState,
    GpuHealthState,
    PowerHealthState,
    PowerStatsState,
    StorageHealthState,
    ThermalHealthState,
    check_cpu_throttle,
    check_fan_spike,
    check_gpu_clock_throttle,
    check_gpu_disappeared,
    check_gpu_ecc,
    check_gpu_hw_throttle,
    check_gpu_retired_pages,
    check_gpu_smi_err,
    check_pcie_downgrade,
    check_rail_cusum,
    check_rail_runrule,
    check_nvme_fio_stall,
    check_storage_pcie_link_downshift,
    check_thermal_drift,
    check_worker_exit,
    update_rail_baseline,
)
from psfu.runners.lifecycle import (
    RunPreparation,
    apply_lobby_config,
    args_to_lobby_values,
    finalize_run,
    prepare_run,
)
from psfu.runners.operator_io import OperatorIO, PlainIO, TuiIO
from psfu.runners.orchestrator import (
    PULSE_OFF_SEC,
    PULSE_ON_SEC,
    RAMP_STEPS,
    build_phase_commands,
)
from psfu.runners.phase_runners import (
    DutRunner,
    FioDutRunner,
    FioPulseRunner,
    build_dut_runner,
)
from psfu.runners.workers.base import StressWorker
from psfu.runners.workers.fio_runner import FioRunner, extract_interval_display_metrics
from psfu.runtime import load_bench_baseline, load_profile


def dut_enabled_from_args(args: argparse.Namespace, profile: Any = None) -> Dict[str, bool]:
    """Build DUT enable map.  If the operator explicitly passed --dut-X or
    --no-dut-X on the CLI, honour that.  Otherwise fall back to whether the
    profile actually exercises that DUT type (has at least one non-idle phase).
    CLI args default to None so we can distinguish "not set" from "set True"."""
    available = profile.available_duts() if profile is not None and hasattr(profile, "available_duts") else {
        "cpu": True, "gpu": True, "mem": True, "storage": True,
    }
    def _pick(key: str, attr: str) -> bool:
        val = getattr(args, attr, None)
        if val is not None:
            return bool(val)
        return available.get(key, True)
    return {
        "cpu": _pick("cpu", "dut_cpu"),
        "gpu": _pick("gpu", "dut_gpu"),
        "mem": _pick("mem", "dut_mem"),
        "storage": _pick("storage", "dut_storage"),
    }


def effective_phase_modes(phase: Any, dut_enabled: Dict[str, bool]) -> Dict[str, str]:
    return {
        "cpu": phase.cpu_mode if dut_enabled.get("cpu", True) else "idle",
        "gpu": phase.gpu_mode if dut_enabled.get("gpu", True) else "idle",
        "mem": phase.mem_mode if dut_enabled.get("mem", True) else "idle",
        "storage": phase.storage_mode if dut_enabled.get("storage", True) else "idle",
    }


def run_test(args: argparse.Namespace) -> int:
    """Main `run` subcommand handler. Wraps prepare -> execute -> finalize."""
    # -- Lobby: collect run config in TUI before prepare_run ---------------
    _lobby_result: Optional[Dict[str, Any]] = None
    _lobby_done = False
    if getattr(args, "tui", True):
        try:
            _lobby_profile = load_profile(args.profile)
        except SystemExit:
            _lobby_profile = None
        if _lobby_profile is not None:
            def _lobby_wrapper(stdscr: "curses._CursesWindow") -> None:
                nonlocal _lobby_result
                _d = TuiDashboard(
                    stdscr,
                    _lobby_profile,
                    psu_serial=getattr(args, "psu_serial", "") or "",
                    app_name=APP_NAME,
                    dut_enabled=dut_enabled_from_args(args, profile=_lobby_profile),
                    initial_values=args_to_lobby_values(args),
                )
                _d.wait_for_start()
                _lobby_result = _d.lobby_config()

            try:
                curses.wrapper(_lobby_wrapper)
            except curses.error:
                pass  # TUI unavailable -- fall through to terminal prompts in prepare_run

            if _lobby_result is not None:
                if _lobby_result.get("aborted"):
                    return 0
                apply_lobby_config(args, _lobby_result)
                _lobby_done = True

    # S5 — destructive mode guards. Must run before prepare_run so no run
    # state (directory, DB) is created on guard failure.
    _destructive_arg = bool(getattr(args, "destructive", False))
    try:
        _early_profile = load_profile(args.profile)
        _profile_destructive = bool(getattr(_early_profile, "destructive", False))
    except SystemExit:
        _profile_destructive = False
    if _destructive_arg or _profile_destructive:
        from psfu.storage.destructive_guards import run_all_guards
        _guard_err = run_all_guards(
            destructive_flag=_destructive_arg,
            profile_destructive=_profile_destructive,
            serial=(getattr(args, "psu_serial", "") or ""),
            devices=list(getattr(args, "storage_dut_path", []) or []),
        )
        if _guard_err:
            print("DESTRUCTIVE RUN ABORTED: {}".format(_guard_err))
            return 1

    prep = prepare_run(args)

    # Apply duration scale from lobby. Profile/Phase are frozen, so we rebuild
    # the profile with new Phase instances instead of mutating in place.
    if _lobby_result is not None:
        try:
            _scale = float(_lobby_result.get("dur_scale") or "1.0")
        except (ValueError, TypeError):
            _scale = 1.0
        if abs(_scale - 1.0) > 0.001 and _scale > 0:
            prep.profile = prep.profile.with_scaled_durations(_scale)
            print("Duration scale {:.2f}x applied -- new total: {}s".format(
                _scale, sum(p.duration_sec for p in prep.profile.phases)))

    profile = prep.profile
    serial = prep.serial
    run_dir = prep.run_dir
    storage_health_start = prep.storage_health_start
    storage_ocp_start = prep.storage_ocp_start
    storage_ocp_latency_start = prep.storage_ocp_latency_start
    storage_ata_start = prep.storage_ata_start
    edac_start = prep.edac_start
    db = prep.db
    run_id = prep.run_id
    run_row = db.fetch_run()
    collector = TelemetryCollector()
    meter_available = prep.meter_available
    scale_actions: List[str] = []
    dut_enabled = dut_enabled_from_args(args, profile=profile)
    bench_baseline = None
    if run_row is not None:
        try:
            hardware = json.loads(run_row["hardware_json"] or "{}")
        except Exception:
            hardware = {}
        board_name = str(((hardware.get("board") or {}) if isinstance(hardware, dict) else {}).get("name") or "")
        gpu_name = "no-gpu"
        if isinstance(hardware, dict):
            commands = hardware.get("commands", {})
            if isinstance(commands, dict):
                entry = commands.get("nvidia_smi_L", {})
                if isinstance(entry, dict):
                    stdout = str(entry.get("stdout") or "")
                    if stdout:
                        first = stdout.splitlines()[0].strip()
                        if ": " in first:
                            first = first.split(": ", 1)[1]
                        if " (UUID:" in first:
                            first = first.split(" (UUID:", 1)[0]
                        if first:
                            gpu_name = first
        if board_name:
            bench_baseline = load_bench_baseline(board_name, gpu_name)

    # Filter out skipped phases. Use a local list — do NOT mutate profile.phases
    # in place. The profile object is shared (e.g. orchestrator reads phase
    # config later) and silent in-place mutation defeats the point of having a
    # typed model.
    skip_set = {s.strip().lower() for s in (getattr(args, "skip_phases", "") or "").split(",") if s.strip()}
    active_phases = list(profile.phases)
    if skip_set:
        original_count = len(active_phases)
        active_phases = [p for p in active_phases if p.name.lower() not in skip_set]
        skipped = original_count - len(active_phases)
        print("Skipping {} phase(s): {}".format(skipped, ", ".join(sorted(skip_set))))

    # Warn if the DUT already failed POST
    _pre_run_row = db.fetch_run()
    if _pre_run_row["post_result"] == "fail":
        print(
            "\n[WARNING] POST result recorded as FAIL. "
            "The DUT did not POST with this PSU installed. "
            "Phases will still run for additional evidence, but expect instability.\n"
        )

    last_gasp_path = run_dir / "last_gasp.jsonl"
    # Opened inside the try below so the finally block can always close it.
    # Must not open here — collectors started between here and the try block
    # can raise, which would leak the handle.
    last_gasp_file: Optional[Any] = None

    current_phase_holder = {"name": "pre-run"}

    _active_workers: List[StressWorker] = []
    _io_holder: Dict[str, Optional[OperatorIO]] = {"io": None}

    _orig_sigint = signal.getsignal(signal.SIGINT)
    def _sigint_handler(sig: int, frame: object) -> None:  # noqa: E306
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _sigint_handler)

    kernel_watcher = KernelWatcher(db, current_phase_fn=lambda: current_phase_holder["name"])
    watcher_err = kernel_watcher.start()
    if watcher_err:
        print("- kernel watcher: {}".format(watcher_err))
        db.add_event("run", "warn", "kernel_watcher_unavailable", watcher_err)
    else:
        print("- kernel watcher: {}".format(" ".join(kernel_watcher.command)))
        db.add_event("run", "info", "kernel_watcher", " ".join(kernel_watcher.command))

    turbostat = TurbostatCollector(interval=1.0)
    turbostat_err = turbostat.start()
    if turbostat_err:
        print("- turbostat: {}".format(turbostat_err))
        db.add_event("run", "info", "turbostat_unavailable", turbostat_err)
    else:
        print("- turbostat: running (1 Hz)")
        db.add_event("run", "info", "turbostat", "running")

    # Root-free sysfs fallback for CPU MHz / Busy% when turbostat can't
    # read MSRs (no cap_sys_rawio, no sudo). Always merged BEFORE turbostat
    # so turbostat overrides when both are available. cpu_pkg_watt stays
    # None on this path — no MSR access means no package power.
    cpufreq_sysfs = CpuFreqSysfsCollector()
    cpufreq_err = cpufreq_sysfs.start()
    if cpufreq_err:
        db.add_event("run", "info", "cpufreq_sysfs_unavailable", cpufreq_err)
    elif not turbostat.available:
        print("- cpufreq sysfs fallback: active (MHz + Busy% only, no PkgWatt)")
        db.add_event("run", "info", "cpufreq_sysfs", "fallback active")

    # v7 — memory bandwidth sampler. 30 s cadence on its own thread via
    # perf stat uncore IMC events. Graceful tombstone when perf is absent
    # or the PMU isn't accessible; no verdict rule depends on these
    # columns yet, so an unavailable sampler just leaves them NULL.
    mem_bw = MemoryBandwidthCollector(interval_s=30.0)
    mem_bw_err = mem_bw.start()
    if mem_bw_err:
        print("- mem bandwidth: {}".format(mem_bw_err))
        db.add_event("run", "info", "mem_bw_unavailable", mem_bw_err)
    else:
        print("- mem bandwidth: running (30 s cadence)")
        db.add_event("run", "info", "mem_bw", "running")

    # v8 — upstream UPS telemetry via NUT. Auto-discover via upsc -l.
    # If zero UPSes visible (no NUT, no device, no bench UPS), log an
    # info event and leave columns NULL. If one, pin it. If multiple,
    # pick the first deterministically — operator can reorder via NUT
    # config if they want a different default (multi-UPS lobby picker
    # is Rule 5 territory: ships when a profile or rule actually needs
    # the second UPS). No verdict rule consumes any ups_* column yet,
    # so an unavailable collector is harmless.
    ups_names = list_upses()
    ups: Optional[UpsCollector] = None
    if not ups_names:
        db.add_event("run", "info", "ups_unavailable",
                     "no UPS visible via NUT (upsc -l empty)")
    else:
        chosen = ups_names[0]
        if len(ups_names) > 1:
            db.add_event("run", "info", "ups_multiple",
                         "multiple UPSes visible ({}), pinning first: {}".format(
                             ",".join(ups_names), chosen))
        ups = UpsCollector(chosen)
        ups_err = ups.start()
        if ups_err:
            print("- ups '{}': {}".format(chosen, ups_err))
            db.add_event("run", "info", "ups_unavailable", ups_err)
            ups = None
        else:
            print("- ups: {} (1 Hz via upsc)".format(chosen))
            db.add_event("run", "info", "ups", "pinned '{}'".format(chosen))

    nvme_telemetry: Optional[NvmeTelemetryCollector] = None
    pcie_link: Optional[PcieLinkCollector] = None
    nvme_duts = list(getattr(args, "storage_dut_path", []) or [])
    if nvme_duts:
        nvme_telemetry = NvmeTelemetryCollector(nvme_duts, run_dir=run_dir)
        nvme_err = nvme_telemetry.start()
        if nvme_err:
            print("- nvme telemetry: {}".format(nvme_err))
            db.add_event("run", "info", "nvme_telemetry_unavailable", nvme_err)
            nvme_telemetry = None
        else:
            print("- nvme telemetry: running (1 Hz)")
            db.add_event("run", "info", "nvme_telemetry", "running")
        pcie_link = PcieLinkCollector(nvme_duts)
        pcie_link_err = pcie_link.start()
        if pcie_link_err:
            print("- nvme pcie link: {}".format(pcie_link_err))
            db.add_event("run", "info", "nvme_pcie_link_unavailable", pcie_link_err)
            pcie_link = None
        else:
            print("- nvme pcie link: running (1 Hz)")
            db.add_event("run", "info", "nvme_pcie_link", "running")

    def execute_phases(stdscr: Optional["curses._CursesWindow"] = None) -> None:
        """Inner phase executor. Runs either plain (stdscr=None) or TUI-wrapped."""
        io: OperatorIO
        if stdscr is not None:
            try:
                # Flush any stale input buffered between the lobby and live
                # curses sessions (e.g. keystrokes during prepare_run output).
                curses.flushinp()
                dashboard = TuiDashboard(
                    stdscr,
                    profile,
                    psu_serial=serial,
                    app_name=APP_NAME,
                    dut_enabled=dut_enabled,
                )
                if _lobby_done:
                    dashboard.awaiting_start = False
                io = TuiIO(dashboard, stdscr)
                if not _lobby_done:
                    if io.wait_for_start():
                        raise KeyboardInterrupt
                    selected = io.selected_duts() or {}
                    dut_enabled.update(selected)
                    args.dut_cpu = dut_enabled["cpu"]
                    args.dut_gpu = dut_enabled["gpu"]
                    args.dut_mem = dut_enabled["mem"]
                    args.dut_storage = dut_enabled["storage"]
            except Exception:
                io = PlainIO()
        else:
            io = PlainIO()
        io.start()
        _io_holder["io"] = io

        def emit_event(
            severity: str,
            phase: str,
            kind: str,
            message: str,
            *,
            tui_text: Optional[str] = None,
        ) -> None:
            """Fan one event out to the DB and the live operator display."""
            db.add_event(phase, severity, kind, message)
            display = tui_text if tui_text is not None else "{}: {}".format(kind, message)
            io.show_event(severity, phase, display)

        armed_duts = [label.upper() for label, enabled in dut_enabled.items() if enabled]
        armed_text = ", ".join(armed_duts) if armed_duts else "none"
        emit_event("info", "run", "dut_selection", "armed DUTs: {}".format(armed_text))
        io.log_action("armed DUTs: {}".format(armed_text))

        gpu_hs = GpuHealthState()
        cpu_hs = CpuHealthState()
        thermal_hs = ThermalHealthState()
        power_hs = PowerHealthState()
        power_stats_hs = PowerStatsState()
        storage_hs = StorageHealthState()

        with PowerLimitGuard(1.0, scale_actions):
            for action in scale_actions:
                io.log_action(action)
                emit_event("info", "run", "power_limit", action)
            scale_actions.clear()
            for phase_idx, phase in enumerate(active_phases):
                phase_actions: List[str] = []
                with PowerLimitGuard(phase.gpu_load_fraction, phase_actions):
                    for action in phase_actions:
                        io.log_action(action)
                        emit_event("info", phase.name, "power_limit", action)
                    phase_actions.clear()
                    current_phase_holder["name"] = phase.name
                    _phase_skipped = False
                    io.reset_skip()
                    phase_modes = effective_phase_modes(phase, dut_enabled)
                    io.start_phase(phase, phase_idx, phase_modes)
                    cpu_cmd, gpu_cmd, gpu_cwd, mem_cmd, storage_cmd, fio_spec = build_phase_commands(
                        phase,
                        args.dry_run,
                        storage_paths=args.storage_path,
                        storage_dut_paths=nvme_duts,
                        run_dir=run_dir,
                        cpu_load_fraction=phase.cpu_load_fraction,
                        gpu_load_fraction=phase.gpu_load_fraction,
                        dut_enabled=dut_enabled,
                        destructive=bool(getattr(args, "destructive", False)),
                    )

                    # --- Orchestration mode detection ---
                    phase_is_ramp = phase_modes["cpu"] == "ramp" or phase_modes["mem"] == "ramp"
                    ramp_step_dur: int = 0
                    ramp_fractions: List[float] = []
                    stress_path_for_ramp: Optional[str] = None
                    if phase_is_ramp and not args.dry_run:
                        stress_path_for_ramp = stress_ng_binary_path()
                        ramp_step_dur = max(1, phase.duration_sec // RAMP_STEPS)
                        ramp_fractions = [round((i + 1) / RAMP_STEPS, 2) for i in range(RAMP_STEPS)]

                    # --- Build one DutRunner per axis ---
                    _dut_specs = [
                        ("cpu", cpu_cmd, None),
                        ("gpu", gpu_cmd, gpu_cwd),
                        ("mem", mem_cmd, None),
                        ("storage", storage_cmd, None),
                    ]
                    runners: Dict[str, DutRunner] = {}
                    for _label, _cmd, _cwd in _dut_specs:
                        if _label == "storage" and fio_spec is not None:
                            # S4: fio routing preempts the legacy sha256
                            # filesystem worker. build_phase_commands made
                            # sure exactly one of (storage_cmd, fio_spec)
                            # is set.
                            _fio_runner = FioRunner(
                                fio_spec.job_file,
                                device=fio_spec.device,
                                phase_name=fio_spec.phase_name,
                                run_dir=fio_spec.run_dir,
                                destructive=fio_spec.destructive,
                            )
                            if phase_modes[_label] == "pulse":
                                on_attr = getattr(phase, "{}_pulse_on_sec".format(_label), None)
                                off_attr = getattr(phase, "{}_pulse_off_sec".format(_label), None)
                                on_sec = on_attr if on_attr is not None else PULSE_ON_SEC
                                off_sec = off_attr if off_attr is not None else PULSE_OFF_SEC
                                inverted = _label in getattr(phase, "pulse_invert", [])
                                runners[_label] = FioPulseRunner(
                                    _fio_runner,
                                    on_sec=on_sec,
                                    off_sec=off_sec,
                                    inverted=inverted,
                                    emit_event=emit_event,
                                    status=io,
                                    phase_name=phase.name,
                                    dry_run=args.dry_run,
                                )
                            else:
                                runners[_label] = FioDutRunner(
                                    _fio_runner,
                                    emit_event=emit_event,
                                    status=io,
                                    phase_name=phase.name,
                                )
                            continue
                        runners[_label] = build_dut_runner(
                            _label,
                            phase_modes[_label],
                            _cmd,
                            phase=phase,
                            emit_event=emit_event,
                            status=io,
                            stress_path_for_ramp=stress_path_for_ramp,
                            ramp_step_dur=ramp_step_dur,
                            ramp_fractions=ramp_fractions,
                            cwd=_cwd,
                            dry_run=args.dry_run,
                        )

                    started = time.monotonic()
                    next_tick = started
                    for _r in runners.values():
                        _r.prepare(started)
                    _active_workers[:] = [
                        _r.worker for _r in runners.values() if _r.worker is not None
                    ]

                    while True:
                        now = time.monotonic()
                        elapsed = now - started
                        if elapsed >= phase.duration_sec:
                            break

                        # --- Per-DUT orchestration (pulse toggle, ramp step) ---
                        for _r in runners.values():
                            _r.tick(now, elapsed)
                        # Ramp/pulse may have respawned workers, so refresh
                        # the active-workers list for the finally block.
                        _active_workers[:] = [
                            _r.worker for _r in runners.values() if _r.worker is not None
                        ]

                        # Capacity-based fio phases complete on their own.
                        # When the fio subprocess exits rc=0, end the phase
                        # normally instead of waiting for the wall-clock
                        # duration ceiling and misclassifying the clean exit as
                        # a worker failure.
                        _storage_runner = runners.get("storage")
                        if isinstance(_storage_runner, FioDutRunner):
                            _storage_state = _storage_runner.state()
                            if _storage_state.endswith("rc=0"):
                                break

                        sample = collector.collect()
                        if cpufreq_sysfs.available:
                            collector.merge_turbostat(sample, cpufreq_sysfs.snapshot())
                        if turbostat.available:
                            collector.merge_turbostat(sample, turbostat.snapshot())
                        # Memory bandwidth sampler runs at 30 s cadence on
                        # its own thread — the snapshot call is a cheap
                        # dict read. Values stay at the last probe until
                        # the next one lands (held-between-samples UX).
                        if mem_bw.available:
                            mbw_snap = mem_bw.snapshot()
                            for _key in ("mem_bw_read_mbps", "mem_bw_write_mbps"):
                                _val = mbw_snap.get(_key)
                                if _val is not None:
                                    sample[_key] = _val
                        # v8 — UPS snapshot is a cheap dict read (upsd
                        # already holds the latest values in memory; the
                        # collector thread polls at 1 Hz). All 10 keys
                        # land in every sample, with None where the
                        # vendor omits the variable (Cyberpower lacks
                        # input.frequency + live ups.realpower).
                        if ups is not None and ups.available:
                            sample.update(ups.snapshot())
                        if nvme_telemetry is not None and nvme_telemetry.available:
                            sample.update(nvme_telemetry.snapshot())
                        if pcie_link is not None and pcie_link.available:
                            sample.update(pcie_link.snapshot())
                        sample.update(
                            {
                                "ts": utc_now(),
                                "phase": phase.name,
                            }
                        )
                        db.add_sample(sample)

                        if isinstance(_storage_runner, FioDutRunner):
                            _live_interval = _storage_runner.worker.latest_interval()
                            if isinstance(_live_interval, dict):
                                for _key, _value in extract_interval_display_metrics(_live_interval).items():
                                    if _value is not None:
                                        sample[_key] = _value

                        # Rail baseline tracking
                        update_rail_baseline(power_hs, sample, phase_idx)

                        # Last gasp
                        lg_sample = {
                            "ts": sample["ts"],
                            "12v": sample["rail_12v"],
                            "5v": sample["rail_5v"],
                            "cpu_load": sample["cpu_load_pct"],
                            "gpu_load": sample["gpu_util_pct"],
                            "phase": phase.name,
                        }
                        last_gasp_file.write(json.dumps(lg_sample) + "\n")
                        last_gasp_file.flush()

                        # Health checks
                        check_gpu_disappeared(gpu_hs, collector, phase.name, dut_enabled, emit_event)
                        check_gpu_hw_throttle(gpu_hs, collector, phase.name, phase_modes, dut_enabled, emit_event)
                        check_gpu_smi_err(gpu_hs, collector, phase.name, dut_enabled, emit_event)
                        check_pcie_downgrade(gpu_hs, collector, phase.name, dut_enabled, emit_event)
                        check_cpu_throttle(cpu_hs, sample, phase_idx, phase.name, phase_modes, dut_enabled, emit_event)
                        check_gpu_clock_throttle(gpu_hs, sample, phase_idx, phase.name, phase_modes, dut_enabled, emit_event)
                        check_gpu_ecc(gpu_hs, sample, phase.name, dut_enabled, emit_event)
                        check_gpu_retired_pages(gpu_hs, sample, phase.name, dut_enabled, emit_event)
                        check_fan_spike(thermal_hs, sample, phase_idx, phase.name, emit_event)
                        if pcie_link is not None:
                            check_storage_pcie_link_downshift(
                                storage_hs, pcie_link, phase.name, phase_modes, dut_enabled, emit_event,
                            )
                        # S6 — fio controller-hang watchdog. FioDutRunner's
                        # worker is the FioRunner carrying ios_progress_snapshot();
                        # storage_cmd (legacy sha256) path has no such snapshot,
                        # in which case _fio_runner is None and the check no-ops.
                        _storage_runner = runners.get("storage")
                        _fio_runner = (
                            _storage_runner.worker
                            if isinstance(_storage_runner, FioDutRunner)
                            else None
                        )
                        check_nvme_fio_stall(
                            storage_hs, _fio_runner, phase.name, phase_modes, dut_enabled, emit_event,
                        )
                        check_rail_runrule(
                            power_stats_hs,
                            sample,
                            phase_idx,
                            phase.name,
                            phase_modes,
                            dut_enabled,
                            bench_baseline,
                            emit_event,
                        )
                        check_rail_cusum(
                            power_stats_hs,
                            sample,
                            phase_idx,
                            phase.name,
                            phase_modes,
                            dut_enabled,
                            bench_baseline,
                            emit_event,
                        )

                        # Worker exit checks — each runner supplies its own
                        # current cmd (ramp respawns change the cmd mid-phase).
                        for _r in runners.values():
                            if _r.worker is None:
                                continue
                            if check_worker_exit(
                                _r.worker, _r.current_cmd(), _r.label,
                                phase.name, emit_event,
                            ):
                                pass  # worker.cmd reference cleared inside check

                        io.update_sample(
                            sample,
                            elapsed_phase=elapsed,
                            phase_duration_sec=phase.duration_sec,
                            cpu_state=runners["cpu"].state(),
                            gpu_state=runners["gpu"].state(),
                            mem_state=runners["mem"].state(),
                            storage_state=runners["storage"].state(),
                        )

                        # Operator input: poll once, then branch.
                        io.poll()

                        # User abort (q)
                        if io.check_abort():
                            emit_event("info", phase.name, "user_abort", "operator pressed q")
                            for _r in runners.values():
                                _r.teardown()
                            raise KeyboardInterrupt

                        # User pause (p) — TUI only
                        if io.check_pause():
                            for _r in runners.values():
                                _r.pause()
                            pause_start = time.monotonic()
                            while io.check_pause() and not io.check_abort():
                                io.wait_during_pause()
                                time.sleep(0.1)
                            pause_dur = time.monotonic() - pause_start
                            if not io.check_abort():
                                for _r in runners.values():
                                    _r.resume(pause_dur)
                                next_tick += pause_dur
                                started += pause_dur
                            else:
                                emit_event("info", phase.name, "user_abort", "operator pressed q during pause")
                                for _r in runners.values():
                                    _r.teardown()
                                raise KeyboardInterrupt

                        # User skip phase (n)
                        if io.check_skip():
                            emit_event(
                                "info", phase.name, "phase_skipped",
                                "operator skipped phase '{}' at {:.0f}s/{:.0f}s".format(
                                    phase.name, elapsed, phase.duration_sec),
                                tui_text="phase skipped by operator",
                            )
                            _phase_skipped = True
                            break

                        next_tick += 1.0
                        # Poll input during the inter-tick sleep so P/N/Q
                        # respond promptly instead of waiting up to 1s.
                        while True:
                            sleep_for = next_tick - time.monotonic()
                            if sleep_for <= 0:
                                break
                            time.sleep(min(sleep_for, 0.1))
                            io.poll()

                    io.end_phase()
                    for _r in runners.values():
                        _r.teardown()
                _active_workers.clear()

                # --- Inter-phase multimeter prompts ---
                if meter_available and not args.dry_run:
                    prompt_fn = None
                    if phase_idx == 0:
                        prompt_fn = idle_multimeter_readings
                    elif phase_modes["cpu"] in ("stress", "pulse", "ramp") and not power_hs.load_meter_captured:
                        prompt_fn = load_multimeter_readings

                    if prompt_fn is not None:
                        vals = io.run_blocking_prompt(prompt_fn, args)
                        if any(v is not None for v in vals.values()):
                            db.update_run(run_id, vals)
                        if prompt_fn is load_multimeter_readings:
                            power_hs.load_meter_captured = True

            for action in scale_actions:
                io.log_action(action)
                emit_event("info", "run", "power_limit", action)
            scale_actions.clear()
            last_gasp_file.close()

            # Post-run thermal drift check
            check_thermal_drift(power_hs, emit_event, io.log_action)

    try:
        last_gasp_file = open(last_gasp_path, "w", encoding="utf-8", buffering=1)
        if getattr(args, "tui", False):
            try:
                curses.wrapper(execute_phases)
            except curses.error as exc:
                print("TUI unavailable ({}) -- falling back to plain log mode.".format(exc))
                execute_phases(None)
        else:
            execute_phases(None)
    except KeyboardInterrupt:
        db.add_event("run", "crit", "operator_abort", "Operator aborted the run.")
        print("\nRun aborted by operator.")
    finally:
        _io = _io_holder.get("io")
        if _io is not None:
            try:
                _io.stop()
            except Exception:
                pass
        signal.signal(signal.SIGINT, _orig_sigint)
        for _w in _active_workers:
            try:
                _w.stop()
            except Exception:
                pass
        _active_workers.clear()
        current_phase_holder["name"] = "post-run"
        kernel_watcher.stop()
        turbostat.stop()
        cpufreq_sysfs.stop()
        mem_bw.stop()
        if ups is not None:
            ups.stop()
        if nvme_telemetry is not None:
            nvme_telemetry.stop()
        if pcie_link is not None:
            pcie_link.stop()
        try:
            collector.close()
        except Exception:
            pass
        if last_gasp_file is not None:
            try:
                last_gasp_file.close()
            except Exception:
                pass
        kw_summary = kernel_watcher.summary()
        if kw_summary["available"]:
            print(
                "Kernel watcher: {} critical, {} warnings captured".format(
                    kw_summary["critical_count"],
                    kw_summary["warn_count"],
                )
            )

    return finalize_run(
        args, run_dir, db, run_id, storage_health_start,
        storage_ocp_start=storage_ocp_start,
        storage_ocp_latency_start=storage_ocp_latency_start,
        storage_ata_start=storage_ata_start,
        edac_start=edac_start,
    )
