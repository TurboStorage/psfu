from collections import Counter
import re
import sqlite3
import statistics
from typing import Dict, List, Optional, Tuple
from psfu.baselines import baseline_drift_findings
from psfu.persistence.db import Database
from psfu.collectors.kernel import kernel_kinds_by_default_verdict
from psfu.verdict.evidence import build_evidence, event_ids_for

# Magnitude extractor for ``thermal_drift`` messages. Self-describing strings
# emitted by check_thermal_drift always carry ``drift=<value>V``. Keying the
# FAIL rule off this numeric field instead of the caller-supplied ``severity``
# column removes the emitter-trust coupling that prior reviews flagged (H4).
_THERMAL_DRIFT_MAG_RE = re.compile(r"drift=([\d.]+)\s*V", re.IGNORECASE)
_THERMAL_DRIFT_FAIL_V = 0.05

# Human-readable reasons for specific kernel watcher kinds. Kinds not present
# here fall back to a generic phrasing built from the kind + subsystem — so
# a new kernel pattern added to KERNEL_WATCH_PATTERNS auto-scores with
# prose, without a follow-up edit here.
_KERNEL_KIND_REASONS: Dict[str, str] = {
    "mce":            "Kernel MCE detected during run — hardware fault (v0.8 FAIL #2).",
    "kernel_panic":   "Kernel panic during run — system did not survive stress (v0.8 FAIL #1).",
    "kernel_oops":    "Kernel oops during run — OS trapped a fatal fault (v0.8 FAIL #1).",
    "hardware_error": "Kernel-reported hardware error during run (v0.8 FAIL #1).",
    "nvidia_xid":     "NVIDIA Xid fault during run — GPU driver/hardware fault, correlate with PSU and known-good swap.",
    "undervoltage":   "Kernel under-voltage event — explicit rail collapse signal.",
    "brownout":       "Kernel brownout event — explicit rail collapse signal.",
    "thermal_crit":   "Kernel thermal-critical event — CPU/SOC hit trip threshold.",
    "thermal_warn":   "Kernel thermal-warn event — temperature above warn threshold.",
    "thermal_throttle": "Kernel thermal throttle event — OS forced frequency reduction.",
    "oom":            "Out-of-memory event recorded in kernel log.",
    "oom_killer":     "OOM killer fired during run — memory pressure reached kill threshold.",
    "pcie_bus_error": "PCIe bus error logged by kernel — corroborate with AER and a known-good PSU swap.",
    "edac":           "EDAC-reported memory error in kernel log.",
    "acpi_bios":      "ACPI BIOS error(s) during run. Firmware-level fault — non-conclusive but worth logging.",
    "nvme_io_timeout":       "NVMe I/O timeout during run — controller failed to service a queued command within the kernel timeout window.",
    "nvme_controller_reset": "NVMe controller reset during run — kernel forced a reset after a stalled queue or internal fault.",
    "nvme_aer_error":        "NVMe async event notified error during run — controller self-reported a fault condition.",
}


class VerdictEngine:
    @staticmethod
    def _blank_subsystems() -> Dict[str, Dict[str, object]]:
        return {
            "power": {"status": "PASS", "reasons": [], "evidence": []},
            "thermal": {"status": "PASS", "reasons": [], "evidence": []},
            "memory": {"status": "PASS", "reasons": [], "evidence": []},
            "storage": {"status": "PASS", "reasons": [], "evidence": []},
            "pcie": {"status": "PASS", "reasons": [], "evidence": []},
        }

    @staticmethod
    def _promote_status(subsystems: Dict[str, Dict[str, object]], subsystem: str, status: str) -> None:
        current = str(subsystems[subsystem]["status"])
        if current == "FAIL":
            return
        if status == "FAIL" or (status == "WARN" and current != "FAIL"):
            subsystems[subsystem]["status"] = status

    @staticmethod
    def _record_finding(
        subsystems: Dict[str, Dict[str, object]],
        *,
        subsystem: str,
        status: str,
        reason: str,
        rule_id: Optional[str] = None,
        threshold: Optional[Dict[str, object]] = None,
        observed: Optional[Dict[str, object]] = None,
        events: Optional[List[sqlite3.Row]] = None,
        source: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> None:
        VerdictEngine._promote_status(subsystems, subsystem, status)
        subsystems[subsystem]["reasons"].append(reason)
        if rule_id:
            subsystems[subsystem]["evidence"].append(
                build_evidence(
                    rule_id=rule_id,
                    subsystem=subsystem,
                    status=status,
                    threshold=threshold,
                    observed=observed,
                    event_ids=event_ids_for(events or []),
                    human_reason=reason,
                    source=source,
                    phase=phase,
                )
            )

    @staticmethod
    def evaluate_subsystems(db: Database) -> Dict[str, Dict[str, object]]:
        run_row = db.fetch_run()
        samples = db.fetch_samples()
        events = db.fetch_events()
        subsystems = VerdictEngine._blank_subsystems()

        if not samples:
            for data in subsystems.values():
                data["status"] = "INVALID"
                data["reasons"] = ["No telemetry samples were captured."]
                data["evidence"] = []
            return subsystems

        # GPU disappeared from NVML mid-run: the clearest software-visible sign of
        # a PSU voltage collapse or bus loss under load.
        gpu_disappeared_events = [e for e in events if e["kind"] == "nvidia_gpu_disappeared"]
        if gpu_disappeared_events:
            VerdictEngine._record_finding(
                subsystems,
                subsystem="power",
                status="FAIL",
                rule_id="power.gpu_disappeared.fail",
                reason=(
                    "GPU disappeared from NVML mid-run ({} event(s)). "
                    "Indicates PSU voltage collapse or system instability under load.".format(
                        len(gpu_disappeared_events)
                    )
                ),
                observed={"event_count": len(gpu_disappeared_events)},
                events=gpu_disappeared_events,
                source="event",
            )

        # nvidia-smi ERR! in any field ((verdict rule)): hardware-level fault signal
        # from the driver or GPU power delivery path.
        gpu_smi_err_events = [e for e in events if e["kind"] == "gpu_smi_err"]
        if gpu_smi_err_events:
            VerdictEngine._record_finding(
                subsystems,
                subsystem="power",
                status="FAIL",
                rule_id="power.gpu_smi_err.fail",
                reason=(
                    "nvidia-smi reported ERR! in one or more GPU fields ({} event(s)). "
                    "Indicates a hardware fault or driver error on the GPU or its power "
                    "delivery path.".format(len(gpu_smi_err_events))
                ),
                observed={"event_count": len(gpu_smi_err_events)},
                events=gpu_smi_err_events,
                source="event",
            )

        # GPU HW power brake: PSU cannot sustain GPU power demand under load.
        throttle_events = [e for e in events if e["kind"] == "gpu_hw_power_throttle"]
        if throttle_events:
            phases_hit = sorted({e["phase"] for e in throttle_events})
            VerdictEngine._record_finding(
                subsystems,
                subsystem="power",
                status="WARN",
                rule_id="power.gpu_hw_power_throttle.warn",
                reason=(
                    "GPU HW power brake active in {} stress phase(s) ({}). "
                    "PSU cannot sustain GPU power demand — check 12V headroom.".format(
                        len(phases_hit), ", ".join(phases_hit)
                    )
                ),
                observed={"event_count": len(throttle_events), "phase_count": len(phases_hit)},
                events=throttle_events,
                source="event",
            )

        # PCIe link downgrade: bus renegotiated to lower gen or width under load.
        pcie_downgrade_events = [e for e in events if e["kind"] == "pcie_link_downgrade"]
        if pcie_downgrade_events:
            if subsystems["pcie"]["status"] != "FAIL":
                subsystems["pcie"]["status"] = "WARN"
            subsystems["pcie"]["reasons"].append(
                "PCIe link downgraded during the run ({} event(s)). "
                "Indicates PSU voltage sag or bus instability under load. "
                "Corroborate with kernel AER events and a known-good PSU swap.".format(
                    len(pcie_downgrade_events)
                )
            )

        # PCIe AER: SPECIAL — threshold-based scoring (>3 in one phase → FAIL,
        # otherwise WARN). pcie_bus_error is routed via the generic kernel loop
        # below (FAIL default). Keeping only the threshold logic here.
        pcie_aer_events  = [e for e in events if e["kind"] == "pcie_aer"]
        if pcie_aer_events:
            phase_counts = Counter(e["phase"] for e in pcie_aer_events)
            worst_phase = max(phase_counts, key=phase_counts.__getitem__)
            worst_count = phase_counts[worst_phase]
            total = sum(phase_counts.values())
            if worst_count > 3:
                VerdictEngine._record_finding(
                    subsystems,
                    subsystem="pcie",
                    status="FAIL",
                    rule_id="pcie.aer_corrected.fail",
                    reason=(
                        "PCIe AER corrected errors exceeded 3 in phase '{}' ({} in that phase, "
                        "{} total). Persistent bus instability — test with known-good PSU.".format(
                            worst_phase, worst_count, total
                        )
                    ),
                    threshold={"type": "corrected_aer_per_phase", "fail_gt": 3},
                    observed={"worst_phase": worst_phase, "worst_phase_count": worst_count, "total_count": total},
                    events=pcie_aer_events,
                    source="event",
                    phase=worst_phase,
                )
            elif subsystems["pcie"]["status"] != "FAIL":
                VerdictEngine._record_finding(
                    subsystems,
                    subsystem="pcie",
                    status="WARN",
                    rule_id="pcie.aer_corrected.warn",
                    reason=(
                        "PCIe AER corrected error(s): {} across {} phase(s). "
                        "Monitor — if this recurs, isolate with known-good PSU swap.".format(
                            total, len(phase_counts)
                        )
                    ),
                    threshold={"type": "corrected_aer_per_phase", "warn_ge": 1, "fail_gt": 3},
                    observed={"total_count": total, "phase_count": len(phase_counts)},
                    events=pcie_aer_events,
                    source="event",
                )

        # CPU frequency throttling: avg MHz dropped under load while cores stayed busy.
        cpu_throttle_events = [e for e in events if e["kind"] == "cpu_core_throttle"]
        if cpu_throttle_events:
            phases_hit = sorted({e["phase"] for e in cpu_throttle_events})
            if subsystems["power"]["status"] != "FAIL":
                subsystems["power"]["status"] = "WARN"
            subsystems["power"]["reasons"].append(
                "CPU frequency throttled under load in {} phase(s) ({}). "
                "Indicates insufficient power delivery, VRM thermal limit, "
                "or BIOS power cap.".format(
                    len(phases_hit), ", ".join(phases_hit)
                )
            )

        # GPU clock throttling: SM clock dropped significantly while utilization stayed high.
        gpu_clock_throttle_events = [e for e in events if e["kind"] == "gpu_clock_throttle"]
        if gpu_clock_throttle_events:
            phases_hit = sorted({e["phase"] for e in gpu_clock_throttle_events})
            if subsystems["power"]["status"] != "FAIL":
                subsystems["power"]["status"] = "WARN"
            subsystems["power"]["reasons"].append(
                "GPU SM clock throttled under load in {} phase(s) ({}). "
                "Indicates insufficient power delivery or thermal throttling.".format(
                    len(phases_hit), ", ".join(phases_hit)
                )
            )

        # GPU retired pages (pynvml-only): VRAM cells permanently marked bad.
        gpu_retired_events = [e for e in events if e["kind"] == "gpu_retired_pages"]
        if gpu_retired_events:
            subsystems["memory"]["status"] = "FAIL"
            subsystems["memory"]["reasons"].extend(e["message"] for e in gpu_retired_events)

        # GPU ECC errors (volatile counters from nvidia-smi).
        gpu_ecc_ue_events = [e for e in events if e["kind"] == "gpu_ecc_ue"]
        gpu_ecc_ce_events = [e for e in events if e["kind"] == "gpu_ecc_ce"]
        if gpu_ecc_ue_events:
            subsystems["memory"]["status"] = "FAIL"
            subsystems["memory"]["reasons"].extend(e["message"] for e in gpu_ecc_ue_events)
        if gpu_ecc_ce_events:
            if subsystems["memory"]["status"] != "FAIL":
                subsystems["memory"]["status"] = "WARN"
            subsystems["memory"]["reasons"].extend(e["message"] for e in gpu_ecc_ce_events)

        # Fan RPM spike during stress: WARN only when it co-occurs with a
        # thermal event in the same phase (v0.8 WARN #6 "Fan RPM spike
        # coinciding with thermal event"). A bare fan spike is common under
        # load and is not diagnostic on its own; the signal is the pairing.
        fan_spike_events = [e for e in events if e["kind"] == "fan_rpm_spike"]
        thermal_kinds = ("thermal_crit", "thermal_warn", "thermal_throttle")
        thermal_phases = {e["phase"] for e in events if e["kind"] in thermal_kinds}
        fan_spike_correlated_phases = sorted(
            {e["phase"] for e in fan_spike_events if e["phase"] in thermal_phases}
        )
        if fan_spike_correlated_phases:
            if subsystems["thermal"]["status"] != "FAIL":
                subsystems["thermal"]["status"] = "WARN"
            subsystems["thermal"]["reasons"].append(
                "CPU fan RPM spike coincided with thermal event in {} phase(s) ({}). "
                "May reflect elevated CPU thermals from PSU power-delivery instability. "
                "Corroborate with CPU temp drift and a known-good PSU swap.".format(
                    len(fan_spike_correlated_phases),
                    ", ".join(fan_spike_correlated_phases),
                )
            )

        # Storage SMART failures.
        storage_fail_events = [
            e for e in events
            if e["kind"] in ("storage_smart_fail", "storage_critical_warning", "storage_media_errors", "storage_overtemp")
        ]
        storage_fail_events.extend([e for e in events if e["kind"] == "pcie_link_downshift"])
        # S6 — fio_stall fires when no ios progress for ≥10 consecutive stress
        # ticks (check_nvme_fio_stall in runners/health_checks.py). Controller
        # hang during read-only workload is a hard storage FAIL; no known
        # benign cause at the workload scales PS-FU runs. Corroborate with
        # nvme_io_timeout / nvme_controller_reset events from the kernel watcher.
        storage_fail_events.extend([e for e in events if e["kind"] == "fio_stall"])
        # S5 (v0.20) — fio CRC32c verify mismatch. Any mismatch after the
        # write+verify phase = hard storage FAIL. Silent data corruption, bad
        # blocks, or FTL defects. Emitted by FioDutRunner.teardown() after
        # stop() parses the fio JSON+ summary_rows.
        storage_fail_events.extend([e for e in events if e["kind"] == "fio_verify_mismatch"])
        # S7 (v0.18) — OCP 0xC0 delta FAILs. Any non-zero growth on the four
        # scored counters during a read-only run = hard storage FAIL. Claim
        # scope stays software-triage-only until S11 per-PN baseline lands
        # (same ladder as S6: kernel watcher + fio_stall + OCP growth are
        # all single-rung until cross-checked against a known-good drive).
        _OCP_GROWTH_KINDS = (
            "ssd_ocp_bad_user_nand_growth",
            "ssd_ocp_bad_system_nand_growth",
            "ssd_ocp_xor_recovery_growth",
            "ssd_ocp_uncorrectable_read_growth",
        )
        storage_fail_events.extend([e for e in events if e["kind"] in _OCP_GROWTH_KINDS])
        # S9 (v0.19) — ATA SMART delta FAILs. Reallocated / Pending /
        # Offline-Uncorrectable sector growth on a SATA drive = hard media
        # degradation: remapping events exhaust spare capacity, then the
        # drive goes into read-only or hangs. Any non-zero raw counter
        # delta is a FAIL trigger. UDMA CRC goes through storage_warn
        # below — not the drive's fault, point to cable/connector.
        _ATA_FAIL_KINDS = (
            "ata_reallocated_sector_grew",
            "ata_pending_sector_grew",
            "ata_offline_uncorrectable_grew",
        )
        storage_fail_events.extend([e for e in events if e["kind"] in _ATA_FAIL_KINDS])
        if storage_fail_events:
            subsystems["storage"]["status"] = "FAIL"
            for event in storage_fail_events:
                if event["kind"] == "pcie_link_downshift":
                    rule_suffix = "pcie_link_downshift"
                elif event["kind"] == "fio_stall":
                    rule_suffix = "fio_stall"
                elif event["kind"] == "fio_verify_mismatch":
                    rule_suffix = "fio_verify_mismatch"
                elif event["kind"] in _OCP_GROWTH_KINDS:
                    rule_suffix = event["kind"]
                elif event["kind"] in _ATA_FAIL_KINDS:
                    rule_suffix = event["kind"]
                else:
                    rule_suffix = event["kind"].replace("storage_", "")
                VerdictEngine._record_finding(
                    subsystems,
                    subsystem="storage",
                    status="FAIL",
                    rule_id="storage.{}.fail".format(rule_suffix),
                    reason=event["message"],
                    events=[event],
                    observed={"kind": event["kind"]},
                    source="event",
                    phase=event["phase"] or None,
                )

        storage_warn_events = [
            e for e in events
            if e["kind"] in ("storage_error_log_growth", "storage_unsafe_shutdown", "storage_hot")
        ]
        # S9 (v0.19) — UDMA CRC growth is WARN not FAIL because the
        # failure mode is cable / SATA port / backplane signal integrity,
        # not the drive. FAIL would condemn perfectly good drives on
        # degrading cables. WARN tells the operator to reseat/swap the
        # cable before RMA.
        storage_warn_events.extend([e for e in events if e["kind"] == "ata_udma_crc_grew"])
        # S15 (v0.21) — blocklisted firmware revision. The drive itself is not
        # necessarily defective; the FW rev has known bugs documented in
        # baselines/nvme_fw_blocklist.json. WARN directs the operator to update
        # FW before passing or note the rev in the RMA bundle.
        storage_warn_events.extend([e for e in events if e["kind"] == "nvme_oob_fw_version"])
        # S11 (v0.22) — per-PN baseline-backed pSLC cliff. Generic cliff
        # observation remains info-only; this WARN kind only emits when the
        # measured cliff ratio falls below the stored baseline floor
        # (p5 - 3σ) for the exact vendor/model/capacity/fw tuple.
        storage_warn_events.extend([e for e in events if e["kind"] == "fio_write_cliff_severe"])
        if storage_warn_events:
            if subsystems["storage"]["status"] != "FAIL":
                subsystems["storage"]["status"] = "WARN"
            for event in storage_warn_events:
                if event["kind"] in ("ata_udma_crc_grew", "nvme_oob_fw_version", "fio_write_cliff_severe"):
                    rule_suffix = event["kind"]
                else:
                    rule_suffix = event["kind"].replace("storage_", "")
                VerdictEngine._record_finding(
                    subsystems,
                    subsystem="storage",
                    status="WARN",
                    rule_id="storage.{}.warn".format(rule_suffix),
                    reason=event["message"],
                    events=[event],
                    observed={"kind": event["kind"]},
                    source="event",
                    phase=event["phase"] or None,
                )

        # Kernel watcher: generic scoring driven by the KERNEL_WATCH_PATTERNS
        # table in psfu/collectors/kernel.py. Each kernel kind carries a
        # subsystem + default_verdict in the table; this block iterates
        # FAIL-default kinds and scores them uniformly. Adding a new pattern
        # to the table now auto-scores without an edit here — closes the
        # silent-dead-rule class (C1/C11). SPECIAL kinds (pcie_aer threshold)
        # stay in bespoke blocks below.
        _kernel_routing = kernel_kinds_by_default_verdict()
        for kind, subsystem in _kernel_routing["FAIL"]:
            matching = [e for e in events if e["kind"] == kind]
            if not matching:
                continue
            reason_prefix = _KERNEL_KIND_REASONS.get(
                kind, "Kernel event {!r} during run — routed to {}.".format(kind, subsystem)
            )
            sample_msg = matching[0]["message"][:200] if matching[0]["message"] else ""
            VerdictEngine._record_finding(
                subsystems,
                subsystem=subsystem,
                status="FAIL",
                rule_id="{}.{}.fail".format(subsystem, kind),
                reason="{} ({} event(s)){}".format(
                    reason_prefix,
                    len(matching),
                    ": {}".format(sample_msg) if sample_msg else "",
                ),
                observed={"kind": kind, "event_count": len(matching), "sample_message": sample_msg or None},
                events=matching,
                source="event",
            )

        # Kernel watcher: WARN-default kinds (table-driven, companion to FAIL
        # loop above). Adding a new WARN-default pattern to
        # KERNEL_WATCH_PATTERNS auto-scores here — no edit required.
        for kind, subsystem in _kernel_routing["WARN"]:
            matching = [e for e in events if e["kind"] == kind]
            if not matching:
                continue
            reason_prefix = _KERNEL_KIND_REASONS.get(
                kind, "Kernel event {!r} during run — routed to {}.".format(kind, subsystem)
            )
            sample_msg = matching[0]["message"][:200] if matching[0]["message"] else ""
            VerdictEngine._record_finding(
                subsystems,
                subsystem=subsystem,
                status="WARN",
                rule_id="{}.{}.warn".format(subsystem, kind),
                reason="{} ({} event(s)){}".format(
                    reason_prefix,
                    len(matching),
                    ": {}".format(sample_msg) if sample_msg else "",
                ),
                observed={"kind": kind, "event_count": len(matching), "sample_message": sample_msg or None},
                events=matching,
                source="event",
            )

        # Kernel watcher thread died mid-run. Emitted by KernelWatcher's
        # finally block when the reader loop exits before stop() was called.
        # Verdict-wise this is non-conclusive about hardware — we just lost
        # coverage — but it's load-bearing for claim scope: an operator
        # interpreting a software-only PASS should know kernel coverage was
        # incomplete. Routed to power subsystem as WARN.
        watcher_died = [e for e in events if e["kind"] == "kernel_watcher_died"]
        if watcher_died:
            VerdictEngine._record_finding(
                subsystems,
                subsystem="power",
                status="WARN",
                rule_id="coverage.kernel_watcher_died.warn",
                reason=(
                    "Kernel watcher thread exited before end of run — kernel events "
                    "may have been missed. Verdict is non-conclusive for any kernel-"
                    "driven rule after that point."
                ),
                observed={"event_count": len(watcher_died)},
                events=watcher_died,
                source="coverage",
            )

        # Kernel watcher never started. Emitted in prepare_run when journalctl
        # is missing, requires root, or otherwise returns an error. Same class
        # of coverage loss as kernel_watcher_died — rather than dying mid-run,
        # the watcher never came up. Verdict impact is identical: WARN on the
        # power subsystem so an operator reading a PASS verdict knows kernel
        # kind scoring was non-operational for the whole run.
        watcher_unavailable = [e for e in events if e["kind"] == "kernel_watcher_unavailable"]
        if watcher_unavailable:
            watcher_message = (watcher_unavailable[0]["message"] or "")[:160]
            VerdictEngine._record_finding(
                subsystems,
                subsystem="power",
                status="WARN",
                rule_id="coverage.kernel_watcher_unavailable.warn",
                reason=(
                    "Kernel watcher did not start ({}). Kernel-driven rules "
                    "(MCE, Xid, AER, thermal_crit, ...) could not score this run. "
                    "Verdict is non-conclusive for any kernel-driven fault class.".format(
                        watcher_message
                    )
                ),
                observed={"message": watcher_message},
                events=watcher_unavailable,
                source="coverage",
            )

        batch_pattern_events = [e for e in events if e["kind"] == "batch_pattern_detected"]
        if batch_pattern_events:
            VerdictEngine._record_finding(
                subsystems,
                subsystem="power",
                status="FAIL",
                rule_id="power.batch_pattern_detected.fail",
                reason=batch_pattern_events[0]["message"],
                observed={"event_count": len(batch_pattern_events)},
                events=batch_pattern_events,
                source="derived",
            )

        # Stress worker lifecycle events (H2: v0.9).
        # *_worker_start fires only when StressWorker.start() returned an error
        # (the worker failed to launch at all). *_worker_exit fires when a
        # worker's subprocess exited with rc!=0 mid-phase. Both are real fault
        # signals under load — a CPU/GPU stress binary dying mid-stress is the
        # software-visible shape of a thermal kill, OOM, SIGSEGV, or power
        # instability. Routing: cpu/gpu → power (under stress, most worker
        # deaths trace back to PSU/VRM/thermal), mem → memory (memtester SIGBUS
        # is a memory fault), storage → storage. Prior to v0.9 these events
        # were written to events.csv but never scored — same silent-dead-rule
        # class as the kernel watcher gap (C1).
        _worker_lifecycle_routing = (
            ("cpu",     "power",   "CPU stress worker"),
            ("gpu",     "power",   "GPU stress worker"),
            ("mem",     "memory",  "memory stress worker"),
            ("storage", "storage", "storage stress worker"),
        )
        for label, subsystem, human in _worker_lifecycle_routing:
            start_events = [e for e in events if e["kind"] == "{}_worker_start".format(label)]
            exit_events  = [e for e in events if e["kind"] == "{}_worker_exit".format(label)]
            if start_events:
                if subsystems[subsystem]["status"] != "FAIL":
                    subsystems[subsystem]["status"] = "WARN"
                subsystems[subsystem]["reasons"].append(
                    "{} failed to start ({} event(s)). "
                    "Launch-time failure — corroborate with logs and known-good swap.".format(
                        human, len(start_events)
                    )
                )
            if exit_events:
                if subsystems[subsystem]["status"] != "FAIL":
                    subsystems[subsystem]["status"] = "WARN"
                phases_hit = sorted({e["phase"] for e in exit_events})
                subsystems[subsystem]["reasons"].append(
                    "{} exited unexpectedly mid-phase in {} phase(s) ({}). "
                    "Under stress this usually traces to a thermal kill, OOM, "
                    "SIGSEGV, or power instability — correlate with kernel events "
                    "and rail telemetry.".format(
                        human, len(phases_hit), ", ".join(phases_hit)
                    )
                )

        # (oom/oom_killer/edac kernel-log kinds are handled by the table-driven
        # kernel loops above — no bespoke block needed here.)

        # EDAC sysfs counter delta (correctable / uncorrectable error counts).
        edac_ue_events = [e for e in events if e["kind"] == "edac_ue"]
        edac_ce_events = [e for e in events if e["kind"] == "edac_ce"]
        if edac_ue_events:
            subsystems["memory"]["status"] = "FAIL"
            for event in edac_ue_events:
                VerdictEngine._record_finding(
                    subsystems,
                    subsystem="memory",
                    status="FAIL",
                    rule_id="memory.edac_ue.fail",
                    reason=event["message"],
                    events=[event],
                    observed={"kind": event["kind"]},
                    source="event",
                    phase=event["phase"] or None,
                )
        if edac_ce_events:
            if subsystems["memory"]["status"] != "FAIL":
                subsystems["memory"]["status"] = "WARN"
            for event in edac_ce_events:
                VerdictEngine._record_finding(
                    subsystems,
                    subsystem="memory",
                    status="WARN",
                    rule_id="memory.edac_ce.warn",
                    reason=event["message"],
                    events=[event],
                    observed={"kind": event["kind"]},
                    source="event",
                    phase=event["phase"] or None,
                )

        # (thermal_crit/thermal_warn/thermal_throttle are handled by the
        # table-driven kernel loops above — no bespoke block needed here.)

        # Thermal drift events (cold-idle vs hot-idle rail compare, produced
        # by the phase runner after all phases complete). These are rail-
        # behavior symptoms, so they live in the power subsystem rather than
        # thermal — a rail that sags by 50 mV after warmup is a supply-side
        # weakness, not a cooling problem. Previously these events were
        # emitted but never scored, which silently killed the rule.
        thermal_drift_events = [e for e in events if e["kind"] == "thermal_drift"]
        if thermal_drift_events:
            # FAIL decision keys off the magnitude carried in the message
            # ("drift=0.08V"), not the emitter-supplied severity column. If
            # magnitude can't be parsed, fall back to severity so older runs
            # or alternate emitters still score something rather than nothing.
            fail_band = False
            for e in thermal_drift_events:
                match = _THERMAL_DRIFT_MAG_RE.search(e["message"] or "")
                if match:
                    try:
                        if float(match.group(1)) > _THERMAL_DRIFT_FAIL_V:
                            fail_band = True
                            break
                    except ValueError:
                        pass
                elif e["severity"] == "crit":
                    fail_band = True
                    break
            if fail_band:
                subsystems["power"]["status"] = "FAIL"
            elif subsystems["power"]["status"] != "FAIL":
                subsystems["power"]["status"] = "WARN"
            for event in thermal_drift_events:
                drift_v = None
                match = _THERMAL_DRIFT_MAG_RE.search(event["message"] or "")
                if match:
                    try:
                        drift_v = float(match.group(1))
                    except ValueError:
                        drift_v = None
                VerdictEngine._record_finding(
                    subsystems,
                    subsystem="power",
                    status="FAIL" if (drift_v is not None and drift_v > _THERMAL_DRIFT_FAIL_V) or (drift_v is None and event["severity"] == "crit") else "WARN",
                    rule_id="power.thermal_drift.fail" if (drift_v is not None and drift_v > _THERMAL_DRIFT_FAIL_V) or (drift_v is None and event["severity"] == "crit") else "power.thermal_drift.warn",
                    reason=event["message"],
                    threshold={"type": "absolute_voltage_drift", "fail_gt_v": _THERMAL_DRIFT_FAIL_V},
                    observed={"drift_v": drift_v, "severity": event["severity"]},
                    events=[event],
                    source="derived",
                    phase=event["phase"] or None,
                )

        # 12V rail drift vs soak baseline (software sensor).
        load_12v = [
            sample["rail_12v"]
            for sample in samples
            if sample["phase"] != "soak" and sample["rail_12v"] is not None
        ]
        soak_12v = [
            sample["rail_12v"]
            for sample in samples
            if sample["phase"] == "soak" and sample["rail_12v"] is not None
        ]
        if soak_12v and load_12v:
            baseline = statistics.median(soak_12v)
            observed_min = min(load_12v)
            drop_pct = (baseline - observed_min) / baseline if baseline else 0.0
            if observed_min < baseline * 0.92:
                VerdictEngine._record_finding(
                    subsystems,
                    subsystem="power",
                    status="FAIL",
                    rule_id="power.rail_12v_drift.fail",
                    reason="12V drift exceeded 8% from soak baseline ({:.2f}V → {:.2f}V).".format(
                        baseline, observed_min
                    ),
                    threshold={
                        "type": "percentage_drop_from_soak_median",
                        "warn_pct": 0.04,
                        "fail_pct": 0.08,
                    },
                    observed={
                        "soak_median_v": baseline,
                        "load_min_v": observed_min,
                        "drop_pct": drop_pct,
                    },
                    source="sample",
                )
            elif observed_min < baseline * 0.96:
                VerdictEngine._record_finding(
                    subsystems,
                    subsystem="power",
                    status="WARN",
                    rule_id="power.rail_12v_drift.warn",
                    reason="12V drift reached warning band ({:.2f}V → {:.2f}V).".format(
                        baseline, observed_min
                    ),
                    threshold={
                        "type": "percentage_drop_from_soak_median",
                        "warn_pct": 0.04,
                        "fail_pct": 0.08,
                    },
                    observed={
                        "soak_median_v": baseline,
                        "load_min_v": observed_min,
                        "drop_pct": drop_pct,
                    },
                    source="sample",
                )

        rail_runrule_events = [e for e in events if e["kind"] == "rail_runrule_violation"]
        if rail_runrule_events:
            phases_hit = sorted({e["phase"] for e in rail_runrule_events})
            runrule_fail = len(phases_hit) >= 2 or subsystems["power"]["status"] == "FAIL"
            VerdictEngine._record_finding(
                subsystems,
                subsystem="power",
                status="FAIL" if runrule_fail else "WARN",
                rule_id="power.rail_runrule_violation.fail" if runrule_fail else "power.rail_runrule_violation.warn",
                reason=(
                    "12V rail violated Western Electric run-rules in {} distinct stress phase(s) ({}). "
                    "Persistent statistical sag above the bench noise floor."
                ).format(len(phases_hit), ", ".join(phases_hit))
                if runrule_fail else
                (
                    "12V rail violated Western Electric run-rules in phase '{}'. "
                    "Sag pattern is below the 4%/8% threshold but above the bench noise floor."
                ).format(phases_hit[0]),
                observed={"event_count": len(rail_runrule_events), "phase_count": len(phases_hit)},
                events=rail_runrule_events,
                source="event",
                phase=phases_hit[0] if len(phases_hit) == 1 else None,
            )

        rail_cusum_events = [e for e in events if e["kind"] == "rail_cusum_violation"]
        if rail_cusum_events:
            cusum_phases = sorted({e["phase"] for e in rail_cusum_events})
            runrule_phases = {e["phase"] for e in rail_runrule_events}
            cooccur = bool(runrule_phases.intersection(cusum_phases))
            cusum_fail = len(cusum_phases) >= 2 or cooccur
            VerdictEngine._record_finding(
                subsystems,
                subsystem="power",
                status="FAIL" if cusum_fail else "WARN",
                rule_id="power.rail_cusum_violation.fail" if cusum_fail else "power.rail_cusum_violation.warn",
                reason=(
                    "12V rail CUSUM drift triggered in {} distinct stress phase(s) ({}). "
                    "Cumulative sag persisted beyond the bench noise floor."
                ).format(len(cusum_phases), ", ".join(cusum_phases))
                if not cooccur else
                (
                    "12V rail CUSUM drift co-occurred with a Western Electric violation in phase(s) {}. "
                    "Statistical sag is corroborated by both SPC detectors."
                ).format(", ".join(sorted(runrule_phases.intersection(cusum_phases))))
                if cusum_fail else
                (
                    "12V rail CUSUM drift triggered in phase '{}'. "
                    "Cumulative statistical sag detected above the bench noise floor."
                ).format(cusum_phases[0]),
                observed={
                    "event_count": len(rail_cusum_events),
                    "phase_count": len(cusum_phases),
                    "cooccur_runrule": cooccur,
                },
                events=rail_cusum_events,
                source="event",
                phase=cusum_phases[0] if len(cusum_phases) == 1 else None,
            )

        baseline_findings = baseline_drift_findings(run_row, samples)
        if baseline_findings:
            if subsystems["power"]["status"] != "FAIL":
                subsystems["power"]["status"] = "WARN"
            for finding in baseline_findings:
                VerdictEngine._record_finding(
                    subsystems,
                    subsystem="power",
                    status="WARN",
                    rule_id=str(finding["rule_id"]),
                    reason=str(finding["human_reason"]),
                    threshold=finding.get("threshold"),
                    observed=finding.get("observed"),
                    source=str(finding.get("source") or "derived"),
                )

        # Manual multimeter readings (operator-entered).
        for field_name, floor, ceiling, label in [
            ("manual_idle_12v", 11.4, 12.6, "Manual 12V idle"),
            ("manual_load_12v", 11.4, 12.6, "Manual 12V load"),
            ("manual_idle_5v", 4.75, 5.25, "Manual 5V idle"),
            ("manual_load_5v", 4.75, 5.25, "Manual 5V load"),
        ]:
            value = run_row[field_name]
            if value is None:
                continue
            if value < floor or value > ceiling:
                VerdictEngine._record_finding(
                    subsystems,
                    subsystem="power",
                    status="FAIL",
                    rule_id="power.manual_multimeter_out_of_range.fail",
                    reason="{} out of ATX range ({:.2f}V).".format(label, value),
                    threshold={"type": "inclusive_voltage_range", "min_v": floor, "max_v": ceiling},
                    observed={"field": field_name, "label": label, "value_v": value},
                    source="run_row",
                )

        # Bench observations: POST failure and known-good swap result.
        if run_row["post_result"] == "fail":
            VerdictEngine._record_finding(
                subsystems,
                subsystem="power",
                status="FAIL",
                reason="The DUT failed to POST with the returned PSU installed.",
            )

        if run_row["known_good_swap_result"] == "fault-follows-psu":
            VerdictEngine._record_finding(
                subsystems,
                subsystem="power",
                status="FAIL",
                rule_id="power.known_good_swap_fault_follows_psu.fail",
                reason="Observed instability followed the returned PSU and cleared with a known-good PSU.",
                observed={"known_good_swap_result": run_row["known_good_swap_result"]},
                source="run_row",
            )

        if run_row["known_good_swap_result"] == "fault-remains":
            VerdictEngine._record_finding(
                subsystems,
                subsystem="power",
                status="WARN",
                rule_id="power.known_good_swap_fault_remains.warn",
                reason=(
                    "Known-good PSU substitution did not remove the observed failure. "
                    "Do not attribute the fault to the returned PSU."
                ),
                observed={"known_good_swap_result": run_row["known_good_swap_result"]},
                source="run_row",
            )

        return subsystems

    @staticmethod
    def evaluate(db: Database) -> Tuple[str, List[str], str, Dict[str, Dict[str, object]]]:
        run_row = db.fetch_run()
        subsystems = VerdictEngine.evaluate_subsystems(db)
        reasons: List[str] = []
        claim_scope = "software-only"

        if any(s["status"] == "INVALID" for s in subsystems.values()):
            reasons.append("No telemetry samples were captured.")
            return "INVALID", reasons, claim_scope, subsystems

        for name in ["power", "thermal", "memory", "storage", "pcie"]:
            reasons.extend("[{}] {}".format(name, reason) for reason in subsystems[name]["reasons"])

        if any(s["status"] == "FAIL" for s in subsystems.values()):
            if run_row["known_good_swap_result"] == "fault-follows-psu" and subsystems["power"]["status"] == "FAIL":
                claim_scope = "isolated-by-swap"
            elif run_row["meter_available"] and subsystems["power"]["status"] == "FAIL":
                claim_scope = "electrical-plus-software"
            return "FAIL", reasons, claim_scope, subsystems

        if run_row["known_good_swap_result"] == "fault-remains":
            return "INCONCLUSIVE", reasons, "software-only", subsystems

        if any(s["status"] == "WARN" for s in subsystems.values()):
            return "WARN", reasons, claim_scope, subsystems

        if run_row["meter_available"]:
            return "PASS", ["No fail or warning rules tripped in this run."], "electrical-plus-software", subsystems

        return (
            "PASS",
            [
                "No fail or warning rules tripped in this run.",
                "No multimeter readings were recorded — software-triage pass only.",
            ],
            "software-only",
            subsystems,
        )
