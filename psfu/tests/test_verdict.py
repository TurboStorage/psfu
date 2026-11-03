"""Verdict rule coverage.

One test per distinct path through ``VerdictEngine.evaluate``. The point of
this file is to make it impossible to emit a new ``db.add_event(kind=X)``
without a matching mapping in ``rules.py`` — a regression we shipped twice
before (``thermal_drift`` and the old mock-parser 12V rule).
"""

from __future__ import annotations

from psfu.tests.conftest import _base_sample, make_run_db

from psfu.verdict.rules import VerdictEngine


def _add_soak_and_load_samples(db, *, soak_12v: float = 12.1, load_12v: float = 12.1) -> None:
    """Seed the DB with enough samples that the 12V drift rule can run."""
    for _ in range(5):
        db.add_sample(_base_sample(phase="soak", rail_12v=soak_12v))
    for _ in range(5):
        db.add_sample(_base_sample(phase="steady", rail_12v=load_12v))


def test_pass_software_only(tmp_path):
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "PASS"
    assert scope == "software-only"
    assert all(sub["status"] == "PASS" for sub in subs.values())


def test_pass_electrical_plus_software_when_meter_present(tmp_path):
    db = make_run_db(tmp_path, overrides={"meter_available": 1})
    _add_soak_and_load_samples(db)
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "PASS"
    assert scope == "electrical-plus-software"


def test_invalid_on_no_samples(tmp_path):
    db = make_run_db(tmp_path)
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "INVALID"
    assert all(sub["status"] == "INVALID" for sub in subs.values())


def test_fail_on_mce_via_power_brake_event(tmp_path):
    """GPU HW power-brake in a stress phase bumps the power subsystem to WARN."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "gpu_hw_power_throttle", "brake active")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["power"]["status"] == "WARN"


def test_fail_on_gpu_disappeared(tmp_path):
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "crit", "nvidia_gpu_disappeared", "3 silent ticks")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"


def test_fail_on_pcie_aer_burst(tmp_path):
    """>3 AER corrected errors in one phase → PCIe FAIL."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    for i in range(4):
        db.add_event("steady", "warn", "pcie_aer", "aer #{}".format(i))
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["pcie"]["status"] == "FAIL"


def test_warn_on_single_pcie_aer(tmp_path):
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "pcie_aer", "single aer")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["pcie"]["status"] == "WARN"


def test_fail_on_12v_drift_from_baseline(tmp_path):
    """Software 12V drift rule: load rail 10% below soak baseline → power FAIL."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db, soak_12v=12.10, load_12v=10.80)  # ~10.7% drop
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"


def test_warn_on_12v_drift_warning_band(tmp_path):
    """Software 12V drift rule: ~5% drop → power WARN, not FAIL."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db, soak_12v=12.10, load_12v=11.50)  # ~5% drop
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["power"]["status"] == "WARN"
    assert [item["rule_id"] for item in subs["power"]["evidence"]] == [
        "power.rail_12v_drift.warn",
    ]


def test_verdict_warn_on_single_phase_rail_runrule_violation(tmp_path):
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "rail_runrule_violation", "12V WE-1: seven consecutive samples below mean")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["power"]["status"] == "WARN"
    assert any(item["rule_id"] == "power.rail_runrule_violation.warn" for item in subs["power"]["evidence"])


def test_verdict_fail_on_multi_phase_rail_runrule_violation(tmp_path):
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "rail_runrule_violation", "12V WE-1: phase 1")
    db.add_event("transient", "warn", "rail_runrule_violation", "12V WE-1: phase 2")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"
    assert any(item["rule_id"] == "power.rail_runrule_violation.fail" for item in subs["power"]["evidence"])


def test_verdict_fail_when_runrule_combined_with_existing_power_fail(tmp_path):
    db = make_run_db(tmp_path, overrides={"post_result": "fail"})
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "rail_runrule_violation", "12V WE-2")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"


def test_verdict_escalates_when_cusum_and_runrule_cooccur_same_phase(tmp_path):
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "rail_runrule_violation", "12V WE-2")
    db.add_event("steady", "warn", "rail_cusum_violation", "12V CUSUM downward")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"
    assert any(item["rule_id"] == "power.rail_cusum_violation.fail" for item in subs["power"]["evidence"])


def test_fail_on_thermal_drift_event(tmp_path):
    """Regression: ``thermal_drift`` events must be routed to the power
    subsystem, not silently ignored. This is the bug we shipped twice."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event(
        "run",
        "crit",
        "thermal_drift",
        "12V thermal drift exceeded 0.05V (cold=12.10V hot=12.02V drift=0.08V)",
    )
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"
    assert any("thermal drift" in r.lower() for r in reasons)


def test_warn_on_thermal_drift_warn_event(tmp_path):
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event(
        "run",
        "warn",
        "thermal_drift",
        "12V thermal drift reached warning band (drift=0.04V)",
    )
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["power"]["status"] == "WARN"


def test_thermal_drift_fail_keys_off_magnitude_not_severity(tmp_path):
    """H4 regression: a ``crit`` severity with a sub-threshold drift (0.02V)
    must NOT produce FAIL — the rule keys off the numeric magnitude parsed
    from the message, not the emitter-supplied severity column."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event(
        "run",
        "crit",  # emitter lies or has a bug — severity says crit
        "thermal_drift",
        "12V thermal drift observed (drift=0.02V)",  # but magnitude is fine
    )
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["power"]["status"] == "WARN"


def test_thermal_drift_fail_on_magnitude_even_if_severity_is_warn(tmp_path):
    """Complement of the above: a ``warn`` severity with drift >0.05V must
    still FAIL. Magnitude is authoritative."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event(
        "run",
        "warn",  # emitter tagged only warn
        "thermal_drift",
        "12V thermal drift exceeded 0.05V (cold=12.10V hot=12.02V drift=0.08V)",
    )
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"


def test_thermal_drift_unparseable_message_falls_back_to_severity(tmp_path):
    """Backward compat: a malformed message (no ``drift=`` token) falls back
    to the severity column so pre-H4 runs still score."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event(
        "run",
        "crit",
        "thermal_drift",
        "legacy format without magnitude token",
    )
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"


def test_fail_on_post_failure(tmp_path):
    """DUT didn't POST with the returned PSU → power FAIL regardless of telemetry."""
    db = make_run_db(tmp_path, overrides={"post_result": "fail"})
    _add_soak_and_load_samples(db)
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"


def test_claim_scope_isolated_by_swap(tmp_path):
    """Known-good swap cleared the fault → claim_scope = isolated-by-swap."""
    db = make_run_db(
        tmp_path,
        overrides={
            "post_result": "fail",
            "known_good_swap_performed": 1,
            "known_good_swap_result": "fault-follows-psu",
        },
    )
    _add_soak_and_load_samples(db)
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert scope == "isolated-by-swap"


def test_inconclusive_when_fault_remains_after_swap(tmp_path):
    """Swap did not remove the observed failure → verdict INCONCLUSIVE."""
    db = make_run_db(
        tmp_path,
        overrides={
            "known_good_swap_performed": 1,
            "known_good_swap_result": "fault-remains",
        },
    )
    _add_soak_and_load_samples(db)
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "INCONCLUSIVE"


def test_fail_on_manual_rail_out_of_spec(tmp_path):
    """Operator-entered multimeter reading outside ATX range → power FAIL."""
    db = make_run_db(
        tmp_path,
        overrides={
            "meter_available": 1,
            "manual_load_12v": 10.9,  # below 11.4 ATX floor
        },
    )
    _add_soak_and_load_samples(db)
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"
    assert any(
        item["rule_id"] == "power.manual_multimeter_out_of_range.fail"
        and item["observed"]["field"] == "manual_load_12v"
        for item in subs["power"]["evidence"]
    )


def test_fail_on_storage_smart(tmp_path):
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("run", "crit", "storage_smart_fail", "/dev/nvme0n1 SMART failed")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["storage"]["status"] == "FAIL"


def test_fail_on_storage_pcie_link_downshift(tmp_path):
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "crit", "pcie_link_downshift",
                 "NVMe PCIe link downshift persisted for 3 consecutive ticks in phase 'steady'.")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["storage"]["status"] == "FAIL"
    assert [item["rule_id"] for item in subs["storage"]["evidence"]] == [
        "storage.pcie_link_downshift.fail",
    ]


def test_fail_on_batch_pattern_detected(tmp_path):
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event(
        "run",
        "crit",
        "batch_pattern_detected",
        "Batch pattern detected for lot LOT-X: 2 FAIL run(s) recorded (SN-A, SN-B). Pull remaining inventory and escalate per Loop 3.",
    )
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"
    assert [item["rule_id"] for item in subs["power"]["evidence"]] == [
        "power.batch_pattern_detected.fail",
    ]


def test_fail_on_edac_memory_error(tmp_path):
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "edac", "EDAC MC0: 1 CE ...")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["memory"]["status"] == "FAIL"


# ---------------------------------------------------------------------------
# Placeholder-serial guard. Prevents smoke/dry-run/demo serials from polluting
# the real batch tracker via update_batch_tracker().
# ---------------------------------------------------------------------------


def test_placeholder_serial_recognises_exact_tokens():
    from psfu.reports.markdown import is_placeholder_serial
    for s in ("", "   ", "TEST", "test", "dummy", "SN000001", "tbd", "n/a", "?", "???"):
        assert is_placeholder_serial(s), "should be placeholder: {!r}".format(s)


def test_placeholder_serial_recognises_substring_markers():
    """Regression: "SMOKE-TEST-001" was accepted by update_batch_tracker and
    wrote a row into the real batch tracker during a smoke test."""
    from psfu.reports.markdown import is_placeholder_serial
    for s in (
        "SMOKE-TEST-001",
        "HX1200-SAMPLE",
        "DEMO-UNIT-42",
        "my-dry-run-1",
        "FAKE-SN-9",
    ):
        assert is_placeholder_serial(s), "should be placeholder: {!r}".format(s)


def test_placeholder_serial_accepts_real_looking_serials():
    from psfu.reports.markdown import is_placeholder_serial
    for s in ("HX1200-4242-ABCDE", "CP-1000X-42069", "CORSAIR-C0FFEE"):
        assert not is_placeholder_serial(s), "should NOT be placeholder: {!r}".format(s)


# ---------------------------------------------------------------------------
# EDAC sysfs counter rules (edac_ue / edac_ce).
# These are separate from the kernel-log "edac" kind — they come from the
# pre/post sysfs snapshot delta, not from journalctl pattern matching.
# ---------------------------------------------------------------------------


def test_fail_on_edac_ue_counter_delta(tmp_path):
    """Any uncorrectable EDAC error delta during a run is a hard FAIL."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("run", "crit", "edac_ue",
                 "EDAC uncorrectable memory error(s) detected during run: 1 UE across all controllers.")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["memory"]["status"] == "FAIL"
    assert any("uncorrectable" in r.lower() for r in subs["memory"]["reasons"])


def test_warn_on_edac_ce_counter_delta(tmp_path):
    """Correctable EDAC error delta = WARN (DIMM degrading, ECC masking it)."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("run", "warn", "edac_ce",
                 "EDAC correctable memory error(s) detected during run: 3 CE total.")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["memory"]["status"] == "WARN"
    assert any("correctable" in r.lower() for r in subs["memory"]["reasons"])


# ---------------------------------------------------------------------------
# GPU clock throttle rule (gpu_clock_throttle).
# ---------------------------------------------------------------------------


def test_warn_on_cpu_core_throttle(tmp_path):
    """CPU frequency throttled under load → power WARN."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "cpu_core_throttle",
                 "CPU avg frequency dropped 50% from stress baseline (4800 → 2400 MHz) "
                 "while busy at 95%.")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["power"]["status"] == "WARN"
    assert any("cpu" in r.lower() and "throttle" in r.lower() for r in subs["power"]["reasons"])


def test_warn_on_gpu_clock_throttle(tmp_path):
    """GPU SM clock throttled under load → power WARN."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "gpu_clock_throttle",
                 "GPU SM clock dropped 45% from soak baseline (2100 → 1155 MHz) "
                 "while utilization at 98%.")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["power"]["status"] == "WARN"
    assert any("clock throttle" in r.lower() for r in subs["power"]["reasons"])


# ---------------------------------------------------------------------------
# GPU ECC error rules (gpu_ecc_ue / gpu_ecc_ce).
# ---------------------------------------------------------------------------


def test_fail_on_gpu_ecc_ue(tmp_path):
    """GPU uncorrectable ECC error → memory FAIL."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "crit", "gpu_ecc_ue",
                 "GPU uncorrectable ECC error(s) detected: 1 new UE (total 1). "
                 "VRAM data corruption — GPU is failing.")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["memory"]["status"] == "FAIL"
    assert any("uncorrectable" in r.lower() for r in subs["memory"]["reasons"])


def test_fail_on_gpu_retired_pages(tmp_path):
    """GPU retired pages increasing mid-run → memory FAIL."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "crit", "gpu_retired_pages",
                 "GPU retired pages increased by 2 during run (was 0, now 2). "
                 "VRAM cells are being permanently marked bad — GPU is actively failing.")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["memory"]["status"] == "FAIL"
    assert any("retired pages" in r.lower() for r in subs["memory"]["reasons"])


def test_warn_on_gpu_ecc_ce(tmp_path):
    """GPU correctable ECC error → memory WARN."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "gpu_ecc_ce",
                 "GPU correctable ECC error(s) detected: 2 new CE (total 2). "
                 "VRAM is degrading — ECC is masking errors now.")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["memory"]["status"] == "WARN"
    assert any("correctable" in r.lower() for r in subs["memory"]["reasons"])


# ---------------------------------------------------------------------------
# Kernel watcher rules (v0.9, adds scoring for kinds emitted by
# psfu/collectors/kernel.py KERNEL_WATCH_PATTERNS). Before v0.9 these were
# written to the DB but never scored — the MCE/panic/xid gap that let a
# recovered hardware fault yield PASS mid-run.
#
# Every kind in KERNEL_WATCH_PATTERNS now has a consumer here. If you add a
# new pattern to kernel.py, add a test here too.
# ---------------------------------------------------------------------------


def test_fail_on_kernel_mce(tmp_path):
    """v0.8 FAIL #2: Kernel MCE event must trip power FAIL."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "crit", "mce",
                 "mce: [Hardware Error]: CPU 12: Machine Check Exception: 0 Bank 1")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"
    assert any("mce" in r.lower() for r in subs["power"]["reasons"])


def test_fail_on_kernel_panic(tmp_path):
    """v0.8 FAIL #1: kernel panic during stress → power FAIL."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "crit", "kernel_panic",
                 "Kernel panic - not syncing: Attempted to kill init!")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"
    assert any("panic" in r.lower() for r in subs["power"]["reasons"])


def test_fail_on_kernel_oops(tmp_path):
    """v0.8 FAIL #1: kernel oops during stress → power FAIL."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "crit", "kernel_oops",
                 "Oops: 0002 [#1] SMP PTI")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"
    assert any("oops" in r.lower() for r in subs["power"]["reasons"])


def test_fail_on_hardware_error(tmp_path):
    """v0.8 FAIL #1: generic Hardware Error line → power FAIL."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "crit", "hardware_error",
                 "Hardware Error: Corrected error, no action required.")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"
    assert any("hardware" in r.lower() for r in subs["power"]["reasons"])


def test_fail_on_nvidia_xid(tmp_path):
    """Xid events during stress are almost always PSU/GPU fault → power FAIL."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "crit", "nvidia_xid",
                 "NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"
    assert any("xid" in r.lower() for r in subs["power"]["reasons"])


def test_fail_on_kernel_undervoltage(tmp_path):
    """Explicit rail collapse signal from kernel → power FAIL."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "crit", "undervoltage",
                 "Under-voltage detected! (0x00050005)")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"
    assert any("under-voltage" in r.lower() or "undervoltage" in r.lower() or "rail collapse" in r.lower()
               for r in subs["power"]["reasons"])


def test_fail_on_kernel_brownout(tmp_path):
    """Brownout kernel event → power FAIL."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "crit", "brownout",
                 "brownout detected on rail")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"
    assert any("brownout" in r.lower() or "rail collapse" in r.lower()
               for r in subs["power"]["reasons"])


def test_warn_on_acpi_bios_error(tmp_path):
    """ACPI BIOS error = firmware fault, non-conclusive → power WARN."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "acpi_bios",
                 "ACPI BIOS Error (bug): Could not resolve symbol [\\_SB.PCI0._OSC]")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["power"]["status"] == "WARN"
    assert any("acpi" in r.lower() for r in subs["power"]["reasons"])


def test_fail_precedence_kernel_fail_beats_acpi_warn(tmp_path):
    """If both a FAIL-tier kernel event and an ACPI WARN occur, FAIL wins."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "acpi_bios", "ACPI BIOS Error")
    db.add_event("steady", "crit", "mce", "mce: [Hardware Error]: Bank 5")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"


def test_fail_on_nvme_io_timeout(tmp_path):
    """S6 (v0.17): NVMe I/O timeout during stress → storage FAIL.
    Controller failed to service a queued command within the kernel timeout
    window — always a hard storage fault at the workload scales PS-FU runs."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "crit", "nvme_io_timeout",
                 "nvme nvme0: I/O 123 QID 4 timeout, aborting")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["storage"]["status"] == "FAIL"
    assert "storage.nvme_io_timeout.fail" in [
        item["rule_id"] for item in subs["storage"]["evidence"]
    ]


def test_fail_on_nvme_controller_reset(tmp_path):
    """S6 (v0.17): NVMe controller reset during stress → storage FAIL.
    Kernel forced a reset after a stalled queue or internal controller fault."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "crit", "nvme_controller_reset",
                 "nvme nvme0: resetting controller")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["storage"]["status"] == "FAIL"
    assert "storage.nvme_controller_reset.fail" in [
        item["rule_id"] for item in subs["storage"]["evidence"]
    ]


def test_fail_on_nvme_aer_error(tmp_path):
    """S6 (v0.17): NVMe async event notified error → storage FAIL.
    Controller self-reported a fault condition via async event."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "crit", "nvme_aer_error",
                 "nvme nvme0: async event result 00000300 error")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["storage"]["status"] == "FAIL"
    assert "storage.nvme_aer_error.fail" in [
        item["rule_id"] for item in subs["storage"]["evidence"]
    ]


def test_fail_on_fio_stall(tmp_path):
    """S6 (v0.17): fio_stall event → storage FAIL with dedicated rule_id.
    Emitted by check_nvme_fio_stall after 10 consecutive 1 Hz ticks with
    no ios progress. Routed via storage_fail_events fold-in, not the
    generic kernel loop, so rule_id is storage.fio_stall.fail (not
    storage.fio_stall.fail via kernel routing)."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "crit", "fio_stall",
                 "fio IO progress stalled for 10 consecutive ticks in phase 'steady'.")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["storage"]["status"] == "FAIL"
    assert "storage.fio_stall.fail" in [
        item["rule_id"] for item in subs["storage"]["evidence"]
    ]


def test_fail_on_ata_reallocated_sector_grew(tmp_path):
    """S9 (v0.19): reallocated sector growth → storage FAIL.
    Remapping events exhaust spare sectors; hard media degradation.
    rule_id preserves full ATA kind (storage.ata_reallocated_sector_grew.fail)."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("run", "crit", "ata_reallocated_sector_grew",
                 "/dev/sda ATA reallocated_sector_ct grew by 3 (0 -> 3) during run.")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["storage"]["status"] == "FAIL"
    assert "storage.ata_reallocated_sector_grew.fail" in [
        item["rule_id"] for item in subs["storage"]["evidence"]
    ]


def test_fail_on_ata_pending_sector_grew(tmp_path):
    """S9 (v0.19): pending sector growth → storage FAIL.
    Sectors flagged for remap on next write; drive knows they're bad."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("run", "crit", "ata_pending_sector_grew",
                 "/dev/sda ATA current_pending_sector grew by 1 (0 -> 1) during run.")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["storage"]["status"] == "FAIL"
    assert "storage.ata_pending_sector_grew.fail" in [
        item["rule_id"] for item in subs["storage"]["evidence"]
    ]


def test_fail_on_ata_offline_uncorrectable_grew(tmp_path):
    """S9 (v0.19): offline uncorrectable growth → storage FAIL.
    ECC couldn't recover the sector during an offline scan."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("run", "crit", "ata_offline_uncorrectable_grew",
                 "/dev/sda ATA offline_uncorrectable grew by 2 (0 -> 2) during run.")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["storage"]["status"] == "FAIL"
    assert "storage.ata_offline_uncorrectable_grew.fail" in [
        item["rule_id"] for item in subs["storage"]["evidence"]
    ]


def test_warn_not_fail_on_ata_udma_crc_grew(tmp_path):
    """S9 (v0.19) critical regression guard: UDMA CRC growth must emit
    WARN, never FAIL. The failure mode is cable / SATA port / backplane
    signal integrity — not the drive. FAIL would condemn good drives on
    degrading cables."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("run", "warning", "ata_udma_crc_grew",
                 "/dev/sda ATA udma_crc_error_count grew by 10 (5 -> 15) during run.")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    # Must NOT be FAIL.
    assert subs["storage"]["status"] == "WARN"
    assert "storage.ata_udma_crc_grew.warn" in [
        item["rule_id"] for item in subs["storage"]["evidence"]
    ]
    # No FAIL rule_id for the UDMA kind.
    fail_ids = [item["rule_id"] for item in subs["storage"]["evidence"]
                if item.get("rule_id", "").endswith(".fail")]
    assert not any("udma_crc" in rid for rid in fail_ids)


def test_warn_on_baseline_backed_fio_write_cliff_severe(tmp_path):
    """S11 (v0.22): baseline-backed pSLC cliff is a storage WARN, not FAIL."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event(
        "cliff-write",
        "warning",
        "fio_write_cliff_severe",
        "/dev/nvme0n1 write cliff ratio 0.080 below baseline floor 0.330; baseline-backed severe cliff scored as WARN.",
    )
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["storage"]["status"] == "WARN"
    assert "storage.fio_write_cliff_severe.warn" in [
        item["rule_id"] for item in subs["storage"]["evidence"]
    ]


def test_kernel_watcher_kinds_have_verdict_consumers():
    """Regression guard: every kind in KERNEL_WATCH_PATTERNS must be scored
    somewhere in rules.py. If a new pattern is added to the watcher without a
    corresponding verdict consumer, this test fails — the same silent-dead-rule
    pattern that caused the v0.8 → v0.9 bump."""
    from psfu.collectors.kernel import KERNEL_WATCH_PATTERNS
    import inspect
    from psfu.verdict import rules as rules_mod

    source = inspect.getsource(rules_mod)
    unscored = []
    for _pattern, _severity, kind, _subsystem, _default in KERNEL_WATCH_PATTERNS:
        # The kind must appear as a string literal in rules.py somewhere.
        # We look for the quoted kind — simple but sufficient: the verdict
        # engine always matches on e["kind"] == "name" or `in (...)` tuples.
        if '"{}"'.format(kind) not in source and "'{}'".format(kind) not in source:
            unscored.append(kind)
    assert not unscored, (
        "Kernel watcher kinds with no verdict consumer: {}. "
        "Every kind emitted by KernelWatcher must be scored in rules.py — "
        "otherwise it's written to events.csv but silently ignored.".format(unscored)
    )


# H3 (validation 2026-04-14): broader AST-level guard. The test above only
# covers KERNEL_WATCH_PATTERNS; every other ``emit_event`` kind in
# runners/*.py was un-guarded, and kernel_watcher_unavailable slipped through
# exactly that gap. This test AST-walks every ``emit_event(...)``,
# ``self._emit(...)``, and ``self._emit_event(...)`` call, extracts the 3rd
# positional argument (the kind), expands label-templated kinds across the
# known DUT label set, and asserts every resulting kind is either:
#
#   (a) present as a string literal in rules.py, or
#   (b) explicitly listed in _INFO_ONLY_KINDS below (operator/flow events
#       that are intentionally unscored).
#
# Any addition to runners/ that invents a new kind must either be consumed in
# rules.py or added to _INFO_ONLY_KINDS in the same commit. Whitelisting is
# a visible code edit, reviewable in diff.

# Operator/flow events written to the DB for audit but deliberately not scored
# by the verdict engine. Anything here is safe-listed — but adding to this set
# is a review-visible action.
_INFO_ONLY_KINDS = frozenset({
    "dut_selection",      # which DUTs the operator armed at run start
    "power_limit",        # nvidia-smi -pl adjustments (informational)
    "baseline_unavailable",  # SPC degraded to within-run soak baseline
    "pcie_link_capability_unknown",  # NVMe capability unavailable; rule disabled
    "fio_write_cliff_observed",  # S11 generic pSLC cliff without matching per-PN baseline is informational only
    "user_abort",          # operator pressed q
    "phase_skipped",       # operator pressed n
    "pulse_cycle",         # transient-pulse runner on/off beats
    "ramp_step",           # ramp runner step announcements
    # kernel_watcher / turbostat startup/info events emitted via raw
    # db.add_event in prepare_run (before emit_event closure exists), so
    # they don't show up in this AST walk — listed here for completeness
    # only if a future edit routes them through emit_event.
})

# DUT labels used by phase_runners.py to template worker lifecycle kinds.
# Kept in sync with the _worker_lifecycle_routing tuple in rules.py — any
# new label here must also be added there, or this test fails.
_DUT_LABELS = ("cpu", "gpu", "mem", "storage")


def _extract_emit_kinds_from(path):
    """Return a (literals, templates) pair of kind forms emitted from one
    runners/*.py file.

    Handles the three call surfaces that fan into the same underlying
    ``emit_event`` closure:
      - bare ``emit_event(severity, phase, kind, ...)``
      - ``self._emit(severity, phase, kind, ...)`` (phase_runners.py)
      - ``self._emit_event(severity, phase, kind, ...)`` (phase_runners.py)

    Also understands two kind-template shapes that appear in phase_runners:
      - ``"{}_worker_start".format(self.label)`` → template ``{}_worker_start``
      - f-strings with a single formatted field → same template form

    Literals go into the ``literals`` set verbatim. Templates go into the
    ``templates`` set as a string containing a single ``{}`` placeholder —
    the consumer-check then verifies rules.py contains *either* the expanded
    form for each DUT label *or* the template itself (rules.py has the same
    ``"{}_worker_start".format(label)`` pattern, which is legitimate).
    """
    import ast

    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=str(path))

    literals = set()
    templates = set()

    def _record_kind(kind_node):
        """Extract literal/templated kind values from the 3rd positional arg."""
        if isinstance(kind_node, ast.Constant) and isinstance(kind_node.value, str):
            literals.add(kind_node.value)
            return
        # "{}_worker_start".format(self.label) → Call(func=Attribute(Constant, "format"))
        if (
            isinstance(kind_node, ast.Call)
            and isinstance(kind_node.func, ast.Attribute)
            and kind_node.func.attr == "format"
            and isinstance(kind_node.func.value, ast.Constant)
            and isinstance(kind_node.func.value.value, str)
        ):
            template = kind_node.func.value.value
            if template.count("{}") == 1:
                templates.add(template)
                return
        # f-string with a single FormattedValue
        if isinstance(kind_node, ast.JoinedStr):
            parts = []
            fmt_count = 0
            for piece in kind_node.values:
                if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                    parts.append(piece.value)
                elif isinstance(piece, ast.FormattedValue):
                    parts.append("{}")
                    fmt_count += 1
                else:
                    return  # unsupported shape; skip silently
            template = "".join(parts)
            if fmt_count == 1 and "{}" in template:
                templates.add(template)
                return
        # Anything else (variable, complex call, concatenation) is not
        # statically extractable. That's a design smell — kinds should be
        # either literal or a template the walker can expand. Raise so the
        # test fails rather than silently skipping an unknowable kind.
        raise AssertionError(
            "emit_event kind in {} at line {} is not a string literal or "
            "recognized single-field template: {}. Kinds must be statically "
            "extractable so the verdict-consumer guard can cover them."
            .format(path, getattr(kind_node, "lineno", "?"), ast.dump(kind_node))
        )

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node):
            callee_name = None
            if isinstance(node.func, ast.Name) and node.func.id == "emit_event":
                callee_name = "emit_event"
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in ("_emit", "_emit_event")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):
                callee_name = node.func.attr
            if callee_name is not None and len(node.args) >= 3:
                _record_kind(node.args[2])
            self.generic_visit(node)

    _Visitor().visit(tree)
    return literals, templates


def test_every_runner_emit_event_kind_has_verdict_consumer():
    """Broader-scope regression guard: every static kind emitted via
    ``emit_event`` / ``self._emit`` / ``self._emit_event`` from any
    ``psfu/runners/*.py`` file must either be consumed in ``rules.py`` or
    explicitly whitelisted in ``_INFO_ONLY_KINDS``.

    This closes the gap that let ``kernel_watcher_unavailable`` and
    ``thermal_drift`` ship unscored in prior versions."""
    import inspect
    from pathlib import Path

    from psfu.verdict import rules as rules_mod

    source = inspect.getsource(rules_mod)

    runners_dir = Path(inspect.getfile(rules_mod)).parent.parent / "runners"
    runner_files = sorted(runners_dir.glob("*.py"))
    assert runner_files, "No runner files found at {}".format(runners_dir)

    all_literals = set()
    all_templates = set()
    for runner_path in runner_files:
        literals, templates = _extract_emit_kinds_from(runner_path)
        all_literals |= literals
        all_templates |= templates

    assert all_literals or all_templates, (
        "AST walker found zero emit_event kinds — walker is broken"
    )

    # Expand templates into concrete kinds via the known DUT label set.
    # Each expanded kind is acceptable if either (a) the expanded literal is
    # in rules.py, or (b) the template itself is in rules.py (rules.py uses
    # the same ``"{}_worker_start".format(label)`` pattern when iterating).
    unscored = []
    for kind in sorted(all_literals):
        if kind in _INFO_ONLY_KINDS:
            continue
        if '"{}"'.format(kind) in source or "'{}'".format(kind) in source:
            continue
        unscored.append(kind)

    for template in sorted(all_templates):
        template_literal_in_rules = (
            '"{}"'.format(template) in source or "'{}'".format(template) in source
        )
        for label in _DUT_LABELS:
            expanded = template.replace("{}", label)
            if expanded in _INFO_ONLY_KINDS:
                continue
            if template_literal_in_rules:
                continue
            if '"{}"'.format(expanded) in source or "'{}'".format(expanded) in source:
                continue
            unscored.append(expanded)

    assert not unscored, (
        "emit_event kinds with no verdict consumer in rules.py: {}. "
        "Every non-info kind must appear as a string literal in rules.py "
        "(or its matching ``{{}}``-template must be there), or be added to "
        "_INFO_ONLY_KINDS in this test file with a comment explaining why "
        "it's intentionally unscored.".format(unscored)
    )


# -----------------------------------------------------------------------------
# H2: Stress worker lifecycle event scoring (v0.9)
# -----------------------------------------------------------------------------
# *_worker_start fires when StressWorker.start() returns an error (failed to
# launch). *_worker_exit fires when the subprocess dies with rc!=0 mid-phase.
# Before v0.9 these kinds were emitted but not scored.


def test_warn_on_cpu_worker_exit(tmp_path):
    """CPU stress worker dying mid-phase → power WARN (PSU/VRM/thermal suspect)."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "cpu_worker_exit",
                 "rc=-11 (SIGSEGV) after 342s")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["power"]["status"] == "WARN"
    assert any("cpu stress worker" in r.lower() and "exited" in r.lower()
               for r in subs["power"]["reasons"])


def test_warn_on_gpu_worker_exit(tmp_path):
    """gpu-burn dying mid-phase → power WARN."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "gpu_worker_exit", "rc=1 after 120s")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["power"]["status"] == "WARN"
    assert any("gpu stress worker" in r.lower() for r in subs["power"]["reasons"])


def test_warn_on_mem_worker_exit_routes_to_memory(tmp_path):
    """memtester/stressapptest SIGBUS is a memory fault signal — route to memory."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "mem_worker_exit",
                 "rc=-7 (SIGBUS) after 80s")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["memory"]["status"] == "WARN"
    assert subs["power"]["status"] == "PASS"
    assert any("memory stress worker" in r.lower() for r in subs["memory"]["reasons"])


def test_warn_on_storage_worker_exit_routes_to_storage(tmp_path):
    """Storage read worker dying → storage WARN."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "storage_worker_exit", "rc=1 after 60s")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["storage"]["status"] == "WARN"
    assert any("storage stress worker" in r.lower()
               for r in subs["storage"]["reasons"])


def test_warn_on_cpu_worker_start_failure(tmp_path):
    """stress-ng failed to launch → power WARN (launch-time failure)."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "cpu_worker_start",
                 "stress-ng not found on PATH")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["power"]["status"] == "WARN"
    assert any("failed to start" in r.lower() for r in subs["power"]["reasons"])


def test_warn_on_mem_worker_start_failure_routes_to_memory(tmp_path):
    """memtester failed to launch → memory WARN (not power)."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "mem_worker_start", "memtester not found")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["memory"]["status"] == "WARN"
    assert subs["power"]["status"] == "PASS"


def test_worker_exit_across_multiple_phases_reported(tmp_path):
    """Multiple phase hits aggregate into one reason line naming each phase."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "gpu_worker_exit", "rc=1")
    db.add_event("transient", "warn", "gpu_worker_exit", "rc=-11")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["power"]["status"] == "WARN"
    reason_text = " ".join(subs["power"]["reasons"]).lower()
    assert "steady" in reason_text and "transient" in reason_text


def test_worker_exit_does_not_override_fail(tmp_path):
    """A FAIL-tier event (e.g. MCE) must dominate a worker_exit WARN."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "crit", "mce", "mce: [Hardware Error]: Bank 5")
    db.add_event("steady", "warn", "cpu_worker_exit", "rc=-9")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"


# -----------------------------------------------------------------------------
# M3: Fan RPM spike WARN #6 requires co-occurrence with a thermal event
# in the same phase. A bare fan spike under stress is not diagnostic.
# -----------------------------------------------------------------------------


def test_fan_spike_alone_does_not_warn(tmp_path):
    """Fan spike with no thermal event in the same phase should not trigger WARN #6."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "fan_rpm_spike", "CPU fan 600 -> 2100 RPM")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert subs["thermal"]["status"] == "PASS"
    assert not any("fan rpm spike" in r.lower() for r in subs["thermal"]["reasons"])


def test_fan_spike_with_thermal_same_phase_warns(tmp_path):
    """Fan spike + thermal event in the same phase → thermal WARN."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "fan_rpm_spike", "CPU fan 600 -> 2100 RPM")
    db.add_event("steady", "warn", "thermal_warn", "CPU temp 86C")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["thermal"]["status"] == "WARN"
    assert any("coincided with thermal event" in r.lower()
               for r in subs["thermal"]["reasons"])


def test_fan_spike_different_phase_than_thermal_does_not_warn(tmp_path):
    """Fan spike in steady and thermal in ramp are not correlated → no WARN."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("steady", "warn", "fan_rpm_spike", "CPU fan 600 -> 2100 RPM")
    db.add_event("ramp", "warn", "thermal_warn", "CPU temp 86C")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    # Thermal still warns from the thermal_warn event, but the fan reason line
    # must not appear — the spike and thermal are in different phases.
    assert not any("fan rpm spike" in r.lower() for r in subs["thermal"]["reasons"])


# -----------------------------------------------------------------------------
# H2 (validation 2026-04-14): kernel_watcher_unavailable must be consumed.
# -----------------------------------------------------------------------------
# Emitted by prepare_run when journalctl can't start (missing binary, permissions,
# etc). Before this fix it was written to the DB but never scored — a run with
# zero kernel coverage would pass with no operator-visible warning about the gap.


def test_warn_on_kernel_watcher_unavailable(tmp_path):
    """kernel_watcher_unavailable → power WARN with an operator-visible reason
    explaining that kernel-driven rules could not score this run."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("run", "warn", "kernel_watcher_unavailable",
                 "journalctl not available on this system")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "WARN"
    assert subs["power"]["status"] == "WARN"
    assert any("kernel watcher did not start" in r.lower()
               for r in subs["power"]["reasons"])
    assert [item["rule_id"] for item in subs["power"]["evidence"]] == [
        "coverage.kernel_watcher_unavailable.warn",
    ]


def test_kernel_watcher_unavailable_preserves_fail(tmp_path):
    """A concurrent FAIL must not be downgraded by a watcher_unavailable WARN."""
    db = make_run_db(tmp_path)
    _add_soak_and_load_samples(db)
    db.add_event("run", "warn", "kernel_watcher_unavailable", "journalctl not available")
    db.add_event("run", "crit", "nvidia_gpu_disappeared", "GPU vanished from NVML")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert verdict == "FAIL"
    assert subs["power"]["status"] == "FAIL"
