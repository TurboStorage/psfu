"""Operator-facing prompts and bench observation capture.

All the stdin-driven metadata collection for a run: PSU serial / model / lot,
operator name, bench info, paperclip / POST / known-good-swap / multimeter
readings, and the narrow-scope defensible-conclusion derivation that turns a
verdict into a sentence an operator can show a manufacturer.

Flow:
  1. prepare_run calls pre_run_observations()  — meter availability, paperclip, POST
  2. execute_phases calls idle_multimeter_readings() after soak phase (if meter)
  3. execute_phases calls load_multimeter_readings() after first stress phase (if meter)
  4. finalize_run calls post_run_observations() — known-good swap, original retest
  5. finalize_run calls final_multimeter_readings() as fallback for TUI runs or
     any readings that didn't get captured mid-run (e.g. operator aborted early)
"""

from __future__ import annotations

import argparse
import sqlite3
from typing import Dict, List, Optional
from psfu.storage.devices import StorageDUT, parse_storage_dut_paths, validate_storage_dut_paths


def prompt_if_missing(
    value: Optional[str],
    label: str,
    default: str = "",
    *,
    allow_blank: bool = False,
) -> str:
    if value:
        return value
    prompt = label
    if default:
        prompt += " [{}]".format(default)
    prompt += ": "
    entered = input(prompt).strip()
    if allow_blank and not entered:
        return ""
    return entered or default


def prompt_int_if_missing(value: Optional[str], label: str, default: int = 0) -> int:
    if value is not None:
        try:
            return int(str(value).strip() or str(default))
        except ValueError:
            pass

    while True:
        entered = input("{} [{}]: ".format(label, default)).strip()
        chosen = entered or str(default)
        try:
            return int(chosen)
        except ValueError:
            print("Invalid integer: '{}'. Enter a whole number.".format(chosen))


def prompt_choice(
    value: Optional[str],
    label: str,
    allowed: List[str],
    default: str,
) -> str:
    """Constrained-choice prompt with operator-friendly aliases.

    Aliases exist so operators can type `y`/`n`/`nr`/`na` at the bench instead
    of the full canonical values. If an alias resolves to a canonical value,
    returns the canonical form.
    """
    _ALIASES: Dict[str, str] = {
        "na": "not-applicable",
        "n/a": "not-applicable",
        "nr": "not-run",
        "sf": "same-failure",
        "p": "pass",
        "f": "fail",
        "y": "yes",
        "n": "no",
        "nt": "not-tested",
        "ff": "fault-follows-psu",
        "fr": "fault-remains",
        "nd": "no-difference",
        "nc": "not-clear",
    }

    def _resolve(raw: str) -> Optional[str]:
        if raw in allowed:
            return raw
        resolved = _ALIASES.get(raw)
        if resolved and resolved in allowed:
            return resolved
        return None

    if value is not None:
        resolved = _resolve(value.strip().lower())
        if resolved:
            return resolved

    allowed_display = "/".join(allowed)
    while True:
        entered = input("{} [{}] ({}): ".format(label, default, allowed_display)).strip().lower()
        chosen = entered or default
        resolved = _resolve(chosen)
        if resolved:
            return resolved
        print("Invalid choice: '{}'. Valid: {}".format(chosen, allowed_display))


def _prompt_float(label: str, current: Optional[float] = None) -> Optional[float]:
    """Single float prompt. Returns current if already set, None on blank entry."""
    if current is not None:
        return current
    raw = input("{} (blank to skip): ".format(label)).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print("Invalid number. Leaving blank.")
        return None


# ---------------------------------------------------------------------------
# Pre-run observations — called in prepare_run before phases start.
# Collects things that are physically done before the software test:
#   - whether a multimeter is available (so the operator knows to have it ready)
#   - paperclip test result (done before wiring the PSU into the bench)
#   - POST result (does the DUT POST with the returned PSU installed)
# ---------------------------------------------------------------------------

def pre_run_observations(args: argparse.Namespace) -> Dict[str, object]:
    """Collect bench observations that happen before the software run starts."""
    meter_available = args.meter_available
    if meter_available is None:
        meter_available = "no"

    paperclip_result = args.paperclip_result or "not-run"
    post_result = args.post_result or "not-tested"
    return {
        "meter_available": 1 if meter_available == "yes" else 0,
        "paperclip_result": paperclip_result,
        "post_result": post_result,
    }


# ---------------------------------------------------------------------------
# Mid-run multimeter prompts — called from execute_phases between phases.
# Fires in both plain and TUI modes; execute.py suspends curses around the
# prompt (def_prog_mode / endwin / reset_prog_mode) so the operator can type
# readings at the real terminal and the dashboard resumes afterwards.
# ---------------------------------------------------------------------------

def idle_multimeter_readings(args: argparse.Namespace) -> Dict[str, Optional[float]]:
    """Prompted right after soak phase: PSU at idle, take 12V+5V readings."""
    print("\n[Meter] Soak complete. Probe Molex connector now — PSU at idle.")
    return {
        "manual_idle_12v": _prompt_float("  12V idle", args.manual_idle_12v),
        "manual_idle_5v": _prompt_float("  5V idle", args.manual_idle_5v),
    }


def load_multimeter_readings(args: argparse.Namespace) -> Dict[str, Optional[float]]:
    """Prompted right after the first stress phase: PSU hot, take 12V+5V load readings."""
    print("\n[Meter] Load phase complete. Probe Molex connector now — PSU still hot.")
    return {
        "manual_load_12v": _prompt_float("  12V load", args.manual_load_12v),
        "manual_load_5v": _prompt_float("  5V load", args.manual_load_5v),
    }


def final_multimeter_readings(
    args: argparse.Namespace,
    run_row: sqlite3.Row,
) -> Dict[str, Optional[float]]:
    """Fallback called in finalize_run for any readings not yet captured mid-run.
    Only prompts for fields that are still None — skips anything already stored
    (e.g. from mid-run prompts in a plain-mode run)."""
    result: Dict[str, Optional[float]] = {}
    pairs = [
        ("manual_idle_12v", "Manual 12V idle", args.manual_idle_12v),
        ("manual_idle_5v", "Manual 5V idle", args.manual_idle_5v),
        ("manual_load_12v", "Manual 12V load", args.manual_load_12v),
        ("manual_load_5v", "Manual 5V load", args.manual_load_5v),
    ]
    for field, label, cli_val in pairs:
        if run_row[field] is None:
            result[field] = _prompt_float(label, cli_val)
    return result


# ---------------------------------------------------------------------------
# Post-run observations — called in finalize_run after phases complete.
# Collects things that happen after the software test or as a separate step:
#   - known-good PSU swap (was one performed, and what happened)
#   - original system retest result
# ---------------------------------------------------------------------------

def post_run_observations(args: argparse.Namespace) -> Dict[str, object]:
    """Collect bench observations that happen after the software run completes."""
    swap_performed = args.known_good_swap_performed
    if swap_performed is None:
        swap_performed = "no"

    swap_result = args.known_good_swap_result
    if swap_performed == "yes":
        swap_result = prompt_choice(
            swap_result,
            "Known-good PSU swap result",
            ["fault-follows-psu", "fault-remains", "no-difference", "not-clear"],
            "fault-follows-psu",
        )
    else:
        swap_result = "not-done"

    original_retest = args.original_system_retest_result or "not-run"

    return {
        "known_good_swap_performed": 1 if swap_performed == "yes" else 0,
        "known_good_swap_result": swap_result,
        "original_system_retest_result": original_retest,
    }


def prompt_storage_dut_paths(
    value: Optional[List[str]],
    inventory: List[StorageDUT],
    *,
    required: bool,
) -> List[str]:
    if value:
        return list(value)

    print("\n[Storage DUTs]")
    if inventory:
        for idx, device in enumerate(inventory, start=1):
            mounts = ",".join(device.mountpoints) if device.mountpoints else "-"
            print(
                "  {}. {}  {} {} [{}] mounts={}".format(
                    idx,
                    device.path,
                    device.vendor or "?",
                    device.model or "?",
                    device.role,
                    mounts,
                )
            )
    else:
        print("  No storage devices discovered.")

    while True:
        raw = input("Select DUT path(s), comma-separated{}: ".format(" (required)" if required else " (blank to skip)")).strip()
        chosen = parse_storage_dut_paths(raw)
        chosen = validate_storage_dut_paths(chosen, inventory)
        if chosen:
            return chosen
        if not required:
            return []
        if inventory and raw:
            print("Selection did not match any discovered storage DUT.")
        else:
            print("At least one storage DUT is required for this profile.")


def defensible_conclusion(run_row: sqlite3.Row) -> str:
    """Turn the verdict + swap / POST / meter state into a claim an operator
    can defend to a manufacturer. Deliberately narrows the scope when evidence
    is weak — never overclaims."""
    verdict = run_row["verdict"] or "UNKNOWN"
    meter_available = bool(run_row["meter_available"])
    swap_result = run_row["known_good_swap_result"] or "not-done"

    if verdict == "FAIL" and swap_result == "fault-follows-psu" and meter_available:
        return (
            "Returned PSU failed bench isolation and the observed fault cleared with a known-good PSU. "
            "Manual rail readings were recorded for this run. This is suitable for an RMA claim that the returned PSU is defective."
        )
    if verdict == "FAIL" and swap_result == "fault-follows-psu":
        return (
            "Returned PSU failed bench isolation and the observed fault cleared with a known-good PSU. "
            "No multimeter readings were recorded, so the defensible claim is system-level isolation failure, not electrical spec failure."
        )
    if verdict == "FAIL" and run_row["post_result"] == "fail":
        return (
            "Returned PSU failed to POST the known-good bench. If a known-good PSU restores normal operation, this supports an RMA claim for PSU failure."
        )
    if verdict == "INCONCLUSIVE":
        return "This run does not isolate the returned PSU as the fault source. Do not submit an RMA based on this evidence alone."
    if verdict == "WARN":
        return "This run captured warning-level behavior. Retest with a known-good PSU substitution before making an RMA claim."
    if verdict == "PASS" and meter_available:
        return (
            "Returned PSU completed this bench run without captured faults and manual rail readings were recorded. "
            "This run alone does not support an RMA claim."
        )
    if verdict == "PASS":
        return (
            "Returned PSU completed software triage without captured faults. No multimeter data was recorded, so this is not evidence of electrical compliance."
        )
    return "No defensible conclusion is available for this run."
