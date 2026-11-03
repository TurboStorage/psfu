import json
import os
import re
import signal
import subprocess
import threading
from typing import Dict, List, Optional, Tuple
from psfu.common import command_exists, utc_now
from psfu.persistence.db import Database

# Kernel/journal patterns we care about during stress phases.
#
# Schema:
#   (pattern, severity, kind, subsystem, default_verdict)
#
#   subsystem       — one of "power" | "thermal" | "memory" | "storage" | "pcie"
#   default_verdict — "FAIL" | "WARN" | "SPECIAL"
#
# "SPECIAL" means the verdict engine applies non-trivial logic (threshold
# counts, cross-correlation with other events, etc.) and the generic
# subsystem/verdict columns are hints only. Generic kinds route straight
# through a single scoring loop in verdict/rules.py — prior to this table
# extension, each simple kind needed its own hand-written block, which is
# how silent dead rules like ``thermal_drift`` escaped review (C11).
KERNEL_WATCH_PATTERNS: List[Tuple[str, str, str, str, str]] = [
    (r"NVRM:\s*Xid",                              "crit", "nvidia_xid",      "power",   "FAIL"),
    (r"\bKernel panic\b",                         "crit", "kernel_panic",    "power",   "FAIL"),
    (r"\bOops:\b",                                "crit", "kernel_oops",     "power",   "FAIL"),
    (r"\bmce:\s",                                 "crit", "mce",             "power",   "FAIL"),
    (r"Hardware Error",                           "crit", "hardware_error",  "power",   "FAIL"),
    (r"above critical temp",                      "crit", "thermal_crit",    "thermal", "FAIL"),
    (r"CPU\d+: Core temperature above threshold", "warn", "thermal_warn",    "thermal", "WARN"),
    (r"thermal.*?throttl",                        "warn", "thermal_throttle","thermal", "WARN"),
    (r"Out of memory",                            "warn", "oom",             "memory",  "WARN"),
    (r"oom[-_]killer",                            "warn", "oom_killer",      "memory",  "WARN"),
    (r"PCIe Bus Error",                           "crit", "pcie_bus_error",  "pcie",    "FAIL"),
    (r"AER.*error",                               "warn", "pcie_aer",        "pcie",    "SPECIAL"),
    (r"EDAC.*error",                              "warn", "edac",            "memory",  "FAIL"),
    (r"ACPI BIOS Error",                          "warn", "acpi_bios",       "power",   "WARN"),
    (r"under-voltage",                            "crit", "undervoltage",    "power",   "FAIL"),
    (r"brownout",                                 "crit", "brownout",        "power",   "FAIL"),
    # S6 — NVMe controller watchdog. Canonical dmesg strings:
    #   nvme nvme0: I/O 123 QID 4 timeout, aborting
    #   nvme nvme0: resetting controller
    #   nvme nvme0: async event result 00000300
    # All three route through the generic FAIL loop; any occurrence during a
    # storage-stress phase is a hard FAIL. Sub-string "nvme" gate (vs. the
    # broader "controller" / "timeout") keeps this from matching SATA / SAS /
    # SCSI noise — NVMe is the only subsystem S6 covers.
    (r"\bnvme\b.*?I/O.*?timeout",                 "crit", "nvme_io_timeout",       "storage", "FAIL"),
    (r"\bnvme\b.*?reset(ting)?\s+controller",     "crit", "nvme_controller_reset", "storage", "FAIL"),
    (r"\bnvme\b.*?async\s+event.*?error",         "crit", "nvme_aer_error",        "storage", "FAIL"),
]

# Compiled tuples used by KernelWatcher (only needs pattern+severity+kind).
KERNEL_WATCH_COMPILED: List[Tuple["re.Pattern[str]", str, str]] = [
    (re.compile(pat, re.IGNORECASE), severity, kind)
    for pat, severity, kind, _subsystem, _default in KERNEL_WATCH_PATTERNS
]


def kernel_kinds_by_default_verdict() -> Dict[str, List[Tuple[str, str]]]:
    """Return ``{"FAIL": [(kind, subsystem), ...], "WARN": [...]}`` for use by
    the verdict engine's generic scoring loop. ``SPECIAL`` kinds are excluded
    — the engine handles them in bespoke blocks."""
    out: Dict[str, List[Tuple[str, str]]] = {"FAIL": [], "WARN": []}
    for _pat, _sev, kind, subsystem, default in KERNEL_WATCH_PATTERNS:
        if default in out:
            out[default].append((kind, subsystem))
    return out


class KernelWatcher:
    def __init__(self, db: Database, current_phase_fn) -> None:
        self.db = db
        self._current_phase = current_phase_fn
        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.critical_events: List[Dict[str, str]] = []
        self.warn_events: List[Dict[str, str]] = []
        self.available = False
        self.command: List[str] = []

    def _pick_command(self) -> Optional[List[str]]:
        # JSON output + _TRANSPORT filter lets us reject /dev/kmsg injections
        # from unprivileged userspace. Kernel-origin messages have no _UID
        # field; kmsg writes from userspace carry the writer's _UID/_PID.
        # Without this filter, any user can `echo "mce: ..." >/dev/kmsg` to
        # force a FAIL verdict on a healthy PSU. See _is_trusted_kernel_entry.
        if command_exists("journalctl"):
            return [
                "journalctl",
                "-k",
                "-f",
                "--since",
                "now",
                "--no-pager",
                "-o",
                "json",
            ]
        # dmesg path does not distinguish injection — documented limitation.
        # Prefer journalctl on any modern distro.
        if command_exists("dmesg"):
            return ["dmesg", "-w", "-T"]
        return None

    @staticmethod
    def _is_trusted_kernel_entry(entry: Dict[str, object]) -> bool:
        """True only if the entry was actually emitted by the kernel, not
        forged via /dev/kmsg from an unprivileged process.

        Real kernel messages have ``_TRANSPORT=kernel`` and lack ``_UID`` /
        ``_PID`` fields (they originate from ring buffer, not a process).
        Userspace writes to /dev/kmsg also get _TRANSPORT=kernel but carry
        the writer's ``_UID``. Reject anything with a non-zero ``_UID``.
        """
        if entry.get("_TRANSPORT") != "kernel":
            return False
        uid = entry.get("_UID")
        if uid is not None and str(uid) != "0":
            return False
        return True

    def start(self) -> Optional[str]:
        cmd = self._pick_command()
        if not cmd:
            return "neither journalctl nor dmesg available; kernel watcher disabled"
        self.command = cmd
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,  # line-buffered
                preexec_fn=os.setsid,
            )
        except Exception as exc:
            return "failed to spawn kernel watcher: {}".format(exc)
        self.available = True
        self.thread = threading.Thread(target=self._reader_loop, name="kernel-watcher", daemon=True)
        self.thread.start()
        return None

    def _reader_loop(self) -> None:
        if not self.proc or not self.proc.stdout:
            self.available = False
            return
        is_json = "json" in self.command
        died_cleanly = False
        try:
            for raw in self.proc.stdout:
                if self._stop.is_set():
                    break
                raw = raw.rstrip()
                if not raw:
                    continue
                if is_json:
                    # journalctl -o json path: parse, verify kernel origin,
                    # then match against patterns using the MESSAGE field.
                    try:
                        entry = json.loads(raw)
                    except ValueError:
                        continue
                    if not self._is_trusted_kernel_entry(entry):
                        continue
                    message = entry.get("MESSAGE")
                    if not isinstance(message, str):
                        continue
                    line = message
                else:
                    # dmesg fallback: no origin check possible, just the line.
                    line = raw
                for compiled, severity, kind in KERNEL_WATCH_COMPILED:
                    if compiled.search(line):
                        phase = "run"
                        try:
                            phase = self._current_phase() or "run"
                        except Exception:
                            pass
                        trimmed = line[:500]
                        try:
                            self.db.add_event(phase, severity, kind, trimmed)
                        except Exception:
                            pass
                        event = {"phase": phase, "severity": severity, "kind": kind, "message": trimmed}
                        with self._lock:
                            if severity == "crit":
                                self.critical_events.append(event)
                            else:
                                self.warn_events.append(event)
                        break
            # Normal iterator exit: journalctl/dmesg closed stdout (process ended).
            died_cleanly = True
        except Exception:
            # stdout iterator raised — unicode decode error, pipe EIO, etc.
            # Swallow and fall through to the finally block; we still want to
            # record that the watcher is no longer collecting.
            pass
        finally:
            # Flip availability off and emit a diagnostic event so the verdict
            # engine (or a future health check) knows kernel coverage ended
            # mid-run. Without this, self.available stayed True forever after
            # an iterator exception and lost events were invisible.
            if not self._stop.is_set():
                self.available = False
                phase = "run"
                try:
                    phase = self._current_phase() or "run"
                except Exception:
                    pass
                reason = "stdout closed" if died_cleanly else "reader thread exception"
                try:
                    self.db.add_event(
                        phase, "warn", "kernel_watcher_died",
                        "kernel watcher thread exited mid-run ({}) — kernel events "
                        "may have been missed from this point on.".format(reason),
                    )
                except Exception:
                    pass

    def stop(self) -> None:
        self._stop.set()
        if self.proc:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    self.proc.terminate()
                except Exception:
                    pass
            try:
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def summary(self) -> Dict[str, object]:
        with self._lock:
            return {
                "available": self.available,
                "command": " ".join(self.command) if self.command else "",
                "critical_count": len(self.critical_events),
                "warn_count": len(self.warn_events),
                "critical_events": list(self.critical_events),
                "warn_events": list(self.warn_events),
            }
