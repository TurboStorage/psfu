"""OCP NVMe Cloud SSD SMART / Health Information Extended log page (0xC0).

Pre/post snapshot pair captured at run start and run end. Delta is scored by
``storage_ocp_events`` which returns ``(severity, kind, message)`` tuples for
the verdict engine — same shape as ``storage_health.storage_health_events``.

S7 (v0.18) scope — four FAIL triggers on any non-zero growth during a
read-only run:

  ssd_ocp_bad_user_nand_growth       — bad user NAND blocks grew
  ssd_ocp_bad_system_nand_growth     — bad system NAND blocks grew
  ssd_ocp_xor_recovery_growth        — reads required XOR parity recovery
  ssd_ocp_uncorrectable_read_growth  — unrecovered read errors

Plus one info-only event:

  ssd_ocp_not_supported              — drive does not expose OCP 0xC0

Why events instead of columns: mirrors S6 fio_stall — no schema migration,
no CURRENT_SCHEMA_VERSION bump, raw pre/post values captured in the event
message for post-run forensics. Verdict rules pattern-match on kind like
every other storage_fail_events entry. Raw columns can land in S12 when
the endurance-ratio rule actually needs them across runs.

Byte offsets follow OCP Datacenter NVMe SSD Specification 2.0, §5.12
"SMART / Health Information Extended (Log Identifier 0xC0)". Log Page
Version at bytes 494-495 must be 2 or 3 (OCP 1.0 and 2.0); anything
else → treat as unsupported and emit ``ssd_ocp_not_supported``.
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from psfu.common import utc_now
from psfu.hardware import nvme_cli_binary_path


# OCP 0xC0 log page is 512 bytes. Anything shorter = unsupported drive
# returning stub data; anything longer = we only look at the first 512.
_OCP_LOG_PAGE_BYTES = 512

# Supported Log Page Versions (bytes 494-495, little-endian u16).
# OCP 1.0 → version 2, OCP 2.0 → version 3, OCP 2.5 → version 4.
# We parse versions 2 / 3 / 4 identically (the five fields we use are at
# the same offsets across all three). Version 0 or 1 = pre-spec vendor
# garbage; version > 4 = future, we don't know the layout yet.
_OCP_LOG_PAGE_VERSIONS_SUPPORTED = (2, 3, 4)


def _parse_ocp_log_page(data: bytes) -> Optional[Dict[str, int]]:
    """Parse the OCP 0xC0 binary log page into a dict of the 5 fields we
    score, plus ``physical_media_units_written`` (info-only for S12 later).

    Returns ``None`` if the page is too short, has an unsupported version,
    or fails a sanity check. Never raises — callers expect a clean None.
    """
    if not data or len(data) < _OCP_LOG_PAGE_BYTES:
        return None
    try:
        # Log Page Version at bytes 494-495 (LE u16).
        version = struct.unpack_from("<H", data, 494)[0]
        if version not in _OCP_LOG_PAGE_VERSIONS_SUPPORTED:
            return None

        # Physical Media Units Written — bytes 0-15, 128-bit LE. We keep
        # only the low 64 bits (the high 64 remain 0 for the foreseeable
        # future; endurance classes are in petabytes, not exabytes).
        pmu_written_low = struct.unpack_from("<Q", data, 0)[0]

        # 6-byte LE counters (spec uses 6 raw + 2 normalized; we only
        # score on raw deltas).
        bad_user_nand = int.from_bytes(data[32:38], "little")
        bad_system_nand = int.from_bytes(data[40:46], "little")

        # 8-byte LE counters.
        xor_recovery = struct.unpack_from("<Q", data, 48)[0]
        uncorrectable_read = struct.unpack_from("<Q", data, 56)[0]

        return {
            "physical_media_units_written": pmu_written_low,
            "bad_user_nand_blocks": bad_user_nand,
            "bad_system_nand_blocks": bad_system_nand,
            "xor_recovery_count": xor_recovery,
            "uncorrectable_read_errors": uncorrectable_read,
            "log_page_version": version,
        }
    except (struct.error, IndexError, ValueError):
        return None


def _read_ocp_log_raw(device: str) -> Optional[bytes]:
    """Run ``nvme get-log -n 1 -i 0xC0 -l 512 -b <dev>`` and return the
    raw bytes. ``-b`` emits binary; nvme-cli's text/json output for OCP
    is plugin-dependent and flakier than binary parse. Uses subprocess
    directly because ``common.run_capture`` is text-mode only."""
    nvme_bin = nvme_cli_binary_path()
    if not nvme_bin:
        return None
    try:
        proc = subprocess.run(
            [nvme_bin, "get-log", "-n", "1", "-i", "0xC0", "-l", "512", "-b", device],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    return bytes(proc.stdout)


def capture_storage_ocp(devices: Optional[List[str]] = None) -> Dict[str, object]:
    """Snapshot OCP 0xC0 for every NVMe device.

    ``devices`` defaults to every ``/dev/nvme*n1`` discovered under
    ``/sys/block``. Each device entry has ``supported: bool`` + the 5
    counter fields (or Nones when unsupported).
    """
    if devices is None:
        devices = _discover_nvme_namespaces()
    payload: Dict[str, object] = {
        "captured_at": utc_now(),
        "devices": [],
    }
    for dev in devices:
        entry: Dict[str, object] = {
            "device": dev,
            "supported": False,
            "physical_media_units_written": None,
            "bad_user_nand_blocks": None,
            "bad_system_nand_blocks": None,
            "xor_recovery_count": None,
            "uncorrectable_read_errors": None,
            "log_page_version": None,
        }
        raw = _read_ocp_log_raw(dev)
        parsed = _parse_ocp_log_page(raw) if raw is not None else None
        if parsed is not None:
            entry.update(parsed)
            entry["supported"] = True
        payload["devices"].append(entry)
    return payload


def _discover_nvme_namespaces() -> List[str]:
    """Return ``/dev/nvme*n1`` paths discovered under /sys/block."""
    sys_block = Path("/sys/block")
    if not sys_block.exists():
        return []
    out: List[str] = []
    try:
        for item in sorted(sys_block.iterdir()):
            name = item.name
            if name.startswith("nvme") and "n" in name:
                out.append("/dev/{}".format(name))
    except OSError:
        return []
    return out


# ---------------------------------------------------------------------------
# Event emitter — called at run finalization. Mirrors
# ``storage_health_events`` in shape and contract.
# ---------------------------------------------------------------------------

_GROWTH_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("bad_user_nand_blocks",     "ssd_ocp_bad_user_nand_growth"),
    ("bad_system_nand_blocks",   "ssd_ocp_bad_system_nand_growth"),
    ("xor_recovery_count",       "ssd_ocp_xor_recovery_growth"),
    ("uncorrectable_read_errors", "ssd_ocp_uncorrectable_read_growth"),
)


def storage_ocp_events(
    start_payload: Dict[str, object],
    end_payload: Dict[str, object],
) -> List[Tuple[str, str, str]]:
    """Compute delta between pre/post OCP snapshots and return verdict
    events. Any non-zero growth on the four scored counters = one FAIL
    event per growth field per device. A drive that was unsupported at
    start OR end gets a single info-only ``ssd_ocp_not_supported`` event
    (routed via _INFO_ONLY_KINDS in verdict/rules.py).
    """
    events: List[Tuple[str, str, str]] = []
    start_devices = {
        item.get("device"): item
        for item in start_payload.get("devices", [])
        if isinstance(item, dict)
    }
    end_devices = {
        item.get("device"): item
        for item in end_payload.get("devices", [])
        if isinstance(item, dict)
    }
    for device, end in end_devices.items():
        if not isinstance(device, str):
            continue
        start = start_devices.get(device, {})
        if not isinstance(start, dict):
            start = {}

        start_ok = bool(start.get("supported"))
        end_ok = bool(end.get("supported"))
        if not (start_ok and end_ok):
            events.append((
                "info",
                "ssd_ocp_not_supported",
                "{} does not expose OCP 0xC0 log page — S7 deltas unavailable.".format(device),
            ))
            continue

        for field, kind in _GROWTH_FIELDS:
            start_val = start.get(field)
            end_val = end.get(field)
            if not isinstance(start_val, (int, float)):
                continue
            if not isinstance(end_val, (int, float)):
                continue
            if end_val <= start_val:
                continue
            events.append((
                "crit",
                kind,
                "{} OCP {} grew by {} ({} -> {}) during run.".format(
                    device,
                    field,
                    int(end_val) - int(start_val),
                    int(start_val),
                    int(end_val),
                ),
            ))
    return events
