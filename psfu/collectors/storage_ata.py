"""ATA (SATA/HDD) SMART attribute pre/post delta scoring.

Pre/post snapshot pair captured at run start and run end via
``smartctl -a --json -T permissive <dev>``. Delta is scored by
``storage_ata_events`` which returns ``(severity, kind, message)`` tuples
for the verdict engine — same shape as ``storage_ocp_events`` and
``storage_health_events``.

S9 (v0.19) scope — four growth triggers on the canonical ATA SMART
attributes that predict drive death or cable trouble:

  ata_reallocated_sector_grew     (FAIL) — Attribute 5  (Reallocated_Sector_Ct)
  ata_pending_sector_grew         (FAIL) — Attribute 197 (Current_Pending_Sector)
  ata_offline_uncorrectable_grew  (FAIL) — Attribute 198 (Offline_Uncorrectable)
  ata_udma_crc_grew               (WARN) — Attribute 199 (UDMA_CRC_Error_Count)

Plus one info-only event:

  ata_smart_not_supported         — smartctl couldn't read ATA attributes
                                    (device not present, non-ATA transport,
                                    smartctl missing, bridge that hides
                                    SMART).

Why UDMA CRC is WARN not FAIL: UDMA CRC errors indicate the cable / SATA
port / backplane connector is degrading the signal — NOT the drive. A
failing SATA cable will retire a perfectly good drive if this is read as
FAIL. WARN tells the operator to reseat/swap the cable before condemning
the drive.

Why raw only (``raw.value``), never ``value``/``worst``: vendor
normalization is a lie. Every vendor implements the 0–255 normalized
scale differently; some decrement linearly, some only at drive-defined
thresholds, some never change raw->normalized at all. The only
comparable-across-vendors number is ``raw.value``, which is the actual
incrementing counter. Every SMART-triage tool of record (smartmontools
docs, every RMA procedure I've seen) scores raw.

Why events instead of columns: mirrors S7 (OCP). No schema migration,
no CURRENT_SCHEMA_VERSION bump, raw pre/post preserved in event message
for post-run forensics. 20 ``ata_*_raw`` columns from the original S9
outline deferred until a rule actually consumes within-run samples
(none yet; all four rules are boundary deltas).

Vendor attribute ID conflicts: attributes 5 / 197 / 198 / 199 are in the
common set — every SATA SSD and HDD vendor of record maps them to the
same semantics (SAS is different, not in scope here). Rarer attributes
(1, 196, 200) have vendor drift and are deferred.

USB-bridge passthrough: ``smartctl`` auto-tries ``-d sat`` on
USB-SATA bridges for most well-known bridge chipsets. When it fails,
we fall through to ``ata_smart_not_supported`` — we do NOT iterate
over every ``-d`` value (too many false positives, too much noise).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from psfu.common import utc_now
from psfu.hardware import smartctl_binary_path


# Attribute ID → (field key in snapshot dict, event kind, severity).
# Severity "crit" routes to storage_fail_events; "warning" routes to
# storage_warn_events (see verdict/rules.py).
_SCORED_ATTRIBUTES: Dict[int, Tuple[str, str, str]] = {
    5:   ("reallocated_sector_ct", "ata_reallocated_sector_grew",     "crit"),
    197: ("current_pending_sector", "ata_pending_sector_grew",        "crit"),
    198: ("offline_uncorrectable",  "ata_offline_uncorrectable_grew", "crit"),
    199: ("udma_crc_error_count",   "ata_udma_crc_grew",              "warning"),
}


def _parse_smartctl_json(data: Optional[str]) -> Optional[Dict[str, object]]:
    """Parse ``smartctl -a --json`` stdout into a dict of scored fields.

    Returns ``None`` when smartctl didn't report any ATA attributes
    (common on NVMe-over-smartctl, unsupported USB bridges, failed
    reads). Never raises — callers expect a clean None.
    """
    if not data:
        return None
    try:
        doc = json.loads(data)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if not isinstance(doc, dict):
        return None
    attrs_block = doc.get("ata_smart_attributes")
    if not isinstance(attrs_block, dict):
        return None
    table = attrs_block.get("table")
    if not isinstance(table, list):
        return None

    fields: Dict[str, object] = {key: None for key, _, _ in _SCORED_ATTRIBUTES.values()}
    seen_any = False
    for row in table:
        if not isinstance(row, dict):
            continue
        attr_id = row.get("id")
        if not isinstance(attr_id, int):
            continue
        spec = _SCORED_ATTRIBUTES.get(attr_id)
        if spec is None:
            continue
        raw_block = row.get("raw")
        if not isinstance(raw_block, dict):
            continue
        raw_value = raw_block.get("value")
        if not isinstance(raw_value, int):
            continue
        field_key, _kind, _sev = spec
        fields[field_key] = raw_value
        seen_any = True

    if not seen_any:
        return None

    fields["model"] = _str_or_none(doc.get("model_name"))
    fields["serial"] = _str_or_none(doc.get("serial_number"))
    fields["firmware"] = _str_or_none(doc.get("firmware_version"))
    return fields


def _str_or_none(val: object) -> Optional[str]:
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def _read_smartctl(device: str) -> Optional[str]:
    """Run ``smartctl -a --json -T permissive <dev>`` and return stdout.

    ``-T permissive`` lets us read attributes from drives reporting
    marginal/non-standard SMART headers (common on older SSDs). Timeout
    after 15 s — a hung smartctl shouldn't stall run startup.
    """
    smartctl_bin = smartctl_binary_path()
    if not smartctl_bin:
        return None
    try:
        proc = subprocess.run(
            [smartctl_bin, "-a", "--json", "-T", "permissive", device],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    # smartctl uses bit-field return codes; bit 0 = command line parse,
    # bits 1-2 = open failures. Anything >= 4 is "SMART read had content
    # but saw errors" — the content is still usable. We only bail on
    # truly empty stdout.
    if not proc.stdout:
        return None
    return proc.stdout


def capture_storage_ata(devices: Optional[List[str]] = None) -> Dict[str, object]:
    """Snapshot ATA SMART attributes for every SATA-class device.

    ``devices`` defaults to every ``/dev/sd?`` discovered under
    ``/sys/block`` with a usable ``device`` node. Each entry has
    ``supported: bool`` + the four scored fields + identity strings.
    """
    if devices is None:
        devices = _discover_sata_devices()
    payload: Dict[str, object] = {
        "captured_at": utc_now(),
        "devices": [],
    }
    for dev in devices:
        entry: Dict[str, object] = {
            "device": dev,
            "supported": False,
            "reallocated_sector_ct": None,
            "current_pending_sector": None,
            "offline_uncorrectable": None,
            "udma_crc_error_count": None,
            "model": None,
            "serial": None,
            "firmware": None,
        }
        raw = _read_smartctl(dev)
        parsed = _parse_smartctl_json(raw) if raw is not None else None
        if parsed is not None:
            entry.update(parsed)
            entry["supported"] = True
        payload["devices"].append(entry)
    return payload


def _discover_sata_devices() -> List[str]:
    """Return ``/dev/sd?`` whole-device paths from /sys/block.

    Filters out partitions (``sda1``) by enforcing the ``/sys/block``
    top-level only. NVMe (``nvme*``), loop (``loop*``), ram (``ram*``),
    and dm (``dm-*``) devices are skipped.
    """
    sys_block = Path("/sys/block")
    if not sys_block.exists():
        return []
    out: List[str] = []
    try:
        for item in sorted(sys_block.iterdir()):
            name = item.name
            if not name.startswith("sd"):
                continue
            # Whole-device names are exactly 3 chars on modern kernels
            # (sda, sdb, ...); but be permissive for 4+ (sdaa, sdab, ...)
            # on big enclosures. Exclude partitions (end in digit).
            if name[-1].isdigit():
                continue
            out.append("/dev/{}".format(name))
    except OSError:
        return []
    return out


# ---------------------------------------------------------------------------
# Event emitter — mirrors storage_ocp_events contract.
# ---------------------------------------------------------------------------

# (field_key, event_kind, severity) tuples, stable-ordered so per-device
# events emit deterministically across runs.
_GROWTH_FIELDS: Tuple[Tuple[str, str, str], ...] = tuple(
    (field_key, kind, sev) for (field_key, kind, sev) in _SCORED_ATTRIBUTES.values()
)


def storage_ata_events(
    start_payload: Dict[str, object],
    end_payload: Dict[str, object],
) -> List[Tuple[str, str, str]]:
    """Compute delta between pre/post ATA snapshots and return verdict
    events. Any non-zero growth = one event per growth field per device.
    A drive unsupported at start OR end = one info-only
    ``ata_smart_not_supported`` event.
    """
    events: List[Tuple[str, str, str]] = []
    start_devices = {
        item.get("device"): item
        for item in _devices(start_payload)
        if isinstance(item.get("device"), str)
    }
    end_devices = {
        item.get("device"): item
        for item in _devices(end_payload)
        if isinstance(item.get("device"), str)
    }
    for device, end in end_devices.items():
        start = start_devices.get(device, {})
        if not isinstance(start, dict):
            start = {}

        start_ok = bool(start.get("supported"))
        end_ok = bool(end.get("supported"))
        if not (start_ok and end_ok):
            events.append((
                "info",
                "ata_smart_not_supported",
                "{} does not expose ATA SMART attributes — S9 deltas unavailable.".format(device),
            ))
            continue

        for field, kind, severity in _GROWTH_FIELDS:
            start_val = start.get(field)
            end_val = end.get(field)
            if not isinstance(start_val, (int, float)):
                continue
            if not isinstance(end_val, (int, float)):
                continue
            if end_val <= start_val:
                continue
            events.append((
                severity,
                kind,
                "{} ATA {} grew by {} ({} -> {}) during run.".format(
                    device,
                    field,
                    int(end_val) - int(start_val),
                    int(start_val),
                    int(end_val),
                ),
            ))
    return events


def _devices(payload: Dict[str, object]) -> List[Dict[str, object]]:
    raw = payload.get("devices") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]
