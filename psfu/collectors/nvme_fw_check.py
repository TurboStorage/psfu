"""NVMe firmware version blocklist checker.

S15 (v0.21) — reads baselines/nvme_fw_blocklist.json and checks each NVMe
device's firmware revision against it. Any match emits a
``nvme_oob_fw_version`` WARN event.

WARN (not FAIL) — a blocklisted FW rev is an operator advisory. The drive may
have known bugs at that FW level but the drive itself is not necessarily failed.
The operator should update firmware before passing a unit or note the rev in
the RMA bundle.

Blocklist entry format:
  {
    "vendor_id":      "optional hex VendorID e.g. '0x144d' (Samsung)",
    "model_pattern":  "optional substring matched case-insensitively against id-ctrl 'mn'",
    "fw_rev":         "required — exact match against id-ctrl 'fr' (stripped)",
    "severity":       "warn",
    "reason":         "human-readable explanation",
    "reference":      "URL or ticket"
  }

All provided fields must match. Missing vendor_id / model_pattern = match any.
fw_rev is always required. One event emitted per device maximum (first match
wins; enumerate entries in priority order).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from psfu.common import run_capture
from psfu.config import BASELINES_DIR
from psfu.hardware import nvme_cli_binary_path


_BLOCKLIST_PATH = BASELINES_DIR / "nvme_fw_blocklist.json"


def load_fw_blocklist(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load the FW blocklist from disk. Returns [] if missing or invalid."""
    target = path if path is not None else _BLOCKLIST_PATH
    if not target.exists():
        return []
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = data.get("entries", []) if isinstance(data, dict) else []
    return [e for e in entries if isinstance(e, dict) and "fw_rev" in e]


def _read_id_ctrl(nvme_bin: str, device: str) -> Optional[Dict[str, Any]]:
    """Run 'nvme id-ctrl <dev> -o json' and return parsed dict or None."""
    code, stdout, _ = run_capture([nvme_bin, "id-ctrl", device, "-o", "json"], timeout=10)
    if code != 0 or not stdout:
        return None
    try:
        data = json.loads(stdout)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _entry_matches(ctrl: Dict[str, Any], entry: Dict[str, Any]) -> bool:
    """Return True when ctrl data matches all constraints in entry."""
    fw_rev = str(ctrl.get("fr", "")).strip()
    if fw_rev != str(entry.get("fw_rev", "")).strip():
        return False

    model_pattern = entry.get("model_pattern", "")
    if model_pattern:
        model = str(ctrl.get("mn", "")).strip()
        if model_pattern.lower() not in model.lower():
            return False

    vendor_id = entry.get("vendor_id", "")
    if vendor_id:
        vid = ctrl.get("vid", "")
        vid_str = hex(vid) if isinstance(vid, int) else str(vid)
        if vid_str.lower() != str(vendor_id).lower():
            return False

    return True


def check_nvme_fw_revisions(
    devices: Optional[List[str]] = None,
    *,
    blocklist_path: Optional[Path] = None,
) -> List[Tuple[str, str, str]]:
    """Check each device's FW revision against the blocklist.

    Returns list of (severity, kind, message) triples, one per matched
    device. Returns [] when blocklist is empty, devices is empty/None,
    or nvme-cli is not available.
    """
    if not devices:
        return []
    blocklist = load_fw_blocklist(blocklist_path)
    if not blocklist:
        return []
    nvme_bin = nvme_cli_binary_path()
    if not nvme_bin:
        return []

    events: List[Tuple[str, str, str]] = []
    for device in devices:
        ctrl = _read_id_ctrl(nvme_bin, device)
        if ctrl is None:
            continue
        fw_rev = str(ctrl.get("fr", "")).strip()
        model = str(ctrl.get("mn", "")).strip()
        for entry in blocklist:
            if _entry_matches(ctrl, entry):
                reason = entry.get("reason", "")
                reference = entry.get("reference", "")
                ref_suffix = " (ref: {})".format(reference) if reference else ""
                message = "NVMe {} ({}) firmware {} is blocklisted. {}{}".format(
                    device, model, fw_rev, reason, ref_suffix,
                ).strip()
                events.append(("warn", "nvme_oob_fw_version", message))
                break  # first match per device
    return events
