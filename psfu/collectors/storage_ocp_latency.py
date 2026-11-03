"""OCP NVMe Latency Monitor log page (0xC3) — pre/post snapshot pair.

Captured at run start and end exactly like OCP 0xC0. Delta scoring emits
info-only events; there are no FAIL/WARN verdict rules for 0xC3 — the log
is diagnostic evidence, not a hard fault signal.

Event kinds:
  ssd_ocp_latency_not_supported  — drive does not expose OCP 0xC3 (info)
  ssd_ocp_latency_captured       — log captured successfully (info)

Artifacts saved to:
  <run_dir>/storage/nvme/ocp-latency-monitor-start.json
  <run_dir>/storage/nvme/ocp-latency-monitor-end.json

Command used:
  nvme ocp latency-monitor-log <dev> -o json

Supported by drives with nvme-cli OCP plugin (built into nvme-cli ≥ 2.4 or
as a plugin). Falls back gracefully when the command is not supported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from psfu.common import run_capture, utc_now
from psfu.hardware import nvme_cli_binary_path


def _read_ocp_latency_log(device: str) -> Optional[Dict[str, object]]:
    """Run ``nvme ocp latency-monitor-log <dev> -o json`` and return the
    parsed JSON. Returns None if the command fails or output is not JSON."""
    nvme_bin = nvme_cli_binary_path()
    if not nvme_bin:
        return None
    code, stdout, _ = run_capture(
        [nvme_bin, "ocp", "latency-monitor-log", device, "-o", "json"],
        timeout=15,
    )
    if code != 0 or not stdout:
        return None
    try:
        data = json.loads(stdout)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def capture_storage_ocp_latency(devices: Optional[List[str]] = None) -> Dict[str, object]:
    """Snapshot OCP 0xC3 latency monitor log for every NVMe device.

    ``devices`` defaults to every ``/dev/nvme*n1`` under /sys/block.
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
            "data": None,
        }
        data = _read_ocp_latency_log(dev)
        if data is not None:
            entry["supported"] = True
            entry["data"] = data
        payload["devices"].append(entry)
    return payload


def write_ocp_latency_artifacts(run_dir: Path, devices: Optional[List[str]], stage: str) -> Dict[str, object]:
    """Capture and write OCP 0xC3 snapshot to <run_dir>/storage/nvme/ocp-latency-monitor-<stage>.json.
    Returns the snapshot payload."""
    payload = capture_storage_ocp_latency(devices or None)
    out_dir = run_dir / "storage" / "nvme"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ocp-latency-monitor-{}.json".format(stage)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def storage_ocp_latency_events(
    start_payload: Dict[str, object],
    end_payload: Dict[str, object],
) -> List[Tuple[str, str, str]]:
    """Emit info-only events per device.

    Unsupported drives get ``ssd_ocp_latency_not_supported``; supported drives
    get ``ssd_ocp_latency_captured`` (info, no verdict impact). All events are
    info-only — 0xC3 is diagnostic evidence, not a fault signal.
    """
    events: List[Tuple[str, str, str]] = []
    end_devices = {
        item.get("device"): item
        for item in end_payload.get("devices", [])
        if isinstance(item, dict)
    }
    start_devices = {
        item.get("device"): item
        for item in start_payload.get("devices", [])
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
                "ssd_ocp_latency_not_supported",
                "{} does not expose OCP 0xC3 latency monitor log.".format(device),
            ))
        else:
            events.append((
                "info",
                "ssd_ocp_latency_captured",
                "{} OCP 0xC3 latency monitor log captured at start and end.".format(device),
            ))
    return events
