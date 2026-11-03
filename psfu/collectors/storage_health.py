"""Storage health capture via `nvme smart-log` / `smartctl -Aj`, plus the diff
that turns start/end snapshots into verdict events.

Captured at run start and run end; `storage_health_events` derives a list of
(severity, kind, message) tuples based on the delta — e.g., media_errors
going up, new critical_warning, SMART status flipping to failed, overtemp.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from psfu.common import run_capture, utc_now
from psfu.hardware import nvme_cli_binary_path, smartctl_binary_path


def storage_health_devices() -> List[str]:
    devices: List[str] = []
    sys_block = Path("/sys/block")
    if not sys_block.exists():
        return devices
    try:
        for item in sorted(sys_block.iterdir()):
            name = item.name
            if name.startswith(("loop", "ram", "zram", "dm-", "md")):
                continue
            devices.append("/dev/{}".format(name))
    except Exception:
        return []
    return devices


def _parse_smartctl_health(payload: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "available": True,
        "temperature_c": None,
        "critical_warning": None,
        "media_errors": None,
        "error_log_entries": None,
        "unsafe_shutdowns": None,
        "smart_passed": None,
    }
    smart_status = payload.get("smart_status")
    if isinstance(smart_status, dict):
        out["smart_passed"] = smart_status.get("passed")
    temperature = payload.get("temperature")
    if isinstance(temperature, dict):
        current = temperature.get("current")
        if isinstance(current, (int, float)):
            out["temperature_c"] = float(current)
    nvme = payload.get("nvme_smart_health_information_log")
    if isinstance(nvme, dict):
        for src, dst in [
            ("critical_warning", "critical_warning"),
            ("media_errors", "media_errors"),
            ("num_err_log_entries", "error_log_entries"),
            ("unsafe_shutdowns", "unsafe_shutdowns"),
            ("temperature", "temperature_c"),
        ]:
            value = nvme.get(src)
            if isinstance(value, (int, float)):
                out[dst] = value
    return out


def _parse_nvme_health(payload: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "available": True,
        "temperature_c": None,
        "critical_warning": None,
        "media_errors": None,
        "error_log_entries": None,
        "unsafe_shutdowns": None,
        "smart_passed": None,
    }
    for src, dst in [
        ("critical_warning", "critical_warning"),
        ("media_errors", "media_errors"),
        ("num_err_log_entries", "error_log_entries"),
        ("unsafe_shutdowns", "unsafe_shutdowns"),
        ("temperature", "temperature_c"),
    ]:
        value = payload.get(src)
        if isinstance(value, (int, float)):
            out[dst] = value
    return out


def capture_storage_health() -> Dict[str, object]:
    devices = storage_health_devices()
    payload: Dict[str, object] = {
        "captured_at": utc_now(),
        "devices": [],
    }
    for dev in devices:
        entry: Dict[str, object] = {
            "device": dev,
            "tool": None,
            "available": False,
            "temperature_c": None,
            "critical_warning": None,
            "media_errors": None,
            "error_log_entries": None,
            "unsafe_shutdowns": None,
            "smart_passed": None,
        }
        nvme_bin = nvme_cli_binary_path()
        if nvme_bin and "/dev/nvme" in dev:
            code, stdout, _ = run_capture([nvme_bin, "smart-log", "-o", "json", dev], timeout=10)
            if code == 0 and stdout:
                try:
                    parsed = json.loads(stdout)
                    entry.update(_parse_nvme_health(parsed))
                    entry["tool"] = "nvme"
                except Exception:
                    pass
        smartctl_bin = smartctl_binary_path()
        if not entry["available"] and smartctl_bin:
            code, stdout, _ = run_capture([smartctl_bin, "-Aj", dev], timeout=15)
            if code in (0, 4) and stdout:
                try:
                    parsed = json.loads(stdout)
                    entry.update(_parse_smartctl_health(parsed))
                    entry["tool"] = "smartctl"
                except Exception:
                    pass
        payload["devices"].append(entry)
    return payload


def storage_health_events(
    start_payload: Dict[str, object],
    end_payload: Dict[str, object],
) -> List[Tuple[str, str, str]]:
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
        if end.get("smart_passed") is False:
            events.append(("crit", "storage_smart_fail", "{} SMART health check failed.".format(device)))
        crit_warn = end.get("critical_warning")
        if isinstance(crit_warn, (int, float)) and crit_warn:
            events.append(("crit", "storage_critical_warning", "{} reported NVMe critical_warning={}.".format(device, int(crit_warn))))
        start_media = start.get("media_errors")
        end_media = end.get("media_errors")
        if isinstance(start_media, (int, float)) and isinstance(end_media, (int, float)) and end_media > start_media:
            events.append(("crit", "storage_media_errors", "{} media_errors increased from {} to {}.".format(device, int(start_media), int(end_media))))
        start_errs = start.get("error_log_entries")
        end_errs = end.get("error_log_entries")
        if isinstance(start_errs, (int, float)) and isinstance(end_errs, (int, float)) and end_errs > start_errs:
            events.append(("warn", "storage_error_log_growth", "{} error log entries increased from {} to {}.".format(device, int(start_errs), int(end_errs))))
        start_unsafe = start.get("unsafe_shutdowns")
        end_unsafe = end.get("unsafe_shutdowns")
        if isinstance(start_unsafe, (int, float)) and isinstance(end_unsafe, (int, float)) and end_unsafe > start_unsafe:
            events.append(("warn", "storage_unsafe_shutdown", "{} unsafe shutdown count increased from {} to {}.".format(device, int(start_unsafe), int(end_unsafe))))
        temp = end.get("temperature_c")
        if isinstance(temp, (int, float)):
            if temp >= 80:
                events.append(("crit", "storage_overtemp", "{} reached {:.1f}C.".format(device, float(temp))))
            elif temp >= 70:
                events.append(("warn", "storage_hot", "{} reached {:.1f}C.".format(device, float(temp))))
    return events
