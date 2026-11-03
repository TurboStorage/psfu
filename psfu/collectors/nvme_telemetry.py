from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from psfu.common import run_capture, utc_now
from psfu.hardware import nvme_cli_binary_path


_DEFAULT_INTERVAL_S = 1.0

_SAMPLE_COLUMNS = [
    "nvme_composite_temp_c",
    "nvme_warning_temp_time_min",
    "nvme_critical_temp_time_min",
    "nvme_percentage_used",
    "nvme_available_spare",
    "nvme_available_spare_threshold",
    "nvme_data_units_read",
    "nvme_data_units_written",
    "nvme_host_read_cmds",
    "nvme_host_write_cmds",
    "nvme_controller_busy_time_min",
    "nvme_power_cycles",
    "nvme_unsafe_shutdowns",
    "nvme_media_errors",
    "nvme_num_err_log_entries",
    "nvme_pcie_link_gen",
    "nvme_pcie_link_width",
]


class NvmeTelemetryCollector:
    def __init__(
        self,
        device_paths: List[str],
        *,
        interval_s: float = _DEFAULT_INTERVAL_S,
        run_dir: Optional[Path] = None,
    ) -> None:
        self.device_paths = [path for path in device_paths if str(path).startswith("/dev/nvme")]
        self.interval_s = interval_s
        self.run_dir = run_dir
        self.available: bool = False
        self._latest_by_device: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> Optional[str]:
        if not self.device_paths:
            return "no NVMe DUTs selected"
        if not nvme_cli_binary_path():
            return "nvme-cli not available"
        probe_ok = False
        for path in self.device_paths:
            payload = _poll_device(path)
            if payload is not None:
                probe_ok = True
                with self._lock:
                    self._latest_by_device[path] = payload
                self._append_jsonl(path, payload)
        if not probe_ok:
            return "no NVMe DUT responded to nvme smart-log"
        self.available = True
        self._thread = threading.Thread(target=self._loop, name="nvme-telemetry", daemon=True)
        self._thread.start()
        return None

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            if not self._latest_by_device:
                return {key: None for key in _SAMPLE_COLUMNS}
            first = self._latest_by_device.get(self.device_paths[0]) or next(iter(self._latest_by_device.values()))
            return {key: first.get(key) for key in _SAMPLE_COLUMNS}

    def _loop(self) -> None:
        while not self._stop.is_set():
            for path in self.device_paths:
                payload = _poll_device(path)
                if payload is not None:
                    with self._lock:
                        self._latest_by_device[path] = payload
                    self._append_jsonl(path, payload)
            self._stop.wait(self.interval_s)

    def _append_jsonl(self, path: str, payload: Dict[str, Any]) -> None:
        if self.run_dir is None:
            return
        out_dir = self.run_dir / "storage" / "nvme" / Path(path).name
        out_dir.mkdir(parents=True, exist_ok=True)
        line = dict(payload)
        line["ts"] = utc_now()
        with (out_dir / "telemetry.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")


def _poll_device(path: str) -> Optional[Dict[str, Any]]:
    code, stdout, _ = run_capture([nvme_cli_binary_path() or "nvme", "smart-log", path, "-o", "json"], timeout=5)
    if code != 0 or not stdout:
        return None
    try:
        data = json.loads(stdout)
    except Exception:
        return None
    return _parse_smart_log_sample(data)


def _parse_smart_log_sample(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "nvme_composite_temp_c": _num(data, "temperature"),
        "nvme_warning_temp_time_min": _num(data, "warning_temp_time"),
        "nvme_critical_temp_time_min": _num(data, "critical_comp_time"),
        "nvme_percentage_used": _num(data, "percent_used", "percentage_used"),
        "nvme_available_spare": _num(data, "avail_spare", "available_spare"),
        "nvme_available_spare_threshold": _num(data, "spare_thresh", "available_spare_threshold"),
        "nvme_data_units_read": _int(data, "data_units_read"),
        "nvme_data_units_written": _int(data, "data_units_written"),
        "nvme_host_read_cmds": _int(data, "host_reads", "host_read_commands"),
        "nvme_host_write_cmds": _int(data, "host_writes", "host_write_commands"),
        "nvme_controller_busy_time_min": _num(data, "controller_busy_time"),
        "nvme_power_cycles": _int(data, "power_cycles"),
        "nvme_unsafe_shutdowns": _int(data, "unsafe_shutdowns"),
        "nvme_media_errors": _int(data, "media_errors"),
        "nvme_num_err_log_entries": _int(data, "num_err_log_entries"),
        "nvme_pcie_link_gen": None,
        "nvme_pcie_link_width": None,
    }


def _num(data: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _int(data: Dict[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                continue
    return None
