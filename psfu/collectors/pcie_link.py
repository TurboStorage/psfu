from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from psfu.common import command_exists, run_capture


_DEFAULT_INTERVAL_S = 1.0
_BDF_RE = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")
_LNKCAP_RE = re.compile(r"LnkCap:.*?Speed\s+([0-9.]+)GT/s,?\s+Width\s+x(\d+)", re.IGNORECASE)
_LNKSTA_RE = re.compile(r"LnkSta:.*?Speed\s+([0-9.]+)GT/s.*?Width\s+x(\d+)", re.IGNORECASE)
_GTPS_TO_GEN = {
    2.5: 1,
    5.0: 2,
    8.0: 3,
    16.0: 4,
    32.0: 5,
    64.0: 6,
}


class PcieLinkCollector:
    def __init__(
        self,
        device_paths: List[str],
        *,
        interval_s: float = _DEFAULT_INTERVAL_S,
    ) -> None:
        self.device_paths = [path for path in device_paths if str(path).startswith("/dev/nvme")]
        self.interval_s = interval_s
        self.available: bool = False
        self._bdf_by_device: Dict[str, str] = {}
        self._latest_by_device: Dict[str, Dict[str, Optional[int]]] = {}
        self._capability_by_device: Dict[str, Dict[str, Optional[int]]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> Optional[str]:
        if not self.device_paths:
            return "no NVMe DUTs selected"
        if not command_exists("lspci"):
            return "lspci not installed"

        for path in self.device_paths:
            bdf = _device_bdf(path)
            if bdf:
                self._bdf_by_device[path] = bdf
        if not self._bdf_by_device:
            return "no NVMe DUT PCIe addresses resolved"

        probe_ok = False
        for path, bdf in self._bdf_by_device.items():
            payload = _poll_bdf(bdf)
            if payload is None:
                continue
            probe_ok = True
            with self._lock:
                self._latest_by_device[path] = {
                    "nvme_pcie_link_gen": payload["current_gen"],
                    "nvme_pcie_link_width": payload["current_width"],
                }
                self._capability_by_device[path] = {
                    "max_gen": payload["cap_gen"],
                    "max_width": payload["cap_width"],
                }

        if not probe_ok:
            return "no NVMe DUT responded to lspci -vv"

        self.available = True
        self._thread = threading.Thread(target=self._loop, name="nvme-pcie-link", daemon=True)
        self._thread.start()
        return None

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def snapshot(self) -> Dict[str, Optional[int]]:
        with self._lock:
            if not self._latest_by_device:
                return {
                    "nvme_pcie_link_gen": None,
                    "nvme_pcie_link_width": None,
                }
            first = self._latest_by_device.get(self.device_paths[0]) or next(iter(self._latest_by_device.values()))
            return dict(first)

    def first_capability(self) -> Dict[str, Optional[int]]:
        with self._lock:
            if not self._capability_by_device:
                return {"max_gen": None, "max_width": None}
            first = self._capability_by_device.get(self.device_paths[0]) or next(iter(self._capability_by_device.values()))
            return dict(first)

    def _loop(self) -> None:
        while not self._stop.is_set():
            for path, bdf in self._bdf_by_device.items():
                payload = _poll_bdf(bdf)
                if payload is None:
                    continue
                with self._lock:
                    self._latest_by_device[path] = {
                        "nvme_pcie_link_gen": payload["current_gen"],
                        "nvme_pcie_link_width": payload["current_width"],
                    }
                    self._capability_by_device[path] = {
                        "max_gen": payload["cap_gen"],
                        "max_width": payload["cap_width"],
                    }
            self._stop.wait(self.interval_s)


def _device_bdf(device_path: str) -> Optional[str]:
    node = Path("/sys/block") / Path(device_path).name / "device"
    try:
        resolved = node.resolve()
    except Exception:
        return None
    for candidate in (resolved, *resolved.parents):
        if _BDF_RE.match(candidate.name):
            return candidate.name
    return None


def _poll_bdf(bdf: str) -> Optional[Dict[str, Optional[int]]]:
    code, stdout, _ = run_capture(["lspci", "-s", bdf, "-vv"], timeout=5)
    if code != 0 or not stdout:
        return None
    return _parse_lspci_link_state(stdout)


def _parse_lspci_link_state(stdout: str) -> Optional[Dict[str, Optional[int]]]:
    cap_match = _LNKCAP_RE.search(stdout)
    sta_match = _LNKSTA_RE.search(stdout)
    if not cap_match and not sta_match:
        return None
    cap_gen, cap_width = _parse_speed_width(cap_match)
    cur_gen, cur_width = _parse_speed_width(sta_match)
    return {
        "cap_gen": cap_gen,
        "cap_width": cap_width,
        "current_gen": cur_gen,
        "current_width": cur_width,
    }


def _parse_speed_width(match: Optional[re.Match[str]]) -> Tuple[Optional[int], Optional[int]]:
    if match is None:
        return None, None
    speed_raw = match.group(1)
    width_raw = match.group(2)
    try:
        speed = float(speed_raw)
    except (TypeError, ValueError):
        speed = None
    try:
        width = int(width_raw)
    except (TypeError, ValueError):
        width = None
    gen = _speed_to_gen(speed)
    return gen, width


def _speed_to_gen(speed: Optional[float]) -> Optional[int]:
    if speed is None:
        return None
    for gtps, gen in _GTPS_TO_GEN.items():
        if abs(speed - gtps) < 0.01:
            return gen
    return None
