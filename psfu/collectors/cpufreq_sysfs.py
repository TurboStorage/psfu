"""Root-free CPU frequency + busy% fallback collector.

Used when turbostat is unavailable (no cap_sys_rawio, no sudo) so the CPU
panel can still show Avg_MHz, Max_MHz, and Busy%. Reads only world-readable
paths — no MSR access, no capabilities, no sudo.

Provides a subset of the turbostat snapshot keys:
  - cpu_avg_mhz            from /sys/.../cpufreq/scaling_cur_freq (mean)
  - cpu_max_mhz            from /sys/.../cpufreq/scaling_cur_freq (max)
  - cpu_busy_pct_turbo     from /proc/stat delta between ticks

Does NOT provide (MSR-only; stays None):
  - cpu_core_watt
  - cpu_pkg_watt_turbo

Caveat: ``scaling_cur_freq`` is the governor's reported frequency, not the
MSR-measured actual delivered frequency. Close enough for RMA throttle
detection; less precise than turbostat's Avg_MHz under bursty loads.
"""

from __future__ import annotations

import glob
import threading
import time
from typing import Dict, List, Optional, Tuple


_CPUFREQ_GLOB = "/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"
_PROC_STAT = "/proc/stat"


class CpuFreqSysfsCollector:
    """Cheap, root-free partial replacement for TurbostatCollector.

    No subprocess, no thread — reads on demand via ``snapshot()``. The
    busy% calculation needs deltas, so the collector keeps the last
    ``/proc/stat`` reading for comparison.
    """

    def __init__(self) -> None:
        self.available: bool = False
        self._lock = threading.Lock()
        self._last_stat: Optional[Tuple[float, int, int]] = None  # (ts, idle, total)
        # Probe once — if sysfs paths exist, mark available.
        if glob.glob(_CPUFREQ_GLOB):
            self.available = True

    def start(self) -> Optional[str]:
        """Parity with TurbostatCollector.start(). Returns None on success."""
        if not self.available:
            return "cpufreq sysfs not present"
        # Seed the /proc/stat baseline so the first snapshot has a delta.
        seed = self._read_proc_stat()
        if seed is not None:
            with self._lock:
                self._last_stat = seed
        return None

    def stop(self) -> None:
        """Parity with TurbostatCollector.stop(). No-op."""
        return

    def snapshot(self) -> Dict[str, Optional[float]]:
        """Return latest CPU metrics. Missing keys stay None."""
        avg_mhz, max_mhz = self._read_cpufreq()
        busy_pct = self._read_busy_pct()
        return {
            "cpu_avg_mhz": avg_mhz,
            "cpu_max_mhz": max_mhz,
            "cpu_busy_pct_turbo": busy_pct,
            "cpu_core_watt": None,
            "cpu_pkg_watt_turbo": None,
        }

    def _read_cpufreq(self) -> Tuple[Optional[float], Optional[float]]:
        """Mean + max across online CPU scaling_cur_freq (kHz → MHz)."""
        paths = glob.glob(_CPUFREQ_GLOB)
        if not paths:
            return None, None
        khz_values: List[int] = []
        for p in paths:
            try:
                with open(p, "r") as f:
                    khz_values.append(int(f.read().strip()))
            except (OSError, ValueError):
                continue
        if not khz_values:
            return None, None
        avg_mhz = sum(khz_values) / len(khz_values) / 1000.0
        max_mhz = max(khz_values) / 1000.0
        return avg_mhz, max_mhz

    def _read_proc_stat(self) -> Optional[Tuple[float, int, int]]:
        """Read /proc/stat cpu aggregate line → (ts, idle_jiffies, total_jiffies)."""
        try:
            with open(_PROC_STAT, "r") as f:
                first = f.readline()
        except OSError:
            return None
        parts = first.split()
        if not parts or parts[0] != "cpu":
            return None
        try:
            fields = [int(x) for x in parts[1:]]
        except ValueError:
            return None
        if len(fields) < 5:
            return None
        # /proc/stat cpu fields: user nice system idle iowait irq softirq steal ...
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        total = sum(fields)
        return time.monotonic(), idle, total

    def _read_busy_pct(self) -> Optional[float]:
        """Busy% over the delta since the last snapshot. None on first call."""
        now = self._read_proc_stat()
        if now is None:
            return None
        with self._lock:
            prev = self._last_stat
            self._last_stat = now
        if prev is None:
            return None
        dt_idle = now[1] - prev[1]
        dt_total = now[2] - prev[2]
        if dt_total <= 0:
            return None
        busy = 1.0 - (dt_idle / dt_total)
        return max(0.0, min(100.0, busy * 100.0))
