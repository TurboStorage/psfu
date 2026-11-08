"""Long-lived turbostat subprocess collector.

Spawns ``turbostat`` once and parses its periodic table output to expose
per-package and per-core CPU metrics that RAPL alone cannot provide:
average MHz, busy%, per-core wattage, and package wattage.

Requires root (``sudo turbostat``) to read MSRs.  If sudo is unavailable
or turbostat is not installed, the collector degrades gracefully — all
fields return ``None`` and ``available`` stays ``False``.

AMD Zen 5 quirks:
- ``PkgTmp`` not exposed — AMD reports die temps via k10temp, not MSR.
- ``RAMWatt`` is Intel-only.
- Working column set: ``Avg_MHz,Busy%,CorWatt,PkgWatt``.

Output format: turbostat prints a header line then per-core rows every
``-i`` seconds.  The first column group is always package/core/CPU IDs.
A row where Core == ``-`` is a package summary.  We parse by column
header name, not position, so it tolerates kernel and CPU variations.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
from typing import Dict, List, Optional, Tuple

from psfu.common import turbostat_sudo_path


# Fallback sanity ceiling for Avg_MHz when cpufreq sysfs is absent.
# Covers Intel Core i9-14900KS (6.2 GHz stock boost) with OC headroom.
# Instances prefer a hardware-aware ceiling derived from
# ``/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq × 1.15`` —
# this fallback only fires when that sysfs path is missing.
#
# Why the guard exists: turbostat occasionally emits a header line
# mid-table (observed 2026-04-14 on TRX50 AI TOP). That leaves
# ``col_map`` pointing at the wrong column for one row, and a PkgWatt
# or per-core Avg_MHz value gets interpreted as the package Avg_MHz
# (observed 7148, 12037, 6475, 6014 MHz on a 7980X whose real max boost
# is 5.3 GHz). The row is discarded entirely; drift corrupts every
# field, not just MHz.
_MAX_PLAUSIBLE_MHZ_FALLBACK = 7200.0
# Back-compat re-export for tests that still import the old name.
_MAX_PLAUSIBLE_MHZ = _MAX_PLAUSIBLE_MHZ_FALLBACK

# Hardware-aware ceiling = advertised max boost × this multiplier.
# 15% covers operator-facing BIOS overclocks + turbostat's tendency
# to report peak boost slightly above cpuinfo_max_freq (typical
# observation: +2-5%). Tighter multipliers caught legitimate 14900KS
# peak-boost rows as drift in bench testing.
_MHZ_CEILING_MULTIPLIER = 1.15


def _read_cpu_max_boost_mhz() -> Optional[float]:
    """Return advertised max CPU boost in MHz from cpufreq sysfs.

    Reads ``cpuinfo_max_freq`` for every CPU and returns the max
    (in case asymmetric big.LITTLE-style designs ever reach x86). On
    any failure returns ``None`` so the caller can fall back to the
    flat ceiling.
    """
    try:
        # Use ``scaling_max_freq`` first — some platforms cap this below
        # ``cpuinfo_max_freq`` via BIOS and we want the real ceiling in
        # effect, not the silicon cap. ``cpuinfo_max_freq`` is the
        # hardware-advertised max and is the right fallback.
        candidates = (
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
        )
        values_khz: List[int] = []
        for path in candidates:
            try:
                with open(path, "r") as fh:
                    values_khz.append(int(fh.read().strip()))
            except (OSError, ValueError):
                continue
        if not values_khz:
            return None
        max_khz = max(values_khz)
        if max_khz <= 0:
            return None
        return float(max_khz) / 1000.0
    except Exception:
        return None

# Sanity ceiling for PkgWatt. Single-socket Threadripper 7000 tops out
# ~400 W; EPYC tops ~500 W. 750 W covers future bumps without admitting
# the column-drift glitches we observed (1473 W and 718 W spikes during
# rail-swap and transient phases on TRX50 AI TOP runs on 2026-04-14).
# Discards the row, same rule as _MAX_PLAUSIBLE_MHZ.
_MAX_PLAUSIBLE_PKG_WATT = 750.0


class TurbostatCollector:
    """Spawn turbostat as a daemon, expose latest parsed snapshot."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.available: bool = False
        self.proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        # Per-bench Avg_MHz sanity ceiling. Reads cpufreq sysfs at init
        # so the guard scales to 14900KS (6.2 GHz stock → 7130 MHz
        # ceiling) on Intel benches while staying tight on Threadripper
        # (~5.3 GHz → 6095 MHz ceiling). The old flat 6500 MHz let the
        # observed 6475 MHz spikes on a 7980X slip through — a value
        # 22% above real peak boost — and poisoned the throttle
        # baseline. Falls back to the flat ceiling when sysfs is
        # absent (containers, WSL, etc.).
        max_boost = _read_cpu_max_boost_mhz()
        if max_boost is not None:
            self._max_plausible_mhz = max_boost * _MHZ_CEILING_MULTIPLIER
        else:
            self._max_plausible_mhz = _MAX_PLAUSIBLE_MHZ_FALLBACK

        # Latest parsed snapshot (package-level aggregates).
        self._pkg_avg_mhz: Optional[float] = None
        self._pkg_max_mhz: Optional[float] = None
        self._pkg_busy_pct: Optional[float] = None
        self._pkg_core_watt: Optional[float] = None
        self._pkg_watt: Optional[float] = None
        # v7 additions — per-core MHz σ + deepest-C-state residency.
        self._pkg_freq_spread_mhz: Optional[float] = None
        self._pkg_c6_pct: Optional[float] = None

        # Per-core data for post-run analysis and throttle detection.
        # List of dicts, one per logical CPU, keyed by column header.
        self._per_core: List[Dict[str, float]] = []

        # Column header indices — parsed once from the first header line.
        self._col_map: Dict[str, int] = {}

    def start(self) -> Optional[str]:
        """Start the turbostat subprocess.  Returns an error string on failure."""
        # Same absolute path the sudoers rule names (psfu.common), so the
        # call matches the rule even when PATH or secure_path differ.
        ts_path = turbostat_sudo_path()
        if not ts_path:
            return "turbostat not installed"

        # Try the v7 column set (adds CPU%c6 for deepest-C-state residency)
        # first; if that turbostat build rejects the column fall back to
        # the v6 set. Older AMD turbostat builds (pre-5.15 kernel tree)
        # don't expose CPU%c6 and exit immediately with rc != 0 when the
        # column is named in --show.
        for show in (
            "Core,CPU,Avg_MHz,Busy%,CorWatt,PkgWatt,CPU%c6",
            "Core,CPU,Avg_MHz,Busy%,CorWatt,PkgWatt",
        ):
            cmd = [
                "sudo", "-n",  # non-interactive — fail immediately if password needed
                ts_path,
                "--show", show,
                "-i", str(self.interval),
            ]
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                    preexec_fn=os.setsid,
                )
            except OSError as exc:
                return "failed to spawn turbostat: {}".format(exc)

            # Check if it died immediately (sudo password required,
            # permission denied, or --show rejected a column).
            try:
                rc = self.proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                break  # Still running — good.
            # Immediate exit — retry with the fallback column set unless
            # this was already the fallback.
            if show == "Core,CPU,Avg_MHz,Busy%,CorWatt,PkgWatt":
                return "turbostat exited immediately (rc={})".format(rc)
            # else: fall through to next iteration with the v6 set.

        self.available = True
        self._thread = threading.Thread(
            target=self._reader_loop, name="turbostat-collector", daemon=True
        )
        self._thread.start()
        return None

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
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def snapshot(self) -> Dict[str, Optional[float]]:
        """Return the latest package-level turbostat readings."""
        with self._lock:
            return {
                "cpu_avg_mhz": self._pkg_avg_mhz,
                "cpu_max_mhz": self._pkg_max_mhz,
                "cpu_busy_pct_turbo": self._pkg_busy_pct,
                "cpu_core_watt": self._pkg_core_watt,
                "cpu_pkg_watt_turbo": self._pkg_watt,
                # v7 — per-core frequency σ and C6% residency. Both are
                # None when turbostat fell back to the v6 column set
                # (older AMD builds without CPU%c6) or when fewer than
                # two cores reported Avg_MHz this snapshot (σ of <2
                # samples is meaningless).
                "cpu_freq_spread_mhz": self._pkg_freq_spread_mhz,
                "cpu_c6_pct": self._pkg_c6_pct,
            }

    def per_core_snapshot(self) -> List[Dict[str, float]]:
        """Return per-core data from the latest turbostat table."""
        with self._lock:
            return list(self._per_core)

    def _reader_loop(self) -> None:
        if not self.proc or not self.proc.stdout:
            return
        try:
            current_block: List[str] = []
            for line in self.proc.stdout:
                if self._stop.is_set():
                    break
                line = line.rstrip()
                if not line:
                    continue

                # turbostat prints a header line then data rows.  A new header
                # (containing "Avg_MHz" or "Core") signals a new table snapshot.
                if self._is_header(line):
                    if current_block:
                        self._parse_block(current_block)
                    current_block = [line]
                else:
                    current_block.append(line)

            # Parse any remaining block.
            if current_block:
                self._parse_block(current_block)
        except Exception:
            return

    @staticmethod
    def _is_header(line: str) -> bool:
        """Detect a turbostat header line."""
        return "Avg_MHz" in line or (line.startswith("Core") and "CPU" in line)

    def _parse_block(self, lines: List[str]) -> None:
        """Parse a complete turbostat table (header + data rows)."""
        if not lines:
            return

        header = lines[0]
        cols = header.split()
        col_map = {name: idx for idx, name in enumerate(cols)}

        # Require at least Avg_MHz to do anything useful.
        if "Avg_MHz" not in col_map:
            return

        per_core: List[Dict[str, float]] = []
        pkg_rows: List[Dict[str, float]] = []

        for data_line in lines[1:]:
            parts = data_line.split()
            if len(parts) < len(cols):
                continue

            row: Dict[str, float] = {}
            for name in ("Avg_MHz", "Busy%", "CorWatt", "PkgWatt", "CPU%c6"):
                idx = col_map.get(name)
                if idx is not None and idx < len(parts):
                    try:
                        row[name] = float(parts[idx])
                    except ValueError:
                        pass

            # Column-drift guard: if Avg_MHz or PkgWatt exceeds any
            # realistic value, the row hit a header-in-stream glitch
            # and all its columns are misaligned. Drop the whole row.
            mhz_val = row.get("Avg_MHz")
            if mhz_val is not None and mhz_val > self._max_plausible_mhz:
                continue
            pkg_watt_val = row.get("PkgWatt")
            if pkg_watt_val is not None and pkg_watt_val > _MAX_PLAUSIBLE_PKG_WATT:
                continue

            # Determine if this is a package summary row or a per-core row.
            # Package summary: Core column is "-".
            core_idx = col_map.get("Core")
            is_pkg_row = False
            if core_idx is not None and core_idx < len(parts):
                if parts[core_idx] == "-":
                    is_pkg_row = True

            if is_pkg_row:
                pkg_rows.append(row)
            elif row:
                per_core.append(row)

        # Max MHz across all per-core rows (shows peak boost, not dragged
        # down by idle cores like Avg_MHz is).
        all_mhz = [r["Avg_MHz"] for r in per_core if "Avg_MHz" in r]
        max_mhz = max(all_mhz) if all_mhz else None

        # v7: per-core frequency spread (σ). Legitimate signal for
        # thermal-throttle heterogeneity — one cooked CCD dragging
        # its cores down while the rest run at boost. Requires at
        # least 2 cores; σ of <2 is meaningless.
        if len(all_mhz) >= 2:
            mean_mhz = sum(all_mhz) / len(all_mhz)
            var = sum((v - mean_mhz) ** 2 for v in all_mhz) / len(all_mhz)
            freq_spread = round(var ** 0.5, 1)
        else:
            freq_spread = None

        # v7: package C6% residency. Pulled from the package summary row
        # when turbostat emits one, else averaged across per-core rows.
        c6_pkg: Optional[float] = None
        if pkg_rows and "CPU%c6" in pkg_rows[0]:
            c6_pkg = pkg_rows[0]["CPU%c6"]
        else:
            c6_vals = [r["CPU%c6"] for r in per_core if "CPU%c6" in r]
            if c6_vals:
                c6_pkg = round(sum(c6_vals) / len(c6_vals), 2)

        with self._lock:
            self._per_core = per_core
            self._pkg_max_mhz = max_mhz
            self._pkg_freq_spread_mhz = freq_spread
            self._pkg_c6_pct = c6_pkg

            if pkg_rows:
                # Use the first package summary row.
                pkg = pkg_rows[0]
                self._pkg_avg_mhz = pkg.get("Avg_MHz")
                self._pkg_busy_pct = pkg.get("Busy%")
                self._pkg_core_watt = pkg.get("CorWatt")
                self._pkg_watt = pkg.get("PkgWatt")
            elif per_core:
                # No package summary (single-socket, some turbostat versions).
                # Aggregate from per-core rows.
                avg_mhz_vals = all_mhz
                busy_vals = [r["Busy%"] for r in per_core if "Busy%" in r]
                corwatt_vals = [r["CorWatt"] for r in per_core if "CorWatt" in r]
                pkgwatt_vals = [r["PkgWatt"] for r in per_core if "PkgWatt" in r]
                self._pkg_avg_mhz = (
                    round(sum(avg_mhz_vals) / len(avg_mhz_vals), 1) if avg_mhz_vals else None
                )
                self._pkg_busy_pct = (
                    round(sum(busy_vals) / len(busy_vals), 1) if busy_vals else None
                )
                self._pkg_core_watt = (
                    round(sum(corwatt_vals), 1) if corwatt_vals else None
                )
                self._pkg_watt = pkgwatt_vals[0] if pkgwatt_vals else None

    def summary(self) -> Dict[str, object]:
        with self._lock:
            return {
                "available": self.available,
                "pkg_avg_mhz": self._pkg_avg_mhz,
                "pkg_busy_pct": self._pkg_busy_pct,
                "pkg_core_watt": self._pkg_core_watt,
                "pkg_watt": self._pkg_watt,
                "core_count": len(self._per_core),
            }
