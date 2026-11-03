"""Memory bandwidth sampler via ``perf stat`` uncore IMC events.

Runs a subprocess-backed sampler on its own thread at a much lower cadence
than the 1 Hz telemetry loop (30 s by default). Each sample runs ``perf
stat`` for one second against the uncore memory-controller read/write
events, parses the resulting counts, and converts them to MB/s.

Why a separate thread and a slow cadence:
  * ``perf stat`` with a 1 s window adds ~30-40 ms of subprocess overhead
    per invocation and consumes counters that the 1 Hz loop does not need
    to touch.
  * DIMM bandwidth on stress runs changes slowly compared to thermal or
    rail signals. A 30 s window captures phase transitions (soak → steady
    → transient) with plenty of resolution.
  * Between samples we hold the last value so the TUI sparkline renders
    flat rather than gappy — same UX contract as the other collectors.

PMU availability:
  * AMD Zen 4/5 exposes uncore memory counters via the ``amd_df`` PMU.
  * Intel Sapphire Rapids / Granite Rapids expose ``uncore_imc_N`` PMUs.
  * Event names vary per microarchitecture. We try a short list and
    accept the first one that ``perf stat`` returns non-zero counts for.
  * Some kernels require ``perf_event_paranoid <= 1`` or root for uncore
    events. We run under ``sudo -n`` if the caller is root-capable, else
    plain perf.

When no event produces numbers, the collector stays ``available = False``
and ``snapshot()`` returns ``{None, None}`` forever — no raises, no spam.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

from psfu.common import command_exists


# Candidate (read_event, write_event) pairs. Tried in order; the first
# pair that returns non-zero counts on a probe run wins. Names follow
# what ``perf list`` reports on representative kernels.
_CANDIDATE_EVENTS: List[Tuple[str, str]] = [
    # AMD Zen 4/5 Data Fabric events — reported by perf on kernel 6.x.
    ("amd_df/event=0x1f,umask=0x00/", "amd_df/event=0x2f,umask=0x00/"),
    # Newer perf-tool naming on Zen 5:
    ("amd_df/mem_bw_read/", "amd_df/mem_bw_write/"),
    # Intel uncore_imc — cast_read / cast_write per IMC PMU.
    ("uncore_imc/cas_count_read/", "uncore_imc/cas_count_write/"),
]

# Width of the perf-stat window per probe, seconds. 1 s is long enough
# to smooth out per-core traffic bursts but short enough to not block
# the sampler thread.
_PROBE_WINDOW_S = 1.0

# How many seconds between probes. Overridable for tests.
_DEFAULT_INTERVAL_S = 30.0

# Bytes per uncore read/write event on Zen 4/5 — each DF transaction is
# a 64-byte cache line. Intel uncore cas_count is already in cache-line
# units. Stays 64 either way. If a future platform differs, this is the
# single constant to change.
_BYTES_PER_EVENT = 64


class MemoryBandwidthCollector:
    """Threaded ``perf stat`` sampler exposing a latest-value snapshot.

    Lifecycle mirrors ``TurbostatCollector``: ``start()`` to spawn the
    thread (returns an error string on unrecoverable failure), ``stop()``
    to shut down, ``snapshot()`` to pull the latest reading.
    """

    def __init__(self, interval_s: float = _DEFAULT_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self.available: bool = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        # Latest sampled values (held between probes).
        self._read_mbps: Optional[float] = None
        self._write_mbps: Optional[float] = None

        # Selected event pair — probed on first iteration, cached.
        self._events: Optional[Tuple[str, str]] = None

    def start(self) -> Optional[str]:
        if not command_exists("perf"):
            return "perf not installed"
        self._thread = threading.Thread(
            target=self._loop, name="mem-bandwidth-collector", daemon=True,
        )
        self._thread.start()
        return None

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def snapshot(self) -> Dict[str, Optional[float]]:
        with self._lock:
            return {
                "mem_bw_read_mbps":  self._read_mbps,
                "mem_bw_write_mbps": self._write_mbps,
            }

    # ---- internals ----

    def _loop(self) -> None:
        # First iteration probes event availability; subsequent iterations
        # use the cached pair. If probing fails, we enter a tombstone state
        # where the loop keeps ticking (so stop() joins cleanly) but no
        # samples are taken.
        tombstone = False
        while not self._stop.is_set():
            if tombstone:
                # Wait out the interval in small chunks so stop() is fast.
                self._stop.wait(timeout=min(self.interval_s, 5.0))
                continue
            if self._events is None:
                picked = self._probe_events()
                if picked is None:
                    tombstone = True
                    continue
                self._events = picked
                self.available = True
            r_mbps, w_mbps = self._sample_once(self._events)
            if r_mbps is not None or w_mbps is not None:
                with self._lock:
                    if r_mbps is not None:
                        self._read_mbps = r_mbps
                    if w_mbps is not None:
                        self._write_mbps = w_mbps
            # Interruptible sleep for the remainder of the interval.
            self._stop.wait(timeout=self.interval_s)

    def _probe_events(self) -> Optional[Tuple[str, str]]:
        """Return the first event pair that produces non-zero counts.

        Probe runs are the same as a real sample — 1 s perf stat. We
        accept zero counts from the write event (some benches have
        write-heavy vs read-heavy phases at probe time) as long as the
        read event produced something, and vice versa.
        """
        for read_ev, write_ev in _CANDIDATE_EVENTS:
            r, w = self._sample_once((read_ev, write_ev))
            if (r or 0) > 0 or (w or 0) > 0:
                return read_ev, write_ev
        return None

    def _sample_once(
        self, events: Tuple[str, str],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Run perf stat once and return (read_mbps, write_mbps)."""
        read_ev, write_ev = events
        cmd = [
            "perf", "stat",
            "-a", "-x", ",",
            "-e", read_ev,
            "-e", write_ev,
            "sleep", str(_PROBE_WINDOW_S),
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=_PROBE_WINDOW_S + 5.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None, None
        # perf -x,: CSV format — counts on stderr:
        #   <count>,<unit>,<event>,<runtime>,<pct_running>,...
        return _parse_perf_csv(proc.stderr, read_ev, write_ev, _PROBE_WINDOW_S)


# ---- pure-function parser (exported for tests) ----

def _parse_perf_csv(
    stderr_text: str,
    read_ev: str,
    write_ev: str,
    window_s: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Parse ``perf stat -x,`` output into (read_mbps, write_mbps).

    Each counted event produces one CSV line; multiplexed events with
    zero runtime are reported as ``<not counted>`` and skipped.
    """
    read_count: Optional[int] = None
    write_count: Optional[int] = None
    for raw in stderr_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        count_raw = parts[0].strip()
        event_name = parts[2].strip() if len(parts) >= 3 else ""
        # "<not counted>" / "<not supported>" → skip.
        if count_raw.startswith("<"):
            continue
        try:
            count = int(count_raw)
        except ValueError:
            # Some perf builds emit floats for sub-second runs.
            try:
                count = int(float(count_raw))
            except ValueError:
                continue
        # Match the event tag against the supplied event spec. perf
        # sometimes normalizes the name (strips /event=.../), so we
        # also accept substring matches against the base PMU name.
        if event_name == read_ev or _event_matches(event_name, read_ev):
            read_count = count
        elif event_name == write_ev or _event_matches(event_name, write_ev):
            write_count = count
    if read_count is None and write_count is None:
        return None, None
    r_mbps = _to_mbps(read_count, window_s) if read_count is not None else None
    w_mbps = _to_mbps(write_count, window_s) if write_count is not None else None
    return r_mbps, w_mbps


def _event_matches(reported: str, requested: str) -> bool:
    # Extract the PMU prefix (before ``/`` or ``:``). Accept substring
    # match on the base PMU name OR on the requested-event body.
    rep = reported.lower()
    req = requested.lower()
    pmu = re.split(r"[/:]", req)[0]
    return pmu and pmu in rep


def _to_mbps(count: Optional[int], window_s: float) -> Optional[float]:
    if count is None or count < 0 or window_s <= 0:
        return None
    bytes_per_sec = count * _BYTES_PER_EVENT / window_s
    return round(bytes_per_sec / 1e6, 1)
