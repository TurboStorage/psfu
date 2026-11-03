"""Upstream UPS telemetry via Network UPS Tools (NUT).

Polls a single named UPS via ``upsc <name>[@<host>]`` at 1 Hz on its own
thread, parses the ``key: value`` lines NUT exposes, and surfaces a
flattened snapshot to the 1 Hz telemetry loop.

Why NUT instead of vendor daemons:
  * One protocol covers Cyberpower (PR1500LCD via the ``usbhid-ups``
    driver) and APC (Smart-UPS family via ``usbhid-ups`` or ``apcsmart``)
    without a per-vendor code branch. ``apcupsd`` and ``pwrstat`` would
    each need their own parser and daemon contract.
  * NUT runs as a system service (``upsd``); we are a read-only client
    via the ``upsc`` CLI — no permissions to acquire, no USB device to
    open, no race with other consumers of the UPS.
  * ``upsc`` reads in-memory state from ``upsd`` so 1 Hz polling is
    cheap (no USB transaction per call). Still ran on its own thread
    so a NUT hiccup cannot stall the main telemetry loop.

Data surfaced (subset of NUT variables, mapped to flat columns):
  * input.voltage         → ups_input_voltage_v
  * input.frequency       → ups_input_freq_hz
  * output.voltage        → ups_output_voltage_v
  * ups.load              → ups_load_pct
  * ups.realpower         → ups_realpower_w         (APC family only)
  * ups.realpower.nominal → ups_realpower_nominal_w (plate rating)
  * battery.charge        → ups_battery_charge_pct
  * battery.runtime       → ups_battery_runtime_s
  * ups.status            → ups_status_raw          (e.g. "OL", "OB DISCHRG")
  * derived               → ups_on_battery          (1 if status contains "OB")
  * derived               → ups_transfer_count      (cumulative OL→OB edges)
  * derived               → ups_realpower_est_w     (load% × nominal_w / 100)
  * derived               → ups_apparent_nominal_va (parsed from model name)
  * derived               → ups_apparent_est_va     (load% × nominal_va / 100)

Why derive the est columns: CyberPower CP1500 family firmware exposes
``ups.realpower.nominal`` (rated plate) and ``ups.load`` (live percent)
but never ``ups.realpower`` directly — the Windows PowerPanel app shows
live watts by computing ``plate × load% / 100``, which is what we
mirror here so the live tick has a wattage number on both APC and
CyberPower benches. On APC, ``ups_realpower_w`` is the authoritative
live value and ``ups_realpower_est_w`` is kept as a cross-check.

Why ``ups_transfer_count`` lives in the collector, not in a verdict
rule: a transfer is a multi-sample event spanning at least two ticks.
Computing it here means the verdict layer just reads a monotonically
increasing counter and can compare endpoints across phases without
re-walking the status history.

Verdict consumers: NONE YET. By design, every event kind
fed into the verdict engine must have a consumer. This collector emits
ZERO events and only writes columns. Adding a ``ups_transfer_during_run``
or ``mains_sag_during_run`` event is a separate commit that ships with
its rule together (no half-done feature paths).

When NUT is not installed, the named UPS is offline, or no UPS is
configured, the collector stays ``available = False`` and ``snapshot()``
returns all-None forever — no raises, no log spam.
"""

from __future__ import annotations

import re
import threading
from typing import Dict, List, Optional

from psfu.common import command_exists, run_capture


_DEFAULT_INTERVAL_S = 1.0

# Keys NUT exposes that we map straight through to columns. The mapping
# is one-to-one (NUT variable on the left, PS-FU column on the right).
#
# ``ups.realpower`` is live draw in watts (APC Smart-UPS family emits it,
# CyberPower CP1500 family omits it). ``ups.realpower.nominal`` is the
# rated plate wattage (CyberPower does emit this, so we can *derive*
# live draw as ``load% × nominal``). Both are mapped here so the raw
# source is preserved in the DB; the derived column lives in snapshot().
_NUT_TO_COLUMN: Dict[str, str] = {
    "input.voltage":         "ups_input_voltage_v",
    "input.frequency":       "ups_input_freq_hz",
    "output.voltage":        "ups_output_voltage_v",
    "ups.load":              "ups_load_pct",
    "ups.realpower":         "ups_realpower_w",
    "ups.realpower.nominal": "ups_realpower_nominal_w",
    "battery.charge":        "ups_battery_charge_pct",
    "battery.runtime":       "ups_battery_runtime_s",
}

# All numeric column names — used to coerce parsed strings to float and
# to enumerate the snapshot shape so missing keys become explicit Nones.
_NUMERIC_COLUMNS: List[str] = list(_NUT_TO_COLUMN.values())

# Derived columns added at snapshot() time, not stored in _latest. Kept
# as a manifest so downstream code that enumerates the snapshot shape
# sees a stable contract.
_DERIVED_COLUMNS: List[str] = [
    "ups_realpower_est_w",
    "ups_apparent_nominal_va",
    "ups_apparent_est_va",
]

# Status flag tokens NUT emits inside ups.status. We only key off "OB"
# (on battery) for transfer detection. The rest stay in the raw string
# for post-run analysis.
_ON_BATTERY_TOKEN = "OB"

# Model-VA parser bounds. CyberPower + APC + Eaton + Tripp Lite name
# models with the nominal VA embedded (CP1500PFCRM2U, Smart-UPS X 1500,
# Back-UPS 650, SMT2200, etc.). Small numbers like "2U" (rack height),
# firmware IDs, and PDU slot counts must not match; real UPS VA starts
# at ~250 VA (smallest OEM UPS on the market) and tops out around 20 kVA
# for anything a single workstation would see.
_MIN_PLAUSIBLE_VA = 200
_MAX_PLAUSIBLE_VA = 20000
_MODEL_NUMBER_RE = re.compile(r"\d+")


def _parse_model_va(model: Optional[str]) -> Optional[float]:
    """Extract nominal VA from a UPS model string.

    Picks the largest integer token in range ``[_MIN_PLAUSIBLE_VA,
    _MAX_PLAUSIBLE_VA]``. "Largest in range" covers the case where the
    model has multiple numbers (``CP1500PFCRM2U`` → 1500 not 2;
    ``Smart-UPS 750 XL`` → 750). Returns ``None`` when no plausible VA
    is embedded.
    """
    if not model:
        return None
    best: Optional[int] = None
    for match in _MODEL_NUMBER_RE.finditer(model):
        try:
            n = int(match.group(0))
        except ValueError:
            continue
        if n < _MIN_PLAUSIBLE_VA or n > _MAX_PLAUSIBLE_VA:
            continue
        if best is None or n > best:
            best = n
    return float(best) if best is not None else None


def _derive_est_power(
    load_pct: Optional[float], nominal: Optional[float],
) -> Optional[float]:
    """Return ``load_pct × nominal / 100`` when both inputs are present.

    Shared by both the watts (nominal_w × load%) and VA (nominal_va ×
    load%) derivations — ``ups.load`` is the same percent for both.
    """
    if load_pct is None or nominal is None:
        return None
    return float(load_pct) * float(nominal) / 100.0


def list_upses() -> List[str]:
    """Return UPS names visible to NUT, e.g. ``["cyberpower", "apc-smt"]``.

    Returns an empty list when ``upsc`` is missing, ``upsd`` is not
    running, or no UPSes are configured. Never raises — discovery
    failure is the same shape as "no UPS available."
    """
    if not command_exists("upsc"):
        return []
    rc, stdout, _ = run_capture(["upsc", "-l"], timeout=3)
    if rc != 0 or not stdout:
        return []
    names: List[str] = []
    for line in stdout.splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            names.append(name)
    return names


class UpsCollector:
    """Threaded ``upsc`` poller exposing a latest-value snapshot.

    Lifecycle mirrors ``MemoryBandwidthCollector``:
      * ``start()`` — spawn the thread; returns ``None`` on success or
        an error string when the prerequisites are absent. Caller logs
        the string as a run-level info event and continues.
      * ``stop()`` — signal shutdown, join with a 2 s timeout.
      * ``snapshot()`` — cheap dict read of the latest poll result.

    The poll interval defaults to 1 Hz to match the main telemetry
    loop. Probing is light (in-memory read from upsd over a local
    socket) so we do not need a slower cadence here.
    """

    def __init__(
        self,
        ups_name: str,
        interval_s: float = _DEFAULT_INTERVAL_S,
    ) -> None:
        self.ups_name = ups_name
        self.interval_s = interval_s
        self.available: bool = False

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._latest: Dict[str, Optional[float]] = {
            col: None for col in _NUMERIC_COLUMNS
        }
        self._status_raw: Optional[str] = None
        self._on_battery: Optional[int] = None
        self._transfer_count: int = 0
        # Tracks the previous status to detect OL→OB edges. None until
        # the first successful poll lands.
        self._prev_on_battery: Optional[int] = None
        # Nominal VA parsed from ups.model / device.model. Cached so the
        # regex runs once per model-string change, not every snapshot.
        self._apparent_nominal_va: Optional[float] = None
        self._model_raw: Optional[str] = None

    def start(self) -> Optional[str]:
        if not command_exists("upsc"):
            return "upsc not installed (NUT)"
        if not self.ups_name:
            return "no UPS name supplied"
        # Probe once synchronously so the caller knows whether the
        # named UPS actually responds before we commit to a thread.
        probe = _poll_ups(self.ups_name)
        if probe is None:
            return "ups '{}' did not respond to upsc".format(self.ups_name)
        self._apply_poll(probe)
        self.available = True
        self._thread = threading.Thread(
            target=self._loop, name="ups-collector", daemon=True,
        )
        self._thread.start()
        return None

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def snapshot(self) -> Dict[str, object]:
        """Return a flat snapshot of all UPS columns.

        Numeric columns return ``Optional[float]``. ``ups_status_raw``
        is a string or ``None``. ``ups_on_battery`` is 0/1 or ``None``.
        ``ups_transfer_count`` is always an int (starts at 0).

        Derived columns (computed fresh on every snapshot):
          * ``ups_realpower_est_w``     — ``load_pct × realpower_nominal_w / 100``
          * ``ups_apparent_nominal_va`` — parsed from ``ups.model`` / ``device.model``
          * ``ups_apparent_est_va``     — ``load_pct × apparent_nominal_va / 100``

        The est_w path is what makes CyberPower runs show live watts:
        the firmware never emits ``ups.realpower`` (only the rated plate
        via ``ups.realpower.nominal``), so we multiply by the live load%
        percentage NUT *does* expose. APC runs have ``ups_realpower_w``
        populated from the firmware directly, so ``ups_realpower_est_w``
        on APC is still computed as a cross-check but the tick display
        prefers the live value.
        """
        with self._lock:
            snap: Dict[str, object] = dict(self._latest)
            snap["ups_status_raw"] = self._status_raw
            snap["ups_on_battery"] = self._on_battery
            snap["ups_transfer_count"] = self._transfer_count
            load_pct = self._latest.get("ups_load_pct")
            nominal_w = self._latest.get("ups_realpower_nominal_w")
            snap["ups_realpower_est_w"] = _derive_est_power(load_pct, nominal_w)
            snap["ups_apparent_nominal_va"] = self._apparent_nominal_va
            snap["ups_apparent_est_va"] = _derive_est_power(
                load_pct, self._apparent_nominal_va,
            )
            return snap

    # ---- internals ----

    def _loop(self) -> None:
        while not self._stop.is_set():
            poll = _poll_ups(self.ups_name)
            if poll is not None:
                self._apply_poll(poll)
            self._stop.wait(timeout=self.interval_s)

    def _apply_poll(self, parsed: Dict[str, object]) -> None:
        on_batt: Optional[int]
        status = parsed.get("ups.status")
        if isinstance(status, str):
            on_batt = 1 if _ON_BATTERY_TOKEN in status.split() else 0
        else:
            on_batt = None

        # Prefer ups.model over device.model — NUT's upstream convention
        # is that ups.model is the UPS-level identifier while device.model
        # may duplicate or normalize it. Either is fine for the VA regex.
        model_raw = parsed.get("ups.model")
        if not isinstance(model_raw, str) or not model_raw:
            model_raw = parsed.get("device.model")
        if not isinstance(model_raw, str):
            model_raw = None

        with self._lock:
            for nut_key, col in _NUT_TO_COLUMN.items():
                val = parsed.get(nut_key)
                if isinstance(val, (int, float)):
                    self._latest[col] = float(val)
                # Else: leave the prior value in place (held-between-samples
                # UX matches mem_bandwidth and turbostat).
            if isinstance(status, str):
                self._status_raw = status
            # Edge-detect OL→OB. A first-ever poll on battery does NOT
            # count as a transfer (we have no prior state to transition
            # from); only a confirmed change does.
            if on_batt is not None:
                if (
                    self._prev_on_battery == 0
                    and on_batt == 1
                ):
                    self._transfer_count += 1
                self._prev_on_battery = on_batt
                self._on_battery = on_batt
            # Re-parse model only if it actually changed — the regex is
            # cheap but the extra state write is wasted under the lock
            # at 1 Hz every tick.
            if model_raw is not None and model_raw != self._model_raw:
                self._model_raw = model_raw
                self._apparent_nominal_va = _parse_model_va(model_raw)


# ---- pure-function poll + parser (exported for tests) ----


def _poll_ups(ups_name: str) -> Optional[Dict[str, object]]:
    """Run ``upsc`` once and return the parsed variable map.

    Returns ``None`` when the command fails (UPS offline, NUT down,
    name unknown). Caller treats ``None`` as "skip this tick."
    """
    rc, stdout, _ = run_capture(["upsc", ups_name], timeout=3)
    if rc != 0 or not stdout:
        return None
    return _parse_upsc_output(stdout)


def _parse_upsc_output(text: str) -> Dict[str, object]:
    """Parse ``upsc`` output into a typed variable map.

    NUT emits one ``key: value`` per line. Numeric variables are coerced
    to float when the value parses cleanly; non-numeric variables stay
    as the raw string (we only need ``ups.status`` as a string today,
    but keeping the rest as strings means a future caller can read
    ``ups.mfr`` / ``ups.model`` / ``device.serial`` without re-parsing).
    """
    parsed: Dict[str, object] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        # Numeric variables we care about live in _NUT_TO_COLUMN.
        # Coerce them to float when possible; otherwise drop. Other
        # variables stay as raw strings.
        if key in _NUT_TO_COLUMN:
            num = _to_float(value)
            if num is not None:
                parsed[key] = num
            continue
        parsed[key] = value
    return parsed


def _to_float(value: str) -> Optional[float]:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None
