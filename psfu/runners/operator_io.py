"""Operator I/O abstraction — TuiIO vs PlainIO.

The live phase loop in ``execute.py`` previously had ~50 ``if dashboard is
not None: ... else: ...`` branches. That made ``execute_phases`` both the
orchestrator AND the renderer — a violation of the project rule (keep
``execute.py`` closed for new features).

This module factors the two rendering modes into a common interface. The
loop owns the simulation (time, workers, samples, events); the IO object
owns how that state is shown and how operator input is collected.

Modes:

* ``TuiIO`` — wraps a live :class:`psfu.ui.tui.TuiDashboard` inside
  ``curses.wrapper``. Keyboard handled by the dashboard's own input poller.
  Used when ``--tui`` (the default).

* ``PlainIO`` — text-mode output to stdout, single-char reads via
  ``termios``/``select``. Used by ``--no-tui`` and CI.

Unattended (headless) runs are intentionally not supported yet; an
operator is expected to be present. Add a ``HeadlessIO`` when a profile
needs one.
"""

from __future__ import annotations

import curses
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

try:  # POSIX terminal control
    import select
    import termios
    import tty
except ImportError:  # Windows
    select = termios = tty = None  # type: ignore[assignment]

try:  # Windows console key polling
    import msvcrt
except ImportError:  # POSIX
    msvcrt = None  # type: ignore[assignment]


class _PlainInputPoller:
    """Non-blocking single-char reader for plain (no-TUI) mode.

    Sets terminal to cbreak on enter, restores on exit. Call
    :meth:`check` each tick to test for a keypress without blocking.
    """

    def __init__(self) -> None:
        self._old_settings: Optional[list] = None
        self._active = False

    def start(self) -> None:
        if termios is None:
            # Windows: msvcrt.kbhit() needs no terminal setup.
            self._active = msvcrt is not None and sys.stdin.isatty()
            return
        try:
            if sys.stdin.isatty():
                self._old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
                self._active = True
        except (OSError, termios.error):
            pass

    def stop(self) -> None:
        if self._old_settings is not None:
            try:
                termios.tcsetattr(
                    sys.stdin, termios.TCSADRAIN, self._old_settings
                )
            except (OSError, termios.error):
                pass
            self._old_settings = None
        self._active = False

    def check(self, *keys: str) -> Optional[str]:
        if not self._active:
            return None
        if termios is None:
            if msvcrt is not None and msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch.lower() in [k.lower() for k in keys]:
                    return ch
            return None
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch.lower() in [k.lower() for k in keys]:
                    return ch
        except (OSError, ValueError):
            pass
        return None


class OperatorIO(ABC):
    """Abstract operator-facing I/O. Subclasses: :class:`TuiIO`,
    :class:`PlainIO`.

    The phase loop calls these methods; it does not know or care which
    mode it's running in.
    """

    # -- Lifecycle ----------------------------------------------------
    @abstractmethod
    def start(self) -> None:
         ...

    @abstractmethod
    def stop(self) -> None:
         ...

    # -- Lobby / run gate ---------------------------------------------
    def wait_for_start(self) -> bool:
        """Block until operator starts the run. Return True if aborted."""
        return False

    def selected_duts(self) -> Optional[Dict[str, bool]]:
        """DUT selection from the lobby, or None if not applicable (plain)."""
        return None

    # -- Display ------------------------------------------------------
    @abstractmethod
    def log_action(self, message: str) -> None:
         ...

    @abstractmethod
    def show_event(
        self, severity: str, phase: str, display: str
    ) -> None:
         ...

    @abstractmethod
    def start_phase(
        self, phase: Any, phase_idx: int, phase_modes: Dict[str, str]
    ) -> None:
         ...

    @abstractmethod
    def set_phase_extra(self, text: str) -> None:
         ...

    def scrollback_note(self, text: str) -> None:
        """Emit a break-out line that lands in scrollback. TUI is a no-op
        (the overlay already shows it); plain prints a marker."""
        return None

    @abstractmethod
    def update_sample(
        self,
        sample: Dict[str, Any],
        *,
        elapsed_phase: float,
        phase_duration_sec: int,
        cpu_state: str,
        gpu_state: str,
        mem_state: str,
        storage_state: str,
    ) -> None:
         ...

    def end_phase(self) -> None:
        """Called after a phase completes (before inter-phase prompts)."""
        return None

    # -- Input --------------------------------------------------------
    def poll(self) -> None:
        """Per-tick input poll. Non-blocking."""
        return None

    def check_abort(self) -> bool:
        return False

    def check_pause(self) -> bool:
        return False

    def check_skip(self) -> bool:
        return False

    def reset_skip(self) -> None:
        return None

    def wait_during_pause(self) -> None:
        """Called in a loop while paused. Renders and polls input."""
        return None

    # -- Blocking prompt ---------------------------------------------
    def run_blocking_prompt(
        self, prompt_fn: Callable[[Any], Dict[str, Any]], args: Any
    ) -> Dict[str, Any]:
        """Run a stdin-consuming prompt. TUI must suspend curses first."""
        return prompt_fn(args)


class TuiIO(OperatorIO):
    """TUI I/O backed by a live :class:`TuiDashboard`.

    Construction is a two-step dance because the dashboard must be built
    inside ``curses.wrapper(fn)``. Caller passes the already-constructed
    dashboard + stdscr.
    """

    def __init__(
        self, dashboard: Any, stdscr: "curses._CursesWindow"
    ) -> None:
        self._d = dashboard
        self._stdscr = stdscr

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def wait_for_start(self) -> bool:
        self._d.wait_for_start()
        return bool(self._d.user_abort)

    def selected_duts(self) -> Optional[Dict[str, bool]]:
        return self._d.selected_duts()

    def log_action(self, message: str) -> None:
        self._d.log_event("info", "run", message)

    def show_event(
        self, severity: str, phase: str, display: str
    ) -> None:
        self._d.log_event(severity, phase, display)

    def start_phase(
        self, phase: Any, phase_idx: int, phase_modes: Dict[str, str]
    ) -> None:
        self._d.start_phase(phase, phase_idx, phase_modes)

    def set_phase_extra(self, text: str) -> None:
        self._d.set_phase_extra(text)

    def update_sample(
        self,
        sample: Dict[str, Any],
        *,
        elapsed_phase: float,
        phase_duration_sec: int,
        cpu_state: str,
        gpu_state: str,
        mem_state: str,
        storage_state: str,
    ) -> None:
        self._d.update_sample(
            sample,
            elapsed_phase=elapsed_phase,
            cpu_state=cpu_state,
            gpu_state=gpu_state,
            mem_state=mem_state,
            storage_state=storage_state,
        )

    def poll(self) -> None:
        self._d._poll_input()

    def check_abort(self) -> bool:
        return bool(self._d.user_abort)

    def check_pause(self) -> bool:
        return bool(self._d.user_paused)

    def check_skip(self) -> bool:
        return bool(self._d.user_skip_phase)

    def reset_skip(self) -> None:
        self._d.user_skip_phase = False

    def wait_during_pause(self) -> None:
        self._d._poll_input()
        self._d._render()

    def run_blocking_prompt(
        self, prompt_fn: Callable[[Any], Dict[str, Any]], args: Any
    ) -> Dict[str, Any]:
        # Must suspend curses so the prompt can read stdin cleanly.
        curses.def_prog_mode()
        curses.endwin()
        try:
            return prompt_fn(args)
        finally:
            curses.reset_prog_mode()
            self._stdscr.refresh()


class PlainIO(OperatorIO):
    """Plain-text I/O to stdout. 1 Hz status line on ``\\r``, periodic
    newline so scrollback records the trajectory. Keyboard via cbreak.
    """

    _NEWLINE_INTERVAL = 10.0

    def __init__(self) -> None:
        self._poller = _PlainInputPoller()
        self._last_newline: float = 0.0
        self._phase_duration_sec: int = 0
        self._skip_requested: bool = False
        self._abort_requested: bool = False

    def start(self) -> None:
        self._poller.start()

    def stop(self) -> None:
        self._poller.stop()

    def wait_for_start(self) -> bool:
        return False

    def selected_duts(self) -> Optional[Dict[str, bool]]:
        return None

    def log_action(self, message: str) -> None:
        print("- {}".format(message))

    def show_event(
        self, severity: str, phase: str, display: str
    ) -> None:
        # Plain mode keeps the status line dense; events go only to DB.
        # Preserving prior behaviour to avoid a wall-of-text regression.
        return None

    def start_phase(
        self, phase: Any, phase_idx: int, phase_modes: Dict[str, str]
    ) -> None:
        print(
            "\n-- {} - {}s - cpu:{} gpu:{} mem:{} sto:{} "
            "-- [n=skip phase, q=abort]".format(
                phase.name,
                phase.duration_sec,
                phase_modes["cpu"],
                phase_modes["gpu"],
                phase_modes["mem"],
                phase_modes["storage"],
            )
        )
        self._last_newline = time.monotonic()
        self._phase_duration_sec = phase.duration_sec
        self._skip_requested = False

    def set_phase_extra(self, text: str) -> None:
        # No-op on plain: TUI gets a status-line overlay, but pulse toggles
        # fire every ~2 s and would flood scrollback. Call sites that need
        # a plain-mode breakout line (e.g. ramp steps) use scrollback_note.
        return None

    def scrollback_note(self, text: str) -> None:
        print("\n  [ramp] {}".format(text))
        self._last_newline = time.monotonic()

    def update_sample(
        self,
        sample: Dict[str, Any],
        *,
        elapsed_phase: float,
        phase_duration_sec: int,
        cpu_state: str,
        gpu_state: str,
        mem_state: str,
        storage_state: str,
    ) -> None:
        def _fv(val: object, fmt: str, fallback: str) -> str:
            try:
                return fmt.format(float(val)) if val is not None else fallback
            except (TypeError, ValueError):
                return fallback

        cpu_pct = _fv(sample["cpu_load_pct"], "{:3.0f}%", "  - ")
        cpu_mhz = _fv(sample.get("cpu_avg_mhz"), "/{:4.0f}M", "     ")
        rapl_w = _fv(sample.get("cpu_pkg_watt"), "/{:4.0f}w", "     ")
        gpu_w = _fv(sample["gpu_power_w"], "{:4.0f}w", "   - ")
        gpu_t = _fv(sample.get("gpu_temp_c"), "/{:3.0f}\u00b0", "    ")
        gpu_sm = _fv(sample.get("gpu_clock_sm_mhz"), "/{:4.0f}M", "     ")
        mem_pct = _fv(sample.get("mem_used_pct"), "{:3.0f}%", "  - ")
        v12 = _fv(sample["rail_12v"], "{:.2f}V", " -   ")
        v5 = _fv(sample.get("rail_5v"), "{:.2f}V", " -  ")
        v33 = _fv(sample.get("rail_33v"), "{:.2f}V", " -  ")
        # UPS wattage: prefer firmware-live `ups_realpower_w` (APC emits
        # it). Fall back to the derived `ups_realpower_est_w` (load% ×
        # nominal_w / 100) for vendors that omit the live variable —
        # CyberPower CP1500 firmware in particular. Same math PowerPanel
        # uses on Windows, so the tick has a wattage number on both
        # benches instead of a dash.
        ups_live_w = sample.get("ups_realpower_w")
        ups_w_val = ups_live_w if ups_live_w is not None else sample.get("ups_realpower_est_w")
        ups_w = _fv(ups_w_val, "{:4.0f}W", "   - ")
        ups_va = _fv(sample.get("ups_apparent_est_va"), "{:5.0f}VA", "    - ")
        remaining = max(0, phase_duration_sec - int(elapsed_phase))

        line = (
            "  t-{:4d}s  cpu {}{}{}  gpu {}{}{}  mem {}"
            "  12V {} 5V {} 3.3V {}"
            "  ups {} {}"
            "  {} {} {} {}".format(
                remaining,
                cpu_pct, cpu_mhz, rapl_w,
                gpu_w, gpu_t, gpu_sm,
                mem_pct,
                v12, v5, v33,
                ups_w, ups_va,
                cpu_state, gpu_state,
                mem_state, storage_state,
            )
        )
        now = time.monotonic()
        if now - self._last_newline >= self._NEWLINE_INTERVAL:
            print(line, flush=True)
            self._last_newline = now
        else:
            print(line, end="\r", flush=True)

    def end_phase(self) -> None:
        print("")

    def poll(self) -> None:
        key = self._poller.check("n", "q")
        if not key:
            return
        low = key.lower()
        if low == "q":
            self._abort_requested = True
        elif low == "n":
            self._skip_requested = True

    def check_abort(self) -> bool:
        return self._abort_requested

    def check_pause(self) -> bool:
        return False

    def check_skip(self) -> bool:
        return self._skip_requested

    def reset_skip(self) -> None:
        self._skip_requested = False

    def wait_during_pause(self) -> None:
        return None

    def run_blocking_prompt(
        self, prompt_fn: Callable[[Any], Dict[str, Any]], args: Any
    ) -> Dict[str, Any]:
        return prompt_fn(args)
