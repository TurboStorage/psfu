"""Per-DUT phase runners — polymorphic dispatch for stress modes.

The live loop in :mod:`psfu.runners.execute` previously had per-DUT
``if phase_modes[X] == "pulse"`` / ``if phase_modes[X] == "ramp"``
branches at every orchestration step (start, per-tick, pause, resume,
teardown). Each new mode piled on more branches across four DUT axes
(cpu / gpu / mem / storage), and the coupling made it hard to add a
mode without touching every site.

This module factors each DUT-axis mode into a :class:`DutRunner`
subclass that owns its own state and orchestration. The per-tick loop
becomes::

    for runner in runners.values():
        runner.tick(now, elapsed)

Runner responsibilities:

* own a :class:`StressWorker` if the mode uses one
* mode-specific state (pulse toggle timers, ramp step index)
* per-tick orchestration (toggle pulse, advance ramp step)
* pause/resume hooks that shift internal timers
* worker-exit surfacing

Initial runner set: :class:`IdleRunner` (no-op), :class:`SteadyRunner`
(stress/idle with a single long-running worker), :class:`PulseRunner`
(SIGSTOP/SIGCONT pulse), :class:`RampRunner` (respawn worker at
fractional load steps). Add new modes here, not in ``execute.py``.

The pause/stop design rule applies: ``StressWorker.pause()/resume()/stop()``
ordering is delicate (SIGSTOP -> SIGCONT -> SIGTERM), and runners MUST
go through those methods, never raw signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

from psfu.runners.orchestrator import (
    PULSE_OFF_SEC,
    PULSE_ON_SEC,
    RAMP_STEPS,
    build_ramp_cpu_cmd,
    build_ramp_mem_cmd,
    cpu_worker_count,
    mem_target_pct,
)
from psfu.storage.cliff import analyze_write_cliff, write_cliff_analysis
from psfu.runners.workers.base import StressWorker
from psfu.runners.workers.fio_runner import FioRunner


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


EmitEvent = Callable[..., None]
"""Signature of the phase loop's emit_event closure. Positional args are
``(severity, phase, kind, message)`` and the keyword ``tui_text`` is
optional — see ``execute_phases`` in ``execute.py``."""


class StatusReporter(Protocol):
    """Subset of :class:`psfu.runners.operator_io.OperatorIO` the runners
    need. Typed as a Protocol so we don't circular-import OperatorIO."""

    def set_phase_extra(self, text: str) -> None: ...

    def scrollback_note(self, text: str) -> None: ...


class WorkerLike(Protocol):
    """Subset of the :class:`StressWorker` / :class:`FioRunner` interface
    that :class:`DutRunner` calls into. Kept deliberately minimal so a new
    worker only has to match ``start / pause / resume / stop / state`` to
    plug into the DUT axis without subclassing ``StressWorker``.

    The pause/stop design rule applies: whoever implements this must honour the
    SIGSTOP → SIGCONT → SIGTERM ordering in ``stop()`` if they use POSIX
    signal process-group control."""

    def start(self) -> Optional[str]: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def stop(self) -> None: ...

    def state(self) -> str: ...


# ---------------------------------------------------------------------------
# Base protocol
# ---------------------------------------------------------------------------


class DutRunner:
    """Abstract base for one DUT axis' per-phase orchestration.

    Lifecycle:
        ``prepare(started_monotonic)`` once at phase start.
        ``tick(now_monotonic, elapsed)`` each 1 Hz tick.
        ``pause() / resume(pause_dur)`` during operator pause.
        ``teardown()`` after the phase ends.

    Concrete subclasses override the methods they need. All are no-ops
    by default so IdleRunner needs almost no code.
    """

    label: str = ""  # "cpu" / "gpu" / "mem" / "storage"

    def __init__(self, label: str) -> None:
        self.label = label
        self.worker: Optional[WorkerLike] = None

    # -- lifecycle ---------------------------------------------------
    def prepare(self, started_monotonic: float) -> None:
        return None

    def tick(self, now: float, elapsed: float) -> None:
        return None

    def pause(self) -> None:
        if self.worker is not None:
            self.worker.pause()

    def resume(self, pause_dur: float) -> None:
        if self.worker is not None:
            self.worker.resume()

    def teardown(self) -> None:
        if self.worker is not None:
            self.worker.stop()

    # -- introspection -----------------------------------------------
    def state(self) -> str:
        if self.worker is None:
            return "idle"
        return self.worker.state()

    def current_cmd(self) -> Optional[List[str]]:
        """Current command line the worker is running, for
        ``check_worker_exit`` diagnostics. Ramp runners respawn with new
        commands; plain/pulse runners keep a fixed one."""
        return None


# ---------------------------------------------------------------------------
# Concrete runners
# ---------------------------------------------------------------------------


class IdleRunner(DutRunner):
    """No worker, no orchestration. Phase modes ``idle`` and anything that
    resolved to an empty command."""

    def __init__(self, label: str) -> None:
        super().__init__(label)


class SteadyRunner(DutRunner):
    """Single long-running worker. Covers ``stress`` and any mode that
    compiles down to one StressWorker with a fixed command."""

    def __init__(
        self,
        label: str,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        emit_event: Optional[EmitEvent] = None,
        phase_name: str = "",
    ) -> None:
        super().__init__(label)
        self._cmd = cmd
        self.worker = StressWorker(cmd, cwd=cwd, label=label)
        self._emit_event = emit_event
        self._phase_name = phase_name

    def prepare(self, started_monotonic: float) -> None:
        err = self.worker.start() if self.worker else None
        if err and self._emit_event is not None:
            self._emit_event(
                "warn",
                self._phase_name,
                "{}_worker_start".format(self.label),
                err,
                tui_text="{}_worker_start: {}".format(self.label, err),
            )

    def current_cmd(self) -> Optional[List[str]]:
        return self._cmd


class FioDutRunner(DutRunner):
    """Storage DUT backed by :class:`FioRunner` (S4+).

    Unlike :class:`SteadyRunner`, the worker is not a ``StressWorker`` —
    it's an ``FioRunner`` that wraps the vendored fio binary with a
    readonly-enforced jobfile and a JSON+ parser that fills
    ``summary_rows`` after ``stop()``. The runner only has to forward
    lifecycle calls; pulse/ramp do not apply to the storage axis."""

    def __init__(
        self,
        fio: FioRunner,
        *,
        emit_event: EmitEvent,
        status: Optional[StatusReporter] = None,
        phase_name: str,
    ) -> None:
        super().__init__("storage")
        self.worker = fio
        self._fio = fio
        self._emit = emit_event
        self._status = status
        self._phase_name = phase_name
        self._last_status_rows = 0

    def prepare(self, started_monotonic: float) -> None:
        if self._fio is None:
            return
        err = self._fio.start()
        if err:
            # Use the same template form as SteadyRunner/PulseRunner/RampRunner
            # (``"{}_worker_start".format(label)``) so the Rule 1 verdict
            # consumer check in test_verdict.py matches via the template path.
            self._emit(
                "warn",
                self._phase_name,
                "{}_worker_start".format(self.label),
                err,
                tui_text="{}_worker_start: {}".format(self.label, err),
            )

    def tick(self, now: float, elapsed: float) -> None:
        if self._fio is None or self._status is None:
            return
        row_count = len(self._fio.interval_rows)
        if row_count < 20 or row_count == self._last_status_rows:
            return
        self._last_status_rows = row_count
        analysis = analyze_write_cliff(
            self._fio.interval_rows,
            phase_name=self._phase_name,
            device=self._fio.device,
        )
        if analysis is None:
            return
        classification = analysis.get("classification") or {}
        label = str(classification.get("label") or "profiling")
        ratio = float(analysis.get("cliff_ratio") or 0.0)
        self._status.set_phase_extra("{} r{:.2f}".format(label, ratio))

    def teardown(self) -> None:
        if self._fio is None:
            return
        self._fio.stop()
        if "cliff" in self._phase_name.lower():
            analysis = analyze_write_cliff(
                self._fio.interval_rows,
                phase_name=self._phase_name,
                device=self._fio.device,
                run_dir=self._fio.run_dir,
            )
            if analysis is not None:
                analysis_path = self._fio.output_dir / "write-cliff-analysis.json"
                write_cliff_analysis(analysis_path, analysis)
                if analysis.get("severe"):
                    baseline_cmp = analysis.get("baseline_comparison") or {}
                    workload_stats = analysis.get("workload_stats") or {}
                    baseline_suffix = ""
                    if isinstance(baseline_cmp, dict) and baseline_cmp:
                        baseline_suffix = (
                            " Baseline floor {:.3f} (p5 {:.3f} - 3σ {:.3f}); below_floor={}."
                        ).format(
                            float(baseline_cmp.get("severe_floor_ratio") or 0.0),
                            float(baseline_cmp.get("cliff_ratio_p5") or 0.0),
                            float(baseline_cmp.get("sigma") or 0.0),
                            "yes" if baseline_cmp.get("below_severe_floor") else "no",
                        )
                    workload_suffix = ""
                    if isinstance(workload_stats, dict) and workload_stats:
                        classification = analysis.get("classification") or {}
                        workload_suffix = (
                            " Class {}. Post-peak CV {:.3f}, tail CV {:.3f}, major drops {}, major rises {}."
                        ).format(
                            str(classification.get("slug") or "unknown"),
                            float(workload_stats.get("post_peak_write_bw_cv") or 0.0),
                            float(workload_stats.get("tail_window_write_bw_cv") or 0.0),
                            int(workload_stats.get("major_drop_count") or 0),
                            int(workload_stats.get("major_rise_count") or 0),
                        )
                    baseline_backed = bool(
                        isinstance(baseline_cmp, dict) and baseline_cmp.get("below_severe_floor")
                    )
                    base_message = (
                        "{} write cliff in phase '{}': tail 10s median {:.0f} KiB/s vs peak 10s window "
                        "{:.0f} KiB/s (ratio {:.3f}, generic severe threshold {:.3f}).{}{}"
                    ).format(
                        self._fio.device,
                        self._phase_name,
                        float(analysis["tail_window_median_kbps"]),
                        float(analysis["peak_window"]["avg_write_bw_kbps"]),
                        float(analysis["cliff_ratio"]),
                        float(analysis["severe_threshold_ratio"]),
                        baseline_suffix,
                        workload_suffix,
                    )
                    if baseline_backed:
                        self._emit(
                            "warning",
                            self._phase_name,
                            "fio_write_cliff_severe",
                            base_message + " Baseline-backed severe cliff scored as storage WARN.",
                        )
                    else:
                        self._emit(
                            "info",
                            self._phase_name,
                            "fio_write_cliff_observed",
                            base_message + " No matching per-PN baseline claim threshold crossed; informational only.",
                        )
        # S5 — emit fio_verify_mismatch for any CRC32c verify errors found in
        # the post-stop summary. verify_mismatches > 0 = hard FAIL (silent data
        # corruption, bad blocks, or FTL defects). Emitted here so the verdict
        # engine scores it via the storage_fail_events path in rules.py.
        for row in self._fio.summary_rows:
            mismatches = row.get("verify_mismatches") or 0
            if mismatches > 0:
                self._emit(
                    "crit",
                    self._phase_name,
                    "fio_verify_mismatch",
                    "fio CRC32c verify mismatch: {} error(s) on {} in phase '{}'. "
                    "Silent data corruption or bad blocks detected — hard storage FAIL.".format(
                        mismatches, self._fio.device, self._phase_name,
                    ),
                )

    def current_cmd(self) -> Optional[List[str]]:
        # FioRunner builds its own command internally; surface the
        # job file + device for diagnostic logs so check_worker_exit
        # and post-run summaries have something human-readable.
        if self._fio is None:
            return None
        return ["fio", str(self._fio.job_file), "--filename={}".format(self._fio.device)]


@dataclass
class _PulseTimings:
    on_sec: float
    off_sec: float


class FioPulseRunner(FioDutRunner):
    """Pulse-mode storage runner backed by :class:`FioRunner`.

    fio is still the worker, but unlike :class:`FioDutRunner` the phase
    alternates between SIGCONT and SIGSTOP using FioRunner.pause/resume so a
    storage ``pulse`` phase stays truthful on the NVMe+fio path.
    """

    def __init__(
        self,
        fio: FioRunner,
        *,
        on_sec: float,
        off_sec: float,
        inverted: bool,
        emit_event: EmitEvent,
        status: StatusReporter,
        phase_name: str,
        dry_run: bool = False,
    ) -> None:
        super().__init__(fio, emit_event=emit_event, status=status, phase_name=phase_name)
        self._t = _PulseTimings(on_sec, off_sec)
        self._is_on = not inverted
        self._state_end: float = 0.0
        self._status = status
        self._dry_run = dry_run

    def prepare(self, started_monotonic: float) -> None:
        super().prepare(started_monotonic)
        if self._dry_run or self.worker is None:
            return
        if self._is_on:
            self._state_end = started_monotonic + self._t.on_sec
        else:
            self.worker.pause()
            self._state_end = started_monotonic + self._t.off_sec

    def tick(self, now: float, elapsed: float) -> None:
        if self._dry_run or self.worker is None:
            return
        if now < self._state_end:
            return
        if self._is_on:
            self.worker.pause()
            self._is_on = False
            self._state_end = now + self._t.off_sec
            direction = "off"
        else:
            self.worker.resume()
            self._is_on = True
            self._state_end = now + self._t.on_sec
            direction = "on"
        self._emit(
            "info",
            self._phase_name,
            "pulse_cycle",
            "STORAGE load {}".format(direction),
            tui_text="pulse: STORAGE {}".format(direction),
        )
        icon = "\u25b6" if direction == "on" else "\u25cf"
        self._status.set_phase_extra(
            "{} pulse STORAGE {}".format(icon, direction.upper())
        )

    def resume(self, pause_dur: float) -> None:
        super().resume(pause_dur)
        self._state_end += pause_dur


class PulseRunner(DutRunner):
    """SIGSTOP/SIGCONT pulse. Starts in an ``on`` state unless the phase
    requests ``pulse_invert`` for this DUT, in which case the worker is
    paused before the first tick. See the pause/resume
    ordering."""

    def __init__(
        self,
        label: str,
        cmd: List[str],
        *,
        on_sec: float,
        off_sec: float,
        inverted: bool,
        cwd: Optional[str] = None,
        emit_event: EmitEvent,
        status: StatusReporter,
        phase_name: str,
        dry_run: bool = False,
    ) -> None:
        super().__init__(label)
        self._cmd = cmd
        self.worker = StressWorker(cmd, cwd=cwd, label=label)
        self._t = _PulseTimings(on_sec, off_sec)
        self._is_on = not inverted
        self._state_end: float = 0.0
        self._emit = emit_event
        self._status = status
        self._phase_name = phase_name
        self._dry_run = dry_run

    def prepare(self, started_monotonic: float) -> None:
        assert self.worker is not None
        err = self.worker.start()
        if err:
            self._emit(
                "warn",
                self._phase_name,
                "{}_worker_start".format(self.label),
                err,
                tui_text="{}_worker_start: {}".format(self.label, err),
            )
        if self._dry_run:
            return
        if self._is_on:
            self._state_end = started_monotonic + self._t.on_sec
        else:
            self.worker.pause()
            self._state_end = started_monotonic + self._t.off_sec

    def tick(self, now: float, elapsed: float) -> None:
        if self._dry_run or self.worker is None:
            return
        if now < self._state_end:
            return
        if self._is_on:
            self.worker.pause()
            self._is_on = False
            self._state_end = now + self._t.off_sec
            direction = "off"
        else:
            self.worker.resume()
            self._is_on = True
            self._state_end = now + self._t.on_sec
            direction = "on"
        big = self.label.upper()
        self._emit(
            "info",
            self._phase_name,
            "pulse_cycle",
            "{} load {}".format(big, direction),
            tui_text="pulse: {} {}".format(big, direction),
        )
        icon = "\u25b6" if direction == "on" else "\u25cf"
        self._status.set_phase_extra(
            "{} pulse {} {}".format(icon, big, direction.upper())
        )

    def resume(self, pause_dur: float) -> None:
        super().resume(pause_dur)
        # Pulse-cycle deadlines drift by the pause duration so the on/off
        # ratio is preserved end-to-end.
        self._state_end += pause_dur

    def current_cmd(self) -> Optional[List[str]]:
        return self._cmd


class RampRunner(DutRunner):
    """Respawn the worker at each ramp step with a heavier command.

    Ramp is time-sliced into :data:`RAMP_STEPS` equal intervals. CPU and
    MEM ramp on separate runners but share the same ``ramp_step_dur``,
    so they advance together. Each runner announces its own step via
    ``emit_event`` — the old execute.py merged them into one message, but
    keeping them separate is the natural cost of the split and keeps the
    runner independent (no coupling to other axes).
    """

    def __init__(
        self,
        label: str,
        *,
        phase: object,
        stress_path: str,
        step_dur: int,
        fractions: List[float],
        emit_event: EmitEvent,
        status: StatusReporter,
        phase_name: str,
        dry_run: bool = False,
    ) -> None:
        super().__init__(label)
        self._phase = phase
        self._stress_path = stress_path
        self._step_dur = step_dur
        self._fractions = fractions
        self._current_step: int = 0
        self._emit = emit_event
        self._status = status
        self._phase_name = phase_name
        self._dry_run = dry_run
        self._current_cmd: Optional[List[str]] = None
        if not dry_run:
            self._current_cmd = self._build_cmd(fractions[0])
            self.worker = StressWorker(self._current_cmd, label=label)

    def _build_cmd(self, frac: float) -> List[str]:
        if self.label == "cpu":
            return build_ramp_cpu_cmd(
                self._stress_path, self._step_dur, frac
            )
        if self.label == "mem":
            return build_ramp_mem_cmd(
                self._stress_path, self._phase, self._step_dur, frac
            )
        raise ValueError(
            "RampRunner only supports cpu/mem labels, got {!r}".format(self.label)
        )

    def _describe(self, frac: float) -> str:
        if self.label == "cpu":
            return "CPU {} workers".format(cpu_worker_count(frac))
        return "MEM {}%".format(mem_target_pct(self._phase, frac))

    def prepare(self, started_monotonic: float) -> None:
        if self.worker is None:
            return
        err = self.worker.start()
        if err:
            self._emit(
                "warn",
                self._phase_name,
                "{}_worker_start".format(self.label),
                err,
            )
        self._announce_step(self._current_step)

    def tick(self, now: float, elapsed: float) -> None:
        if self._dry_run:
            return
        step_for_now = min(
            RAMP_STEPS - 1, int(elapsed / self._step_dur)
        )
        if step_for_now <= self._current_step:
            return
        self._current_step = step_for_now
        frac = self._fractions[step_for_now]
        # Respawn with the heavier command.
        if self.worker is not None:
            self.worker.stop()
        self._current_cmd = self._build_cmd(frac)
        self.worker = StressWorker(self._current_cmd, label=self.label)
        err = self.worker.start()
        if err:
            self._emit(
                "warn",
                self._phase_name,
                "{}_worker_start".format(self.label),
                err,
            )
        self._announce_step(step_for_now)

    def _announce_step(self, step_idx: int) -> None:
        frac = self._fractions[step_idx]
        msg = "ramp step {}/{} -- {:.0%} load ({})".format(
            step_idx + 1, RAMP_STEPS, frac, self._describe(frac)
        )
        self._emit(
            "info",
            self._phase_name,
            "ramp_step",
            msg,
            tui_text="ramp: {}".format(msg),
        )
        self._status.set_phase_extra(
            "ramp {}/{} {:.0%}".format(step_idx + 1, RAMP_STEPS, frac)
        )
        self._status.scrollback_note(msg)

    def current_cmd(self) -> Optional[List[str]]:
        return self._current_cmd


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_dut_runner(
    label: str,
    mode: str,
    cmd: Optional[List[str]],
    *,
    phase: object,
    emit_event: EmitEvent,
    status: StatusReporter,
    stress_path_for_ramp: Optional[str] = None,
    ramp_step_dur: int = 0,
    ramp_fractions: Optional[List[float]] = None,
    cwd: Optional[str] = None,
    dry_run: bool = False,
) -> DutRunner:
    """Construct the :class:`DutRunner` for one DUT axis.

    ``cmd`` is the pre-built command for steady/pulse modes (see
    :func:`psfu.runners.orchestrator.build_phase_commands`); ramp modes
    build their own commands from ``stress_path_for_ramp``.
    """
    phase_name = getattr(phase, "name", "")

    if mode == "idle" or not cmd:
        return IdleRunner(label)

    if mode == "pulse":
        on_attr = getattr(phase, "{}_pulse_on_sec".format(label), None)
        off_attr = getattr(phase, "{}_pulse_off_sec".format(label), None)
        on_sec = on_attr if on_attr is not None else PULSE_ON_SEC
        off_sec = off_attr if off_attr is not None else PULSE_OFF_SEC
        inverted = label in getattr(phase, "pulse_invert", [])
        return PulseRunner(
            label,
            cmd,
            on_sec=on_sec,
            off_sec=off_sec,
            inverted=inverted,
            cwd=cwd,
            emit_event=emit_event,
            status=status,
            phase_name=phase_name,
            dry_run=dry_run,
        )

    if mode == "ramp":
        if stress_path_for_ramp is None or not ramp_fractions:
            # Ramp requires stress-ng; if unavailable fall back to steady.
            return SteadyRunner(
                label, cmd, cwd=cwd, emit_event=emit_event,
                phase_name=phase_name,
            )
        return RampRunner(
            label,
            phase=phase,
            stress_path=stress_path_for_ramp,
            step_dur=ramp_step_dur,
            fractions=ramp_fractions,
            emit_event=emit_event,
            status=status,
            phase_name=phase_name,
            dry_run=dry_run,
        )

    # stress / any other non-empty steady command
    return SteadyRunner(
        label, cmd, cwd=cwd, emit_event=emit_event, phase_name=phase_name,
    )
