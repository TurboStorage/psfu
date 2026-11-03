"""Per-tick health checks extracted from execute.py.

Each check reads mutable state from a subsystem-scoped state object
(GpuHealthState / CpuHealthState / ThermalHealthState / PowerHealthState)
and emits events via a callback. State objects travel across ticks and
phases so checks can track baselines and dedup warnings.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from psfu.runners.workers.base import StressWorker


@dataclass
class GpuHealthState:
    """GPU-intrinsic state: visibility, PCIe link, HW throttle, SM clock,
    ECC, retired pages."""

    gpu_seen_data: bool = False
    gpu_no_data_streak: int = 0
    gpu_disappeared_fired: bool = False
    gpu_smi_err_fired: bool = False
    hw_throttle_warned_phases: Set[str] = field(default_factory=set)
    pcie_baseline: Optional[List[Dict[str, Optional[int]]]] = None
    pcie_drop_fired: bool = False
    soak_sm_clock_samples: List[int] = field(default_factory=list)
    gpu_sm_clock_baseline: Optional[float] = None
    gpu_clock_throttle_warned_phases: Set[str] = field(default_factory=set)
    gpu_ecc_ce_last: int = 0
    gpu_ecc_ue_last: int = 0
    gpu_ecc_ue_fired: bool = False
    gpu_retired_pages_baseline: Optional[int] = None
    gpu_retired_pages_fired: bool = False


@dataclass
class CpuHealthState:
    """CPU frequency throttle tracking (turbostat Avg_MHz)."""

    soak_cpu_mhz_samples: List[float] = field(default_factory=list)
    stress_cpu_mhz_samples: List[float] = field(default_factory=list)
    cpu_mhz_baseline: Optional[float] = None
    cpu_throttle_warned_phases: Set[str] = field(default_factory=set)


@dataclass
class ThermalHealthState:
    """Cooling-system state: fan RPM soak baseline + spike dedup."""

    soak_fan_samples: List[float] = field(default_factory=list)
    fan_rpm_baseline: Optional[float] = None
    fan_spike_warned_phases: Set[str] = field(default_factory=set)


@dataclass
class PowerHealthState:
    """Rail voltages (cold/hot idle) + load-meter capture latch.

    Thermal-drift rule reads cold_idle_*v and hot_idle_*v from here — the
    phenomenon is thermal, the data is power-rail voltages.
    """

    cold_idle_12v: Optional[float] = None
    cold_idle_5v: Optional[float] = None
    hot_idle_12v: Optional[float] = None
    hot_idle_5v: Optional[float] = None
    load_meter_captured: bool = False


@dataclass
class PowerStatsState:
    """SPC state for 12V rail monitoring.

    S1 (Western Electric) and S2 (CUSUM) share the same baseline mean/sigma.
    Prefer the S3 bench calibration file when present; otherwise degrade to a
    within-run soak-derived baseline and surface that degraded mode once.
    """

    soak_rail_12v_samples: List[float] = field(default_factory=list)
    rail_12v_baseline_mean: Optional[float] = None
    rail_12v_baseline_sigma: Optional[float] = None
    baseline_source: Optional[str] = None  # "bench" | "within-run" | None
    baseline_unavailable_emitted: bool = False
    consecutive_below_count: int = 0
    consecutive_same_side_count: int = 0
    last_side: Optional[str] = None
    recent_below_2sigma: List[bool] = field(default_factory=list)
    runrule_fired_phases: Set[str] = field(default_factory=set)
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    cusum_fired_phases: Set[str] = field(default_factory=set)
    last_phase_name: Optional[str] = None


@dataclass
class StorageHealthState:
    """NVMe PCIe link-health state for storage DUTs."""

    pcie_downshift_streak: int = 0
    pcie_downshift_fired_phases: Set[str] = field(default_factory=set)
    pcie_capability_unknown_emitted: bool = False
    last_phase_name: Optional[str] = None
    # S6 — fio stall detection state (check_nvme_fio_stall).
    # fio_stall_startup_ticks: number of consecutive ticks where fio had
    #   no stanza yet (startup grace window; reset once we see a stanza).
    # fio_stall_last_snapshot: last observed (read_ios, write_ios) tuple.
    # fio_stall_streak: consecutive ticks where the snapshot equalled
    #   fio_stall_last_snapshot (= no IO progress).
    fio_stall_startup_ticks: int = 0
    fio_stall_last_snapshot: Optional[Tuple[int, int]] = None
    fio_stall_streak: int = 0
    fio_stall_fired_phases: Set[str] = field(default_factory=set)


# Type alias for the emit_event callback used by all checks.
EmitFn = Callable[..., None]


_POWER_BASELINE_MIN_SAMPLES = 60
_POWER_SIGMA_FLOOR_V = 0.001
_CUSUM_K = 0.5
_CUSUM_H = 5.0
_STORAGE_PCIE_DOWNSHIFT_TICKS = 3
# S6 — fio IO progress stalled for this many consecutive 1 Hz ticks = hard
# FAIL. 10 s tight enough to catch controller hangs (kernel nvme-io-timeout
# defaults at 30 s; PS-FU detects the user-visible stall ~20 s earlier),
# loose enough that queue-depth starvation at phase transitions doesn't
# trip. Tuned against empty tick — first real RMA catch will re-evaluate.
_FIO_STALL_TICKS = 10
# Grace window for fio to emit its first `--status-interval=1` stanza.
# 30 s covers slow-starting io_uring queues on older kernels; past that,
# the check gives up for this phase rather than scoring a false stall.
_FIO_STALL_STARTUP_GRACE_TICKS = 30


def _phase_has_power_stress(phase_modes: Dict[str, str], dut_enabled: Dict[str, bool]) -> bool:
    for label in ("cpu", "gpu", "mem", "storage"):
        if dut_enabled.get(label, True) and phase_modes.get(label) in ("stress", "pulse", "ramp"):
            return True
    return False


def _ensure_power_baseline(
    state: PowerStatsState,
    sample: Dict[str, object],
    phase_idx: int,
    phase_name: str,
    phase_modes: Dict[str, str],
    dut_enabled: Dict[str, bool],
    bench_baseline: Optional[Dict[str, float]],
    emit_event: EmitFn,
) -> bool:
    stress_phase = _phase_has_power_stress(phase_modes, dut_enabled)
    rail_12v = sample.get("rail_12v")
    if rail_12v is not None and phase_idx == 0 and state.rail_12v_baseline_mean is None:
        try:
            state.soak_rail_12v_samples.append(float(rail_12v))
        except (TypeError, ValueError):
            pass

    if state.rail_12v_baseline_mean is not None and state.rail_12v_baseline_sigma is not None:
        if (
            stress_phase
            and state.baseline_source == "within-run"
            and not state.baseline_unavailable_emitted
        ):
            emit_event(
                "info",
                phase_name,
                "baseline_unavailable",
                "Bench baseline unavailable; 12V statistical rules running in degraded within-run mode.",
                tui_text="12V SPC degraded: within-run baseline",
            )
            state.baseline_unavailable_emitted = True
        return True

    if bench_baseline:
        try:
            mean = float(bench_baseline["soak_12v_mean"])
            sigma = max(float(bench_baseline["bench_noise_floor_v"]), _POWER_SIGMA_FLOOR_V)
        except (KeyError, TypeError, ValueError):
            mean = None
            sigma = None
        if mean is not None and sigma is not None:
            state.rail_12v_baseline_mean = mean
            state.rail_12v_baseline_sigma = sigma
            state.baseline_source = "bench"
            return True

    if phase_idx == 0:
        if len(state.soak_rail_12v_samples) >= _POWER_BASELINE_MIN_SAMPLES:
            state.rail_12v_baseline_mean = statistics.mean(state.soak_rail_12v_samples)
            state.rail_12v_baseline_sigma = max(
                statistics.pstdev(state.soak_rail_12v_samples),
                _POWER_SIGMA_FLOOR_V,
            )
            state.baseline_source = "within-run"
            return True
        return False

    if stress_phase:
        if len(state.soak_rail_12v_samples) >= _POWER_BASELINE_MIN_SAMPLES:
            state.rail_12v_baseline_mean = statistics.mean(state.soak_rail_12v_samples)
            state.rail_12v_baseline_sigma = max(
                statistics.pstdev(state.soak_rail_12v_samples),
                _POWER_SIGMA_FLOOR_V,
            )
            state.baseline_source = "within-run"
            if not state.baseline_unavailable_emitted:
                emit_event(
                    "info",
                    phase_name,
                    "baseline_unavailable",
                    "Bench baseline unavailable; 12V statistical rules running in degraded within-run mode.",
                    tui_text="12V SPC degraded: within-run baseline",
                )
                state.baseline_unavailable_emitted = True
            return True
    return False


def _reset_power_stats_phase(state: PowerStatsState, phase_name: str) -> None:
    if state.last_phase_name == phase_name:
        return
    state.last_phase_name = phase_name
    state.consecutive_below_count = 0
    state.consecutive_same_side_count = 0
    state.last_side = None
    state.recent_below_2sigma.clear()
    state.cusum_pos = 0.0
    state.cusum_neg = 0.0


def _reset_storage_phase(state: StorageHealthState, phase_name: str) -> None:
    if state.last_phase_name == phase_name:
        return
    state.last_phase_name = phase_name
    state.pcie_downshift_streak = 0
    # S6 — every phase gets a fresh fio stall accounting window. A stall
    # spanning a phase boundary would fire on the new phase after another
    # _FIO_STALL_TICKS idle ticks, which is the behavior we want: each
    # phase scores independently.
    state.fio_stall_startup_ticks = 0
    state.fio_stall_last_snapshot = None
    state.fio_stall_streak = 0


def check_gpu_disappeared(
    state: GpuHealthState,
    collector: object,
    phase_name: str,
    dut_enabled: Dict[str, bool],
    emit_event: EmitFn,
) -> None:
    """GPU health check 1: NVML disappearance (3+ consecutive silent ticks)."""
    if not dut_enabled["gpu"]:
        return
    if not getattr(collector, "available", {}).get("nvidia-smi"):
        return
    if state.gpu_disappeared_fired:
        return

    if getattr(collector, "last_gpu_visible", False):
        state.gpu_seen_data = True
        state.gpu_no_data_streak = 0
    elif state.gpu_seen_data:
        state.gpu_no_data_streak += 1
        if state.gpu_no_data_streak >= 3:
            msg = (
                "GPU stopped responding to nvidia-smi after {} samples "
                "with data ({} consecutive silent ticks).".format(
                    state.gpu_no_data_streak, state.gpu_no_data_streak
                )
            )
            emit_event(
                "crit", phase_name, "nvidia_gpu_disappeared", msg,
                tui_text="nvidia_gpu_disappeared",
            )
            state.gpu_disappeared_fired = True


def check_gpu_hw_throttle(
    state: GpuHealthState,
    collector: object,
    phase_name: str,
    phase_modes: Dict[str, str],
    dut_enabled: Dict[str, bool],
    emit_event: EmitFn,
) -> None:
    """GPU health check 2: HW power brake slowdown (PSU-side signal)."""
    if not dut_enabled["gpu"]:
        return
    if not getattr(collector, "last_hw_throttle", False):
        return
    if phase_modes["gpu"] not in ("stress", "pulse", "ramp"):
        return
    if phase_name in state.hw_throttle_warned_phases:
        return

    emit_event(
        "warn", phase_name, "gpu_hw_power_throttle",
        "GPU HW power brake active during stress phase '{}'. "
        "Indicates PSU cannot sustain GPU power demand. "
        "Check 12V rail and PSU wattage headroom.".format(phase_name),
        tui_text="gpu_hw_power_throttle",
    )
    state.hw_throttle_warned_phases.add(phase_name)


def check_gpu_smi_err(
    state: GpuHealthState,
    collector: object,
    phase_name: str,
    dut_enabled: Dict[str, bool],
    emit_event: EmitFn,
) -> None:
    """GPU health check 2b: nvidia-smi ERR! field ((verdict rule))."""
    if not dut_enabled["gpu"]:
        return
    if not getattr(collector, "last_gpu_smi_err", False):
        return
    if state.gpu_smi_err_fired:
        return

    emit_event(
        "crit", phase_name, "gpu_smi_err",
        "nvidia-smi reported ERR! in one or more GPU fields during "
        "phase '{}'. Indicates a hardware fault or driver error "
        "on the GPU or its power delivery path.".format(phase_name),
        tui_text="gpu_smi_err: nvidia-smi ERR!",
    )
    state.gpu_smi_err_fired = True


def check_pcie_downgrade(
    state: GpuHealthState,
    collector: object,
    phase_name: str,
    dut_enabled: Dict[str, bool],
    emit_event: EmitFn,
) -> None:
    """GPU health check 3: PCIe link downgrade under load."""
    if not dut_enabled["gpu"]:
        return
    pcie_states = getattr(collector, "last_pcie_states", None)
    if not pcie_states:
        return
    if state.pcie_drop_fired:
        return

    if state.pcie_baseline is None:
        state.pcie_baseline = [dict(s) for s in pcie_states]
        return

    for gpu_idx, (cur, base) in enumerate(zip(pcie_states, state.pcie_baseline)):
        c_gen = cur.get("gen")
        b_gen = base.get("gen")
        c_width = cur.get("width")
        b_width = base.get("width")
        if c_gen is not None and b_gen is not None and c_gen < b_gen:
            emit_event(
                "warn", phase_name, "pcie_link_downgrade",
                "GPU {} PCIe link gen dropped from Gen{} to Gen{} "
                "under load. Indicates PSU voltage sag or bus "
                "instability.".format(gpu_idx, b_gen, c_gen),
                tui_text="pcie_link_downgrade",
            )
            state.pcie_drop_fired = True
            return
        if c_width is not None and b_width is not None and c_width < b_width:
            emit_event(
                "warn", phase_name, "pcie_link_downgrade",
                "GPU {} PCIe link width dropped from x{} to x{} "
                "under load. Indicates PSU voltage sag or bus "
                "instability.".format(gpu_idx, b_width, c_width),
                tui_text="pcie_link_downgrade",
            )
            state.pcie_drop_fired = True
            return


def check_fan_spike(
    state: ThermalHealthState,
    sample: Dict[str, object],
    phase_idx: int,
    phase_name: str,
    emit_event: EmitFn,
) -> None:
    """GPU health check 4: fan RPM spike relative to soak baseline."""
    fan_rpm_now = sample.get("fan_cpu_rpm")
    if phase_idx == 0:
        if fan_rpm_now is not None:
            try:
                state.soak_fan_samples.append(float(fan_rpm_now))
            except (TypeError, ValueError):
                pass
        return

    if state.fan_rpm_baseline is None and state.soak_fan_samples:
        state.fan_rpm_baseline = statistics.median(state.soak_fan_samples)

    if (
        state.fan_rpm_baseline is not None
        and state.fan_rpm_baseline > 0
        and phase_name not in state.fan_spike_warned_phases
        and fan_rpm_now is not None
    ):
        try:
            fan_now_f = float(fan_rpm_now)
            spike_thresh = max(state.fan_rpm_baseline * 2.5, state.fan_rpm_baseline + 2000)
            if fan_now_f > spike_thresh:
                emit_event(
                    "warn", phase_name, "fan_rpm_spike",
                    "CPU fan RPM spike: {:.0f} RPM during '{}' vs "
                    "soak baseline {:.0f} RPM. May indicate sustained "
                    "thermal stress.".format(fan_now_f, phase_name, state.fan_rpm_baseline),
                    tui_text="fan spike {:.0f}\u2192{:.0f} RPM".format(
                        state.fan_rpm_baseline, fan_now_f,
                    ),
                )
            state.fan_spike_warned_phases.add(phase_name)
        except (TypeError, ValueError):
            pass


def check_storage_pcie_link_downshift(
    state: StorageHealthState,
    collector: object,
    phase_name: str,
    phase_modes: Dict[str, str],
    dut_enabled: Dict[str, bool],
    emit_event: EmitFn,
) -> None:
    """Storage health check: sustained NVMe PCIe downshift for 3+ ticks."""
    _reset_storage_phase(state, phase_name)
    if not dut_enabled.get("storage", True):
        state.pcie_downshift_streak = 0
        return
    if phase_modes.get("storage") not in ("stress", "pulse", "ramp"):
        state.pcie_downshift_streak = 0
        return
    if not getattr(collector, "available", False):
        state.pcie_downshift_streak = 0
        return
    if phase_name in state.pcie_downshift_fired_phases:
        return

    current = collector.snapshot()
    capability = collector.first_capability()
    cur_gen = current.get("nvme_pcie_link_gen")
    cur_width = current.get("nvme_pcie_link_width")
    cap_gen = capability.get("max_gen")
    cap_width = capability.get("max_width")

    downshift_gen = cap_gen is not None and cur_gen is not None and cur_gen < cap_gen
    downshift_width = cap_width is not None and cur_width is not None and cur_width < cap_width
    capability_known = (cap_gen is not None and cur_gen is not None) or (cap_width is not None and cur_width is not None)

    if not capability_known:
        state.pcie_downshift_streak = 0
        if not state.pcie_capability_unknown_emitted:
            emit_event(
                "info",
                phase_name,
                "pcie_link_capability_unknown",
                "NVMe PCIe capability unavailable; downshift rule disabled for this run.",
                tui_text="nvme PCIe capability unknown",
            )
            state.pcie_capability_unknown_emitted = True
        return

    if downshift_gen or downshift_width:
        state.pcie_downshift_streak += 1
    else:
        state.pcie_downshift_streak = 0
        return

    if state.pcie_downshift_streak < _STORAGE_PCIE_DOWNSHIFT_TICKS:
        return

    details: List[str] = []
    if downshift_gen:
        details.append("Gen{} capability running at Gen{}".format(cap_gen, cur_gen))
    if downshift_width:
        details.append("x{} capability running at x{}".format(cap_width, cur_width))
    emit_event(
        "crit",
        phase_name,
        "pcie_link_downshift",
        "NVMe PCIe link downshift persisted for {} consecutive ticks in phase '{}': {}.".format(
            state.pcie_downshift_streak,
            phase_name,
            "; ".join(details),
        ),
        tui_text="nvme pcie_link_downshift",
    )
    state.pcie_downshift_fired_phases.add(phase_name)


def check_nvme_fio_stall(
    state: StorageHealthState,
    fio_runner: Optional[object],
    phase_name: str,
    phase_modes: Dict[str, str],
    dut_enabled: Dict[str, bool],
    emit_event: EmitFn,
) -> None:
    """S6 — controller-hang watchdog.

    Reads ``fio_runner.ios_progress_snapshot()`` (populated by the reader
    thread consuming ``--status-interval=1`` stanzas). If the cumulative
    ``(read_ios, write_ios)`` tuple is unchanged for
    ``_FIO_STALL_TICKS`` consecutive 1 Hz ticks in a stress-class phase,
    emits a ``fio_stall`` event which the verdict engine scores FAIL
    (``storage.fio_stall.fail``). Corroborates the kernel watcher's
    ``nvme_io_timeout`` / ``nvme_controller_reset`` signals — either can
    land first depending on which timeout wins.
    """
    _reset_storage_phase(state, phase_name)
    if not dut_enabled.get("storage", True):
        state.fio_stall_streak = 0
        return
    if phase_modes.get("storage") not in ("stress", "pulse", "ramp"):
        state.fio_stall_streak = 0
        return
    if fio_runner is None:
        state.fio_stall_streak = 0
        return
    if phase_name in state.fio_stall_fired_phases:
        return
    # SIGSTOP from the pulse harness freezes fio's stdout emission. That is
    # an operator-induced pause, not a hardware hang — skip this tick so the
    # streak does not accumulate while paused. FioRunner.state() returns
    # ``"storage:pausd"`` when SIGSTOPped (see fio_runner.py state()).
    try:
        worker_state = str(fio_runner.state())
    except Exception:
        worker_state = ""
    if ":pausd" in worker_state:
        return
    snapshot_fn = getattr(fio_runner, "ios_progress_snapshot", None)
    if not callable(snapshot_fn):
        return
    snapshot = snapshot_fn()
    if snapshot is None:
        # fio has not emitted its first stanza yet. Accumulate grace ticks;
        # after _FIO_STALL_STARTUP_GRACE_TICKS give up scoring this phase.
        state.fio_stall_startup_ticks += 1
        if state.fio_stall_startup_ticks >= _FIO_STALL_STARTUP_GRACE_TICKS:
            state.fio_stall_streak = 0
        return
    state.fio_stall_startup_ticks = 0
    if state.fio_stall_last_snapshot is None:
        state.fio_stall_last_snapshot = snapshot
        return
    if snapshot == state.fio_stall_last_snapshot:
        state.fio_stall_streak += 1
    else:
        state.fio_stall_streak = 0
        state.fio_stall_last_snapshot = snapshot
        return

    if state.fio_stall_streak < _FIO_STALL_TICKS:
        return

    read_ios, write_ios = snapshot
    emit_event(
        "crit",
        phase_name,
        "fio_stall",
        "fio IO progress stalled for {} consecutive ticks in phase '{}' "
        "(read_ios={}, write_ios={}) — likely NVMe controller hang. "
        "Correlate with kernel nvme_io_timeout / nvme_controller_reset "
        "events and a bench-swap of the DUT.".format(
            state.fio_stall_streak, phase_name, read_ios, write_ios,
        ),
        tui_text="fio_stall {}t".format(state.fio_stall_streak),
    )
    state.fio_stall_fired_phases.add(phase_name)


def update_rail_baseline(
    state: PowerHealthState,
    sample: Dict[str, object],
    phase_idx: int,
) -> None:
    """Track cold/hot idle rail voltages for thermal drift check."""
    if phase_idx == 0 and state.cold_idle_12v is None:
        state.cold_idle_12v = sample.get("rail_12v")
        state.cold_idle_5v = sample.get("rail_5v")
    state.hot_idle_12v = sample.get("rail_12v")
    state.hot_idle_5v = sample.get("rail_5v")


def check_thermal_drift(
    state: PowerHealthState,
    emit_event: EmitFn,
    log_action: Callable[[str], None],
) -> None:
    """Post-run thermal drift check on 12V and 5V rails.

    Goes through ``emit_event`` (not raw ``db.add_event``) so the TUI event
    log shows the drift alongside the DB write.
    """
    if state.cold_idle_12v is not None and state.hot_idle_12v is not None:
        drift_12v = state.cold_idle_12v - state.hot_idle_12v
        if drift_12v > 0.05:
            emit_event(
                "crit", "run", "thermal_drift",
                "12V thermal drift exceeded 0.05V (cold={:.2f}V hot={:.2f}V drift={:.2f}V)".format(
                    state.cold_idle_12v, state.hot_idle_12v, drift_12v,
                ),
                tui_text="12V drift {:.2f}V (FAIL)".format(drift_12v),
            )
            log_action("FAIL: 12V thermal drift exceeded 0.05V")
        elif drift_12v > 0.03:
            emit_event(
                "warn", "run", "thermal_drift",
                "12V thermal drift reached warning band (drift={:.2f}V)".format(drift_12v),
                tui_text="12V drift {:.2f}V (warn)".format(drift_12v),
            )
    if state.cold_idle_5v is not None and state.hot_idle_5v is not None:
        drift_5v = state.cold_idle_5v - state.hot_idle_5v
        if drift_5v > 0.05:
            emit_event(
                "crit", "run", "thermal_drift",
                "5V thermal drift exceeded 0.05V (cold={:.2f}V hot={:.2f}V drift={:.2f}V)".format(
                    state.cold_idle_5v, state.hot_idle_5v, drift_5v,
                ),
                tui_text="5V drift {:.2f}V (FAIL)".format(drift_5v),
            )
            log_action("FAIL: 5V thermal drift exceeded 0.05V")


def check_rail_runrule(
    state: PowerStatsState,
    sample: Dict[str, object],
    phase_idx: int,
    phase_name: str,
    phase_modes: Dict[str, str],
    dut_enabled: Dict[str, bool],
    bench_baseline: Optional[Dict[str, float]],
    emit_event: EmitFn,
) -> None:
    """Western Electric run-rules on 12V under stress phases."""
    if not _ensure_power_baseline(
        state, sample, phase_idx, phase_name, phase_modes, dut_enabled, bench_baseline, emit_event
    ):
        return
    if not _phase_has_power_stress(phase_modes, dut_enabled):
        return
    if phase_name in state.runrule_fired_phases:
        return
    _reset_power_stats_phase(state, phase_name)
    rail_12v = sample.get("rail_12v")
    if rail_12v is None or state.rail_12v_baseline_mean is None or state.rail_12v_baseline_sigma is None:
        return
    try:
        value = float(rail_12v)
    except (TypeError, ValueError):
        return

    mean = state.rail_12v_baseline_mean
    sigma = state.rail_12v_baseline_sigma
    below = value < mean
    if below:
        state.consecutive_below_count += 1
    else:
        state.consecutive_below_count = 0

    side = "below" if value < mean else "above" if value > mean else "center"
    if side == state.last_side:
        state.consecutive_same_side_count += 1
    else:
        state.last_side = side
        state.consecutive_same_side_count = 1

    below_2sigma = value < (mean - (2.0 * sigma))
    state.recent_below_2sigma.append(below_2sigma)
    if len(state.recent_below_2sigma) > 3:
        state.recent_below_2sigma.pop(0)

    rule_id = None
    human_reason = None
    if value < (mean - (3.0 * sigma)):
        rule_id = 3
        human_reason = "one point crossed baseline-3sigma ({:.3f}V)".format(mean - (3.0 * sigma))
    elif len(state.recent_below_2sigma) == 3 and sum(1 for hit in state.recent_below_2sigma if hit) >= 2:
        rule_id = 2
        human_reason = "two of the last three samples fell below baseline-2sigma ({:.3f}V)".format(
            mean - (2.0 * sigma)
        )
    elif state.consecutive_below_count >= 7:
        rule_id = 1
        human_reason = "seven consecutive samples stayed below the baseline mean"

    if rule_id is None:
        return

    emit_event(
        "warn",
        phase_name,
        "rail_runrule_violation",
        (
            "12V WE-{}: {} (baseline={:.3f}V sigma={:.3f}V, sample={:.3f}V, source={}). "
            "Slow sag below the 4%/8% threshold but above the statistical noise floor."
        ).format(rule_id, human_reason, mean, sigma, value, state.baseline_source or "unknown"),
        tui_text="12V WE-{} violation".format(rule_id),
    )
    state.runrule_fired_phases.add(phase_name)


def check_rail_cusum(
    state: PowerStatsState,
    sample: Dict[str, object],
    phase_idx: int,
    phase_name: str,
    phase_modes: Dict[str, str],
    dut_enabled: Dict[str, bool],
    bench_baseline: Optional[Dict[str, float]],
    emit_event: EmitFn,
) -> None:
    """CUSUM drift detection on 12V under stress phases."""
    if not _ensure_power_baseline(
        state, sample, phase_idx, phase_name, phase_modes, dut_enabled, bench_baseline, emit_event
    ):
        return
    if not _phase_has_power_stress(phase_modes, dut_enabled):
        return
    _reset_power_stats_phase(state, phase_name)
    rail_12v = sample.get("rail_12v")
    if rail_12v is None or state.rail_12v_baseline_mean is None or state.rail_12v_baseline_sigma is None:
        return
    try:
        value = float(rail_12v)
    except (TypeError, ValueError):
        return

    z_i = (value - state.rail_12v_baseline_mean) / state.rail_12v_baseline_sigma
    state.cusum_pos = max(0.0, state.cusum_pos + z_i - _CUSUM_K)
    state.cusum_neg = max(0.0, state.cusum_neg - z_i - _CUSUM_K)

    if phase_name in state.cusum_fired_phases:
        return
    if state.cusum_pos <= _CUSUM_H and state.cusum_neg <= _CUSUM_H:
        return

    direction = "downward" if state.cusum_neg > _CUSUM_H else "upward"
    score = max(state.cusum_pos, state.cusum_neg)
    emit_event(
        "warn",
        phase_name,
        "rail_cusum_violation",
        (
            "12V CUSUM {} shift detected (score={:.2f}, baseline={:.3f}V sigma={:.3f}V, "
            "sample={:.3f}V, source={})."
        ).format(
            direction,
            score,
            state.rail_12v_baseline_mean,
            state.rail_12v_baseline_sigma,
            value,
            state.baseline_source or "unknown",
        ),
        tui_text="12V CUSUM {}".format(direction),
    )
    state.cusum_fired_phases.add(phase_name)


# How many stress-phase samples we collect before scoring begins. At 1 Hz
# this is a 30 s warm-up — long enough to see real boost behavior without
# delaying detection of genuine throttling.
_CPU_BASELINE_MIN_SAMPLES = 30

# Percentile used as the throttle baseline. p90 is robust to the single
# outlier that used to poison the previous ratchet-max logic (one bogus
# 12000 MHz read would lock the baseline there for the rest of the run).
_CPU_BASELINE_PERCENTILE = 0.90


def _percentile(values: List[float], pct: float) -> float:
    """Nearest-rank percentile. Returns the value at rank ceil(pct * n)."""
    ordered = sorted(values)
    rank = max(1, int(round(pct * len(ordered))))
    return ordered[min(rank, len(ordered)) - 1]


def check_cpu_throttle(
    state: CpuHealthState,
    sample: Dict[str, object],
    phase_idx: int,
    phase_name: str,
    phase_modes: Dict[str, str],
    dut_enabled: Dict[str, bool],
    emit_event: EmitFn,
) -> None:
    """Detect CPU frequency throttling under load via turbostat Avg_MHz.

    Baseline is established from the p90 of the first N stress-phase
    samples (see _CPU_BASELINE_*). Once set, it does not change — throttle
    detection needs a fixed reference. Under load, if avg MHz drops >40%
    while Busy% stays >80%, the CPU is throttled.

    p90 instead of max because a single glitched reading (turbostat column
    drift, timing artifact) used to ratchet the baseline to an impossible
    value and then fire WARN for the rest of the run on every legitimate
    sample.
    """
    if not dut_enabled.get("cpu", True):
        return
    avg_mhz = sample.get("cpu_avg_mhz")
    if avg_mhz is None:
        return

    try:
        mhz_f = float(avg_mhz)
    except (TypeError, ValueError):
        return

    # Soak (phase_idx 0): informational capture only. Baseline is derived
    # from stress-phase samples, not idle samples.
    if phase_idx == 0:
        if mhz_f > 0:
            state.soak_cpu_mhz_samples.append(mhz_f)
        return

    # Only consider CPU-stress phases for both baseline capture and scoring.
    if phase_modes.get("cpu") not in ("stress", "pulse", "ramp"):
        return

    if mhz_f > 0:
        state.stress_cpu_mhz_samples.append(mhz_f)

    # Establish baseline once enough stress samples exist. Lock it.
    if state.cpu_mhz_baseline is None:
        if len(state.stress_cpu_mhz_samples) >= _CPU_BASELINE_MIN_SAMPLES:
            state.cpu_mhz_baseline = _percentile(
                state.stress_cpu_mhz_samples, _CPU_BASELINE_PERCENTILE,
            )
        return

    if phase_name in state.cpu_throttle_warned_phases:
        return

    busy_pct = sample.get("cpu_busy_pct_turbo")
    try:
        busy_f = float(busy_pct) if busy_pct is not None else 0.0
    except (TypeError, ValueError):
        busy_f = 0.0

    drop_pct = (1.0 - mhz_f / state.cpu_mhz_baseline) * 100.0
    if drop_pct > 40.0 and busy_f > 80.0:
        emit_event(
            "warn", phase_name, "cpu_core_throttle",
            "CPU avg frequency dropped {:.0f}% from stress baseline ({:.0f} → {:.0f} MHz) "
            "while busy at {:.0f}%. CPU is being throttled under load — "
            "check VRM thermals, power delivery, or BIOS power limits.".format(
                drop_pct, state.cpu_mhz_baseline, mhz_f, busy_f,
            ),
            tui_text="cpu_throttle: {:.0f}→{:.0f} MHz".format(
                state.cpu_mhz_baseline, mhz_f,
            ),
        )
        state.cpu_throttle_warned_phases.add(phase_name)


def check_gpu_clock_throttle(
    state: GpuHealthState,
    sample: Dict[str, object],
    phase_idx: int,
    phase_name: str,
    phase_modes: Dict[str, str],
    dut_enabled: Dict[str, bool],
    emit_event: EmitFn,
) -> None:
    """Detect GPU SM clock throttling under load.

    During soak (phase_idx 0), collect SM clock samples to establish a
    baseline. During stress phases, if SM clock drops >30% from baseline
    while GPU utilization stays >80%, the GPU is being throttled — power,
    thermal, or both. Emit WARN with the specific throttle reason from
    the collector's bitmask.
    """
    if not dut_enabled.get("gpu", True):
        return
    sm_clock = sample.get("gpu_clock_sm_mhz")
    if sm_clock is None:
        return

    try:
        sm_clock_int = int(sm_clock)
    except (TypeError, ValueError):
        return

    if phase_idx == 0:
        if sm_clock_int > 0:
            state.soak_sm_clock_samples.append(sm_clock_int)
        return

    if state.gpu_sm_clock_baseline is None and state.soak_sm_clock_samples:
        state.gpu_sm_clock_baseline = statistics.median(state.soak_sm_clock_samples)

    if state.gpu_sm_clock_baseline is None or state.gpu_sm_clock_baseline <= 0:
        return
    if phase_modes.get("gpu") not in ("stress", "pulse", "ramp"):
        return
    if phase_name in state.gpu_clock_throttle_warned_phases:
        return

    gpu_util = sample.get("gpu_util_pct")
    try:
        util_f = float(gpu_util) if gpu_util is not None else 0.0
    except (TypeError, ValueError):
        util_f = 0.0

    drop_pct = (1.0 - sm_clock_int / state.gpu_sm_clock_baseline) * 100.0
    if drop_pct > 30.0 and util_f > 80.0:
        emit_event(
            "warn", phase_name, "gpu_clock_throttle",
            "GPU SM clock dropped {:.0f}% from soak baseline ({:.0f} → {} MHz) "
            "while utilization at {:.0f}%. GPU is being throttled under load.".format(
                drop_pct, state.gpu_sm_clock_baseline, sm_clock_int, util_f,
            ),
            tui_text="gpu_clock_throttle: SM {:.0f}→{} MHz".format(
                state.gpu_sm_clock_baseline, sm_clock_int,
            ),
        )
        state.gpu_clock_throttle_warned_phases.add(phase_name)


def check_gpu_ecc(
    state: GpuHealthState,
    sample: Dict[str, object],
    phase_name: str,
    dut_enabled: Dict[str, bool],
    emit_event: EmitFn,
) -> None:
    """Detect GPU ECC errors (volatile counters from nvidia-smi).

    On ECC-capable GPUs (A-series, H-series, Quadro), nvidia-smi exposes
    correctable and uncorrectable error counters. UE = FAIL (data
    corruption). CE = WARN (degrading VRAM).
    """
    if not dut_enabled.get("gpu", True):
        return
    if state.gpu_ecc_ue_fired:
        return

    ecc_ue_raw = sample.get("gpu_ecc_ue")
    ecc_ce_raw = sample.get("gpu_ecc_ce")

    try:
        ecc_ue = int(ecc_ue_raw) if ecc_ue_raw is not None else 0
    except (TypeError, ValueError):
        ecc_ue = 0
    try:
        ecc_ce = int(ecc_ce_raw) if ecc_ce_raw is not None else 0
    except (TypeError, ValueError):
        ecc_ce = 0

    if ecc_ue > state.gpu_ecc_ue_last:
        delta = ecc_ue - state.gpu_ecc_ue_last
        emit_event(
            "crit", phase_name, "gpu_ecc_ue",
            "GPU uncorrectable ECC error(s) detected: {} new UE (total {}). "
            "VRAM data corruption — GPU is failing.".format(delta, ecc_ue),
            tui_text="GPU ECC UE: +{} uncorrectable".format(delta),
        )
        state.gpu_ecc_ue_last = ecc_ue
        state.gpu_ecc_ue_fired = True

    if ecc_ce > state.gpu_ecc_ce_last:
        delta = ecc_ce - state.gpu_ecc_ce_last
        emit_event(
            "warn", phase_name, "gpu_ecc_ce",
            "GPU correctable ECC error(s) detected: {} new CE (total {}). "
            "VRAM is degrading — ECC is masking errors now.".format(delta, ecc_ce),
            tui_text="GPU ECC CE: +{} correctable".format(delta),
        )
        state.gpu_ecc_ce_last = ecc_ce


def check_gpu_retired_pages(
    state: GpuHealthState,
    sample: Dict[str, object],
    phase_name: str,
    dut_enabled: Dict[str, bool],
    emit_event: EmitFn,
) -> None:
    """Detect GPU retired pages increasing during the run.

    Retired pages are VRAM cells permanently marked bad by the driver due to
    ECC errors. If the count increases during a run, the GPU VRAM is actively
    failing. This is a FAIL-level event — the GPU needs replacement.

    Only available when pynvml is the GPU backend; the subprocess path cannot
    query retired pages.
    """
    if not dut_enabled.get("gpu", True):
        return
    if state.gpu_retired_pages_fired:
        return

    retired_raw = sample.get("gpu_retired_pages")
    if retired_raw is None:
        return

    try:
        retired = int(retired_raw)
    except (TypeError, ValueError):
        return

    if state.gpu_retired_pages_baseline is None:
        state.gpu_retired_pages_baseline = retired
        return

    if retired > state.gpu_retired_pages_baseline:
        delta = retired - state.gpu_retired_pages_baseline
        emit_event(
            "crit", phase_name, "gpu_retired_pages",
            "GPU retired pages increased by {} during run (was {}, now {}). "
            "VRAM cells are being permanently marked bad — GPU is actively failing.".format(
                delta, state.gpu_retired_pages_baseline, retired,
            ),
            tui_text="GPU retired pages: +{} (VRAM failing)".format(delta),
        )
        state.gpu_retired_pages_fired = True


def check_worker_exit(
    worker: "StressWorker",
    cmd: object,
    label: str,
    phase_name: str,
    emit_event: EmitFn,
) -> bool:
    """Check if a stress worker exited unexpectedly. Returns True if the
    worker's command should be cleared (worker died)."""
    state_str = worker.state()
    if cmd and "rc=" in state_str:
        if state_str.endswith("rc=0"):
            return False
        emit_event(
            "warn", phase_name, "{}_worker_exit".format(label), state_str,
            tui_text="{}_worker_exit: {}".format(label, state_str),
        )
        return True
    return False
