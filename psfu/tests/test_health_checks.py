"""Unit tests for psfu.runners.health_checks — CPU throttle baseline logic.

Covers the 2026-04-14 fix: a single outlier turbostat sample used to
ratchet the throttle baseline to an impossible value (e.g. 12037 MHz on
Threadripper 7000, which caps ~5.3 GHz) and then fire WARN on every
legitimate sample afterward. Baseline is now p90 of the first
_CPU_BASELINE_MIN_SAMPLES stress-phase readings, fixed once set.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from psfu.runners.health_checks import (
    _CPU_BASELINE_MIN_SAMPLES,
    _POWER_BASELINE_MIN_SAMPLES,
    CpuHealthState,
    PowerStatsState,
    StorageHealthState,
    check_cpu_throttle,
    check_rail_cusum,
    check_rail_runrule,
    check_nvme_fio_stall,
    check_storage_pcie_link_downshift,
)


def _feed(
    state: CpuHealthState,
    mhz_values: List[float],
    *,
    phase_idx: int = 1,
    phase_name: str = "cpu-only",
    busy: float = 95.0,
) -> List[Tuple[str, str, str, str]]:
    """Drive check_cpu_throttle with a list of samples, return emitted events."""
    events: List[Tuple[str, str, str, str]] = []

    def emit(level: str, phase: str, kind: str, message: str, **_: Any) -> None:
        events.append((level, phase, kind, message))

    phase_modes = {"cpu": "stress"}
    dut_enabled = {"cpu": True}

    for mhz in mhz_values:
        sample: Dict[str, Any] = {
            "cpu_avg_mhz": mhz,
            "cpu_busy_pct_turbo": busy,
        }
        check_cpu_throttle(
            state, sample, phase_idx, phase_name, phase_modes, dut_enabled, emit,
        )
    return events


def test_outlier_does_not_poison_baseline():
    """One 12037 MHz glitch among 4900 MHz samples should not fire WARN
    on subsequent legitimate 4800 MHz readings.

    Regression for the 4/5 WARN rate on known-good PSUs observed
    2026-04-13 — baseline ratcheted up on the outlier, then every real
    sample tripped the 40% drop threshold.
    """
    state = CpuHealthState()
    # 29 legit samples + 1 outlier = 30 samples → baseline is established
    # at p90 of 30 ≈ 27th value, which will be ~4900 (not 12037).
    warmup = [4900.0] * 29 + [12037.0]
    events = _feed(state, warmup)
    assert events == []
    # Baseline should be ~4900, not the outlier.
    assert state.cpu_mhz_baseline is not None
    assert state.cpu_mhz_baseline < 6000.0

    # Subsequent legitimate readings at 4800 should NOT fire
    # (4800/4900 = 2% drop, well under 40%).
    events = _feed(state, [4800.0] * 10, phase_name="cpu-only-b")
    assert events == []


def test_real_throttle_still_fires():
    """A legitimate >40% frequency drop under load must still WARN.

    After baseline is established from healthy samples, a sustained
    drop to ~2400 MHz at high busy% is real throttling.
    """
    state = CpuHealthState()
    # Establish baseline around 4900 MHz.
    healthy = [4900.0] * _CPU_BASELINE_MIN_SAMPLES
    assert _feed(state, healthy) == []
    assert state.cpu_mhz_baseline is not None

    # Now feed a throttled reading.
    events = _feed(state, [2400.0], phase_name="transient", busy=95.0)
    assert len(events) == 1
    level, phase, kind, _message = events[0]
    assert level == "warn"
    assert phase == "transient"
    assert kind == "cpu_core_throttle"


def test_baseline_locked_once_set():
    """Once established, baseline must not drift upward with later
    higher samples. Throttle detection needs a fixed reference."""
    state = CpuHealthState()
    # Establish baseline at ~4900.
    assert _feed(state, [4900.0] * _CPU_BASELINE_MIN_SAMPLES) == []
    first_baseline = state.cpu_mhz_baseline
    assert first_baseline is not None

    # Feed higher legit samples (CPU boosted past initial distribution).
    _feed(state, [5200.0] * 20, phase_name="cpu-only-b")
    assert state.cpu_mhz_baseline == first_baseline


def test_soak_samples_do_not_establish_baseline():
    """Phase 0 (soak) samples must not contribute to the throttle
    baseline. Soak is idle; baseline must come from stress."""
    state = CpuHealthState()
    # Feed only soak samples — baseline must stay None.
    _feed(state, [200.0] * 60, phase_idx=0, phase_name="baseline")
    assert state.cpu_mhz_baseline is None
    assert len(state.soak_cpu_mhz_samples) == 60
    assert len(state.stress_cpu_mhz_samples) == 0


def test_warning_deduped_per_phase():
    """Two throttle samples in the same phase produce one event."""
    state = CpuHealthState()
    assert _feed(state, [4900.0] * _CPU_BASELINE_MIN_SAMPLES) == []
    events = _feed(state, [2400.0] * 5, phase_name="cpu-only", busy=95.0)
    assert len(events) == 1

    # New phase, same drop → fires again once.
    events = _feed(state, [2400.0] * 5, phase_name="transient", busy=95.0)
    assert len(events) == 1


def _power_emit_collector():
    events: List[Tuple[str, str, str, str]] = []

    def emit(level: str, phase: str, kind: str, message: str, **_: Any) -> None:
        events.append((level, phase, kind, message))

    return events, emit


def _power_sample(rail_12v: float) -> Dict[str, Any]:
    return {"rail_12v": rail_12v}


def _feed_soak_baseline(state: PowerStatsState, emit, *, value: float = 12.0) -> None:
    phase_modes = {"cpu": "idle", "gpu": "idle", "mem": "idle", "storage": "idle"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for _ in range(_POWER_BASELINE_MIN_SAMPLES):
        sample = _power_sample(value)
        check_rail_runrule(state, sample, 0, "soak", phase_modes, dut_enabled, None, emit)
        check_rail_cusum(state, sample, 0, "soak", phase_modes, dut_enabled, None, emit)


def test_we_rule_1_seven_consecutive_below():
    state = PowerStatsState()
    events, emit = _power_emit_collector()
    bench = {"soak_12v_mean": 12.0, "bench_noise_floor_v": 0.05}
    phase_modes = {"cpu": "stress", "gpu": "idle", "mem": "idle", "storage": "idle"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for idx in range(6):
        check_rail_runrule(state, _power_sample(11.98), 1, "steady", phase_modes, dut_enabled, bench, emit)
        assert [e for e in events if e[2] == "rail_runrule_violation"] == []
    check_rail_runrule(state, _power_sample(11.98), 1, "steady", phase_modes, dut_enabled, bench, emit)
    violations = [e for e in events if e[2] == "rail_runrule_violation"]
    assert len(violations) == 1
    assert "WE-1" in violations[0][3]


def test_we_rule_2_two_of_three_beyond_2sigma():
    state = PowerStatsState()
    events, emit = _power_emit_collector()
    bench = {"soak_12v_mean": 12.0, "bench_noise_floor_v": 0.05}
    phase_modes = {"cpu": "stress", "gpu": "idle", "mem": "idle", "storage": "idle"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for value in (11.89, 11.97, 11.89):
        check_rail_runrule(state, _power_sample(value), 1, "steady", phase_modes, dut_enabled, bench, emit)
    violations = [e for e in events if e[2] == "rail_runrule_violation"]
    assert len(violations) == 1
    assert "WE-2" in violations[0][3]


def test_we_rule_3_one_point_beyond_3sigma():
    state = PowerStatsState()
    events, emit = _power_emit_collector()
    bench = {"soak_12v_mean": 12.0, "bench_noise_floor_v": 0.05}
    phase_modes = {"cpu": "stress", "gpu": "idle", "mem": "idle", "storage": "idle"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    check_rail_runrule(state, _power_sample(11.84), 1, "steady", phase_modes, dut_enabled, bench, emit)
    violations = [e for e in events if e[2] == "rail_runrule_violation"]
    assert len(violations) == 1
    assert "WE-3" in violations[0][3]


def test_we_rule_dedup_within_phase():
    state = PowerStatsState()
    events, emit = _power_emit_collector()
    bench = {"soak_12v_mean": 12.0, "bench_noise_floor_v": 0.05}
    phase_modes = {"cpu": "stress", "gpu": "idle", "mem": "idle", "storage": "idle"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for _ in range(20):
        check_rail_runrule(state, _power_sample(11.98), 1, "steady", phase_modes, dut_enabled, bench, emit)
    assert len([e for e in events if e[2] == "rail_runrule_violation"]) == 1


def test_we_rule_separate_phases_fire_separately():
    state = PowerStatsState()
    events, emit = _power_emit_collector()
    bench = {"soak_12v_mean": 12.0, "bench_noise_floor_v": 0.05}
    phase_modes = {"cpu": "stress", "gpu": "idle", "mem": "idle", "storage": "idle"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for phase in ("steady", "transient"):
        for _ in range(7):
            check_rail_runrule(state, _power_sample(11.98), 1, phase, phase_modes, dut_enabled, bench, emit)
    assert len([e for e in events if e[2] == "rail_runrule_violation"]) == 2


def test_rail_runrule_uses_bench_baseline_when_present():
    state = PowerStatsState()
    events, emit = _power_emit_collector()
    bench = {"soak_12v_mean": 12.222, "bench_noise_floor_v": 0.018}
    phase_modes = {"cpu": "stress", "gpu": "idle", "mem": "idle", "storage": "idle"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    check_rail_runrule(state, _power_sample(12.20), 1, "steady", phase_modes, dut_enabled, bench, emit)
    assert state.rail_12v_baseline_mean == 12.222
    assert state.rail_12v_baseline_sigma == 0.018
    assert state.baseline_source == "bench"


def test_rail_runrule_degrades_to_within_run_without_baseline():
    state = PowerStatsState()
    events, emit = _power_emit_collector()
    _feed_soak_baseline(state, emit, value=12.0)
    phase_modes = {"cpu": "stress", "gpu": "idle", "mem": "idle", "storage": "idle"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    check_rail_runrule(state, _power_sample(11.98), 1, "steady", phase_modes, dut_enabled, None, emit)
    assert state.baseline_source == "within-run"
    baseline_events = [e for e in events if e[2] == "baseline_unavailable"]
    assert len(baseline_events) == 1


def test_cusum_fires_on_half_sigma_persistent_shift():
    state = PowerStatsState()
    events, emit = _power_emit_collector()
    bench = {"soak_12v_mean": 12.0, "bench_noise_floor_v": 0.05}
    phase_modes = {"cpu": "stress", "gpu": "idle", "mem": "idle", "storage": "idle"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    fire_idx = None
    for idx in range(1, 80):
        check_rail_cusum(state, _power_sample(11.96875), 1, "steady", phase_modes, dut_enabled, bench, emit)
        if any(e[2] == "rail_cusum_violation" for e in events):
            fire_idx = idx
            break
    assert fire_idx is not None
    assert 35 <= fire_idx <= 60


def test_cusum_does_not_fire_on_noise():
    state = PowerStatsState()
    events, emit = _power_emit_collector()
    bench = {"soak_12v_mean": 12.0, "bench_noise_floor_v": 0.05}
    phase_modes = {"cpu": "stress", "gpu": "idle", "mem": "idle", "storage": "idle"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for idx in range(1000):
        value = 12.0 + (0.005 if idx % 2 == 0 else -0.005)
        check_rail_cusum(state, _power_sample(value), 1, "steady", phase_modes, dut_enabled, bench, emit)
    assert [e for e in events if e[2] == "rail_cusum_violation"] == []


def test_cusum_resets_at_phase_boundary():
    state = PowerStatsState()
    events, emit = _power_emit_collector()
    bench = {"soak_12v_mean": 12.0, "bench_noise_floor_v": 0.05}
    phase_modes = {"cpu": "stress", "gpu": "idle", "mem": "idle", "storage": "idle"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for _ in range(5):
        check_rail_cusum(state, _power_sample(11.95), 1, "steady", phase_modes, dut_enabled, bench, emit)
    assert state.cusum_neg > 0
    prev = state.cusum_neg
    check_rail_cusum(state, _power_sample(11.95), 1, "transient", phase_modes, dut_enabled, bench, emit)
    assert state.cusum_neg < prev
    assert math.isclose(state.cusum_pos, 0.0, abs_tol=1e-9)


def test_cusum_dedup_within_phase():
    state = PowerStatsState()
    events, emit = _power_emit_collector()
    bench = {"soak_12v_mean": 12.0, "bench_noise_floor_v": 0.05}
    phase_modes = {"cpu": "stress", "gpu": "idle", "mem": "idle", "storage": "idle"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for _ in range(100):
        check_rail_cusum(state, _power_sample(11.95), 1, "steady", phase_modes, dut_enabled, bench, emit)
    assert len([e for e in events if e[2] == "rail_cusum_violation"]) == 1


class _FakePcieCollector:
    def __init__(self, snapshots, capability):
        self.available = True
        self._snapshots = list(snapshots)
        self._capability = dict(capability)

    def snapshot(self):
        if len(self._snapshots) > 1:
            return self._snapshots.pop(0)
        return self._snapshots[0]

    def first_capability(self):
        return dict(self._capability)


def test_storage_pcie_link_downshift_fires_after_3_ticks():
    state = StorageHealthState()
    events, emit = _power_emit_collector()
    collector = _FakePcieCollector(
        [
            {"nvme_pcie_link_gen": 3, "nvme_pcie_link_width": 4},
            {"nvme_pcie_link_gen": 3, "nvme_pcie_link_width": 4},
            {"nvme_pcie_link_gen": 3, "nvme_pcie_link_width": 4},
        ],
        {"max_gen": 5, "max_width": 4},
    )
    phase_modes = {"cpu": "idle", "gpu": "idle", "mem": "idle", "storage": "stress"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for _ in range(3):
        check_storage_pcie_link_downshift(state, collector, "storage", phase_modes, dut_enabled, emit)
    events = [e for e in events if e[2] == "pcie_link_downshift"]
    assert len(events) == 1


def test_storage_pcie_link_single_tick_blip_does_not_fire():
    state = StorageHealthState()
    events, emit = _power_emit_collector()
    collector = _FakePcieCollector(
        [
            {"nvme_pcie_link_gen": 3, "nvme_pcie_link_width": 4},
            {"nvme_pcie_link_gen": 5, "nvme_pcie_link_width": 4},
            {"nvme_pcie_link_gen": 3, "nvme_pcie_link_width": 4},
            {"nvme_pcie_link_gen": 5, "nvme_pcie_link_width": 4},
        ],
        {"max_gen": 5, "max_width": 4},
    )
    phase_modes = {"cpu": "idle", "gpu": "idle", "mem": "idle", "storage": "stress"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for _ in range(4):
        check_storage_pcie_link_downshift(state, collector, "storage", phase_modes, dut_enabled, emit)
    assert [e for e in events if e[2] == "pcie_link_downshift"] == []


class _FakeFioRunner:
    """Minimal stub for check_nvme_fio_stall — scripted
    ios_progress_snapshot() returns + controllable worker state string."""

    def __init__(self, snapshots, worker_state="storage:run"):
        self._snapshots = list(snapshots)
        self._worker_state = worker_state

    def ios_progress_snapshot(self):
        if not self._snapshots:
            return None
        if len(self._snapshots) > 1:
            return self._snapshots.pop(0)
        return self._snapshots[0]

    def state(self):
        return self._worker_state


def test_fio_stall_fires_after_10_idle_ticks():
    """Same (read_ios, write_ios) tuple for 10 consecutive stress ticks in
    an active storage-stress phase must emit one `fio_stall` event."""
    state = StorageHealthState()
    events, emit = _power_emit_collector()
    fio = _FakeFioRunner([(1000, 0)] * 20)
    phase_modes = {"cpu": "idle", "gpu": "idle", "mem": "idle", "storage": "stress"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for _ in range(15):
        check_nvme_fio_stall(state, fio, "steady", phase_modes, dut_enabled, emit)
    stalls = [e for e in events if e[2] == "fio_stall"]
    assert len(stalls) == 1
    assert stalls[0][0] == "crit"
    assert "1000" in stalls[0][3]


def test_fio_stall_does_not_fire_when_ios_grow():
    """Growing IOs = healthy controller → no stall event."""
    state = StorageHealthState()
    events, emit = _power_emit_collector()
    snapshots = [(i * 100, 0) for i in range(20)]
    fio = _FakeFioRunner(snapshots)
    phase_modes = {"cpu": "idle", "gpu": "idle", "mem": "idle", "storage": "stress"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for _ in range(20):
        check_nvme_fio_stall(state, fio, "steady", phase_modes, dut_enabled, emit)
    assert [e for e in events if e[2] == "fio_stall"] == []


def test_fio_stall_dedups_within_phase():
    state = StorageHealthState()
    events, emit = _power_emit_collector()
    fio = _FakeFioRunner([(1000, 0)] * 40)
    phase_modes = {"cpu": "idle", "gpu": "idle", "mem": "idle", "storage": "stress"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for _ in range(40):
        check_nvme_fio_stall(state, fio, "steady", phase_modes, dut_enabled, emit)
    assert len([e for e in events if e[2] == "fio_stall"]) == 1


def test_fio_stall_does_not_fire_when_paused():
    """SIGSTOPped fio (pulse harness) must not count as a controller hang.
    state() returning ':pausd' short-circuits the check."""
    state = StorageHealthState()
    events, emit = _power_emit_collector()
    fio = _FakeFioRunner([(1000, 0)] * 30, worker_state="storage:pausd")
    phase_modes = {"cpu": "idle", "gpu": "idle", "mem": "idle", "storage": "pulse"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for _ in range(30):
        check_nvme_fio_stall(state, fio, "pulse1", phase_modes, dut_enabled, emit)
    assert [e for e in events if e[2] == "fio_stall"] == []


def test_fio_stall_idle_phase_is_noop():
    state = StorageHealthState()
    events, emit = _power_emit_collector()
    fio = _FakeFioRunner([(0, 0)] * 30)
    phase_modes = {"cpu": "idle", "gpu": "idle", "mem": "idle", "storage": "idle"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for _ in range(30):
        check_nvme_fio_stall(state, fio, "baseline", phase_modes, dut_enabled, emit)
    assert events == []


def test_fio_stall_fallback_path_no_runner_is_noop():
    """When storage runs on the legacy sha256 path, fio_runner is None and
    the check must no-op without exceptions."""
    state = StorageHealthState()
    events, emit = _power_emit_collector()
    phase_modes = {"cpu": "idle", "gpu": "idle", "mem": "idle", "storage": "stress"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for _ in range(15):
        check_nvme_fio_stall(state, None, "steady", phase_modes, dut_enabled, emit)
    assert events == []


def test_fio_stall_startup_grace_does_not_fire():
    """fio runner that never emits a stanza (ios_progress_snapshot returns
    None) should not fire a stall — give it up for the phase after the
    grace window rather than falsely scoring."""
    state = StorageHealthState()
    events, emit = _power_emit_collector()
    fio = _FakeFioRunner([])  # ios_progress_snapshot returns None forever
    phase_modes = {"cpu": "idle", "gpu": "idle", "mem": "idle", "storage": "stress"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for _ in range(60):
        check_nvme_fio_stall(state, fio, "steady", phase_modes, dut_enabled, emit)
    assert events == []


def test_storage_pcie_link_capability_unknown_emits_info_once():
    state = StorageHealthState()
    events, emit = _power_emit_collector()
    collector = _FakePcieCollector(
        [{"nvme_pcie_link_gen": 3, "nvme_pcie_link_width": 2}],
        {"max_gen": None, "max_width": None},
    )
    phase_modes = {"cpu": "idle", "gpu": "idle", "mem": "idle", "storage": "stress"}
    dut_enabled = {"cpu": True, "gpu": True, "mem": True, "storage": True}
    for _ in range(3):
        check_storage_pcie_link_downshift(state, collector, "storage", phase_modes, dut_enabled, emit)
    info_events = [e for e in events if e[2] == "pcie_link_capability_unknown"]
    assert len(info_events) == 1
