"""Header run-timer accuracy (``_run_total_display``).

The top progress bar must not hit 100 % before the last phase finishes.
It used to, because ``run_total_seconds`` was a fixed sum of phase
target durations and didn't account for:

* inter-phase gaps — worker teardown + startup takes ~3-6 s per
  transition; psu-v1 has 5 transitions so wall clock drifted ~20 s
  past the profile-declared total (observed on run-20260416-123146).
* operator-skipped phases — a baseline skipped at 7 / 15 saves 8 s that
  the fixed sum doesn't reclaim.

``_run_total_display`` is recomputed on every ``start_phase`` call as
``wall_clock_elapsed_at_phase_start + current_phase + future_phases``,
so it drifts outward to absorb gaps and inward to reclaim skip savings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List

from psfu.ui.tui import TuiDashboard


@dataclass
class _FakePhase:
    name: str
    duration_sec: int
    cpu_mode: str = "idle"
    gpu_mode: str = "idle"
    mem_mode: str = "idle"
    storage_mode: str = "idle"


@dataclass
class _FakeProfile:
    name: str
    phases: List[_FakePhase]


def _bare_dashboard(phases: List[_FakePhase]) -> Any:
    """Instantiate a TuiDashboard without touching curses so the test
    doesn't need a TTY. We only exercise timer state, not rendering."""
    dash = TuiDashboard.__new__(TuiDashboard)
    dash.profile = _FakeProfile("test", phases)
    dash.run_total_seconds = max(1, sum(p.duration_sec for p in phases))
    dash._run_total_display = dash.run_total_seconds
    dash._run_start_mono = time.monotonic()
    dash.phase_idx = 0
    dash.current_phase = None
    dash.phase_elapsed = 0.0
    dash.phase_total = 0
    dash._phase_extra = ""
    # Stub out log_event + _render so start_phase() doesn't try to draw.
    dash.log_event = lambda *a, **k: None
    dash._render = lambda: None
    return dash


def test_display_total_matches_profile_sum_at_start() -> None:
    """Fresh dashboard: display total equals the profile's declared sum."""
    phases = [_FakePhase("a", 10), _FakePhase("b", 20), _FakePhase("c", 30)]
    dash = _bare_dashboard(phases)
    assert dash._run_total_display == 60


def test_display_total_absorbs_inter_phase_gap() -> None:
    """Simulate a 5 s gap before phase 2 starts. The display total must
    grow by 5 s so the header bar doesn't pre-emptively hit 100 %."""
    phases = [_FakePhase("a", 10), _FakePhase("b", 20), _FakePhase("c", 30)]
    dash = _bare_dashboard(phases)
    # Phase 1 starts immediately — total unchanged.
    dash.start_phase(phases[0], 0)
    assert 59 <= dash._run_total_display <= 61  # allow for monotonic jitter

    # Simulate phase 1 running its 10 s PLUS a 5 s teardown gap.
    dash._run_start_mono = time.monotonic() - 15.0
    dash.start_phase(phases[1], 1)
    # 15 s elapsed + 20 (current) + 30 (future) = 65
    assert 64 <= dash._run_total_display <= 66


def test_display_total_reclaims_operator_skip() -> None:
    """If a phase ends early, the next start_phase recomputes the total
    based on the shorter-than-expected wall clock — skip savings get
    pulled forward. E.g. a 15 s baseline skipped at 7 s must NOT leave
    the display total inflated by 8 s."""
    phases = [_FakePhase("baseline", 15), _FakePhase("soak", 600)]
    dash = _bare_dashboard(phases)
    dash.start_phase(phases[0], 0)
    assert 614 <= dash._run_total_display <= 616

    # Operator skipped baseline at 7 s wall-clock.
    dash._run_start_mono = time.monotonic() - 7.0
    dash.start_phase(phases[1], 1)
    # 7 s elapsed + 600 current + 0 future = 607 — 8 s shorter than original.
    assert 606 <= dash._run_total_display <= 608


def test_display_total_never_less_than_wall_clock() -> None:
    """Sanity: the displayed total must always be ≥ wall-clock elapsed
    at the moment of computation, otherwise run_pct could exceed 100 %
    inside the current phase."""
    phases = [_FakePhase("a", 5), _FakePhase("b", 10)]
    dash = _bare_dashboard(phases)
    # Phase 2 starts 20 s in (massively over budget on phase 1).
    dash._run_start_mono = time.monotonic() - 20.0
    dash.start_phase(phases[1], 1)
    run_el = time.monotonic() - dash._run_start_mono
    assert dash._run_total_display >= int(run_el)


def test_lobby_total_unchanged_by_dynamic_updates() -> None:
    """The lobby-facing ``run_total_seconds`` must stay the static
    profile sum — operators setting up the next run shouldn't see a
    moving target because the previous run's display total drifted."""
    phases = [_FakePhase("a", 10), _FakePhase("b", 20)]
    dash = _bare_dashboard(phases)
    original = dash.run_total_seconds
    dash._run_start_mono = time.monotonic() - 30.0  # well past total
    dash.start_phase(phases[1], 1)
    assert dash.run_total_seconds == original
    # Dynamic total absorbed the overrun; static total stayed put.
    assert dash._run_total_display > original
