"""Tests for the bench power budget gate ."""

from __future__ import annotations

import pytest

from psfu import power_budget
from psfu.power_budget import (
    BENCH_OVERHEAD_WATTS,
    FALLBACK_CPU_TDP_WATTS,
    PowerBudget,
    compute_power_budget,
    gpu_transient_multiplier_for,
)


def _stub_detect(monkeypatch, cpu_tdp: int, cpu_source: str, gpu_max: int, gpu_model):
    monkeypatch.setattr(
        power_budget, "detect_cpu_tdp_watts", lambda: (cpu_tdp, cpu_source)
    )
    monkeypatch.setattr(power_budget, "nvidia_gpu_max_power_watts", lambda: gpu_max)
    monkeypatch.setattr(power_budget, "detect_gpu_model", lambda: gpu_model)


def test_undersized_psu_bench_would_block(monkeypatch):
    """The original failing config: 5090 (575W) + 64T TR (350W) on SF1000 (1000W).
    Must fail the sustained check — otherwise the gate that exists to prevent
    an undersized-PSU incident is broken."""
    _stub_detect(monkeypatch, 350, "known_model", 575, "NVIDIA GeForce RTX 5090")
    budget = compute_power_budget(psu_rated_watts=1000)
    assert budget.sustained_required == 350 + 575 + BENCH_OVERHEAD_WATTS
    assert not budget.is_adequate()
    assert not budget.has_transient_headroom()
    assert budget.shortfall_watts() == budget.sustained_required - 1000


def test_adequate_1600w_for_5090_bench(monkeypatch):
    """1600W ATX 3.1 bench covers sustained and transient for the same config."""
    _stub_detect(monkeypatch, 350, "known_model", 575, "NVIDIA GeForce RTX 5090")
    budget = compute_power_budget(psu_rated_watts=1600)
    assert budget.is_adequate()
    assert budget.has_transient_headroom()
    assert budget.shortfall_watts() == 0


def test_unknown_gpu_gets_small_transient_multiplier(monkeypatch):
    """Unknown NVIDIA cards still get a mild transient multiplier."""
    _stub_detect(monkeypatch, 250, "known_model", 200, "NVIDIA RTX A2000")
    budget = compute_power_budget(psu_rated_watts=850)
    assert budget.gpu_transient_multiplier == pytest.approx(1.3)
    assert budget.is_adequate()


def test_no_gpu_no_multiplier(monkeypatch):
    """Headless / CPU-only bench — no transient multiplier applied."""
    _stub_detect(monkeypatch, 250, "known_model", 0, None)
    budget = compute_power_budget(psu_rated_watts=500)
    assert budget.gpu_transient_multiplier == 1.0
    assert budget.sustained_required == 250 + 0 + BENCH_OVERHEAD_WATTS
    assert budget.is_adequate()


def test_cpu_tdp_fallback_is_conservative(monkeypatch):
    """Unknown CPU falls back to a high TDP so the gate errors toward blocking."""
    _stub_detect(monkeypatch, FALLBACK_CPU_TDP_WATTS, "fallback", 300, "GeForce RTX 4080")
    budget = compute_power_budget(psu_rated_watts=600)
    assert budget.cpu_source == "fallback"
    assert not budget.is_adequate()


def test_transient_multiplier_keywords_case_insensitive():
    assert gpu_transient_multiplier_for("NVIDIA GeForce RTX 5090") == pytest.approx(1.9)
    assert gpu_transient_multiplier_for("rtx 4090 founders") == pytest.approx(1.8)
    assert gpu_transient_multiplier_for(None) == 1.0
    assert gpu_transient_multiplier_for("") == 1.0


def test_psu_zero_reports_unknown_verdict(monkeypatch):
    """Zero rated wattage surfaces in the formatted report so the caller can
    prompt; the data object itself still reports not-adequate."""
    _stub_detect(monkeypatch, 250, "known_model", 300, "RTX 4080")
    budget = compute_power_budget(psu_rated_watts=0)
    report = power_budget.format_budget_report(budget)
    assert "UNKNOWN" in report
    assert not budget.is_adequate()
