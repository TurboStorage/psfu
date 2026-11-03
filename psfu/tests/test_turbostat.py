"""Unit tests for psfu.collectors.turbostat — block parser.

Covers the 2026-04-14 column-drift guard: turbostat occasionally emits a
row whose Avg_MHz column gets misaligned (header line appears mid-table;
parser's col_map points at PkgWatt instead of the actual frequency). The
resulting implausible value (>6500 MHz) used to poison the throttle
baseline downstream. Parser now discards such rows entirely.
"""

from __future__ import annotations

from psfu.collectors.turbostat import (
    TurbostatCollector,
    _MAX_PLAUSIBLE_MHZ,
    _MAX_PLAUSIBLE_MHZ_FALLBACK,
    _MAX_PLAUSIBLE_PKG_WATT,
    _MHZ_CEILING_MULTIPLIER,
)
from psfu.collectors import turbostat as tmod


def _parse(block: str) -> TurbostatCollector:
    c = TurbostatCollector()
    c._parse_block(block.strip("\n").splitlines())
    return c


def test_parse_normal_block():
    """Happy path: a single package-summary row sets pkg_avg_mhz."""
    block = (
        "Core\tCPU\tAvg_MHz\tBusy%\tCorWatt\tPkgWatt\n"
        "-\t-\t4850\t95.1\t180.0\t220.0\n"
        "0\t0\t4900\t96.0\t5.5\t220.0\n"
        "0\t1\t4800\t94.0\t5.3\t220.0\n"
    )
    c = _parse(block)
    assert c._pkg_avg_mhz == 4850
    assert c._pkg_max_mhz == 4900


def test_discards_row_with_implausible_mhz():
    """Column drift: a per-core row reports Avg_MHz=12037. Parser
    must discard the row entirely so pkg_max_mhz comes from the
    legitimate rows only.

    Regression for the 2026-04-13/14 false-positive throttle WARNs
    observed on known-good PSUs where the baseline captured 12037 MHz
    and then every legit 4800 MHz reading fired as a 60% drop.
    """
    block = (
        "Core\tCPU\tAvg_MHz\tBusy%\tCorWatt\tPkgWatt\n"
        "-\t-\t4850\t95.1\t180.0\t220.0\n"
        "0\t0\t4900\t96.0\t5.5\t220.0\n"
        "0\t1\t12037\t96.0\t5.3\t220.0\n"  # glitched row
        "0\t2\t4800\t94.0\t5.3\t220.0\n"
    )
    c = _parse(block)
    # pkg_max_mhz must come from the good rows — no 12037.
    assert c._pkg_max_mhz is not None
    assert c._pkg_max_mhz <= _MAX_PLAUSIBLE_MHZ
    assert c._pkg_max_mhz == 4900


def test_ceiling_is_above_real_cpus():
    """Sanity: the ceiling must stay above plausible real-world Avg_MHz.
    Threadripper 7000 ~5.3 GHz single-core boost; bump the constant
    and this test when 7 GHz+ CPUs ship."""
    assert _MAX_PLAUSIBLE_MHZ > 5500
    assert _MAX_PLAUSIBLE_MHZ < 10000


def test_discards_row_with_implausible_pkg_watt():
    """Column drift can also land on PkgWatt — observed 1473 W and
    718 W spikes on TRX50 runs 2026-04-14. Those rows must drop."""
    block = (
        "Core\tCPU\tAvg_MHz\tBusy%\tCorWatt\tPkgWatt\n"
        "-\t-\t4850\t95.1\t180.0\t220.0\n"
        "0\t0\t4900\t96.0\t5.5\t1473.0\n"  # glitched PkgWatt
        "0\t1\t4800\t94.0\t5.3\t220.0\n"
    )
    c = _parse(block)
    # pkg_avg_mhz comes from the legit pkg row; no 1473 W leak.
    assert c._pkg_avg_mhz == 4850
    assert c._pkg_watt is not None
    assert c._pkg_watt <= _MAX_PLAUSIBLE_PKG_WATT


def test_pkg_watt_ceiling_above_real_cpus():
    """Sanity: ceiling must be above realistic PkgWatt but below the
    observed glitch values. EPYC 9004 tops ~500 W; bump when CPUs ship
    higher."""
    assert _MAX_PLAUSIBLE_PKG_WATT > 500
    assert _MAX_PLAUSIBLE_PKG_WATT < 1000


def test_discards_glitched_pkg_row():
    """If the package summary row itself is glitched, it must be
    discarded — the entire row's field mapping is corrupt, so
    pkg_avg_mhz falls back to the per-core aggregate."""
    block = (
        "Core\tCPU\tAvg_MHz\tBusy%\tCorWatt\tPkgWatt\n"
        "-\t-\t9999\t95.1\t180.0\t220.0\n"  # glitched pkg row
        "0\t0\t4900\t96.0\t5.5\t220.0\n"
        "0\t1\t4800\t94.0\t5.3\t220.0\n"
    )
    c = _parse(block)
    # pkg_avg_mhz comes from aggregation of per-core rows (no pkg row left).
    assert c._pkg_avg_mhz is not None
    assert c._pkg_avg_mhz <= _MAX_PLAUSIBLE_MHZ
    # Aggregate of 4900 and 4800 = 4850.
    assert 4700 <= c._pkg_avg_mhz <= 4950


def test_instance_ceiling_uses_cpufreq_sysfs(monkeypatch):
    """HW-aware ceiling: read cpuinfo_max_freq and multiply by 1.15.
    On a 7980X (5.3 GHz max boost) the ceiling should land at ~6095 MHz,
    catching the 6475 MHz drift observed on 2026-04-16."""
    monkeypatch.setattr(tmod, "_read_cpu_max_boost_mhz", lambda: 5300.0)
    c = TurbostatCollector()
    expected = 5300.0 * _MHZ_CEILING_MULTIPLIER
    assert abs(c._max_plausible_mhz - expected) < 0.01
    # Ceiling must catch 6475 MHz drift on TR 7980X.
    assert c._max_plausible_mhz < 6475


def test_instance_ceiling_intel_14900ks(monkeypatch):
    """14900KS boosts to 6.2 GHz stock — ceiling ~7130 MHz.
    Must NOT flag a legitimate 6200 MHz boost reading as drift."""
    monkeypatch.setattr(tmod, "_read_cpu_max_boost_mhz", lambda: 6200.0)
    c = TurbostatCollector()
    assert c._max_plausible_mhz > 6200
    assert c._max_plausible_mhz < 8000


def test_instance_ceiling_falls_back_when_sysfs_missing(monkeypatch):
    """When cpufreq sysfs is absent (containers, WSL, BSD), fall back
    to the flat fallback ceiling."""
    monkeypatch.setattr(tmod, "_read_cpu_max_boost_mhz", lambda: None)
    c = TurbostatCollector()
    assert c._max_plausible_mhz == _MAX_PLAUSIBLE_MHZ_FALLBACK


def test_instance_ceiling_applied_to_parser(monkeypatch):
    """Drift row at 6475 MHz must be discarded on a bench whose
    cpuinfo_max_freq is 5.3 GHz — the flat 7200 MHz fallback would
    have let it through."""
    monkeypatch.setattr(tmod, "_read_cpu_max_boost_mhz", lambda: 5300.0)
    c = TurbostatCollector()
    block = (
        "Core\tCPU\tAvg_MHz\tBusy%\tCorWatt\tPkgWatt\n"
        "-\t-\t4850\t95.1\t180.0\t220.0\n"
        "0\t0\t4900\t96.0\t5.5\t220.0\n"
        "0\t1\t6475\t96.0\t5.3\t220.0\n"  # TR 7980X drift, observed 2026-04-16
        "0\t2\t4800\t94.0\t5.3\t220.0\n"
    )
    c._parse_block(block.strip("\n").splitlines())
    assert c._pkg_max_mhz == 4900


def test_fallback_ceiling_accommodates_14900ks():
    """Sanity check the fallback ceiling: must stay above Intel 14900KS
    stock boost (6.2 GHz) so benches without cpufreq sysfs don't false-
    flag legitimate Intel peak-boost samples."""
    assert _MAX_PLAUSIBLE_MHZ_FALLBACK >= 6300
