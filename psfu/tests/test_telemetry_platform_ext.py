"""Platform thermal + link-health extensions (v5 schema).

New columns added 2026-04-16 to surface RMA-relevant signals that were
previously sitting unused on the DUT:

  * mobo_vrm_temp_c  — motherboard VRM temp (``vrm temp`` in sensors)
  * mobo_case_temp_c — board ambient (``case temp``)
  * cpu_ccd_max_c / cpu_ccd_spread_c — Zen 4/5 per-CCD (Tccd*) max + spread
  * ssd_temp_ctrl_c / ssd_temp_nand_c — split NVMe sensors (temp2/temp3)
  * pcie_aer_corr_total — sum of correctable AER counters across all PCI

Each extractor must degrade cleanly to None on hardware that doesn't
expose the source — Intel benches, boards without VRM sensors, kernels
built without PCIEAER_ROOTPORT, etc.
"""

from __future__ import annotations

from psfu.collectors.telemetry import TelemetryCollector


def test_extract_ccd_stats_returns_max_and_spread() -> None:
    """Zen 5 Threadripper exposes Tccd1..Tccd8 via k10temp. We aggregate
    to max (hottest CCD) and spread (hottest - coolest) because raw
    per-CCD values are over-granular for a 1 Hz TUI, and spread is the
    real signal for uneven cooling / failing VRM phase."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    sensors = {
        "tccd1": 62.5,
        "tccd2": 65.0,
        "tccd3": 70.5,  # hottest
        "tccd4": 61.0,  # coolest
        # unrelated keys to make sure they don't leak in
        "tctl": 55.0,
        "vrm temp": 48.0,
    }
    ccd_max, spread = collector._extract_ccd_stats(sensors)
    assert ccd_max == 70.5
    assert spread == 9.5


def test_extract_ccd_stats_none_when_no_ccd_keys_even_with_tctl() -> None:
    """Threadripper 7980X Storm Peak on the TRX50 AI TOP bench kernel
    only exposes ``tctl`` — no per-CCD labels. We intentionally do NOT
    fall back to Tctl: ``cpu_temp_c`` already shows Tctl, so duplicating
    it as CCDmax would imply we have per-die data we don't. Dashed row
    is more honest."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    sensors = {"tctl": 55.0, "package": 60.0}
    ccd_max, spread = collector._extract_ccd_stats(sensors)
    assert ccd_max is None
    assert spread is None


def test_extract_ccd_stats_none_when_no_cpu_sensors_at_all() -> None:
    """Intel with no CPU temps in this dict — (None, None) cleanly."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    sensors = {"fan1_input": 800.0, "in1_input": 12.0}
    ccd_max, spread = collector._extract_ccd_stats(sensors)
    assert ccd_max is None
    assert spread is None


def test_extract_ccd_stats_accepts_ccd_without_t_prefix() -> None:
    """Some drivers expose labels as ``CCD1`` rather than ``Tccd1``.
    Our reader lowercases keys, so we match both forms."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    sensors = {"ccd1": 60.0, "ccd2": 68.0, "tctl": 55.0}
    ccd_max, spread = collector._extract_ccd_stats(sensors)
    assert ccd_max == 68.0
    assert spread == 8.0


def test_extract_ccd_stats_ignores_ccd_prefixed_non_sensor_keys() -> None:
    """``ccdparent`` or similar non-numeric-suffix keys must not be
    confused for CCD temp labels. With no real per-CCD keys and no
    fallback, the result is (None, None)."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    sensors = {"ccdparent": 45.0, "tctl": 50.0}
    ccd_max, spread = collector._extract_ccd_stats(sensors)
    assert ccd_max is None
    assert spread is None


def test_extract_ccd_stats_ignores_nonpositive_values() -> None:
    """A CCD reporting 0 or negative is unpopulated / sensor failure.
    Skip rather than drag the min down and fake an inflated spread."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    sensors = {
        "tccd1": 60.0,
        "tccd2": 65.0,
        "tccd3": 0.0,      # unpopulated
        "tccd4": -128.0,   # sensor error
    }
    ccd_max, spread = collector._extract_ccd_stats(sensors)
    assert ccd_max == 65.0
    assert spread == 5.0  # 65 - 60, not 65 - (-128)


def test_extract_ccd_stats_single_ccd_produces_zero_spread() -> None:
    """Desktop Ryzen with only Tccd1 — spread = 0 (max == min), not None."""
    collector = TelemetryCollector.__new__(TelemetryCollector)
    sensors = {"tccd1": 58.0}
    ccd_max, spread = collector._extract_ccd_stats(sensors)
    assert ccd_max == 58.0
    assert spread == 0.0


def test_read_pcie_aer_corr_total_returns_none_when_no_sysfs(tmp_path, monkeypatch) -> None:
    """Kernels without PCIEAER_ROOTPORT or systems where /sys/bus/pci
    isn't readable — return None cleanly, never raise."""
    import psfu.collectors.telemetry as tel

    class FakePath:
        def __init__(self, p: str) -> None:
            self._p = p

        def exists(self) -> bool:
            return False

    monkeypatch.setattr(tel, "Path", lambda p: FakePath(p))
    collector = TelemetryCollector.__new__(TelemetryCollector)
    assert collector._read_pcie_aer_corr_total() is None


def test_read_pcie_aer_corr_total_sums_all_devices(tmp_path) -> None:
    """On a real sysfs tree, we sum the numeric-second-column values
    across every aer_dev_correctable file. Each file contains key/value
    rows like ``RxErr 5`` — the sum is absolute; the consumer diffs
    against a phase-start baseline to detect growth."""
    pci_root = tmp_path / "sys" / "bus" / "pci" / "devices"
    dev1 = pci_root / "0000_00_00.0"
    dev2 = pci_root / "0000_01_00.0"
    dev1.mkdir(parents=True)
    dev2.mkdir(parents=True)
    (dev1 / "aer_dev_correctable").write_text(
        "RxErr 2\nBadTLP 0\nBadDLLP 3\nRollover 0\nTimeout 0\nNonFatalErr 1\nCorrIntErr 0\nHeaderOF 0\nTOTAL_ERR_COR 6\n"
    )
    (dev2 / "aer_dev_correctable").write_text(
        "RxErr 0\nBadTLP 0\nBadDLLP 0\nRollover 0\nTimeout 0\nNonFatalErr 0\nCorrIntErr 0\nHeaderOF 0\nTOTAL_ERR_COR 0\n"
    )

    import psfu.collectors.telemetry as tel
    _orig_path = tel.Path
    # Point Path("/sys/bus/pci/devices") at our tmp_path.
    def fake_path(p):
        if p == "/sys/bus/pci/devices":
            return pci_root
        return _orig_path(p)
    tel.Path = fake_path
    try:
        collector = TelemetryCollector.__new__(TelemetryCollector)
        total = collector._read_pcie_aer_corr_total()
    finally:
        tel.Path = _orig_path

    # dev1: 2 + 0 + 3 + 0 + 0 + 1 + 0 + 0 + 6 = 12
    # dev2: 0
    assert total == 12


def test_read_nvme_temps_splits_sensors(tmp_path) -> None:
    """NVMe hwmon exposes temp1_input (composite), temp2_input (sensor 1,
    typically controller), temp3_input (sensor 2, typically NAND). Values
    are in milli-degrees. SATA drivetemp only contributes to composite."""
    hwmon = tmp_path / "sys" / "class" / "hwmon"
    nvme_dev = hwmon / "hwmon3"
    sata_dev = hwmon / "hwmon5"
    nvme_dev.mkdir(parents=True)
    sata_dev.mkdir(parents=True)
    (nvme_dev / "name").write_text("nvme\n")
    (nvme_dev / "temp1_input").write_text("45000\n")   # composite 45 C
    (nvme_dev / "temp2_input").write_text("52000\n")   # controller 52 C
    (nvme_dev / "temp3_input").write_text("48000\n")   # NAND 48 C
    (sata_dev / "name").write_text("drivetemp\n")
    (sata_dev / "temp1_input").write_text("40000\n")   # sata ssd 40 C

    import psfu.collectors.telemetry as tel
    _orig_path = tel.Path
    def fake_path(p):
        if p == "/sys/class/hwmon":
            return hwmon
        return _orig_path(p)
    tel.Path = fake_path
    try:
        collector = TelemetryCollector.__new__(TelemetryCollector)
        composite, ctrl, nand = collector._read_nvme_temps()
    finally:
        tel.Path = _orig_path

    # Composite takes the hottest across nvme.temp1 + sata.temp1.
    assert composite == 45.0
    assert ctrl == 52.0
    assert nand == 48.0


def test_read_nvme_temps_none_when_no_hwmon(tmp_path) -> None:
    """No /sys/class/hwmon — all three values None, no exception."""
    import psfu.collectors.telemetry as tel
    _orig_path = tel.Path
    missing = tmp_path / "nope"
    def fake_path(p):
        if p == "/sys/class/hwmon":
            return missing
        return _orig_path(p)
    tel.Path = fake_path
    try:
        collector = TelemetryCollector.__new__(TelemetryCollector)
        composite, ctrl, nand = collector._read_nvme_temps()
    finally:
        tel.Path = _orig_path

    assert (composite, ctrl, nand) == (None, None, None)
