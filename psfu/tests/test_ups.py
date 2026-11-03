"""Unit tests for the NUT (Network UPS Tools) collector.

Covers the pure-function parser (``_parse_upsc_output``) against real-
shape ``upsc`` output for both vendors PS-FU has on the bench:
  * Cyberpower PR1500LCD (via the NUT ``usbhid-ups`` driver)
  * APC Smart-UPS X 1500 (also via ``usbhid-ups``)

Plus the stateful pieces of ``UpsCollector`` that are not exercised by
the parser alone:
  * OL→OB transfer edge detection (cumulative counter)
  * Held-between-samples behavior when a poll drops a key
  * Snapshot shape always includes every documented column

The collector's threading and subprocess paths are intentionally NOT
exercised here — they sit behind the ``upsc`` CLI which is a Linux-only
runtime dep. The parser + state machine carry the failure-mode load,
and that is what the tests target.
"""

from __future__ import annotations

import pytest

from psfu.collectors.ups import (
    UpsCollector,
    _DERIVED_COLUMNS,
    _NUT_TO_COLUMN,
    _derive_est_power,
    _parse_model_va,
    _parse_upsc_output,
    _to_float,
)


# ---- fixtures: real-shape upsc output ----


CYBERPOWER_PR1500LCD = """
battery.charge: 100
battery.charge.low: 20
battery.charge.warning: 35
battery.mfr.date: CPS
battery.runtime: 2700
battery.runtime.low: 300
battery.type: PbAcid
battery.voltage: 27.0
battery.voltage.nominal: 24
device.mfr: CPS
device.model: CP1500PFCRM2U
device.serial: ABCD123456
device.type: ups
driver.name: usbhid-ups
driver.version: 2.7.4
input.frequency: 60.0
input.transfer.high: 140
input.transfer.low: 90
input.voltage: 121.0
input.voltage.nominal: 120
output.voltage: 121.0
ups.beeper.status: enabled
ups.delay.shutdown: 20
ups.delay.start: 30
ups.load: 13
ups.mfr: CPS
ups.model: CP1500PFCRM2U
ups.productid: 0501
ups.realpower.nominal: 1000
ups.serial: ABCD123456
ups.status: OL
ups.test.result: Done and passed
ups.timer.shutdown: -60
ups.timer.start: -60
ups.vendorid: 0764
""".strip()


APC_SMART_UPS = """
battery.charge: 100
battery.runtime: 1234
battery.voltage: 27.4
device.mfr: American Power Conversion
device.model: Smart-UPS X 1500
device.type: ups
input.frequency: 60.0
input.voltage: 119.6
output.voltage: 119.6
ups.firmware: UPS 09.3 / ID=18
ups.load: 22
ups.realpower: 264
ups.status: OL CHRG
""".strip()


# ---- parser ----


class TestParseUpscOutput:

    def test_cyberpower_pr1500lcd_shape(self) -> None:
        parsed = _parse_upsc_output(CYBERPOWER_PR1500LCD)
        # Every numeric column we map should land as a float.
        assert parsed["input.voltage"] == 121.0
        assert parsed["input.frequency"] == 60.0
        assert parsed["output.voltage"] == 121.0
        assert parsed["ups.load"] == 13.0
        assert parsed["battery.charge"] == 100.0
        assert parsed["battery.runtime"] == 2700.0
        # ups.status remains a string.
        assert parsed["ups.status"] == "OL"
        # Non-mapped strings are preserved (so a future caller can read
        # ups.mfr / device.serial without re-parsing).
        assert parsed["device.model"] == "CP1500PFCRM2U"
        assert parsed["device.serial"] == "ABCD123456"

    def test_apc_smart_ups_with_realpower(self) -> None:
        parsed = _parse_upsc_output(APC_SMART_UPS)
        assert parsed["input.voltage"] == 119.6
        # APC reports realpower live; Cyberpower only reports the nominal.
        assert parsed["ups.realpower"] == 264.0
        # Multi-token status survives untouched for downstream analysis.
        assert parsed["ups.status"] == "OL CHRG"

    def test_skips_blank_and_comment_lines(self) -> None:
        text = "\n".join([
            "# vendor banner",
            "",
            "input.voltage: 120.0",
            "   ",
            "# trailing comment",
        ])
        parsed = _parse_upsc_output(text)
        assert parsed == {"input.voltage": 120.0}

    def test_skips_lines_without_colon(self) -> None:
        text = "garbage line with no colon\ninput.voltage: 119.0"
        parsed = _parse_upsc_output(text)
        assert parsed == {"input.voltage": 119.0}

    def test_drops_numeric_column_with_unparseable_value(self) -> None:
        # A NUT bug or a non-numeric vendor variant should not crash;
        # the column simply does not appear and the collector will hold
        # its prior value.
        text = "input.voltage: not-a-number\nups.load: 17"
        parsed = _parse_upsc_output(text)
        assert "input.voltage" not in parsed
        assert parsed["ups.load"] == 17.0

    def test_empty_input_returns_empty_map(self) -> None:
        assert _parse_upsc_output("") == {}


class TestToFloat:

    @pytest.mark.parametrize("value,expected", [
        ("121.0", 121.0),
        ("0", 0.0),
        ("-60", -60.0),
        ("60", 60.0),
        ("", None),
        ("abc", None),
        ("12.3.4", None),
    ])
    def test_coercion(self, value: str, expected) -> None:
        assert _to_float(value) == expected


# ---- collector state machine ----


def _make_collector_with_state() -> UpsCollector:
    """A collector instance with no thread or subprocess dependencies.

    We never call ``start()`` so no thread spawns and no ``upsc``
    invocation occurs. We exercise ``_apply_poll`` directly to drive
    the state machine.
    """
    return UpsCollector(ups_name="test-ups")


class TestUpsCollectorState:

    def test_snapshot_shape_includes_every_column(self) -> None:
        col = _make_collector_with_state()
        snap = col.snapshot()
        # Every numeric column the parser knows about must appear in
        # the snapshot, even before the first poll lands.
        for column in _NUT_TO_COLUMN.values():
            assert column in snap, "missing column in snapshot: " + column
            assert snap[column] is None
        # Status / on-battery / transfer-count fields all present.
        assert snap["ups_status_raw"] is None
        assert snap["ups_on_battery"] is None
        assert snap["ups_transfer_count"] == 0

    def test_first_poll_populates_columns(self) -> None:
        col = _make_collector_with_state()
        parsed = _parse_upsc_output(CYBERPOWER_PR1500LCD)
        col._apply_poll(parsed)
        snap = col.snapshot()
        assert snap["ups_input_voltage_v"] == 121.0
        assert snap["ups_load_pct"] == 13.0
        assert snap["ups_battery_charge_pct"] == 100.0
        assert snap["ups_status_raw"] == "OL"
        assert snap["ups_on_battery"] == 0
        # First-ever poll on mains is NOT a transfer.
        assert snap["ups_transfer_count"] == 0

    def test_held_between_samples_when_key_drops(self) -> None:
        # A subsequent poll missing a numeric key must NOT clobber the
        # prior value — same UX contract as mem_bandwidth + turbostat.
        col = _make_collector_with_state()
        col._apply_poll(_parse_upsc_output(CYBERPOWER_PR1500LCD))
        col._apply_poll({"ups.status": "OL"})  # no numeric keys
        snap = col.snapshot()
        assert snap["ups_input_voltage_v"] == 121.0
        assert snap["ups_load_pct"] == 13.0

    def test_transfer_edge_detection_ol_to_ob(self) -> None:
        col = _make_collector_with_state()
        # Tick 1: OL — establishes prior_on_battery = 0.
        col._apply_poll({"ups.status": "OL"})
        assert col.snapshot()["ups_transfer_count"] == 0
        # Tick 2: OB — counted.
        col._apply_poll({"ups.status": "OB DISCHRG"})
        snap = col.snapshot()
        assert snap["ups_on_battery"] == 1
        assert snap["ups_transfer_count"] == 1
        # Tick 3: still OB — NOT counted (level, not edge).
        col._apply_poll({"ups.status": "OB DISCHRG"})
        assert col.snapshot()["ups_transfer_count"] == 1
        # Tick 4: OL CHRG — back on mains, no transfer increment.
        col._apply_poll({"ups.status": "OL CHRG"})
        snap = col.snapshot()
        assert snap["ups_on_battery"] == 0
        assert snap["ups_transfer_count"] == 1
        # Tick 5: OB again — second transfer counted.
        col._apply_poll({"ups.status": "OB"})
        assert col.snapshot()["ups_transfer_count"] == 2

    def test_first_poll_on_battery_is_not_a_transfer(self) -> None:
        # If we wake up to a UPS already on battery (e.g. mains was
        # already out when PS-FU started), do NOT count that as a
        # transfer — there is no prior OL state to transition from.
        col = _make_collector_with_state()
        col._apply_poll({"ups.status": "OB DISCHRG"})
        snap = col.snapshot()
        assert snap["ups_on_battery"] == 1
        assert snap["ups_transfer_count"] == 0

    def test_status_with_multiple_tokens(self) -> None:
        # NUT emits compound status tokens like "OL CHRG", "OB LB",
        # "OL OVER". Match on whole-token "OB" only — substring match
        # would false-positive on hypothetical statuses like "BOOST".
        col = _make_collector_with_state()
        col._apply_poll({"ups.status": "OL CHRG"})
        assert col.snapshot()["ups_on_battery"] == 0
        col._apply_poll({"ups.status": "OB LB"})
        assert col.snapshot()["ups_on_battery"] == 1

    def test_status_without_ob_token_is_on_mains(self) -> None:
        col = _make_collector_with_state()
        for status in ("OL", "OL CHRG", "OL BOOST", "OL TRIM"):
            col._apply_poll({"ups.status": status})
            assert col.snapshot()["ups_on_battery"] == 0, (
                "status {!r} misclassified as on-battery".format(status)
            )

    def test_apply_poll_with_no_status_leaves_on_battery_unchanged(self) -> None:
        # Missing ups.status should not flip the flag — leave the prior
        # state intact. Important so a NUT hiccup that drops one field
        # does not look like a transfer.
        col = _make_collector_with_state()
        col._apply_poll({"ups.status": "OB"})
        first = col.snapshot()
        col._apply_poll({"input.voltage": 119.0})  # no status field
        second = col.snapshot()
        assert second["ups_on_battery"] == first["ups_on_battery"] == 1
        assert second["ups_transfer_count"] == first["ups_transfer_count"] == 0


# ---- model-VA parser ----


class TestParseModelVa:

    @pytest.mark.parametrize("model,expected", [
        # CyberPower — rack and tower forms both embed the VA.
        ("CP1500PFCLCD",   1500.0),
        ("CP1500PFCRM2U",  1500.0),  # "2U" rack-height must NOT win
        ("CP900AVR",        900.0),
        ("CP1350PFCLCD",   1350.0),
        # APC Smart-UPS and Back-UPS.
        ("Smart-UPS X 1500", 1500.0),
        ("Smart-UPS 750",     750.0),
        ("Back-UPS 650",      650.0),
        ("SMT2200",          2200.0),
        ("SMT3000RM2U",      3000.0),  # "2U" excluded, 3000 wins
        # Eaton / Tripp Lite / Vertiv.
        ("5PX 2200",         2200.0),
        ("SMART1500LCD",     1500.0),
        ("GXT4-2000RT120",   2000.0),  # "120" is voltage, under the 200 floor — ignored; "2000" wins
        # Larger workstation-scale units.
        ("SURT10000XLT",    10000.0),
    ])
    def test_parses_plausible_va_from_model(self, model: str, expected: float) -> None:
        assert _parse_model_va(model) == expected

    @pytest.mark.parametrize("model", [
        "",
        "TX",                 # no digits
        "Smart-UPS",          # no digits
        "Rack-2U",            # "2" well under _MIN_PLAUSIBLE_VA
        "FW-1.2.3",           # small version numbers only
    ])
    def test_returns_none_when_no_plausible_va(self, model: str) -> None:
        assert _parse_model_va(model) is None

    def test_returns_none_on_none_input(self) -> None:
        assert _parse_model_va(None) is None

    def test_returns_none_when_only_tiny_numbers(self) -> None:
        # Every number in range [0, _MIN_PLAUSIBLE_VA) — all rejected.
        assert _parse_model_va("Model-2U-Rev3-Fw12") is None

    def test_returns_none_when_number_exceeds_plausible_max(self) -> None:
        # 50,000 VA is a data-center PDU, not a workstation UPS.
        # Outside [200, 20000] range so the regex ignores it.
        assert _parse_model_va("Industrial-50000VA-3Phase") is None

    def test_picks_largest_in_range_when_multiple_numbers(self) -> None:
        # Belt-and-suspenders for the "2U rack height" class of confusion.
        # 2 is out of range; 1500 is the only plausible number.
        assert _parse_model_va("CP1500PFCRM2U") == 1500.0
        # Both in range — largest wins (the VA is the headline number).
        assert _parse_model_va("SomeModel 1000 with 1500 VA") == 1500.0


# ---- derived power helper ----


class TestDeriveEstPower:

    def test_straight_multiplication(self) -> None:
        # 13 % of a 1000 W plate = 130 W. Same math PowerPanel uses.
        assert _derive_est_power(13.0, 1000.0) == 130.0

    def test_zero_load_is_zero(self) -> None:
        assert _derive_est_power(0.0, 1500.0) == 0.0

    def test_full_load_is_nominal(self) -> None:
        assert _derive_est_power(100.0, 1500.0) == 1500.0

    @pytest.mark.parametrize("load,nominal", [
        (None, 1000.0),
        (13.0, None),
        (None, None),
    ])
    def test_returns_none_when_input_missing(
        self, load, nominal,
    ) -> None:
        assert _derive_est_power(load, nominal) is None


# ---- collector-level derived columns in snapshot() ----


class TestSnapshotDerivedColumns:

    def test_derived_columns_in_snapshot_shape(self) -> None:
        # Every derived column must appear in the snapshot, even before
        # any poll lands — callers rely on the stable dict shape.
        col = _make_collector_with_state()
        snap = col.snapshot()
        for column in _DERIVED_COLUMNS:
            assert column in snap, "missing derived column: " + column
            assert snap[column] is None, (
                "{} should be None pre-poll, got {!r}".format(column, snap[column])
            )

    def test_cyberpower_populates_est_watts_via_nominal(self) -> None:
        # CP1500PFCRM2U:
        #   ups.load               = 13
        #   ups.realpower.nominal  = 1000  (plate wattage — firmware plate)
        #   ups.model              = "CP1500PFCRM2U" → 1500 VA parsed
        # Expected derived:
        #   est_w  = 13 × 1000 / 100   = 130.0
        #   est_va = 13 × 1500 / 100   = 195.0
        # Live ups_realpower_w stays None because CyberPower firmware
        # doesn't emit it — this is the central regression test.
        col = _make_collector_with_state()
        col._apply_poll(_parse_upsc_output(CYBERPOWER_PR1500LCD))
        snap = col.snapshot()
        assert snap["ups_realpower_w"] is None
        assert snap["ups_realpower_nominal_w"] == 1000.0
        assert snap["ups_realpower_est_w"] == 130.0
        assert snap["ups_apparent_nominal_va"] == 1500.0
        assert snap["ups_apparent_est_va"] == 195.0

    def test_apc_populates_live_and_derived_both(self) -> None:
        # APC Smart-UPS X 1500:
        #   ups.load              = 22
        #   ups.realpower         = 264  (firmware live)
        #   ups.model             = "Smart-UPS X 1500" → 1500 VA parsed
        # Nominal wattage is NOT emitted by APC in this fixture, so
        # est_w must degrade to None cleanly — we never fall back to a
        # wrong number.
        col = _make_collector_with_state()
        col._apply_poll(_parse_upsc_output(APC_SMART_UPS))
        snap = col.snapshot()
        assert snap["ups_realpower_w"] == 264.0
        assert snap["ups_realpower_nominal_w"] is None
        assert snap["ups_realpower_est_w"] is None
        assert snap["ups_apparent_nominal_va"] == 1500.0
        # 22 % × 1500 VA = 330 VA
        assert snap["ups_apparent_est_va"] == 330.0

    def test_model_parse_cached_and_survives_poll_without_model(self) -> None:
        # Second poll drops ups.model (NUT hiccup) — apparent_nominal_va
        # must remain populated from the first poll's cache so est_va
        # stays live. Same "held-between-samples" contract as the numeric
        # columns and mem_bandwidth/turbostat.
        col = _make_collector_with_state()
        col._apply_poll(_parse_upsc_output(CYBERPOWER_PR1500LCD))
        first = col.snapshot()
        assert first["ups_apparent_nominal_va"] == 1500.0
        # Load changes to 20%, model string absent on this poll.
        col._apply_poll({"ups.load": 20.0, "ups.status": "OL"})
        snap = col.snapshot()
        assert snap["ups_apparent_nominal_va"] == 1500.0
        # 20 % × 1500 VA = 300 VA
        assert snap["ups_apparent_est_va"] == 300.0
        # ups.realpower.nominal was held from the first poll too.
        assert snap["ups_realpower_nominal_w"] == 1000.0
        # 20 % × 1000 W = 200 W
        assert snap["ups_realpower_est_w"] == 200.0

    def test_zero_load_derives_zero_watts_not_none(self) -> None:
        # A legitimately idle UPS (load=0) must derive 0 W, not None —
        # None would make operators think the probe is broken.
        col = _make_collector_with_state()
        col._apply_poll({
            "ups.load": 0.0,
            "ups.realpower.nominal": 1000.0,
            "ups.model": "CP1500PFCLCD",
            "ups.status": "OL",
        })
        snap = col.snapshot()
        assert snap["ups_realpower_est_w"] == 0.0
        assert snap["ups_apparent_est_va"] == 0.0

    def test_missing_nominal_means_est_w_is_none(self) -> None:
        # No way to derive watts without a nominal plate — must not
        # hallucinate a number.
        col = _make_collector_with_state()
        col._apply_poll({
            "ups.load": 25.0,
            "ups.status": "OL",
            # no ups.realpower.nominal, no ups.model
        })
        snap = col.snapshot()
        assert snap["ups_realpower_est_w"] is None
        assert snap["ups_apparent_est_va"] is None
