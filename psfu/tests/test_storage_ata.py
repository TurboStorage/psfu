"""S9 (v0.19) — ATA SMART attribute parser + event emitter tests."""

from __future__ import annotations

import json

import pytest

from psfu.collectors.storage_ata import (
    _parse_smartctl_json,
    storage_ata_events,
)


def _build_smartctl_doc(
    *,
    reallocated: int = 0,
    pending: int = 0,
    offline_uncorr: int = 0,
    udma_crc: int = 0,
    model: str = "Samsung SSD 860 EVO 1TB",
    serial: str = "S4ABCDEFGH",
    firmware: str = "RVT04B6Q",
    include_extra_attrs: bool = True,
) -> str:
    """Build synthetic smartctl -a --json output with the 4 scored ATA
    attributes. include_extra_attrs adds a few unrelated attribute rows
    to make sure the parser filters correctly."""
    table = [
        {"id": 5,   "name": "Reallocated_Sector_Ct",  "raw": {"value": reallocated, "string": str(reallocated)}},
        {"id": 197, "name": "Current_Pending_Sector", "raw": {"value": pending,     "string": str(pending)}},
        {"id": 198, "name": "Offline_Uncorrectable",  "raw": {"value": offline_uncorr, "string": str(offline_uncorr)}},
        {"id": 199, "name": "UDMA_CRC_Error_Count",   "raw": {"value": udma_crc,    "string": str(udma_crc)}},
    ]
    if include_extra_attrs:
        # Unrelated attributes that should be ignored by the parser.
        table.extend([
            {"id": 9,   "name": "Power_On_Hours",       "raw": {"value": 12345, "string": "12345"}},
            {"id": 12,  "name": "Power_Cycle_Count",    "raw": {"value": 200,   "string": "200"}},
            {"id": 177, "name": "Wear_Leveling_Count",  "raw": {"value": 5,     "string": "5"}},
        ])
    return json.dumps({
        "model_name": model,
        "serial_number": serial,
        "firmware_version": firmware,
        "ata_smart_attributes": {"table": table},
    })


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parser_extracts_all_scored_attributes() -> None:
    doc = _build_smartctl_doc(
        reallocated=3, pending=1, offline_uncorr=2, udma_crc=7,
    )
    parsed = _parse_smartctl_json(doc)
    assert parsed is not None
    assert parsed["reallocated_sector_ct"] == 3
    assert parsed["current_pending_sector"] == 1
    assert parsed["offline_uncorrectable"] == 2
    assert parsed["udma_crc_error_count"] == 7
    assert parsed["model"] == "Samsung SSD 860 EVO 1TB"
    assert parsed["firmware"] == "RVT04B6Q"


def test_parser_ignores_unrelated_attributes() -> None:
    """Power-on hours and wear-leveling count in the same table must
    NOT leak into the scored field dict."""
    doc = _build_smartctl_doc(include_extra_attrs=True)
    parsed = _parse_smartctl_json(doc)
    assert parsed is not None
    # Scored keys plus identity only — no Power_On_Hours, etc.
    expected_keys = {
        "reallocated_sector_ct",
        "current_pending_sector",
        "offline_uncorrectable",
        "udma_crc_error_count",
        "model",
        "serial",
        "firmware",
    }
    assert set(parsed.keys()) == expected_keys


def test_parser_returns_none_when_no_ata_block() -> None:
    """NVMe drives through smartctl have `nvme_smart_health_information_log`
    but no `ata_smart_attributes` key — must return None, not a dict of
    None values."""
    doc = json.dumps({
        "model_name": "Samsung 990 PRO 2TB",
        "nvme_smart_health_information_log": {"critical_warning": 0},
    })
    assert _parse_smartctl_json(doc) is None


def test_parser_returns_none_on_empty_table() -> None:
    """Some USB bridges hand smartctl an empty attribute table — treat
    as unsupported, don't emit zero-valued fields."""
    doc = json.dumps({
        "model_name": "USB Bridge",
        "ata_smart_attributes": {"table": []},
    })
    assert _parse_smartctl_json(doc) is None


def test_parser_returns_none_on_malformed_json() -> None:
    assert _parse_smartctl_json("not json") is None
    assert _parse_smartctl_json("") is None
    assert _parse_smartctl_json(None) is None


def test_parser_skips_rows_without_integer_raw_value() -> None:
    """Some smartctl builds emit string raw values for certain vendor
    attrs (e.g. temperature strings). Must not crash, must skip."""
    doc = json.dumps({
        "ata_smart_attributes": {
            "table": [
                {"id": 5,   "name": "Reallocated", "raw": {"value": "0", "string": "0"}},
                {"id": 197, "name": "Pending",    "raw": {"value": 2,   "string": "2"}},
            ],
        },
    })
    parsed = _parse_smartctl_json(doc)
    # Attr 5 had string raw value — None. Attr 197 parsed — 2.
    assert parsed is not None
    assert parsed["reallocated_sector_ct"] is None
    assert parsed["current_pending_sector"] == 2


def test_parser_handles_all_zero_healthy_drive() -> None:
    """Brand-new drive: all four scored counters at 0 must still parse
    cleanly, not False-y out."""
    doc = _build_smartctl_doc()
    parsed = _parse_smartctl_json(doc)
    assert parsed is not None
    assert parsed["reallocated_sector_ct"] == 0
    assert parsed["udma_crc_error_count"] == 0


def test_parser_handles_large_raw_counters() -> None:
    """UDMA CRC can count into the tens of thousands on a decade-old
    cable — confirm we don't clamp or truncate."""
    doc = _build_smartctl_doc(udma_crc=50_000)
    parsed = _parse_smartctl_json(doc)
    assert parsed is not None
    assert parsed["udma_crc_error_count"] == 50_000


# ---------------------------------------------------------------------------
# storage_ata_events emitter
# ---------------------------------------------------------------------------


def _snapshot(devices: list) -> dict:
    return {"captured_at": "2026-04-19T00:00:00Z", "devices": devices}


def _dev(device: str, *, supported: bool = True, **fields) -> dict:
    base = {
        "device": device,
        "supported": supported,
        "reallocated_sector_ct": None,
        "current_pending_sector": None,
        "offline_uncorrectable": None,
        "udma_crc_error_count": None,
        "model": None,
        "serial": None,
        "firmware": None,
    }
    if supported:
        base.update({
            "reallocated_sector_ct": 0,
            "current_pending_sector": 0,
            "offline_uncorrectable": 0,
            "udma_crc_error_count": 0,
            "model": "Samsung SSD 860 EVO 1TB",
            "serial": "S4ABCDEFGH",
            "firmware": "RVT04B6Q",
        })
    base.update(fields)
    return base


def test_events_empty_when_no_growth() -> None:
    start = _snapshot([_dev("/dev/sda")])
    end = _snapshot([_dev("/dev/sda")])
    assert storage_ata_events(start, end) == []


def test_events_fail_on_reallocated_sector_growth() -> None:
    start = _snapshot([_dev("/dev/sda", reallocated_sector_ct=0)])
    end = _snapshot([_dev("/dev/sda", reallocated_sector_ct=3)])
    events = storage_ata_events(start, end)
    kinds = [(s, k) for s, k, _m in events]
    assert ("crit", "ata_reallocated_sector_grew") in kinds
    msg = next(m for s, k, m in events if k == "ata_reallocated_sector_grew")
    assert "/dev/sda" in msg
    assert "0 -> 3" in msg or "0 → 3" in msg


def test_events_fail_on_pending_sector_growth() -> None:
    start = _snapshot([_dev("/dev/sda", current_pending_sector=0)])
    end = _snapshot([_dev("/dev/sda", current_pending_sector=1)])
    events = storage_ata_events(start, end)
    assert any(s == "crit" and k == "ata_pending_sector_grew" for s, k, _m in events)


def test_events_fail_on_offline_uncorrectable_growth() -> None:
    start = _snapshot([_dev("/dev/sda", offline_uncorrectable=0)])
    end = _snapshot([_dev("/dev/sda", offline_uncorrectable=5)])
    events = storage_ata_events(start, end)
    assert any(s == "crit" and k == "ata_offline_uncorrectable_grew" for s, k, _m in events)


def test_events_warn_not_fail_on_udma_crc_growth() -> None:
    """Critical regression guard: UDMA CRC must be WARN ("warning"),
    never FAIL ("crit"). Cable issue, not drive issue — wrong severity
    would condemn good drives."""
    start = _snapshot([_dev("/dev/sda", udma_crc_error_count=5)])
    end = _snapshot([_dev("/dev/sda", udma_crc_error_count=42)])
    events = storage_ata_events(start, end)
    udma_events = [(s, k) for s, k, _m in events if k == "ata_udma_crc_grew"]
    assert udma_events == [("warning", "ata_udma_crc_grew")]


def test_events_multiple_growth_fields_emit_multiple_events() -> None:
    start = _snapshot([_dev("/dev/sda",
                             reallocated_sector_ct=0,
                             udma_crc_error_count=0)])
    end = _snapshot([_dev("/dev/sda",
                           reallocated_sector_ct=2,
                           udma_crc_error_count=10)])
    events = storage_ata_events(start, end)
    kinds = sorted(k for _s, k, _m in events)
    assert kinds == [
        "ata_reallocated_sector_grew",
        "ata_udma_crc_grew",
    ]


def test_events_info_on_unsupported_at_end() -> None:
    start = _snapshot([_dev("/dev/sda")])
    end = _snapshot([_dev("/dev/sda", supported=False)])
    events = storage_ata_events(start, end)
    assert len(events) == 1
    sev, kind, _msg = events[0]
    assert sev == "info"
    assert kind == "ata_smart_not_supported"


def test_events_info_on_unsupported_at_start() -> None:
    """USB-SATA bridge that becomes visible mid-run — still info-only,
    no FAIL (no comparable baseline)."""
    start = _snapshot([_dev("/dev/sda", supported=False)])
    end = _snapshot([_dev("/dev/sda")])
    events = storage_ata_events(start, end)
    assert len(events) == 1
    assert events[0][1] == "ata_smart_not_supported"


def test_events_never_fire_on_counter_rollback() -> None:
    """A lower post-run value (drive replaced, SMART cleared by vendor
    tool) must NOT trigger any growth event. Growth only."""
    start = _snapshot([_dev("/dev/sda", reallocated_sector_ct=100)])
    end = _snapshot([_dev("/dev/sda", reallocated_sector_ct=0)])
    events = storage_ata_events(start, end)
    assert events == []


def test_events_multi_device_independent() -> None:
    start = _snapshot([
        _dev("/dev/sda", reallocated_sector_ct=0),
        _dev("/dev/sdb", reallocated_sector_ct=0),
    ])
    end = _snapshot([
        _dev("/dev/sda", reallocated_sector_ct=0),
        _dev("/dev/sdb", reallocated_sector_ct=4),
    ])
    events = storage_ata_events(start, end)
    growth = [m for _s, k, m in events if k == "ata_reallocated_sector_grew"]
    assert len(growth) == 1
    assert "/dev/sdb" in growth[0]
    assert "/dev/sda" not in growth[0]


def test_events_fail_plus_warn_mixed() -> None:
    """Real-world case: drive with growing reallocations AND a failing
    cable. Must emit one FAIL + one WARN event, both scoped to same
    device."""
    start = _snapshot([_dev("/dev/sda",
                             reallocated_sector_ct=0,
                             udma_crc_error_count=0)])
    end = _snapshot([_dev("/dev/sda",
                           reallocated_sector_ct=2,
                           udma_crc_error_count=10)])
    events = storage_ata_events(start, end)
    by_kind = {k: s for s, k, _m in events}
    assert by_kind["ata_reallocated_sector_grew"] == "crit"
    assert by_kind["ata_udma_crc_grew"] == "warning"
