"""S7 (v0.18) — OCP 0xC0 SMART Extended parser + event emitter tests."""

from __future__ import annotations

import struct

import pytest

from psfu.collectors.storage_ocp import (
    _parse_ocp_log_page,
    storage_ocp_events,
)


def _build_ocp_page(
    *,
    pmu_written: int = 0,
    bad_user_nand: int = 0,
    bad_system_nand: int = 0,
    xor_recovery: int = 0,
    uncorrectable_read: int = 0,
    version: int = 3,
    size: int = 512,
) -> bytes:
    """Build a synthetic 512 B OCP 0xC0 log page from field values.

    Uses OCP 2.0 byte offsets (§5.12). Fields not specified default to
    zero. Used by parser tests and event-emitter tests both."""
    buf = bytearray(size)
    # Physical Media Units Written: bytes 0-15, 128-bit LE; low 64 bits.
    struct.pack_into("<Q", buf, 0, pmu_written & 0xFFFFFFFFFFFFFFFF)
    # Bad User NAND Blocks Raw: bytes 32-37, 6 bytes LE.
    buf[32:38] = (bad_user_nand & ((1 << 48) - 1)).to_bytes(6, "little")
    # Bad System NAND Blocks Raw: bytes 40-45, 6 bytes LE.
    buf[40:46] = (bad_system_nand & ((1 << 48) - 1)).to_bytes(6, "little")
    # XOR Recovery Count: bytes 48-55, 8 bytes LE.
    struct.pack_into("<Q", buf, 48, xor_recovery & 0xFFFFFFFFFFFFFFFF)
    # Uncorrectable Read Error Count: bytes 56-63, 8 bytes LE.
    struct.pack_into("<Q", buf, 56, uncorrectable_read & 0xFFFFFFFFFFFFFFFF)
    # Log Page Version: bytes 494-495, u16 LE.
    if size >= 496:
        struct.pack_into("<H", buf, 494, version & 0xFFFF)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parser_extracts_all_scored_fields() -> None:
    page = _build_ocp_page(
        pmu_written=12_345,
        bad_user_nand=3,
        bad_system_nand=1,
        xor_recovery=7,
        uncorrectable_read=2,
        version=3,
    )
    parsed = _parse_ocp_log_page(page)
    assert parsed is not None
    assert parsed["physical_media_units_written"] == 12_345
    assert parsed["bad_user_nand_blocks"] == 3
    assert parsed["bad_system_nand_blocks"] == 1
    assert parsed["xor_recovery_count"] == 7
    assert parsed["uncorrectable_read_errors"] == 2
    assert parsed["log_page_version"] == 3


def test_parser_rejects_short_page() -> None:
    page = _build_ocp_page(version=3)[:400]
    assert _parse_ocp_log_page(page) is None


def test_parser_rejects_empty_input() -> None:
    assert _parse_ocp_log_page(b"") is None
    assert _parse_ocp_log_page(None) is None  # type: ignore[arg-type]


@pytest.mark.parametrize("version", [0, 1, 5, 99])
def test_parser_rejects_unsupported_version(version: int) -> None:
    """OCP 0.x / pre-spec vendor stubs (version 0/1) and future layouts
    (version > 4) must be rejected — we don't know the field offsets."""
    page = _build_ocp_page(version=version)
    assert _parse_ocp_log_page(page) is None


@pytest.mark.parametrize("version", [2, 3, 4])
def test_parser_accepts_ocp_v2_v3_v4(version: int) -> None:
    """OCP 1.0 (v2), 2.0 (v3), and 2.5 (v4) share the five scored offsets."""
    page = _build_ocp_page(bad_user_nand=5, version=version)
    parsed = _parse_ocp_log_page(page)
    assert parsed is not None
    assert parsed["log_page_version"] == version
    assert parsed["bad_user_nand_blocks"] == 5


def test_parser_handles_all_zero_healthy_page() -> None:
    """A brand-new drive's OCP page is all zeros except version — must parse
    cleanly, not False-y out."""
    page = _build_ocp_page(version=3)
    parsed = _parse_ocp_log_page(page)
    assert parsed is not None
    assert parsed["bad_user_nand_blocks"] == 0
    assert parsed["uncorrectable_read_errors"] == 0


def test_parser_handles_max_48bit_user_nand() -> None:
    """Bad User NAND is a 6-byte raw counter — confirm we don't truncate on
    wraparound close to 2^48."""
    page = _build_ocp_page(bad_user_nand=(1 << 48) - 1, version=3)
    parsed = _parse_ocp_log_page(page)
    assert parsed is not None
    assert parsed["bad_user_nand_blocks"] == (1 << 48) - 1


# ---------------------------------------------------------------------------
# storage_ocp_events emitter
# ---------------------------------------------------------------------------


def _snapshot(devices: list) -> dict:
    return {"captured_at": "2026-04-19T00:00:00Z", "devices": devices}


def _dev(device: str, *, supported: bool = True, **fields) -> dict:
    base = {
        "device": device,
        "supported": supported,
        "physical_media_units_written": None,
        "bad_user_nand_blocks": None,
        "bad_system_nand_blocks": None,
        "xor_recovery_count": None,
        "uncorrectable_read_errors": None,
        "log_page_version": None,
    }
    if supported:
        base.update({
            "physical_media_units_written": 0,
            "bad_user_nand_blocks": 0,
            "bad_system_nand_blocks": 0,
            "xor_recovery_count": 0,
            "uncorrectable_read_errors": 0,
            "log_page_version": 3,
        })
    base.update(fields)
    return base


def test_events_empty_when_no_growth() -> None:
    start = _snapshot([_dev("/dev/nvme0n1")])
    end = _snapshot([_dev("/dev/nvme0n1")])
    assert storage_ocp_events(start, end) == []


def test_events_fire_on_bad_user_nand_growth() -> None:
    start = _snapshot([_dev("/dev/nvme0n1", bad_user_nand_blocks=0)])
    end = _snapshot([_dev("/dev/nvme0n1", bad_user_nand_blocks=3)])
    events = storage_ocp_events(start, end)
    kinds = [k for _sev, k, _msg in events]
    assert "ssd_ocp_bad_user_nand_growth" in kinds
    # Message must include the device + delta + raw pre/post for forensics.
    msg = next(m for s, k, m in events if k == "ssd_ocp_bad_user_nand_growth")
    assert "/dev/nvme0n1" in msg
    assert "0 -> 3" in msg or "0 → 3" in msg


def test_events_fire_on_bad_system_nand_growth() -> None:
    start = _snapshot([_dev("/dev/nvme0n1", bad_system_nand_blocks=0)])
    end = _snapshot([_dev("/dev/nvme0n1", bad_system_nand_blocks=1)])
    events = storage_ocp_events(start, end)
    assert any(k == "ssd_ocp_bad_system_nand_growth" for _s, k, _m in events)


def test_events_fire_on_xor_recovery_growth() -> None:
    start = _snapshot([_dev("/dev/nvme0n1", xor_recovery_count=5)])
    end = _snapshot([_dev("/dev/nvme0n1", xor_recovery_count=6)])
    events = storage_ocp_events(start, end)
    assert any(k == "ssd_ocp_xor_recovery_growth" for _s, k, _m in events)


def test_events_fire_on_uncorrectable_read_growth() -> None:
    start = _snapshot([_dev("/dev/nvme0n1", uncorrectable_read_errors=0)])
    end = _snapshot([_dev("/dev/nvme0n1", uncorrectable_read_errors=1)])
    events = storage_ocp_events(start, end)
    assert any(k == "ssd_ocp_uncorrectable_read_growth" for _s, k, _m in events)


def test_events_multiple_growth_fields_emit_multiple_events() -> None:
    start = _snapshot([_dev("/dev/nvme0n1",
                             bad_user_nand_blocks=0,
                             xor_recovery_count=0)])
    end = _snapshot([_dev("/dev/nvme0n1",
                           bad_user_nand_blocks=2,
                           xor_recovery_count=5)])
    events = storage_ocp_events(start, end)
    kinds = sorted(k for _s, k, _m in events)
    assert kinds == [
        "ssd_ocp_bad_user_nand_growth",
        "ssd_ocp_xor_recovery_growth",
    ]


def test_events_info_on_drive_unsupported_at_end() -> None:
    """Drive supported OCP at start but not at end — single info-only event
    per device, never FAIL (no comparable counters)."""
    start = _snapshot([_dev("/dev/nvme0n1")])
    end = _snapshot([_dev("/dev/nvme0n1", supported=False)])
    events = storage_ocp_events(start, end)
    assert len(events) == 1
    sev, kind, _msg = events[0]
    assert sev == "info"
    assert kind == "ssd_ocp_not_supported"


def test_events_info_on_drive_unsupported_at_start() -> None:
    start = _snapshot([_dev("/dev/nvme0n1", supported=False)])
    end = _snapshot([_dev("/dev/nvme0n1")])
    events = storage_ocp_events(start, end)
    assert len(events) == 1
    assert events[0][1] == "ssd_ocp_not_supported"


def test_events_never_fire_on_counter_rollback() -> None:
    """A lower post-run value (drive formatted / firmware-reset mid-run) must
    NOT trigger any growth event. That's a weird state worth logging later
    but absolutely not a storage FAIL — no media actually went bad."""
    start = _snapshot([_dev("/dev/nvme0n1", bad_user_nand_blocks=100)])
    end = _snapshot([_dev("/dev/nvme0n1", bad_user_nand_blocks=0)])
    events = storage_ocp_events(start, end)
    assert events == []


def test_events_multi_device_independent() -> None:
    start = _snapshot([
        _dev("/dev/nvme0n1", bad_user_nand_blocks=0),
        _dev("/dev/nvme1n1", bad_user_nand_blocks=0),
    ])
    end = _snapshot([
        _dev("/dev/nvme0n1", bad_user_nand_blocks=0),
        _dev("/dev/nvme1n1", bad_user_nand_blocks=4),
    ])
    events = storage_ocp_events(start, end)
    growth = [m for _s, k, m in events if k == "ssd_ocp_bad_user_nand_growth"]
    assert len(growth) == 1
    assert "/dev/nvme1n1" in growth[0]
    assert "/dev/nvme0n1" not in growth[0]
