"""Tests for S15 NVMe firmware blocklist checker."""

import json
import pytest
from pathlib import Path

from psfu.collectors.nvme_fw_check import (
    load_fw_blocklist,
    check_nvme_fw_revisions,
    _entry_matches,
)


# ---------------------------------------------------------------------------
# load_fw_blocklist
# ---------------------------------------------------------------------------


def test_load_returns_empty_when_file_missing(tmp_path):
    assert load_fw_blocklist(tmp_path / "nonexistent.json") == []


def test_load_returns_empty_on_invalid_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not-json", encoding="utf-8")
    assert load_fw_blocklist(bad) == []


def test_load_returns_empty_when_entries_key_missing(tmp_path):
    f = tmp_path / "bl.json"
    f.write_text(json.dumps({"_comment": "no entries key"}), encoding="utf-8")
    assert load_fw_blocklist(f) == []


def test_load_filters_out_entries_without_fw_rev(tmp_path):
    data = {"entries": [{"model_pattern": "foo"}, {"fw_rev": "1.0"}]}
    f = tmp_path / "bl.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    result = load_fw_blocklist(f)
    assert len(result) == 1
    assert result[0]["fw_rev"] == "1.0"


def test_load_returns_all_valid_entries(tmp_path):
    entries = [{"fw_rev": "A"}, {"fw_rev": "B", "model_pattern": "SN770"}]
    f = tmp_path / "bl.json"
    f.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    assert len(load_fw_blocklist(f)) == 2


# ---------------------------------------------------------------------------
# _entry_matches
# ---------------------------------------------------------------------------


def _ctrl(fw_rev="1.0", model="WD_BLACK SN850X", vid=0x15b7):
    return {"fr": fw_rev, "mn": model, "vid": vid}


def test_matches_exact_fw_rev():
    assert _entry_matches(_ctrl(), {"fw_rev": "1.0"})


def test_no_match_on_fw_rev_mismatch():
    assert not _entry_matches(_ctrl(), {"fw_rev": "2.0"})


def test_matches_with_model_pattern_substring():
    assert _entry_matches(_ctrl(), {"fw_rev": "1.0", "model_pattern": "SN850X"})


def test_no_match_when_model_pattern_not_in_model():
    assert not _entry_matches(_ctrl(), {"fw_rev": "1.0", "model_pattern": "SN750"})


def test_matches_with_vendor_id_int():
    assert _entry_matches(_ctrl(vid=0x15b7), {"fw_rev": "1.0", "vendor_id": "0x15b7"})


def test_no_match_on_vendor_id_mismatch():
    assert not _entry_matches(_ctrl(vid=0x15b7), {"fw_rev": "1.0", "vendor_id": "0x144d"})


def test_matches_all_fields_present():
    assert _entry_matches(
        _ctrl(fw_rev="1.0", model="WD_BLACK SN850X", vid=0x15b7),
        {"fw_rev": "1.0", "model_pattern": "SN850X", "vendor_id": "0x15b7"},
    )


def test_fw_rev_strip_whitespace():
    assert _entry_matches({"fr": "  1.0  ", "mn": ""}, {"fw_rev": "1.0"})


# ---------------------------------------------------------------------------
# check_nvme_fw_revisions
# ---------------------------------------------------------------------------


def _make_blocklist_file(tmp_path, entries):
    f = tmp_path / "bl.json"
    f.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return f


def test_check_returns_empty_when_no_devices():
    assert check_nvme_fw_revisions(None) == []
    assert check_nvme_fw_revisions([]) == []


def test_check_returns_empty_when_blocklist_empty(tmp_path):
    bl = _make_blocklist_file(tmp_path, [])
    assert check_nvme_fw_revisions(["/dev/nvme0n1"], blocklist_path=bl) == []


def test_check_returns_empty_when_nvme_cli_absent(tmp_path, monkeypatch):
    monkeypatch.setattr("psfu.collectors.nvme_fw_check.nvme_cli_binary_path", lambda: None)
    bl = _make_blocklist_file(tmp_path, [{"fw_rev": "1.0"}])
    assert check_nvme_fw_revisions(["/dev/nvme0n1"], blocklist_path=bl) == []


def test_check_emits_warn_on_matching_fw(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "psfu.collectors.nvme_fw_check.nvme_cli_binary_path",
        lambda: "/usr/bin/nvme",
    )
    ctrl = {"fr": "1.0", "mn": "WD_BLACK SN850X", "vid": 0x15b7}
    monkeypatch.setattr(
        "psfu.collectors.nvme_fw_check._read_id_ctrl",
        lambda _bin, _dev: ctrl,
    )
    bl = _make_blocklist_file(tmp_path, [{"fw_rev": "1.0", "reason": "known bug"}])
    events = check_nvme_fw_revisions(["/dev/nvme0n1"], blocklist_path=bl)
    assert len(events) == 1
    sev, kind, msg = events[0]
    assert sev == "warn"
    assert kind == "nvme_oob_fw_version"
    assert "1.0" in msg
    assert "nvme0n1" in msg


def test_check_skips_device_when_id_ctrl_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "psfu.collectors.nvme_fw_check.nvme_cli_binary_path",
        lambda: "/usr/bin/nvme",
    )
    monkeypatch.setattr(
        "psfu.collectors.nvme_fw_check._read_id_ctrl",
        lambda _bin, _dev: None,
    )
    bl = _make_blocklist_file(tmp_path, [{"fw_rev": "1.0"}])
    assert check_nvme_fw_revisions(["/dev/nvme0n1"], blocklist_path=bl) == []


def test_check_one_event_per_device_first_match_wins(tmp_path, monkeypatch):
    """Two matching entries → only first fires per device."""
    monkeypatch.setattr(
        "psfu.collectors.nvme_fw_check.nvme_cli_binary_path",
        lambda: "/usr/bin/nvme",
    )
    ctrl = {"fr": "1.0", "mn": "SN850X", "vid": 0}
    monkeypatch.setattr(
        "psfu.collectors.nvme_fw_check._read_id_ctrl",
        lambda _bin, _dev: ctrl,
    )
    entries = [
        {"fw_rev": "1.0", "reason": "first"},
        {"fw_rev": "1.0", "reason": "second"},
    ]
    bl = _make_blocklist_file(tmp_path, entries)
    events = check_nvme_fw_revisions(["/dev/nvme0n1"], blocklist_path=bl)
    assert len(events) == 1
    assert "first" in events[0][2]


def test_check_emits_one_event_per_device(tmp_path, monkeypatch):
    """Two matching devices → two events."""
    monkeypatch.setattr(
        "psfu.collectors.nvme_fw_check.nvme_cli_binary_path",
        lambda: "/usr/bin/nvme",
    )
    ctrl = {"fr": "1.0", "mn": "SN850X", "vid": 0}
    monkeypatch.setattr(
        "psfu.collectors.nvme_fw_check._read_id_ctrl",
        lambda _bin, _dev: ctrl,
    )
    bl = _make_blocklist_file(tmp_path, [{"fw_rev": "1.0"}])
    events = check_nvme_fw_revisions(
        ["/dev/nvme0n1", "/dev/nvme1n1"], blocklist_path=bl
    )
    assert len(events) == 2


def test_check_no_event_when_fw_does_not_match(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "psfu.collectors.nvme_fw_check.nvme_cli_binary_path",
        lambda: "/usr/bin/nvme",
    )
    ctrl = {"fr": "2.0", "mn": "SN850X", "vid": 0}
    monkeypatch.setattr(
        "psfu.collectors.nvme_fw_check._read_id_ctrl",
        lambda _bin, _dev: ctrl,
    )
    bl = _make_blocklist_file(tmp_path, [{"fw_rev": "1.0"}])
    assert check_nvme_fw_revisions(["/dev/nvme0n1"], blocklist_path=bl) == []


# ---------------------------------------------------------------------------
# Verdict integration: nvme_oob_fw_version is routed as a storage WARN
# ---------------------------------------------------------------------------


def test_nvme_oob_fw_version_scores_as_storage_warn(tmp_path):
    """nvme_oob_fw_version events must produce a storage WARN, not PASS or FAIL."""
    from psfu.verdict.rules import VerdictEngine
    from psfu.tests.conftest import make_run_db, _base_sample

    def _add_samples(db):
        for _ in range(5):
            db.add_sample(_base_sample())

    db = make_run_db(tmp_path)
    _add_samples(db)
    db.add_event("preflight", "warn", "nvme_oob_fw_version",
                 "NVMe /dev/nvme0n1 (WD_BLACK SN850X) firmware 1.0 is blocklisted.")
    verdict, reasons, scope, subs = VerdictEngine.evaluate(db)
    assert subs["storage"]["status"] == "WARN"
    assert "storage.nvme_oob_fw_version.warn" in [
        item["rule_id"] for item in subs["storage"]["evidence"]
    ]
