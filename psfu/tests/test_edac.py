"""EDAC collector unit tests.

Tests the snapshot, delta, and event generation logic using synthetic data
(no real sysfs needed).
"""

from psfu.collectors.edac import edac_delta, edac_events_from_delta


def _snap(total_ce=0, total_ue=0, csrow_ce=None, csrow_ue=None):
    """Build a synthetic EDAC snapshot."""
    csrows = {}
    if csrow_ce or csrow_ue:
        csrows["csrow0"] = {
            "ce_count": csrow_ce or 0,
            "ue_count": csrow_ue or 0,
        }
    return {
        "available": True,
        "controllers": {
            "mc0": {
                "ce_count": total_ce,
                "ue_count": total_ue,
                "csrows": csrows,
            },
        },
        "total_ce": total_ce,
        "total_ue": total_ue,
    }


def test_delta_zero_when_no_errors():
    start = _snap(total_ce=0, total_ue=0)
    end = _snap(total_ce=0, total_ue=0)
    d = edac_delta(start, end)
    assert d["available"] is True
    assert d["delta_ce"] == 0
    assert d["delta_ue"] == 0


def test_delta_detects_correctable_errors():
    start = _snap(total_ce=5, total_ue=0, csrow_ce=5)
    end = _snap(total_ce=8, total_ue=0, csrow_ce=8)
    d = edac_delta(start, end)
    assert d["delta_ce"] == 3
    assert d["delta_ue"] == 0
    assert d["per_controller"]["mc0"]["csrows"]["csrow0"]["delta_ce"] == 3


def test_delta_detects_uncorrectable_errors():
    start = _snap(total_ce=0, total_ue=0)
    end = _snap(total_ce=0, total_ue=1)
    d = edac_delta(start, end)
    assert d["delta_ue"] == 1


def test_events_empty_when_no_errors():
    d = {"available": True, "delta_ce": 0, "delta_ue": 0, "per_controller": {}}
    events = edac_events_from_delta(d)
    assert events == []


def test_events_crit_on_ue():
    d = {"available": True, "delta_ce": 0, "delta_ue": 2, "per_controller": {}}
    events = edac_events_from_delta(d)
    assert len(events) == 1
    assert events[0]["severity"] == "crit"
    assert events[0]["kind"] == "edac_ue"


def test_events_warn_on_ce():
    d = {
        "available": True,
        "delta_ce": 5,
        "delta_ue": 0,
        "per_controller": {
            "mc0": {
                "delta_ce": 5,
                "delta_ue": 0,
                "csrows": {"csrow0": {"delta_ce": 5, "delta_ue": 0}},
            },
        },
    }
    events = edac_events_from_delta(d)
    assert len(events) == 1
    assert events[0]["severity"] == "warn"
    assert events[0]["kind"] == "edac_ce"
    assert "mc0/csrow0" in events[0]["message"]


def test_events_both_ue_and_ce():
    d = {
        "available": True,
        "delta_ce": 3,
        "delta_ue": 1,
        "per_controller": {},
    }
    events = edac_events_from_delta(d)
    kinds = {e["kind"] for e in events}
    assert "edac_ue" in kinds
    assert "edac_ce" in kinds


def test_unavailable_returns_no_events():
    d = {"available": False, "delta_ce": 0, "delta_ue": 0, "per_controller": {}}
    events = edac_events_from_delta(d)
    assert events == []
