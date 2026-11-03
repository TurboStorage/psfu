"""Golden tests over the deterministic synthetic runs.

These replace recorded-hardware fixtures: the generator in
``psfu.synthetic`` is seeded, so the verdict engine's output for each
scenario is pinned. A rule change that flips one of these is either a
regression or an intentional change that needs the expectations updated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psfu.persistence.db import Database
from psfu.synthetic import SCENARIOS, build_synthetic_run


@pytest.fixture(scope="module", params=SCENARIOS)
def synthetic(request, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("synthetic") / request.param
    result = build_synthetic_run(run_dir, request.param)
    return request.param, run_dir, result


def test_scenario_verdicts_are_pinned(synthetic) -> None:
    scenario, _, result = synthetic
    assert result["status"] == {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[scenario]
    assert result["claim_scope"] == "software-only"


def test_warn_scenario_scopes_finding_to_power(tmp_path: Path) -> None:
    result = build_synthetic_run(tmp_path / "w", "warn")
    subs = result["subsystems"]
    assert subs["power"]["status"] == "WARN"
    assert [e["rule_id"] for e in subs["power"]["evidence"]] == ["power.rail_12v_drift.warn"]
    for name in ("thermal", "memory", "storage", "pcie"):
        assert subs[name]["status"] == "PASS"


def test_fail_scenario_hits_expected_subsystems(tmp_path: Path) -> None:
    subs = build_synthetic_run(tmp_path / "f", "fail")["subsystems"]
    assert {n for n, s in subs.items() if s["status"] == "FAIL"} == {"power", "storage", "pcie"}
    assert subs["thermal"]["status"] == "PASS"
    assert subs["memory"]["status"] == "PASS"


def test_pass_scenario_has_no_evidence(tmp_path: Path) -> None:
    subs = build_synthetic_run(tmp_path / "p", "pass")["subsystems"]
    assert all(s["status"] == "PASS" and s["evidence"] == [] for s in subs.values())


def test_bundle_is_complete_and_hash_covers_it(synthetic) -> None:
    _, run_dir, _ = synthetic
    for name in ("verdict.json", "summary.md", "samples.csv", "events.csv", "bundle.sha256"):
        assert (run_dir / name).is_file(), name
    payload = json.loads((run_dir / "verdict.json").read_text(encoding="utf-8"))
    assert payload["run"]["psu_serial"] == "SYNTH-0001"
    assert "MISSING" not in (run_dir / "bundle.sha256").read_text(encoding="utf-8")


def test_sample_count_and_write_cliff_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "c"
    build_synthetic_run(run_dir, "warn")
    db = Database(run_dir / "run.sqlite")
    try:
        assert len(db.fetch_samples()) == 210
        assert db.fetch_run()["protocol_version"]
    finally:
        db.close()
    payload = json.loads((run_dir / "verdict.json").read_text(encoding="utf-8"))
    assert payload["storage_workload"]["cliff_ratio"] < 0.5


def test_generator_is_deterministic(tmp_path: Path) -> None:
    def rail_trace(name: str):
        d = tmp_path / name
        build_synthetic_run(d, "warn", seed=3, export=False)
        db = Database(d / "run.sqlite")
        try:
            return [s["rail_12v"] for s in db.fetch_samples()]
        finally:
            db.close()

    assert rail_trace("a") == rail_trace("b")


def test_unknown_scenario_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_synthetic_run(tmp_path / "x", "explode")
