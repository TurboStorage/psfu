from __future__ import annotations

import json

from psfu.replay import replay_run
from psfu.tests.conftest import _base_sample, make_run_db


def test_replay_reports_rule_id_growth_when_old_run_had_no_stored_evidence(tmp_path) -> None:
    run_dir = tmp_path
    db = make_run_db(tmp_path)
    for _ in range(5):
        db.add_sample(_base_sample(phase="soak", rail_12v=12.10))
    for _ in range(5):
        db.add_sample(_base_sample(phase="steady", rail_12v=11.50))

    db.update_run(
        db.fetch_run()["id"],
        {
            "verdict": "WARN",
            "telemetry_status": "software-only",
            "verdict_reasons_json": json.dumps([
                "[power] 12V drift reached warning band (12.10V → 11.50V)."
            ]),
            "subsystem_results_json": json.dumps({
                "power": {
                    "status": "WARN",
                    "reasons": ["12V drift reached warning band (12.10V → 11.50V)."],
                },
                "thermal": {"status": "PASS", "reasons": []},
                "memory": {"status": "PASS", "reasons": []},
                "storage": {"status": "PASS", "reasons": []},
                "pcie": {"status": "PASS", "reasons": []},
            }),
        },
    )
    db.close()

    result = replay_run(run_dir)
    report = result["report_path"].read_text(encoding="utf-8")

    assert result["stored_verdict"] == "WARN"
    assert result["current_verdict"] == "WARN"
    assert "| Stored evidence records: 0 |" not in report
    assert "- Stored evidence records: 0" in report
    assert "- Current evidence records: 1" in report
    assert "power.rail_12v_drift.warn" in report
