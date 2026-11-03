from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from psfu.common import slugify, utc_now
from psfu.config import RUNS_DIR
from psfu.persistence.db import Database


def _iter_run_dirs() -> List[Path]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        path for path in RUNS_DIR.rglob("*")
        if path.is_dir() and (path / "run.sqlite").exists()
    )


def _normalize_lot(value: Any) -> str:
    return str(value or "").strip().casefold()


def _run_snapshot(run_dir: Path) -> Optional[Dict[str, Any]]:
    db = Database(run_dir / "run.sqlite")
    try:
        run_row = db.fetch_run()
        if run_row is None:
            return None
        verdict = str(run_row["verdict"] or "UNKNOWN")
        return {
            "run_dir": run_dir,
            "run_name": run_dir.name,
            "started_at": str(run_row["started_at"] or ""),
            "ended_at": str(run_row["ended_at"] or ""),
            "profile_name": str(run_row["profile_name"] or ""),
            "bench_name": str(run_row["bench_name"] or ""),
            "verdict": verdict,
            "claim_scope": str(run_row["telemetry_status"] or ""),
            "psu_make": str(run_row["psu_make"] or ""),
            "psu_model": str(run_row["psu_model"] or ""),
            "psu_wattage": run_row["psu_wattage"],
            "psu_serial": str(run_row["psu_serial"] or ""),
            "psu_lot": str(run_row["psu_lot"] or ""),
        }
    finally:
        db.close()


def scan_batch_runs(lot_code: str, *, exclude_run_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    target = _normalize_lot(lot_code)
    if not target:
        return []
    rows: List[Dict[str, Any]] = []
    for run_dir in _iter_run_dirs():
        if exclude_run_dir is not None and run_dir == exclude_run_dir:
            continue
        row = _run_snapshot(run_dir)
        if row is None:
            continue
        if _normalize_lot(row["psu_lot"]) != target:
            continue
        rows.append(row)
    return rows


def batch_failures_for_lot(lot_code: str) -> List[Dict[str, Any]]:
    return [row for row in scan_batch_runs(lot_code) if row["verdict"] == "FAIL"]


def _snapshot_from_run_row(run_dir: Path, run_row: Any) -> Dict[str, Any]:
    return {
        "run_dir": run_dir,
        "run_name": run_dir.name,
        "started_at": str(run_row["started_at"] or ""),
        "ended_at": str(run_row["ended_at"] or ""),
        "profile_name": str(run_row["profile_name"] or ""),
        "bench_name": str(run_row["bench_name"] or ""),
        "verdict": str(run_row["verdict"] or "UNKNOWN"),
        "claim_scope": str(run_row["telemetry_status"] or ""),
        "psu_make": str(run_row["psu_make"] or ""),
        "psu_model": str(run_row["psu_model"] or ""),
        "psu_wattage": run_row["psu_wattage"],
        "psu_serial": str(run_row["psu_serial"] or ""),
        "psu_lot": str(run_row["psu_lot"] or ""),
    }


def maybe_batch_pattern_event(run_dir: Path, db: Database) -> Optional[Dict[str, str]]:
    run_row = db.fetch_run()
    lot_code = str(run_row["psu_lot"] or "").strip()
    if not lot_code:
        return None
    if str(run_row["verdict"] or "") != "FAIL":
        return None

    current_events = db.fetch_events()
    if any(event["kind"] == "batch_pattern_detected" for event in current_events):
        return None

    failures = [
        row
        for row in scan_batch_runs(lot_code, exclude_run_dir=run_dir)
        if row["verdict"] == "FAIL"
    ]
    current_snapshot = _snapshot_from_run_row(run_dir, run_row)
    if current_snapshot["verdict"] == "FAIL":
        failures.append(current_snapshot)
    if len(failures) < 2:
        return None

    serials = []
    for item in failures:
        serial = str(item["psu_serial"] or item["run_name"])
        if serial not in serials:
            serials.append(serial)
    return {
        "severity": "crit",
        "kind": "batch_pattern_detected",
        "message": (
            "Batch pattern detected for lot {}: {} FAIL run(s) recorded ({}). "
            "Pull remaining inventory and escalate per Loop 3.".format(
                lot_code, len(failures), ", ".join(serials[:6])
            )
        ),
    }


def render_batch_summary(lot_code: str, rows: List[Dict[str, Any]]) -> str:
    fail_rows = [row for row in rows if row["verdict"] == "FAIL"]
    warn_rows = [row for row in rows if row["verdict"] == "WARN"]
    inconclusive_rows = [row for row in rows if row["verdict"] == "INCONCLUSIVE"]
    pass_rows = [row for row in rows if row["verdict"] == "PASS"]

    lines = [
        "# Batch Summary — {}".format(lot_code),
        "",
        "| Metric | Value |",
        "| --- | --- |",
        "| Lot code | {} |".format(lot_code),
        "| Generated at | {} |".format(utc_now()),
        "| Runs scanned | {} |".format(len(rows)),
        "| FAIL | {} |".format(len(fail_rows)),
        "| WARN | {} |".format(len(warn_rows)),
        "| INCONCLUSIVE | {} |".format(len(inconclusive_rows)),
        "| PASS | {} |".format(len(pass_rows)),
        "| Loop 3 trigger | {} |".format("YES" if len(fail_rows) >= 2 else "no"),
        "",
    ]
    if len(fail_rows) >= 2:
        serials = ", ".join(str(row["psu_serial"] or row["run_name"]) for row in fail_rows)
        lines.extend([
            "## Escalation",
            "- Two or more FAIL runs share this lot code.",
            "- Affected serials: {}".format(serials),
            "- Pull remaining inventory and escalate per Loop 3.",
            "",
        ])

    lines.extend([
        "## Runs",
        "| Run | Ended | Serial | Make / Model | Verdict | Scope | Bench | Profile |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in rows:
        make_model = "{} {} ({}W)".format(
            row["psu_make"] or "?",
            row["psu_model"] or "?",
            row["psu_wattage"] if row["psu_wattage"] is not None else "?",
        )
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["run_name"],
                row["ended_at"] or row["started_at"] or "UNKNOWN",
                row["psu_serial"] or "UNKNOWN",
                make_model,
                row["verdict"],
                row["claim_scope"] or "UNKNOWN",
                row["bench_name"] or "UNKNOWN",
                row["profile_name"] or "UNKNOWN",
            )
        )
    if not rows:
        lines.extend([
            "## Runs",
            "- No runs found for this lot code under `runs/`.",
        ])
    return "\n".join(lines) + "\n"


def write_batch_summary(lot_code: str, output_path: Optional[Path] = None) -> Dict[str, Any]:
    rows = scan_batch_runs(lot_code)
    if output_path is None:
        output_path = RUNS_DIR / "batch-{}-summary-{}.md".format(
            slugify(lot_code) or "unknown-lot",
            utc_now().replace(":", "").replace("+00:00", "Z"),
        )
    output_path.write_text(render_batch_summary(lot_code, rows), encoding="utf-8")
    return {
        "output_path": output_path,
        "rows": rows,
        "fail_count": sum(1 for row in rows if row["verdict"] == "FAIL"),
    }
