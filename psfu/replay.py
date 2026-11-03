from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from psfu.common import utc_now
from psfu.config import PROTOCOL_VERSION, RUNS_DIR
from psfu.persistence.db import Database
from psfu.verdict.evidence import flatten_subsystem_evidence, normalize_evidence
from psfu.verdict.rules import VerdictEngine


def _safe_report_stamp() -> str:
    return (
        utc_now()
        .replace(":", "")
        .replace("+00:00", "Z")
    )


def _load_json_file(path: Path) -> Tuple[Any, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, "{}: {}".format(path.name, exc)


def _normalize_reasons(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _normalize_subsystems(value: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(value, dict):
        normalized: Dict[str, Dict[str, Any]] = {}
        for name, payload in value.items():
            if isinstance(payload, dict):
                item = dict(payload)
            else:
                item = {}
            item["reasons"] = _normalize_reasons(item.get("reasons"))
            item["evidence"] = normalize_evidence(item.get("evidence"))
            normalized[str(name)] = item
        return normalized
    return {}


def _normalize_evidence(value: Any) -> List[Dict[str, Any]]:
    return normalize_evidence(value)


def _stored_snapshot(run_dir: Path, db: Database) -> Dict[str, Any]:
    run_row = db.fetch_run()
    if run_row is None:
        raise RuntimeError("No run metadata in {}".format(run_dir / "run.sqlite"))

    parse_notes: List[str] = []
    reasons = []
    subsystems: Dict[str, Dict[str, Any]] = {}

    reasons_json = run_row["verdict_reasons_json"]
    if reasons_json:
        try:
            reasons = _normalize_reasons(json.loads(reasons_json))
        except Exception as exc:
            parse_notes.append("run.verdict_reasons_json parse failed: {}".format(exc))

    subsystems_json = run_row["subsystem_results_json"] if "subsystem_results_json" in run_row.keys() else None
    if subsystems_json:
        try:
            subsystems = _normalize_subsystems(json.loads(subsystems_json))
        except Exception as exc:
            parse_notes.append("run.subsystem_results_json parse failed: {}".format(exc))

    stored = {
        "run_row": run_row,
        "schema_version": getattr(db, "_schema_version", None),
        "protocol_version": run_row["protocol_version"] or "UNKNOWN",
        "verdict": run_row["verdict"] or "UNKNOWN",
        "claim_scope": run_row["telemetry_status"] or "UNKNOWN",
        "reasons": reasons,
        "subsystems": subsystems,
        "evidence": flatten_subsystem_evidence(subsystems),
        "parse_notes": parse_notes,
    }

    if (
        stored["verdict"] != "UNKNOWN"
        and stored["claim_scope"] != "UNKNOWN"
        and reasons_json
        and subsystems_json
    ):
        return stored

    verdict_path = run_dir / "verdict.json"
    if not verdict_path.exists():
        return stored

    payload, error = _load_json_file(verdict_path)
    if error:
        parse_notes.append(error)
        return stored
    if not isinstance(payload, dict):
        parse_notes.append("verdict.json payload was not an object")
        return stored

    overall = payload.get("overall")
    if isinstance(overall, dict):
        if stored["verdict"] == "UNKNOWN":
            stored["verdict"] = str(overall.get("status") or "UNKNOWN")
        if stored["claim_scope"] == "UNKNOWN":
            stored["claim_scope"] = str(overall.get("claim_scope") or "UNKNOWN")
        if not stored["reasons"]:
            stored["reasons"] = _normalize_reasons(overall.get("reasons"))
        if not stored["evidence"]:
            stored["evidence"] = _normalize_evidence(overall.get("evidence"))
    if not stored["subsystems"]:
        stored["subsystems"] = _normalize_subsystems(payload.get("subsystems"))
        if not stored["evidence"]:
            stored["evidence"] = flatten_subsystem_evidence(stored["subsystems"])
    return stored


def _current_snapshot(db: Database) -> Dict[str, Any]:
    verdict, reasons, claim_scope, subsystems = VerdictEngine.evaluate(db)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "verdict": verdict,
        "claim_scope": claim_scope,
        "reasons": list(reasons),
        "subsystems": dict(subsystems),
        "evidence": flatten_subsystem_evidence(subsystems),
    }


def _historical_context_notes(run_dir: Path, run_row: Any) -> List[str]:
    notes: List[str] = []
    haystacks: List[str] = []
    for key in ("notes", "reason"):
        if key in run_row.keys():
            value = run_row[key]
            if value:
                haystacks.append(str(value))
    summary_path = run_dir / "summary.md"
    if summary_path.exists():
        try:
            haystacks.append(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    merged = "\n".join(haystacks)
    if "bench-power-budget" in merged or "--allow-undersized-psu" in merged:
        notes.append(
            "This run was captured under a bench-power-budget exception or undersized-PSU "
            "context. Any verdict delta here is advisory, not clean retroactive evidence "
            "against the original DUT."
        )
    return notes


def _diff_reasons(stored: List[str], current: List[str]) -> Tuple[List[str], List[str]]:
    stored_set = set(stored)
    current_set = set(current)
    added = [reason for reason in current if reason not in stored_set]
    removed = [reason for reason in stored if reason not in current_set]
    return added, removed


def _subsystem_rows(
    stored: Dict[str, Dict[str, Any]],
    current: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str, str, str]]:
    names = sorted(set(stored.keys()) | set(current.keys()))
    rows: List[Tuple[str, str, str, str]] = []
    for name in names:
        stored_status = str(stored.get(name, {}).get("status") or "UNKNOWN")
        current_status = str(current.get(name, {}).get("status") or "UNKNOWN")
        rows.append((name, stored_status, current_status, "yes" if stored_status != current_status else "no"))
    return rows


def _evidence_rule_ids(records: List[Dict[str, Any]]) -> List[str]:
    ids = []
    for record in records:
        rule_id = record.get("rule_id")
        if isinstance(rule_id, str):
            ids.append(rule_id)
    return ids


def _diff_rule_ids(stored: List[Dict[str, Any]], current: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    stored_ids = _evidence_rule_ids(stored)
    current_ids = _evidence_rule_ids(current)
    stored_set = set(stored_ids)
    current_set = set(current_ids)
    added = [rule_id for rule_id in current_ids if rule_id not in stored_set]
    removed = [rule_id for rule_id in stored_ids if rule_id not in current_set]
    return added, removed


def _render_reason_lines(reasons: List[str]) -> List[str]:
    if not reasons:
        return ["- (none)"]
    return ["- {}".format(reason) for reason in reasons]


def _render_rule_lines(rule_ids: List[str]) -> List[str]:
    if not rule_ids:
        return ["- (none)"]
    return ["- {}".format(rule_id) for rule_id in rule_ids]


def _render_rescore_report(run_dir: Path, stored: Dict[str, Any], current: Dict[str, Any]) -> str:
    run_row = stored["run_row"]
    added_reasons, removed_reasons = _diff_reasons(stored["reasons"], current["reasons"])
    added_rule_ids, removed_rule_ids = _diff_rule_ids(stored["evidence"], current["evidence"])
    subsystem_rows = _subsystem_rows(stored["subsystems"], current["subsystems"])
    context_notes = _historical_context_notes(run_dir, run_row)

    lines = [
        "# Replay Audit — {}".format(run_dir.name),
        "",
        "## Snapshot",
        "| Field | Value |",
        "| --- | --- |",
        "| Run dir | `{}` |".format(run_dir),
        "| Started at | {} |".format(run_row["started_at"] or "UNKNOWN"),
        "| Profile | {} |".format(run_row["profile_name"] or "UNKNOWN"),
        "| Stored protocol | {} |".format(stored["protocol_version"]),
        "| Current protocol | {} |".format(current["protocol_version"]),
        "| Stored schema | {} |".format(stored["schema_version"] if stored["schema_version"] is not None else "UNKNOWN"),
        "| Stored verdict | {} |".format(stored["verdict"]),
        "| Current verdict | {} |".format(current["verdict"]),
        "| Verdict changed | {} |".format("yes" if stored["verdict"] != current["verdict"] else "no"),
        "| Stored claim scope | {} |".format(stored["claim_scope"]),
        "| Current claim scope | {} |".format(current["claim_scope"]),
        "| Claim scope changed | {} |".format("yes" if stored["claim_scope"] != current["claim_scope"] else "no"),
        "",
        "## Interpretation Guardrail",
        "- This report is a read-only replay under the current verdict engine.",
        "- Original run artifacts remain authoritative for the original test date.",
        "- A replay change means rule semantics changed, not that the original operator made a mistake.",
        "",
        "## Verdict Diff",
        "- Added reasons: {}".format(len(added_reasons)),
        "- Removed reasons: {}".format(len(removed_reasons)),
        "- Stored evidence records: {}".format(len(stored["evidence"])),
        "- Current evidence records: {}".format(len(current["evidence"])),
        "- Added rule IDs: {}".format(len(added_rule_ids)),
        "- Removed rule IDs: {}".format(len(removed_rule_ids)),
        "",
        "### Added Reasons",
        *_render_reason_lines(added_reasons),
        "",
        "### Removed Reasons",
        *_render_reason_lines(removed_reasons),
        "",
        "### Added Rule IDs",
        *_render_rule_lines(added_rule_ids),
        "",
        "### Removed Rule IDs",
        *_render_rule_lines(removed_rule_ids),
        "",
        "## Stored Reasons",
        *_render_reason_lines(stored["reasons"]),
        "",
        "## Current Reasons",
        *_render_reason_lines(current["reasons"]),
        "",
        "## Evidence Presence",
        "| Snapshot | Count | Rule IDs |",
        "| --- | --- | --- |",
        "| Stored | {} | {} |".format(len(stored["evidence"]), ", ".join(_evidence_rule_ids(stored["evidence"])) or "(none)"),
        "| Current | {} | {} |".format(len(current["evidence"]), ", ".join(_evidence_rule_ids(current["evidence"])) or "(none)"),
        "",
        "## Subsystem Diff",
        "| Subsystem | Stored | Current | Changed |",
        "| --- | --- | --- | --- |",
    ]
    for name, stored_status, current_status, changed in subsystem_rows:
        lines.append("| {} | {} | {} | {} |".format(name, stored_status, current_status, changed))

    if stored["parse_notes"]:
        lines.extend([
            "",
            "## Parse Notes",
            *["- {}".format(note) for note in stored["parse_notes"]],
        ])

    if context_notes:
        lines.extend([
            "",
            "## Historical Context",
            *["- {}".format(note) for note in context_notes],
        ])

    return "\n".join(lines) + "\n"


def replay_run(run_dir: Path) -> Dict[str, Any]:
    source_db_path = run_dir / "run.sqlite"
    if not source_db_path.exists():
        raise RuntimeError("Missing run DB: {}".format(source_db_path))

    with tempfile.TemporaryDirectory(prefix="psfu-replay-") as temp_dir:
        temp_db_path = Path(temp_dir) / "run.sqlite"
        shutil.copy2(source_db_path, temp_db_path)
        db = Database(temp_db_path)
        try:
            stored = _stored_snapshot(run_dir, db)
            current = _current_snapshot(db)
        finally:
            db.close()
    report_path = run_dir / "rescore-{}.md".format(_safe_report_stamp())
    report_path.write_text(_render_rescore_report(run_dir, stored, current), encoding="utf-8")
    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "stored_protocol_version": stored["protocol_version"],
        "current_protocol_version": current["protocol_version"],
        "stored_verdict": stored["verdict"],
        "current_verdict": current["verdict"],
        "verdict_changed": stored["verdict"] != current["verdict"],
        "stored_claim_scope": stored["claim_scope"],
        "current_claim_scope": current["claim_scope"],
        "claim_scope_changed": stored["claim_scope"] != current["claim_scope"],
    }


def replay_all_runs() -> Dict[str, Any]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(
        path for path in RUNS_DIR.rglob("*")
        if path.is_dir() and (path / "run.sqlite").exists()
    )
    results: List[Dict[str, Any]] = []
    failures: List[Tuple[Path, str]] = []
    for run_dir in run_dirs:
        try:
            results.append(replay_run(run_dir))
        except Exception as exc:
            failures.append((run_dir, str(exc)))

    summary_path = RUNS_DIR / "rescore-summary-{}.md".format(_safe_report_stamp())
    changed = sum(1 for item in results if item["verdict_changed"] or item["claim_scope_changed"])
    lines = [
        "# Replay Audit Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        "| Runs scanned | {} |".format(len(run_dirs)),
        "| Reports written | {} |".format(len(results)),
        "| Changed runs | {} |".format(changed),
        "| Failed runs | {} |".format(len(failures)),
        "",
        "## Results",
        "| Run | Stored | Current | Changed | Report |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in results:
        lines.append(
            "| {} | {} | {} | {} | `{}` |".format(
                item["run_dir"].name,
                item["stored_verdict"],
                item["current_verdict"],
                "yes" if item["verdict_changed"] or item["claim_scope_changed"] else "no",
                item["report_path"].name,
            )
        )
    if failures:
        lines.extend(["", "## Failures"])
        for run_dir, error in failures:
            lines.append("- `{}`: {}".format(run_dir.name, error))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "summary_path": summary_path,
        "results": results,
        "failures": failures,
    }
