from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional


EvidenceRecord = Dict[str, Any]


def build_evidence(
    *,
    rule_id: str,
    subsystem: str,
    status: str,
    human_reason: str,
    threshold: Optional[Mapping[str, Any]] = None,
    observed: Optional[Mapping[str, Any]] = None,
    event_ids: Optional[Iterable[int]] = None,
    source: Optional[str] = None,
    phase: Optional[str] = None,
) -> EvidenceRecord:
    record: EvidenceRecord = {
        "rule_id": str(rule_id),
        "subsystem": str(subsystem),
        "status": str(status),
        "threshold": dict(threshold) if isinstance(threshold, Mapping) else None,
        "observed": dict(observed) if isinstance(observed, Mapping) else None,
        "event_ids": _normalize_event_ids(event_ids),
        "human_reason": str(human_reason),
    }
    if source:
        record["source"] = str(source)
    if phase:
        record["phase"] = str(phase)
    return record


def normalize_evidence(value: Any) -> List[EvidenceRecord]:
    if not isinstance(value, list):
        return []
    normalized: List[EvidenceRecord] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        record = build_evidence(
            rule_id=str(item.get("rule_id") or "unknown"),
            subsystem=str(item.get("subsystem") or "unknown"),
            status=str(item.get("status") or "UNKNOWN"),
            human_reason=str(item.get("human_reason") or ""),
            threshold=item.get("threshold") if isinstance(item.get("threshold"), Mapping) else None,
            observed=item.get("observed") if isinstance(item.get("observed"), Mapping) else None,
            event_ids=item.get("event_ids"),
            source=str(item.get("source")) if item.get("source") is not None else None,
            phase=str(item.get("phase")) if item.get("phase") is not None else None,
        )
        normalized.append(record)
    return normalized


def flatten_subsystem_evidence(subsystems: Mapping[str, Any]) -> List[EvidenceRecord]:
    records: List[EvidenceRecord] = []
    for subsystem_name, payload in subsystems.items():
        if not isinstance(payload, Mapping):
            continue
        for record in normalize_evidence(payload.get("evidence")):
            if record.get("subsystem") == "unknown":
                record["subsystem"] = str(subsystem_name)
            records.append(record)
    return records


def event_ids_for(events: Iterable[Any]) -> List[int]:
    ids: List[int] = []
    for event in events:
        if event is None:
            continue
        event_id = None
        if isinstance(event, Mapping):
            event_id = event.get("id")
        else:
            try:
                event_id = event["id"]
            except Exception:
                event_id = None
        if isinstance(event_id, int):
            ids.append(event_id)
    return ids


def _normalize_event_ids(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, int)]
    if isinstance(value, tuple):
        return [item for item in value if isinstance(item, int)]
    return []
