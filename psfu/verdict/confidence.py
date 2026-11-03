from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


_ALL_DUTS = ("cpu", "gpu", "mem", "storage")


def _load_bootstrap(run_dir: Path, run_row: Mapping[str, Any]) -> Dict[str, Any]:
    bootstrap_path = run_dir / "bootstrap.json"
    if bootstrap_path.exists():
        try:
            return json.loads(bootstrap_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    raw = run_row["preflight_json"] if "preflight_json" in run_row.keys() else None
    if raw:
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {}


def _bootstrap_status(bootstrap: Mapping[str, Any], step_name: str) -> str:
    for step in bootstrap.get("steps", []):
        if isinstance(step, dict) and step.get("name") == step_name:
            return str(step.get("status") or "missing")
    return "missing"


def _disabled_duts(events: Iterable[Mapping[str, Any]]) -> List[str]:
    selections = [e for e in events if e["kind"] == "dut_selection"]
    if not selections:
        return []
    message = str(selections[-1]["message"] or "")
    marker = "armed DUTs:"
    if marker not in message:
        return []
    armed = {
        part.strip().lower()
        for part in message.split(marker, 1)[1].split(",")
        if part.strip()
    }
    return [dut for dut in _ALL_DUTS if dut not in armed]


def coverage_confidence(
    run_dir: Path,
    run_row: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    bootstrap = _load_bootstrap(run_dir, run_row)
    score = 1.0
    factors: List[Dict[str, Any]] = []
    event_list = list(events)

    def apply_penalty(code: str, penalty: float, reason: str) -> None:
        nonlocal score
        score -= penalty
        factors.append({
            "code": code,
            "penalty": round(penalty, 2),
            "reason": reason,
        })

    if not run_row["meter_available"]:
        apply_penalty(
            "no_multimeter",
            0.30,
            "No multimeter readings were recorded for this run.",
        )

    if any(e["kind"] == "kernel_watcher_unavailable" for e in event_list):
        apply_penalty(
            "kernel_watcher_unavailable",
            0.20,
            "Kernel watcher did not start, so kernel-driven rules were not fully covered.",
        )

    if _bootstrap_status(bootstrap, "pynvml") != "ready":
        apply_penalty(
            "nvml_unavailable",
            0.15,
            "NVML telemetry was unavailable, reducing GPU rule coverage.",
        )

    if _bootstrap_status(bootstrap, "turbostat") != "ready":
        apply_penalty(
            "turbostat_unavailable",
            0.10,
            "turbostat was unavailable, reducing CPU telemetry coverage.",
        )

    disabled = _disabled_duts(event_list)
    for dut in disabled:
        apply_penalty(
            "dut_disabled_{}".format(dut),
            0.05,
            "{} DUT was disabled for this run.".format(dut.upper()),
        )

    score = max(0.0, min(1.0, score))
    return {
        "score": round(score, 2),
        "factors": factors,
    }
