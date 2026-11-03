from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from psfu.common import slugify, utc_now
from psfu.config import BASELINES_DIR, RUNS_DIR
from psfu.persistence.db import Database


_IDLE_PHASES = {"soak", "baseline"}
_NON_LOAD_PHASES = {"soak", "baseline", "cooldown", "idle"}
_SOAK_12V_WARN_PCT = 0.015
_LOAD_12V_WARN_PCT = 0.02
_CPU_STRESS_MHZ_WARN_PCT = 0.10


def _hardware_json(run_row: Mapping[str, Any]) -> Dict[str, Any]:
    raw = run_row["hardware_json"] if "hardware_json" in run_row.keys() else None
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _hardware_command_stdout(run_row: Mapping[str, Any], key: str) -> str:
    hardware = _hardware_json(run_row)
    commands = hardware.get("commands", {})
    if not isinstance(commands, dict):
        return ""
    entry = commands.get(key, {})
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("stdout") or "")


def gpu_name_for_run(run_row: Mapping[str, Any]) -> str:
    stdout = _hardware_command_stdout(run_row, "nvidia_smi_L")
    if stdout:
        first = stdout.splitlines()[0].strip()
        if ": " in first:
            first = first.split(": ", 1)[1]
        if " (UUID:" in first:
            first = first.split(" (UUID:", 1)[0]
        if first:
            return first
    return "no-gpu"


def baseline_key_for_run(run_row: Mapping[str, Any]) -> str:
    bench = slugify(str(run_row["bench_name"] or "unknown-bench"))
    gpu = slugify(gpu_name_for_run(run_row))
    return "{}-{}".format(bench, gpu)


def baseline_path_for_run(run_row: Mapping[str, Any]) -> Path:
    return BASELINES_DIR / "{}.json".format(baseline_key_for_run(run_row))


def _p90(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.9))))
    return ordered[idx]


def compute_baseline_metrics(samples: List[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    soak_12v = [
        float(sample["rail_12v"])
        for sample in samples
        if sample["phase"] in _IDLE_PHASES and sample["rail_12v"] is not None
    ]
    load_12v = [
        float(sample["rail_12v"])
        for sample in samples
        if sample["phase"] not in _NON_LOAD_PHASES and sample["rail_12v"] is not None
    ]
    load_cpu_mhz = [
        float(sample["cpu_avg_mhz"])
        for sample in samples
        if sample["phase"] not in _NON_LOAD_PHASES and sample["cpu_avg_mhz"] is not None
    ]
    return {
        "soak_12v_median": statistics.median(soak_12v) if soak_12v else None,
        "load_12v_min": min(load_12v) if load_12v else None,
        "load_cpu_avg_mhz_p90": _p90(load_cpu_mhz),
    }


def load_baseline_for_run(run_row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    path = baseline_path_for_run(run_row)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def baseline_drift_reasons(
    run_row: Mapping[str, Any],
    samples: List[Mapping[str, Any]],
) -> List[str]:
    return [finding["human_reason"] for finding in baseline_drift_findings(run_row, samples)]


def baseline_drift_findings(
    run_row: Mapping[str, Any],
    samples: List[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    baseline = load_baseline_for_run(run_row)
    if not baseline:
        return []
    metrics = baseline.get("metrics", {})
    if not isinstance(metrics, dict):
        return []
    current = compute_baseline_metrics(samples)
    findings: List[Dict[str, Any]] = []

    soak_ref = metrics.get("soak_12v_median")
    soak_cur = current.get("soak_12v_median")
    if soak_ref and soak_cur:
        drift = abs(soak_cur - soak_ref) / soak_ref
        if drift > _SOAK_12V_WARN_PCT:
            findings.append(
                {
                    "rule_id": "power.baseline_drift.soak_12v.warn",
                    "human_reason": (
                        "Baseline drift: soak 12V median deviated {:.1f}% from reference ({:.2f}V → {:.2f}V).".format(
                            drift * 100.0, soak_ref, soak_cur
                        )
                    ),
                    "threshold": {
                        "type": "percentage_delta_from_baseline",
                        "warn_pct": _SOAK_12V_WARN_PCT,
                        "metric": "soak_12v_median",
                    },
                    "observed": {
                        "baseline_value_v": soak_ref,
                        "current_value_v": soak_cur,
                        "delta_pct": drift,
                    },
                    "source": "derived",
                }
            )

    load_ref = metrics.get("load_12v_min")
    load_cur = current.get("load_12v_min")
    if load_ref and load_cur:
        drift = abs(load_cur - load_ref) / load_ref
        if drift > _LOAD_12V_WARN_PCT:
            findings.append(
                {
                    "rule_id": "power.baseline_drift.load_12v.warn",
                    "human_reason": (
                        "Baseline drift: load 12V minimum deviated {:.1f}% from reference ({:.2f}V → {:.2f}V).".format(
                            drift * 100.0, load_ref, load_cur
                        )
                    ),
                    "threshold": {
                        "type": "percentage_delta_from_baseline",
                        "warn_pct": _LOAD_12V_WARN_PCT,
                        "metric": "load_12v_min",
                    },
                    "observed": {
                        "baseline_value_v": load_ref,
                        "current_value_v": load_cur,
                        "delta_pct": drift,
                    },
                    "source": "derived",
                }
            )

    cpu_ref = metrics.get("load_cpu_avg_mhz_p90")
    cpu_cur = current.get("load_cpu_avg_mhz_p90")
    if cpu_ref and cpu_cur:
        drift = abs(cpu_cur - cpu_ref) / cpu_ref
        if drift > _CPU_STRESS_MHZ_WARN_PCT:
            findings.append(
                {
                    "rule_id": "power.baseline_drift.cpu_avg_mhz.warn",
                    "human_reason": (
                        "Baseline drift: load CPU avg MHz p90 deviated {:.1f}% from reference ({:.0f} → {:.0f} MHz).".format(
                            drift * 100.0, cpu_ref, cpu_cur
                        )
                    ),
                    "threshold": {
                        "type": "percentage_delta_from_baseline",
                        "warn_pct": _CPU_STRESS_MHZ_WARN_PCT,
                        "metric": "load_cpu_avg_mhz_p90",
                    },
                    "observed": {
                        "baseline_value_mhz": cpu_ref,
                        "current_value_mhz": cpu_cur,
                        "delta_pct": drift,
                    },
                    "source": "derived",
                }
            )
    return findings


def record_baseline(run_dir: Path) -> Path:
    db = Database(run_dir / "run.sqlite")
    try:
        run_row = db.fetch_run()
        if run_row is None:
            raise RuntimeError("No run metadata in {}".format(run_dir / "run.sqlite"))
        if run_row["verdict"] != "PASS":
            raise RuntimeError("Baseline source must be a PASS run: {}".format(run_dir.name))
        payload = {
            "baseline_key": baseline_key_for_run(run_row),
            "captured_at": utc_now(),
            "protocol_version_captured": run_row["protocol_version"],
            "bench_name": run_row["bench_name"],
            "gpu_name": gpu_name_for_run(run_row),
            "profile_name": run_row["profile_name"],
            "source_run": run_dir.name,
            "metrics": compute_baseline_metrics(db.fetch_samples()),
        }
    finally:
        db.close()

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINES_DIR / "{}.json".format(payload["baseline_key"])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def latest_pass_run(profile_name: str) -> Optional[Path]:
    candidates = sorted(
        path for path in RUNS_DIR.rglob("*")
        if path.is_dir() and (path / "run.sqlite").exists()
    )
    for run_dir in reversed(candidates):
        db = Database(run_dir / "run.sqlite")
        try:
            run_row = db.fetch_run()
            if run_row and run_row["verdict"] == "PASS" and run_row["profile_name"] == profile_name:
                return run_dir
        finally:
            db.close()
    return None
