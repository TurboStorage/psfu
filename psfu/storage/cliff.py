from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

from psfu.common import slugify, utc_now
from psfu.config import BASELINES_DIR


WRITE_CLIFF_WINDOW_SEC = 10
WRITE_CLIFF_MIN_WINDOWS = 4
WRITE_CLIFF_SEVERE_RATIO = 0.12
WRITE_CLIFF_TAIL_WINDOWS = 3
WRITE_CLIFF_CHANGEPOINT_DROP_RATIO = 0.15


def extract_fio_interval_row(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        return None
    read_bw_kbps = 0.0
    write_bw_kbps = 0.0
    read_ios = 0
    write_ios = 0
    counted_jobs = 0
    for job in jobs:
        if not isinstance(job, dict):
            continue
        counted_jobs += 1
        read = job.get("read") if isinstance(job.get("read"), dict) else {}
        write = job.get("write") if isinstance(job.get("write"), dict) else {}
        try:
            read_bw_kbps += float(read.get("bw") or 0.0)
        except (TypeError, ValueError):
            pass
        try:
            write_bw_kbps += float(write.get("bw") or 0.0)
        except (TypeError, ValueError):
            pass
        try:
            read_ios += int(read.get("total_ios") or 0)
        except (TypeError, ValueError):
            pass
        try:
            write_ios += int(write.get("total_ios") or 0)
        except (TypeError, ValueError):
            pass
    if counted_jobs == 0:
        return None
    return {
        "captured_at": utc_now(),
        "job_count": counted_jobs,
        "read_bw_kbps": read_bw_kbps,
        "write_bw_kbps": write_bw_kbps,
        "read_total_ios": read_ios,
        "write_total_ios": write_ios,
    }


def build_write_bw_windows(
    interval_rows: List[Dict[str, Any]],
    *,
    window_sec: int = WRITE_CLIFF_WINDOW_SEC,
) -> List[Dict[str, Any]]:
    write_rows = [
        row for row in interval_rows
        if isinstance(row.get("write_bw_kbps"), (int, float)) and float(row.get("write_bw_kbps") or 0.0) > 0.0
    ]
    windows: List[Dict[str, Any]] = []
    if window_sec <= 0:
        return windows
    for idx in range(0, len(write_rows), window_sec):
        chunk = write_rows[idx:idx + window_sec]
        if len(chunk) < window_sec:
            break
        values = [float(row["write_bw_kbps"]) for row in chunk]
        windows.append({
            "window_index": len(windows),
            "sample_count": len(chunk),
            "start_sample_index": idx,
            "end_sample_index": idx + len(chunk) - 1,
            "avg_write_bw_kbps": sum(values) / len(values),
            "min_write_bw_kbps": min(values),
            "max_write_bw_kbps": max(values),
        })
    return windows


def ssd_baseline_path_for_identity(
    *,
    vendor: Optional[str],
    model: Optional[str],
    capacity: Optional[str],
    fw_rev: Optional[str],
) -> Path:
    stem = "{}-{}-{}-{}.json".format(
        slugify(vendor or "unknown-vendor"),
        slugify(model or "unknown-model"),
        slugify(capacity or "unknown-capacity"),
        slugify(fw_rev or "unknown-fw"),
    )
    return BASELINES_DIR / "ssd" / stem


def load_ssd_cliff_baseline(
    *,
    vendor: Optional[str],
    model: Optional[str],
    capacity: Optional[str],
    fw_rev: Optional[str],
) -> Optional[Dict[str, Any]]:
    path = ssd_baseline_path_for_identity(
        vendor=vendor,
        model=model,
        capacity=capacity,
        fw_rev=fw_rev,
    )
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def nvme_identity_from_run(run_dir: Path, device: str) -> Optional[Dict[str, str]]:
    base = run_dir / "storage" / "nvme" / Path(device).name
    id_ctrl_path = base / "start-id-ctrl.json"
    id_ns_path = base / "start-id-ns.json"
    try:
        id_ctrl = json.loads(id_ctrl_path.read_text(encoding="utf-8"))
    except Exception:
        id_ctrl = {}
    try:
        id_ns = json.loads(id_ns_path.read_text(encoding="utf-8"))
    except Exception:
        id_ns = {}
    ctrl = id_ctrl.get("payload") if isinstance(id_ctrl.get("payload"), dict) else {}
    ns = id_ns.get("payload") if isinstance(id_ns.get("payload"), dict) else {}
    if not ctrl and not ns:
        return None
    vendor = ctrl.get("vid")
    if isinstance(vendor, int):
        vendor_text = "vid-{:04x}".format(vendor)
    else:
        vendor_text = str(vendor or "unknown-vendor")
    capacity = ns.get("nsze")
    if isinstance(capacity, int):
        capacity_text = "nsze-{}".format(capacity)
    else:
        capacity_text = str(capacity or "unknown-capacity")
    return {
        "vendor": vendor_text,
        "model": str(ctrl.get("mn") or "unknown-model").strip(),
        "fw_rev": str(ctrl.get("fr") or "unknown-fw").strip(),
        "capacity": capacity_text,
    }


def _baseline_comparison(
    cliff_ratio: float,
    baseline: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(baseline, dict):
        return None
    p5 = baseline.get("cliff_ratio_p5")
    sigma = baseline.get("sigma")
    if not isinstance(p5, (int, float)) or not isinstance(sigma, (int, float)):
        return None
    severe_floor = float(p5) - (3.0 * float(sigma))
    return {
        "cliff_ratio_p5": float(p5),
        "sigma": float(sigma),
        "severe_floor_ratio": severe_floor,
        "below_severe_floor": cliff_ratio < severe_floor,
    }


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def _stddev(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if values else None
    mean_val = _mean(values)
    if mean_val is None:
        return None
    return math.sqrt(sum((value - mean_val) ** 2 for value in values) / float(len(values)))


def _coeff_var(values: List[float]) -> Optional[float]:
    mean_val = _mean(values)
    if mean_val is None or mean_val <= 0.0:
        return None
    stddev_val = _stddev(values)
    if stddev_val is None:
        return None
    return stddev_val / mean_val


def _mad(values: List[float]) -> Optional[float]:
    if not values:
        return None
    med = median(values)
    return median([abs(value - med) for value in values])


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * pct))))
    return ordered[idx]


def _changepoints(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if len(windows) < 2:
        return out
    last_avg = float(windows[0]["avg_write_bw_kbps"])
    for idx in range(1, len(windows)):
        current_avg = float(windows[idx]["avg_write_bw_kbps"])
        if last_avg <= 0:
            last_avg = current_avg
            continue
        delta_ratio = (current_avg - last_avg) / last_avg
        if abs(delta_ratio) >= WRITE_CLIFF_CHANGEPOINT_DROP_RATIO:
            out.append({
                "from_window_index": idx - 1,
                "to_window_index": idx,
                "from_avg_write_bw_kbps": last_avg,
                "to_avg_write_bw_kbps": current_avg,
                "delta_ratio": delta_ratio,
                "direction": "rise" if delta_ratio > 0 else "drop",
            })
        last_avg = current_avg
    return out


def _workload_stats(
    windows: List[Dict[str, Any]],
    *,
    peak_window_index: int,
    tail_window_count: int,
) -> Dict[str, Any]:
    avgs = [float(window["avg_write_bw_kbps"]) for window in windows]
    tail_values = avgs[-tail_window_count:] if tail_window_count > 0 else []
    post_peak_values = avgs[peak_window_index + 1:] if peak_window_index + 1 < len(avgs) else []
    changepoints = _changepoints(windows)
    peak_to_tail_drop = None
    if avgs and tail_values and avgs[peak_window_index] > 0:
        peak_to_tail_drop = 1.0 - (median(tail_values) / avgs[peak_window_index])
    return {
        "window_avg_write_bw_p10_kbps": _percentile(avgs, 0.10),
        "window_avg_write_bw_p50_kbps": _percentile(avgs, 0.50),
        "window_avg_write_bw_p90_kbps": _percentile(avgs, 0.90),
        "window_avg_write_bw_mad_kbps": _mad(avgs),
        "window_avg_write_bw_cv": _coeff_var(avgs),
        "tail_window_write_bw_cv": _coeff_var(tail_values),
        "tail_window_write_bw_mad_kbps": _mad(tail_values),
        "post_peak_window_count": len(post_peak_values),
        "post_peak_write_bw_median_kbps": median(post_peak_values) if post_peak_values else None,
        "post_peak_write_bw_cv": _coeff_var(post_peak_values),
        "peak_to_tail_drop_ratio": peak_to_tail_drop,
        "changepoint_count": len(changepoints),
        "major_drop_count": sum(1 for item in changepoints if item["direction"] == "drop"),
        "major_rise_count": sum(1 for item in changepoints if item["direction"] == "rise"),
        "changepoints": changepoints,
    }


def classify_write_workload(analysis: Dict[str, Any]) -> Dict[str, str]:
    workload = analysis.get("workload_stats") if isinstance(analysis.get("workload_stats"), dict) else {}
    cliff_ratio = float(analysis.get("cliff_ratio") or 0.0)
    tail_cv = float(workload.get("tail_window_write_bw_cv") or 0.0)
    post_peak_cv = float(workload.get("post_peak_write_bw_cv") or 0.0)
    major_drops = int(workload.get("major_drop_count") or 0)
    major_rises = int(workload.get("major_rise_count") or 0)

    if major_drops >= 2 and major_rises >= 1 and max(tail_cv, post_peak_cv) >= 0.10:
        return {
            "slug": "oscillatory_gc",
            "label": "gc oscillating",
            "summary": "Repeated drops and recoveries after peak suggest GC or folding oscillation.",
        }
    if major_drops >= 2 and major_rises == 0 and cliff_ratio < 0.55:
        return {
            "slug": "multi_stage_fold",
            "label": "multi-stage fold",
            "summary": "Bandwidth fell in multiple stages without meaningful recovery.",
        }
    if major_drops >= 1 and cliff_ratio < 0.70 and tail_cv <= 0.08:
        return {
            "slug": "single_cliff_stable_plateau",
            "label": "cliff stable",
            "summary": "One major cliff followed by a relatively stable post-cliff plateau.",
        }
    return {
        "slug": "no_clear_cliff",
        "label": "no clear cliff",
        "summary": "No clear sustained write cliff pattern yet.",
    }


def analyze_write_cliff(
    interval_rows: List[Dict[str, Any]],
    *,
    phase_name: str,
    device: str,
    run_dir: Optional[Path] = None,
    window_sec: int = WRITE_CLIFF_WINDOW_SEC,
    severe_ratio: float = WRITE_CLIFF_SEVERE_RATIO,
) -> Optional[Dict[str, Any]]:
    windows = build_write_bw_windows(interval_rows, window_sec=window_sec)
    if len(windows) < WRITE_CLIFF_MIN_WINDOWS:
        return None
    peak_window = max(windows, key=lambda item: float(item["avg_write_bw_kbps"]))
    peak_window_index = int(peak_window["window_index"])
    tail_windows = windows[-WRITE_CLIFF_TAIL_WINDOWS:]
    tail_median_kbps = median(float(item["avg_write_bw_kbps"]) for item in tail_windows)
    peak_kbps = float(peak_window["avg_write_bw_kbps"])
    if peak_kbps <= 0:
        return None
    cliff_ratio = tail_median_kbps / peak_kbps
    identity = nvme_identity_from_run(run_dir, device) if run_dir is not None else None
    baseline_path = None
    baseline = None
    if identity is not None:
        baseline_path = ssd_baseline_path_for_identity(
            vendor=identity.get("vendor"),
            model=identity.get("model"),
            capacity=identity.get("capacity"),
            fw_rev=identity.get("fw_rev"),
        )
        baseline = load_ssd_cliff_baseline(
            vendor=identity.get("vendor"),
            model=identity.get("model"),
            capacity=identity.get("capacity"),
            fw_rev=identity.get("fw_rev"),
        )
    baseline_comparison = _baseline_comparison(cliff_ratio, baseline)
    workload_stats = _workload_stats(
        windows,
        peak_window_index=peak_window_index,
        tail_window_count=len(tail_windows),
    )
    classification = classify_write_workload(
        {
            "cliff_ratio": cliff_ratio,
            "workload_stats": workload_stats,
        }
    )
    return {
        "captured_at": utc_now(),
        "phase_name": phase_name,
        "device": device,
        "window_sec": window_sec,
        "interval_sample_count": len(interval_rows),
        "full_window_count": len(windows),
        "peak_window": peak_window,
        "tail_windows": tail_windows,
        "tail_window_median_kbps": tail_median_kbps,
        "cliff_ratio": cliff_ratio,
        "severe_threshold_ratio": severe_ratio,
        "severe": cliff_ratio < severe_ratio,
        "baseline_path": str(baseline_path) if baseline_path is not None else None,
        "baseline_loaded": baseline is not None,
        "baseline": baseline,
        "baseline_comparison": baseline_comparison,
        "classification": classification,
        "workload_stats": workload_stats,
        "identity": identity,
        "windows": windows,
    }


def write_cliff_analysis(path: Path, analysis: Dict[str, Any]) -> None:
    path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
