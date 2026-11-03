from __future__ import annotations

import json
import re
import sqlite3
import textwrap
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from psfu.ui.sparkline import render_sparkline
from psfu.verdict.confidence import coverage_confidence


def phase_boundaries(samples: List[sqlite3.Row]) -> List[Tuple[int, str]]:
    boundaries: List[Tuple[int, str]] = []
    current: Optional[str] = None
    for i, sample in enumerate(samples):
        phase = sample["phase"]
        if phase != current:
            boundaries.append((i, phase))
            current = phase
    return boundaries


def format_voltage(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return "{:.2f}V".format(value)


def format_watts(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return "{:.2f}W".format(value)


def format_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return "{:.2f}%".format(value)


def format_celsius(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return "{:.2f}C".format(value)


def min_value(rows: List[sqlite3.Row], field: str) -> Optional[float]:
    values = [row[field] for row in rows if row[field] is not None]
    return min(values) if values else None


def max_value(rows: List[sqlite3.Row], field: str) -> Optional[float]:
    values = [row[field] for row in rows if row[field] is not None]
    return max(values) if values else None


def find_latest_write_cliff_analysis(run_dir: Path) -> Optional[Path]:
    candidates = sorted(run_dir.glob("storage/fio/*/*/write-cliff-analysis.json"))
    return candidates[-1] if candidates else None


def render_storage_workload_section(run_dir: Path) -> List[str]:
    analysis_path = find_latest_write_cliff_analysis(run_dir)
    if analysis_path is None:
        return []
    try:
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    classification = payload.get("classification") if isinstance(payload.get("classification"), dict) else {}
    workload = payload.get("workload_stats") if isinstance(payload.get("workload_stats"), dict) else {}
    peak = payload.get("peak_window") if isinstance(payload.get("peak_window"), dict) else {}
    lines = [
        "## Storage Workload",
        "",
        "| Field | Value |",
        "|---|---|",
        "| Phase | {} |".format(payload.get("phase_name") or "unknown"),
        "| Classification | {} |".format(classification.get("label") or "unknown"),
        "| Cliff ratio | {:.3f} |".format(float(payload.get("cliff_ratio") or 0.0)),
        "| Peak 10s avg write BW | {:.0f} KiB/s |".format(float(peak.get("avg_write_bw_kbps") or 0.0)),
        "| Tail 10s median write BW | {:.0f} KiB/s |".format(float(payload.get("tail_window_median_kbps") or 0.0)),
        "| Major drops | {} |".format(int(workload.get("major_drop_count") or 0)),
        "| Major rises | {} |".format(int(workload.get("major_rise_count") or 0)),
        "| Tail CV | {:.3f} |".format(float(workload.get("tail_window_write_bw_cv") or 0.0)),
        "| Baseline loaded | {} |".format("yes" if payload.get("baseline_loaded") else "no"),
        "| Artifact | `{}` |".format(analysis_path.relative_to(run_dir)),
        "",
    ]
    summary = str(classification.get("summary") or "").strip()
    if summary:
        lines.extend([summary, ""])
    return lines


def render_rma_storage_workload_lines(run_dir: Path) -> List[str]:
    analysis_path = find_latest_write_cliff_analysis(run_dir)
    if analysis_path is None:
        return []
    try:
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    classification = payload.get("classification") if isinstance(payload.get("classification"), dict) else {}
    return [
        "## Storage Workload Characterization",
        "",
        "- Observed sustained-write behavior matched `{}` with cliff ratio {:.3f}.".format(
            classification.get("label") or "unknown",
            float(payload.get("cliff_ratio") or 0.0),
        ),
        "",
    ]


def render_charts(run_dir: Path, samples: List[sqlite3.Row]) -> str:
    if not samples:
        return "_No samples captured - charts unavailable._\n"

    boundaries = phase_boundaries(samples)
    n = len(samples)
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.ticker as mticker  # type: ignore

        def col(key: str) -> List[Optional[float]]:
            return [sample[key] for sample in samples]

        x = list(range(n))
        rail_12v = col("rail_12v")
        rail_5v = col("rail_5v")
        has_12v = any(v is not None for v in rail_12v)
        has_5v = any(v is not None for v in rail_5v)

        if has_12v or has_5v:
            fig, ax = plt.subplots(figsize=(10, 3))
            if has_12v:
                ax.plot(x, [v if v is not None else float("nan") for v in rail_12v], linewidth=0.8, label="12V", color="#1f77b4")
            if has_5v:
                ax.plot(x, [v if v is not None else float("nan") for v in rail_5v], linewidth=0.8, label="5V", color="#ff7f0e")
            ax.set_xlabel("Sample (1 Hz)")
            ax.set_ylabel("Voltage (V)")
            ax.set_title("Rail voltages over time")
            ax.legend(loc="upper right", fontsize=8)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.set_xlim(0, n)
            for idx, name in boundaries:
                ax.axvline(x=idx, color="#aaaaaa", linewidth=0.8, linestyle="--")
                ylim = ax.get_ylim()
                ax.text(idx + max(1, n // 200), ylim[1] - (ylim[1] - ylim[0]) * 0.05, name, fontsize=7, color="#666666", va="top")
            fig.tight_layout()
            chart_rails_path = run_dir / "chart_rails.png"
            fig.savefig(str(chart_rails_path), dpi=100, bbox_inches="tight")
            plt.close(fig)
        else:
            chart_rails_path = None

        gpu_w = col("gpu_power_w")
        cpu_w = col("cpu_load_pct")
        cpu_c = col("cpu_temp_c")
        gpu_c = col("gpu_temp_c")
        has_pwr = any(v is not None for v in gpu_w) or any(v is not None for v in cpu_w)
        has_tmp = any(v is not None for v in cpu_c) or any(v is not None for v in gpu_c)

        if has_pwr or has_tmp:
            n_subplots = sum([has_pwr, has_tmp])
            fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 3 * n_subplots), sharex=True)
            if n_subplots == 1:
                axes = [axes]
            ax_idx = 0

            if has_pwr:
                ax = axes[ax_idx]
                ax_idx += 1
                if any(v is not None for v in gpu_w):
                    ax.plot(x, [v if v is not None else float("nan") for v in gpu_w], linewidth=0.8, label="GPU W", color="#d62728")
                if any(v is not None for v in cpu_w):
                    ax.plot(x, [v if v is not None else float("nan") for v in cpu_w], linewidth=0.8, label="CPU load%", color="#9467bd")
                ax.set_ylabel("GPU W / CPU%")
                ax.set_title("GPU power draw and CPU load over time")
                ax.legend(loc="upper right", fontsize=8)
                ax.set_xlim(0, n)
                for idx, name in boundaries:
                    ylim = ax.get_ylim()
                    ax.axvline(x=idx, color="#aaaaaa", linewidth=0.8, linestyle="--")
                    ax.text(idx + max(1, n // 200), ylim[1] - (ylim[1] - ylim[0]) * 0.05, name, fontsize=7, color="#666666", va="top")

            if has_tmp:
                ax = axes[ax_idx]
                if any(v is not None for v in cpu_c):
                    ax.plot(x, [v if v is not None else float("nan") for v in cpu_c], linewidth=0.8, label="CPU °C", color="#8c564b")
                if any(v is not None for v in gpu_c):
                    ax.plot(x, [v if v is not None else float("nan") for v in gpu_c], linewidth=0.8, label="GPU °C", color="#e377c2")
                ax.set_xlabel("Sample (1 Hz)")
                ax.set_ylabel("Temp (°C)")
                ax.set_title("Temperatures over time")
                ax.legend(loc="upper right", fontsize=8)
                ax.set_xlim(0, n)
                for idx, name in boundaries:
                    ylim = ax.get_ylim()
                    ax.axvline(x=idx, color="#aaaaaa", linewidth=0.8, linestyle="--")
                    ax.text(idx + max(1, n // 200), ylim[1] - (ylim[1] - ylim[0]) * 0.05, name, fontsize=7, color="#666666", va="top")

            fig.tight_layout()
            chart_pt_path = run_dir / "chart_power_temps.png"
            fig.savefig(str(chart_pt_path), dpi=100, bbox_inches="tight")
            plt.close(fig)
        else:
            chart_pt_path = None

        md_lines = []
        if chart_rails_path and chart_rails_path.exists():
            md_lines.append("![Rail voltages over time](chart_rails.png)")
            md_lines.append("")
        if chart_pt_path and chart_pt_path.exists():
            md_lines.append("![Power and temperatures over time](chart_power_temps.png)")
            md_lines.append("")
        if not md_lines:
            md_lines.append("_No chartable rail or power data in this run._")
        return "\n".join(md_lines) + "\n"
    except ImportError:
        pass

    chart_width = 72
    metric_specs = [
        ("CPU%", "cpu_load_pct", "{:>6.1f}"),
        ("RAM%", "mem_used_pct", "{:>6.1f}"),
        ("Disk%", "disk_busy_pct", "{:>6.1f}"),
        ("SSD°C", "ssd_temp_c", "{:>6.1f}"),
        ("GPU W", "gpu_power_w", "{:>6.1f}"),
        ("CPU °C", "cpu_temp_c", "{:>6.1f}"),
        ("GPU °C", "gpu_temp_c", "{:>6.1f}"),
        ("12V", "rail_12v", "{:>6.2f}"),
        ("5V", "rail_5v", "{:>6.2f}"),
    ]
    code_lines = ["```"]
    if boundaries:
        phase_label_line = " " * 8
        for idx, name in boundaries:
            col = int(round(idx / float(n) * chart_width))
            gap = col - (len(phase_label_line) - 8)
            if gap > 0:
                phase_label_line += " " * gap + name[:max(1, gap)]
        code_lines.append("phases  " + phase_label_line[8:])
        code_lines.append("")

    has_any = False
    for label, key, fmt in metric_specs:
        values = [sample[key] for sample in samples]
        cleaned = [v for v in values if v is not None]
        if not cleaned:
            continue
        has_any = True
        spark = render_sparkline(values, chart_width)
        lo = min(cleaned)
        hi = max(cleaned)
        avg = sum(cleaned) / len(cleaned)
        stats = "min {} max {} avg {}".format(
            fmt.format(lo).strip(),
            fmt.format(hi).strip(),
            fmt.format(avg).strip(),
        )
        code_lines.append("{:<7} {}  {}".format(label, spark, stats))

    code_lines.append("```")
    if not has_any:
        return "_No chartable data in this run._\n"
    code_lines.append("_matplotlib not installed - install it for PNG charts: `pip install matplotlib`_")
    return "\n".join(code_lines) + "\n"


def render_summary(
    run_dir: Path,
    run_row: sqlite3.Row,
    samples: List[sqlite3.Row],
    events: List[sqlite3.Row],
    defensible_conclusion: Callable[[sqlite3.Row], str],
) -> str:
    row_keys = set(run_row.keys())
    reasons = json.loads(
        (run_row["verdict_reasons_json"] if "verdict_reasons_json" in row_keys else "[]") or "[]"
    )
    subsystem_results = json.loads(
        (run_row["subsystem_results_json"] if "subsystem_results_json" in row_keys else "{}") or "{}"
    )
    claim_scope = run_row["telemetry_status"] or "unknown"
    confidence = coverage_confidence(run_dir, run_row, events)
    try:
        bootstrap = json.loads((run_dir / "bootstrap.json").read_text())
    except Exception:
        bootstrap = {}
    lines = [
        "# PSU Test Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        "| Verdict | {} |".format(run_row["verdict"] or "UNKNOWN"),
        "| PSU | {} {} |".format(run_row["psu_make"], run_row["psu_model"]),
        "| Wattage | {}W |".format(run_row["psu_wattage"]),
        "| Serial | {} |".format(run_row["psu_serial"]),
        "| Lot | {} |".format(run_row["psu_lot"] or ""),
        "| Operator | {} |".format(run_row["operator"]),
        "| Reason | {} |".format(run_row["reason"]),
        "| Started | {} |".format(run_row["started_at"]),
        "| Ended | {} |".format(run_row["ended_at"] or ""),
        "| Profile | {} ({}) |".format(run_row["profile_name"], run_row["profile_version"]),
        "| Bench | {} |".format(run_row["bench_name"] or ""),
        "| Bench last verified | {} |".format(run_row["bench_last_verified"] or ""),
        "| Ambient temp | {} |".format(format_celsius(run_row["ambient_c"]) if "ambient_c" in set(run_row.keys()) else "n/a"),
        "| Notes | {} |".format(run_row["notes"] if "notes" in set(run_row.keys()) and run_row["notes"] else ""),
        "| Claim scope | {} |".format(claim_scope),
        "| Coverage confidence | {:.2f} |".format(confidence["score"]),
        "",
        "## Telemetry Readiness",
        "",
    ]
    readiness = bootstrap.get("telemetry_readiness", {})
    if readiness:
        lines.extend([
            "| Field | Value |",
            "|---|---|",
            "| Readiness | {} |".format(str(readiness.get("level", "unknown")).upper()),
            "| Notes | {} |".format("; ".join(readiness.get("messages", []))),
            "",
        ])
    if bootstrap.get("steps"):
        lines.extend([
            "| Step | Status | Detail |",
            "|---|---|---|",
        ])
        for step in bootstrap["steps"]:
            lines.append("| {} | {} | {} |".format(step.get("name", ""), step.get("status", ""), step.get("detail", "")))
        lines.append("")
    if confidence["factors"]:
        lines.extend([
            "## Coverage Confidence Factors",
            "",
        ])
        for factor in confidence["factors"]:
            lines.append("- {:.2f}: {}".format(-factor["penalty"], factor["reason"]))
        lines.append("")
    lines.extend([
        "## Bench Observations",
        "",
        "| Check | Result |",
        "|---|---|",
        "| Paperclip test | {} |".format(run_row["paperclip_result"] or "not-run"),
        "| POST with returned PSU | {} |".format(run_row["post_result"] or "not-tested"),
        "| Known-good PSU swap performed | {} |".format("yes" if run_row["known_good_swap_performed"] else "no"),
        "| Known-good PSU swap result | {} |".format(run_row["known_good_swap_result"] or "not-done"),
        "| Original system retest | {} |".format(run_row["original_system_retest_result"] or "not-run"),
        "",
        "## Manual Rail Readings",
        "",
        "| Reading | Value |",
        "|---|---|",
        "| 12V idle | {} |".format(format_voltage(run_row["manual_idle_12v"])),
        "| 12V load | {} |".format(format_voltage(run_row["manual_load_12v"])),
        "| 5V idle | {} |".format(format_voltage(run_row["manual_idle_5v"])),
        "| 5V load | {} |".format(format_voltage(run_row["manual_load_5v"])),
        "",
        "## Verdict Rationale",
        "",
    ])
    if reasons:
        for reason in reasons:
            lines.append("- {}".format(reason))
    else:
        lines.append("- No explicit reasons recorded.")

    lines.extend([
        "",
        "## Subsystem Results",
        "",
        "| Subsystem | Status | Notes |",
        "|---|---|---|",
    ])
    for name in ["power", "thermal", "memory", "storage", "pcie"]:
        data = subsystem_results.get(name, {}) if isinstance(subsystem_results, dict) else {}
        status = data.get("status", "unknown") if isinstance(data, dict) else "unknown"
        notes = "; ".join(data.get("reasons", [])) if isinstance(data, dict) else ""
        lines.append("| {} | {} | {} |".format(name, status, notes or "none"))

    lines.extend([
        "",
        "## Defensible Conclusion",
        "",
        defensible_conclusion(run_row),
        "",
    ])
    lines.extend(render_storage_workload_section(run_dir))
    lines.extend([
        "## Telemetry Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        "| Samples captured | {} |".format(len(samples)),
        "| Events captured | {} |".format(len(events)),
        "| RAM used max | {} |".format(format_pct(max_value(samples, "mem_used_pct"))),
        "| Disk busy max | {} |".format(format_pct(max_value(samples, "disk_busy_pct"))),
        "| SSD temp max | {} |".format(format_celsius(max_value(samples, "ssd_temp_c"))),
        "| HBA count | {} |".format(max_value(samples, "hba_count") or 0),
        "| I2C adapters | {} |".format(max_value(samples, "i2c_adapter_count") or 0),
        "| I3C buses | {} |".format(max_value(samples, "i3c_bus_count") or 0),
        "| 12V min | {} |".format(format_voltage(min_value(samples, "rail_12v"))),
        "| 12V max | {} |".format(format_voltage(max_value(samples, "rail_12v"))),
        "| GPU power max | {} |".format(format_watts(max_value(samples, "gpu_power_total_w"))),
        "| CPU load max | {} |".format(format_pct(max_value(samples, "cpu_load_pct"))),
        "",
        "## Telemetry Charts",
        "",
        render_charts(run_dir, samples),
        "## Scope Note",
        "",
        textwrap.fill(
            "This report is written to preserve only what the run directly supports. "
            "Without multimeter data, the strongest claim is bench isolation by system behavior. "
            "With a known-good PSU swap, the strongest claim is that the observed fault followed the returned PSU on the bench. "
            "Do not claim absolute rail compliance, ripple compliance, or transient-response compliance unless those were actually measured.",
            width=90,
        ),
        "",
    ])
    return "\n".join(lines) + "\n"


def failure_mode_checks(run_row: sqlite3.Row, reasons: List[str]) -> List[Tuple[str, bool]]:
    verdict = run_row["verdict"] or "UNKNOWN"
    post_failed = run_row["post_result"] == "fail"
    swap_followed = run_row["known_good_swap_result"] == "fault-follows-psu"
    has_warning = verdict == "WARN"
    has_fail = verdict == "FAIL"
    text = " ".join(reasons).lower()
    return [
        ("Non-detection (doesn't show in BIOS/lspci)", False),
        ("Detection but errors under load", has_fail and ("drift" in text or "instability" in text)),
        ("Thermal (throttle/shutdown, temp at failure: ___C)", has_fail and "thermal" in text),
        ("Computation errors (stress-ng --verify / gpu_burn)", has_fail and "computation" in text),
        ("MCE / EDAC / dmesg errors", has_fail and ("mce" in text or "edac" in text or "kernel" in text)),
        ("Intermittent (works sometimes, fails sometimes)", has_warning),
        ("DOA (dead on arrival, never worked)", post_failed and swap_followed),
        ("Counterfeit suspect", False),
        ("Compatibility issue (works alone, fails in full config)", False),
        ("Other: bench-isolated PSU fault follows returned unit and clears with known-good PSU", swap_followed),
    ]


def render_fm_entry(
    run_dir: Path,
    run_row: sqlite3.Row,
    samples: List[sqlite3.Row],
    events: List[sqlite3.Row],
    defensible_conclusion: Callable[[sqlite3.Row], str],
    batch_tracker_name: str,
) -> str:
    del samples
    row_keys = set(run_row.keys())
    reasons = json.loads(
        (run_row["verdict_reasons_json"] if "verdict_reasons_json" in row_keys else "[]") or "[]"
    )
    checks = failure_mode_checks(run_row, reasons)
    check_lines = ["  - [{}] {}".format("x" if checked else " ", label) for label, checked in checks]
    evidence_bits = [
        "summary.md",
        "verdict.json",
        "samples.csv",
        "events.csv",
    ]
    notes = [
        defensible_conclusion(run_row),
        "Claim scope: {}.".format(run_row["telemetry_status"] or "unknown"),
        "Coverage confidence: {:.2f}.".format(coverage_confidence(run_dir, run_row, events)["score"]),
    ]
    if run_row["meter_available"]:
        notes.append("Manual rail readings were recorded in the run bundle.")
    else:
        notes.append("No multimeter readings were recorded for this run.")
    if not events:
        notes.append("No event rows were captured in this run.")

    lines = [
        "### FM-XXXX | PSU | {}".format((run_row["ended_at"] or run_row["started_at"] or "")[:10]),
        "- **Part:** {} {} ({}W)".format(run_row["psu_make"], run_row["psu_model"], run_row["psu_wattage"]),
        "- **Serial:** {}".format(run_row["psu_serial"]),
        "- **Batch/Lot:** {}".format(run_row["psu_lot"] or "<!-- NEEDS -->"),
        "- **Supplier:** <!-- NEEDS -->",
        "- **Board:** {}".format(run_row["bench_name"] or "<!-- NEEDS -->"),
        "- **Config:** Bench isolation run via profile `{}` on `{}`".format(run_row["profile_name"], run_row["os_name"]),
        "- **Slot/Position:** N/A",
        "- **Order #:** <!-- NEEDS -->",
        "- **Reported Symptom:** {}".format(run_row["reason"] or "<!-- NEEDS -->"),
        "- **Actual Failure Mode:**",
    ]
    lines.extend(check_lines)
    lines.extend([
        "- **Test Bench:** {} (last verified: {})".format(run_row["bench_name"] or "<!-- NEEDS -->", run_row["bench_last_verified"] or "<!-- NEEDS -->"),
        "- **Evidence:** {}".format(", ".join(evidence_bits)),
        "- **Resolution:** Pending",
        "- **Batch Pattern?** Unknown - cross-check `{}` for same lot/date failures.".format(batch_tracker_name),
        "- **Notes:** {}".format(" ".join(notes)),
        "",
    ])
    return "\n".join(lines)


def render_rma_summary(
    run_dir: Path,
    run_row: sqlite3.Row,
    samples: List[sqlite3.Row],
    events: List[sqlite3.Row],
    defensible_conclusion: Callable[[sqlite3.Row], str],
) -> str:
    del samples
    row_keys = set(run_row.keys())
    reasons = json.loads(
        (run_row["verdict_reasons_json"] if "verdict_reasons_json" in row_keys else "[]") or "[]"
    )
    lines = [
        "# Manufacturer RMA Summary",
        "",
        "## Unit",
        "",
        "- Make/model: {} {}".format(run_row["psu_make"], run_row["psu_model"]),
        "- Wattage: {}W".format(run_row["psu_wattage"]),
        "- Serial: {}".format(run_row["psu_serial"]),
        "- Lot/date code: {}".format(run_row["psu_lot"] or "not recorded"),
        "- Test date: {}".format((run_row["ended_at"] or run_row["started_at"] or "")[:10]),
        "- Ambient temp: {}".format(format_celsius(run_row["ambient_c"]) if "ambient_c" in set(run_row.keys()) else "n/a"),
        "",
        "## Bench Isolation Result",
        "",
        "- Known-good bench: {}".format(run_row["bench_name"] or "recorded in run bundle"),
        "- POST with returned PSU: {}".format(run_row["post_result"] or "not recorded"),
        "- Known-good PSU substitution performed: {}".format("yes" if run_row["known_good_swap_performed"] else "no"),
        "- Known-good PSU substitution result: {}".format(run_row["known_good_swap_result"] or "not recorded"),
        "- Coverage confidence: {:.2f}".format(coverage_confidence(run_dir, run_row, events)["score"]),
        "",
        "## Defensible Statement",
        "",
        defensible_conclusion(run_row),
        "",
    ]
    lines.extend(render_rma_storage_workload_lines(run_dir))
    lines.extend([
        "## Directly Observed Facts",
        "",
    ])
    if reasons:
        for reason in reasons:
            lines.append("- {}".format(reason))
    else:
        lines.append("- No fail conditions were captured in this run.")
    lines.extend([
        "",
        "## Evidence Bundle",
        "",
        "- Summary: {}".format("summary.md"),
        "- Verdict: {}".format("verdict.json"),
        "- Telemetry samples: {}".format("samples.csv"),
        "- Event log: {}".format("events.csv"),
        "",
        "## Scope Limitation",
        "",
        "This statement is intentionally limited to what the bench test directly supports. "
        "If no multimeter readings were recorded, this package does not claim absolute rail-spec compliance, ripple compliance, or transient-response compliance.",
        "",
    ])
    return "\n".join(lines) + "\n"


# Exact matches that mean "no serial yet"
_PLACEHOLDER_SERIAL_EXACT_RE = re.compile(
    r"^(sn\d{6}|test|placeholder|unknown|dummy|example|\?+|tbd|n/?a|none)$",
    re.IGNORECASE,
)

# Substring markers that mean "this is a synthetic / smoke-test serial and
# must NOT pollute the real batch tracker". Any serial containing one of
# these tokens is treated as non-production even if the rest of the string
# looks real (e.g. "SMOKE-TEST-001", "HX1200-SAMPLE", "DEMO-UNIT-42").
_PLACEHOLDER_SERIAL_MARKERS = (
    "smoke",
    "test",
    "sample",
    "demo",
    "dummy",
    "fake",
    "dryrun",
    "dry-run",
)


def is_placeholder_serial(serial: Optional[str]) -> bool:
    if not serial or not serial.strip():
        return True
    cleaned = serial.strip()
    if _PLACEHOLDER_SERIAL_EXACT_RE.match(cleaned):
        return True
    lowered = cleaned.lower()
    return any(marker in lowered for marker in _PLACEHOLDER_SERIAL_MARKERS)
