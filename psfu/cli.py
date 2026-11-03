from __future__ import annotations

import argparse
import html as _html
import json
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from psfu.batch import write_batch_summary
from psfu.baselines import latest_pass_run, record_baseline
from psfu.config import RUNS_DIR, apply_path_overrides
from psfu.synthetic import SCENARIOS as SYNTHETIC_SCENARIOS, build_synthetic_run
from psfu.replay import replay_all_runs, replay_run
from psfu.reports.exporter import export_run_artifacts
from psfu.ui.sparkline import render_sparkline
from psfu.runtime import bootstrap_handler, preflight_handler
from psfu.persistence.db import Database
from psfu.verdict.rules import VerdictEngine


def _run_test_lazy(args: argparse.Namespace) -> int:
    """Defer importing the live execution engine (which requires curses) until
    the `run` subcommand is actually invoked. Lets non-TUI commands (list,
    report, chart, verdict, recover) load on hosts without _curses (Windows
    stock Python)."""
    from psfu.runners.execute import run_test
    return run_test(args)


CommandHandler = Callable[[argparse.Namespace], int]


def _load_run_db(run_dir_arg: str) -> tuple[Path, Database]:
    run_dir = _resolve_run_dir(run_dir_arg)
    db_path = run_dir / "run.sqlite"
    if not db_path.exists():
        raise SystemExit("Missing run DB: {}".format(db_path))
    return run_dir, Database(db_path)


def _resolve_run_dir(run_dir_arg: str) -> Path:
    run_dir = Path(run_dir_arg).resolve()
    # Containment check: ``run_dir`` must be a subdirectory of RUNS_DIR.
    # Without this, a crafted path (e.g. ``--run-dir /etc`` with a planted
    # ``run.sqlite``) would open an arbitrary SQLite file and cause the tool
    # to write evidence back into that directory. RUNS_DIR is the only
    # location the tool is authorized to read/write per design.
    runs_dir_resolved = RUNS_DIR.resolve()
    try:
        run_dir.relative_to(runs_dir_resolved)
    except ValueError:
        raise SystemExit(
            "Refusing to operate outside RUNS_DIR.\n"
            "  run_dir:  {}\n"
            "  RUNS_DIR: {}\n"
            "Pass a path under RUNS_DIR, or run with --run-dir <name> where "
            "the name resolves inside it.".format(run_dir, runs_dir_resolved)
        )
    return run_dir


def _verdict_label_from_payload(payload: dict) -> str:
    if "verdict" in payload:
        return str(payload.get("verdict") or "UNKNOWN")
    overall = payload.get("overall")
    if isinstance(overall, dict):
        return str(overall.get("status") or "UNKNOWN")
    return "UNKNOWN"


def list_runs(_: argparse.Namespace) -> int:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs = sorted(
        path for path in RUNS_DIR.rglob("*")
        if path.is_dir() and (path / "run.sqlite").exists()
    )
    if not runs:
        print("No runs yet.")
        return 0
    for run_dir in runs:
        verdict_path = run_dir / "verdict.json"
        verdict = "UNKNOWN"
        if verdict_path.exists():
            try:
                verdict = _verdict_label_from_payload(json.loads(verdict_path.read_text(encoding="utf-8")))
            except Exception:
                verdict = "UNKNOWN"
        print("{}  {}".format(verdict.ljust(7), run_dir.name))
    return 0


def generate_pdf(run_dir: Path) -> int:
    summary_md = run_dir / "summary.md"
    summary_html = run_dir / "summary.html"
    summary_pdf = run_dir / "summary.pdf"

    if not summary_md.exists():
        print("Cannot generate PDF: summary.md missing.")
        return 1

    md_text = summary_md.read_text(encoding="utf-8")
    html_content = textwrap.dedent("""\
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body { font-family: sans-serif; line-height: 1.4; padding: 40px; }
                pre { background: #f4f4f4; padding: 10px; border-radius: 5px; white-space: pre-wrap; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
                th { background: #eee; }
                h1, h2, h3 { color: #333; }
                .verdict-pass { color: green; font-weight: bold; }
                .verdict-fail { color: red; font-weight: bold; }
            </style>
        </head>
        <body>
    """)
    # Build HTML with state machine so table rows and list items are grouped
    # into single <table>/<ul> elements rather than one per row.
    # All user-supplied content (PSU fields, reasons, etc.) is html.escape()'d.
    parts_list: list = [html_content]
    in_table = False
    in_list = False

    def _close_table() -> None:
        nonlocal in_table
        if in_table:
            parts_list.append("</table>\n")
            in_table = False

    def _close_list() -> None:
        nonlocal in_list
        if in_list:
            parts_list.append("</ul>\n")
            in_list = False

    for line in md_text.splitlines():
        if line.startswith("# "):
            _close_table(); _close_list()
            parts_list.append("<h1>{}</h1>\n".format(_html.escape(line[2:])))
        elif line.startswith("## "):
            _close_table(); _close_list()
            parts_list.append("<h2>{}</h2>\n".format(_html.escape(line[3:])))
        elif line.startswith("### "):
            _close_table(); _close_list()
            parts_list.append("<h3>{}</h3>\n".format(_html.escape(line[4:])))
        elif line.startswith("|"):
            if "---" in line:
                continue
            cells = [p.strip() for p in line.split("|") if p.strip()]
            if not cells:
                continue
            _close_list()
            if not in_table:
                parts_list.append("<table>\n")
                in_table = True
            parts_list.append(
                "<tr>" + "".join("<td>{}</td>".format(_html.escape(c)) for c in cells) + "</tr>\n"
            )
        elif line.startswith("- "):
            _close_table()
            if not in_list:
                parts_list.append("<ul>\n")
                in_list = True
            parts_list.append("<li>{}</li>\n".format(_html.escape(line[2:])))
        elif line.strip():
            _close_table(); _close_list()
            parts_list.append("<p>{}</p>\n".format(_html.escape(line)))
        else:
            _close_table(); _close_list()

    _close_table()
    _close_list()
    parts_list.append("</body></html>")
    summary_html.write_text("".join(parts_list), encoding="utf-8")

    wkhtmltopdf = shutil.which("wkhtmltopdf")
    if not wkhtmltopdf:
        print("wkhtmltopdf not found on PATH; only HTML report generated at: {}".format(summary_html))
        return 0

    try:
        subprocess.run([wkhtmltopdf, "--quiet", str(summary_html), str(summary_pdf)], check=True)
        print("PDF report generated at: {}".format(summary_pdf))
        return 0
    except Exception as exc:
        print("wkhtmltopdf failed: {}".format(exc))
        return 1


def batch_summary(args: argparse.Namespace) -> int:
    result = write_batch_summary(
        args.lot_code,
        Path(args.output).resolve() if args.output else None,
    )
    print("Batch summary: {}".format(result["output_path"]))
    print("Runs scanned: {}".format(len(result["rows"])))
    print("FAIL count: {}".format(result["fail_count"]))
    if args.print:
        print("")
        print(result["output_path"].read_text(encoding="utf-8"))
    return 0


def report_run(args: argparse.Namespace) -> int:
    run_dir, db = _load_run_db(args.run_dir)
    export_run_artifacts(run_dir, db)
    if args.pdf:
        pdf_rc = generate_pdf(run_dir)
        if pdf_rc != 0:
            return pdf_rc
    print((run_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


def chart_run(args: argparse.Namespace) -> int:
    run_dir, db = _load_run_db(args.run_dir)
    run_row = db.fetch_run()
    if run_row is None:
        raise SystemExit("No run metadata in {}".format(run_dir / "run.sqlite"))
    samples = db.fetch_samples()
    events = db.fetch_events()

    term_width = shutil.get_terminal_size((100, 30)).columns
    sl_width = max(20, min(term_width - 40, 80))

    verdict = run_row["verdict"] or "(incomplete)"
    print("Run:     {}  ->  {}".format(run_row["started_at"], run_row["ended_at"] or "(running)"))
    print("Profile: {} (v{})".format(run_row["profile_name"], run_row["profile_version"]))
    print("PSU:     {} {}  {}W  SN:{}".format(
        run_row["psu_make"] or "?",
        run_row["psu_model"] or "?",
        run_row["psu_wattage"] or 0,
        run_row["psu_serial"] or "?",
    ))
    print("Verdict: {}  (claim scope: {})".format(verdict, run_row["telemetry_status"] or "?"))
    print("")

    phase_counts = {}
    phase_order: List[str] = []
    for sample in samples:
        phase_name = sample["phase"]
        if phase_name not in phase_counts:
            phase_counts[phase_name] = 0
            phase_order.append(phase_name)
        phase_counts[phase_name] += 1

    phase_event_counts = {}
    for event in events:
        bucket = phase_event_counts.setdefault(event["phase"], {"crit": 0, "warn": 0, "info": 0})
        severity = event["severity"]
        if severity in bucket:
            bucket[severity] += 1

    if phase_order:
        max_count = max(phase_counts.values())
        print("Phases:")
        for phase_name in phase_order:
            count = phase_counts[phase_name]
            bar_w = max(1, int(round((count / float(max_count)) * 32)))
            bar = "█" * bar_w
            ev = phase_event_counts.get(phase_name, {})
            notes = []
            if ev.get("crit"):
                notes.append("{} crit".format(ev["crit"]))
            if ev.get("warn"):
                notes.append("{} warn".format(ev["warn"]))
            ev_note = ("  <- " + ", ".join(notes)) if notes else ""
            print("  {:<14} [{:>4}s]  {}{}".format(phase_name, count, bar, ev_note))
        print("")

    metric_specs = [
        ("CPU%", "cpu_load_pct", "{:>6.1f}"),
        ("RAM%", "mem_used_pct", "{:>6.1f}"),
        ("Disk%", "disk_busy_pct", "{:>6.1f}"),
        ("SSD °C", "ssd_temp_c", "{:>6.1f}"),
        ("GPU W", "gpu_power_w", "{:>6.1f}"),
        ("RAPL W", "cpu_pkg_watt", "{:>6.1f}"),
        ("CPU °C", "cpu_temp_c", "{:>6.1f}"),
        ("GPU °C", "gpu_temp_c", "{:>6.1f}"),
        ("VRAM °C", "gpu_vram_temp_c", "{:>6.1f}"),
        ("GPU%", "gpu_util_pct", "{:>6.1f}"),
        ("12V", "rail_12v", "{:>6.2f}"),
        ("5V", "rail_5v", "{:>6.2f}"),
        ("3.3V", "rail_33v", "{:>6.2f}"),
        ("FanCPU", "fan_cpu_rpm", "{:>6.0f}"),
        ("FanSYS", "fan_sys_rpm", "{:>6.0f}"),
    ]
    print("Full-run sparklines ({} samples over {} phases):".format(len(samples), len(phase_order)))
    for label, key, fmt in metric_specs:
        values = [sample[key] for sample in samples]
        cleaned = [v for v in values if v is not None]
        if not cleaned:
            print("  {:<7} (no data)".format(label))
            continue
        spark = render_sparkline(values, sl_width)
        lo = min(cleaned)
        hi = max(cleaned)
        avg = sum(cleaned) / float(len(cleaned))
        stats = "min {}  max {}  avg {}".format(
            fmt.format(lo).strip(),
            fmt.format(hi).strip(),
            fmt.format(avg).strip(),
        )
        print("  {:<7} {}  {}".format(label, spark, stats))
    print("")

    if events:
        print("Events ({}):".format(len(events)))
        for event in events[:20]:
            message = (event["message"] or "").replace("\n", " ").strip()
            if len(message) > 70:
                message = message[:67] + "..."
            print("  {:<5} {:<14} {:<18} {}".format(
                event["severity"], event["phase"], event["kind"], message
            ))
        if len(events) > 20:
            print("  ... and {} more".format(len(events) - 20))
        print("")

    print("Verdict: {}".format(verdict))
    reasons_json = run_row["verdict_reasons_json"]
    if reasons_json:
        try:
            reasons = json.loads(reasons_json)
            if isinstance(reasons, list):
                for reason in reasons:
                    print("  - {}".format(reason))
        except Exception:
            pass
    return 0


def rescore_verdict(args: argparse.Namespace) -> int:
    run_dir, db = _load_run_db(args.run_dir)
    run_row = db.fetch_run()
    if run_row is None:
        raise SystemExit("No run metadata in {}".format(run_dir / "run.sqlite"))

    old_verdict = run_row["verdict"] or "(none)"
    old_reasons: List[str] = []
    if run_row["verdict_reasons_json"]:
        try:
            old_reasons = json.loads(run_row["verdict_reasons_json"])
        except Exception:
            pass

    new_verdict, new_reasons, new_scope, new_subsystems = VerdictEngine.evaluate(db)
    db.update_run(
        run_row["id"],
        {
            "verdict": new_verdict,
            "verdict_reasons_json": json.dumps(new_reasons, indent=2),
            "subsystem_results_json": json.dumps(new_subsystems, indent=2),
            "telemetry_status": new_scope,
        },
    )
    export_run_artifacts(run_dir, db)

    verdict_changed = new_verdict != old_verdict
    print("Run dir:  {}".format(run_dir))
    print("Verdict:  {} -> {}{}".format(
        old_verdict, new_verdict, "  (CHANGED)" if verdict_changed else "  (unchanged)"
    ))
    print("")
    if verdict_changed:
        print("Old reasons:")
        for reason in old_reasons:
            print("  - {}".format(reason))
        print("")
    print("New reasons:")
    for reason in new_reasons:
        print("  - {}".format(reason))
    print("")
    print("Scope: {}".format(new_scope))
    print("Artifacts regenerated: summary.md  verdict.json  fm-entry.md  rma-summary.md")
    return 0


def replay_rescore(args: argparse.Namespace) -> int:
    if args.all:
        result = replay_all_runs()
        print("Scanned: {}".format(len(result["results"]) + len(result["failures"])))
        print("Reports: {}".format(len(result["results"])))
        print("Failures: {}".format(len(result["failures"])))
        print("Summary: {}".format(result["summary_path"]))
        return 0 if not result["failures"] else 1

    run_dir = _resolve_run_dir(args.run_dir)
    result = replay_run(run_dir)
    print("Run dir:  {}".format(run_dir))
    print(
        "Verdict:  {} -> {}{}".format(
            result["stored_verdict"],
            result["current_verdict"],
            "  (CHANGED)" if result["verdict_changed"] else "  (unchanged)",
        )
    )
    print(
        "Scope:    {} -> {}{}".format(
            result["stored_claim_scope"],
            result["current_claim_scope"],
            "  (CHANGED)" if result["claim_scope_changed"] else "  (unchanged)",
        )
    )
    print("Protocol: {} -> {}".format(result["stored_protocol_version"], result["current_protocol_version"]))
    print("Report:   {}".format(result["report_path"]))
    return 0


def recover_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    lg_path = run_dir / "last_gasp.jsonl"
    if not lg_path.exists():
        print("No last_gasp.jsonl found in {}".format(run_dir))
        return 1

    last_sample = None
    try:
        with lg_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    last_sample = json.loads(line)
                except Exception:
                    pass
    except Exception as exc:
        print("Error reading {}: {}".format(lg_path, exc))
        return 1

    if last_sample is None:
        print("No valid samples found in {}".format(lg_path))
        return 1

    print("Last recorded sample from crash (run: {}):".format(run_dir.name))
    print(json.dumps(last_sample, indent=2))
    return 0


def baseline_record(args: argparse.Namespace) -> int:
    if args.run_dir:
        run_dir = _resolve_run_dir(args.run_dir)
    else:
        run_dir = latest_pass_run(args.profile)
        if run_dir is None:
            print("No PASS run found for profile: {}".format(args.profile))
            return 1
    path = record_baseline(run_dir)
    print("Baseline source: {}".format(run_dir))
    print("Baseline file:   {}".format(path))
    return 0


def _selftest_handler(args: argparse.Namespace) -> int:
    from psfu.storage.selftest import (
        run_selftest,
        SELFTEST_PASSED,
        SELFTEST_FAILED,
        SELFTEST_TIMEOUT,
        SELFTEST_ABORTED,
        SELFTEST_ERROR,
    )
    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path.cwd()
    result, _ = run_selftest(args.device, short=args.short, out_dir=out_dir, verbose=True)
    if result == SELFTEST_PASSED:
        return 0
    if result in (SELFTEST_FAILED, SELFTEST_ABORTED):
        return 1
    if result == SELFTEST_TIMEOUT:
        print("[self-test] TIMEOUT — test did not complete within the time limit.")
        return 2
    print("[self-test] ERROR — could not start or communicate with device.")
    return 3


def build_parser() -> argparse.ArgumentParser:
    from psfu.config import APP_FULL_NAME, APP_TAGLINE, APP_DESCRIPTION
    parser = argparse.ArgumentParser(
        prog="psfu",
        description="{}\n{}\n\n{}".format(APP_FULL_NAME, APP_TAGLINE, APP_DESCRIPTION),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    paths = parser.add_argument_group(
        "output locations",
        "Precedence: flag > environment variable > default (~/psfu-data).",
    )
    paths.add_argument("--data-dir", metavar="DIR", help="Root for runs/, baselines/, tools/ and logs. Env: PSFU_HOME.")
    paths.add_argument("--runs-dir", metavar="DIR", help="Where run bundles are written. Env: PSFU_RUNS_DIR.")
    paths.add_argument("--baselines-dir", metavar="DIR", help="Baseline store. Env: PSFU_BASELINES_DIR.")
    paths.add_argument("--tools-dir", metavar="DIR", help="Where bootstrap builds stress-ng/fio/etc. Env: PSFU_TOOLS_DIR.")
    paths.add_argument("--profiles-dir", metavar="DIR", help="Profile JSON directory. Env: PSFU_PROFILES_DIR.")
    sub = parser.add_subparsers(dest="command", required=True)

    demo_cmd = sub.add_parser(
        "demo",
        help="Generate a synthetic run (no hardware needed) and score it.",
    )
    demo_cmd.add_argument(
        "--scenario", choices=list(SYNTHETIC_SCENARIOS), default="warn",
        help="Which synthetic fault pattern to generate (default: warn).",
    )
    demo_cmd.add_argument("--seed", type=int, default=7, help="RNG seed (default: 7).")
    demo_cmd.set_defaults(func=demo_run)

    run_cmd = sub.add_parser("run", help="Start a new PSU test run")
    run_cmd.add_argument("--profile", default="system-full-v1")
    run_cmd.add_argument("--psu-make")
    run_cmd.add_argument("--psu-model")
    run_cmd.add_argument("--psu-wattage", help="PSU nameplate wattage. Enforced by the bench power budget check — runs abort if rated wattage is below detected CPU TDP + GPU TGP + overhead .")
    run_cmd.add_argument(
        "--allow-undersized-psu",
        action="store_true",
        help="Override the bench power budget check. Records an override flag in run metadata. Use only when you know the bench has a redundant or supplementary supply.",
    )
    run_cmd.add_argument("--psu-serial")
    run_cmd.add_argument("--psu-lot")
    run_cmd.add_argument("--operator")
    run_cmd.add_argument("--reason")
    run_cmd.add_argument("--bench-name")
    run_cmd.add_argument("--bench-last-verified")
    run_cmd.add_argument("--os-name")
    run_cmd.add_argument("--ambient-c", type=float)
    run_cmd.add_argument("--meter-available", choices=["yes", "no"])
    run_cmd.add_argument("--manual-idle-12v", type=float)
    run_cmd.add_argument("--manual-load-12v", type=float)
    run_cmd.add_argument("--manual-idle-5v", type=float)
    run_cmd.add_argument("--manual-load-5v", type=float)
    run_cmd.add_argument("--paperclip-result", choices=["not-run", "pass", "fail"])
    run_cmd.add_argument("--post-result", choices=["pass", "fail", "not-tested"])
    run_cmd.add_argument("--known-good-swap-performed", choices=["yes", "no"])
    run_cmd.add_argument("--known-good-swap-result", choices=["fault-follows-psu", "fault-remains", "no-difference", "not-clear"])
    run_cmd.add_argument("--original-system-retest-result", choices=["not-run", "same-failure", "passed", "not-applicable"])
    run_cmd.add_argument("--notes")
    run_cmd.add_argument(
        "--storage-path",
        action="append",
        default=[],
        help=(
            "Readable file or directory to use for storage validation read/checksum phases. "
            "Repeat for multiple targets."
        ),
    )
    run_cmd.add_argument(
        "--storage-dut-path",
        action="append",
        default=[],
        help=(
            "Block device path(s) for storage DUT inventory / identify-log capture. "
            "Repeat for multiple devices, e.g. --storage-dut-path /dev/nvme0n1."
        ),
    )
    run_cmd.add_argument(
        "--skip-phases",
        default="",
        help="Comma-separated phase names to skip (e.g. --skip-phases soak,cooldown).",
    )
    run_cmd.add_argument("--dry-run", action="store_true")
    run_cmd.add_argument(
        "--destructive",
        action="store_true",
        help=(
            "Opt in to destructive write fio jobs. REQUIRED when the profile has "
            "\"destructive\": true. Triggers serial re-entry confirmation and mount "
            "check before any run state is created. All data on the DUT will be overwritten."
        ),
    )
    run_cmd.add_argument(
        "--bench-gage-rr",
        action="store_true",
        help=(
            "Route this run under runs/gage-rr/<bench>/ for bench Gauge R&R "
            "calibration work. Does not change verdict semantics."
        ),
    )
    run_cmd.add_argument("--dut-cpu", action=argparse.BooleanOptionalAction, default=None)
    run_cmd.add_argument("--dut-gpu", action=argparse.BooleanOptionalAction, default=None)
    run_cmd.add_argument("--dut-mem", action=argparse.BooleanOptionalAction, default=None)
    run_cmd.add_argument("--dut-storage", action=argparse.BooleanOptionalAction, default=None)
    run_cmd.add_argument(
        "--tui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Render a live curses dashboard during the run with phase/progress "
            "bars, worker state, event counters, and rolling sparklines for the "
            "last 60 samples. Pass --no-tui to force plain-log mode (useful for "
            "headless CI, redirected output, or debugging). Default: on. "
            "Falls back to plain log mode if curses is unavailable."
        ),
    )
    run_cmd.set_defaults(func=_run_test_lazy)

    chart_cmd = sub.add_parser(
        "chart",
        help="Post-hoc terminal chart viewer for an existing run directory.",
    )
    chart_cmd.add_argument("run_dir", help="Path to a runs/<timestamp>/ directory")
    chart_cmd.set_defaults(func=chart_run)

    list_cmd = sub.add_parser("list", help="List completed runs")
    list_cmd.set_defaults(func=list_runs)

    report_cmd = sub.add_parser("report", help="Rebuild and print a run summary")
    report_cmd.add_argument("run_dir")
    report_cmd.add_argument("--pdf", action="store_true", help="Also generate a PDF report via wkhtmltopdf.")
    report_cmd.set_defaults(func=report_run)

    pdf_cmd = sub.add_parser("pdf", help="Generate a PDF report for an existing run directory.")
    pdf_cmd.add_argument("run_dir")
    pdf_cmd.set_defaults(func=lambda args: generate_pdf(Path(args.run_dir).resolve()))

    recover_cmd = sub.add_parser("recover", help="Read the last gasp sample from a crashed run.")
    recover_cmd.add_argument("run_dir")
    recover_cmd.set_defaults(func=recover_run)

    preflight_cmd = sub.add_parser("preflight", help="Show dependency availability")
    preflight_cmd.set_defaults(func=preflight_handler)

    bootstrap_cmd = sub.add_parser(
        "bootstrap",
        help="Prep a DUT: build gpu-burn, try sensor fallbacks, print readiness (no run).",
    )
    bootstrap_cmd.add_argument(
        "--no-auto-install",
        action="store_true",
        help="Skip apt-install / clone+build; only report what's already present.",
    )
    bootstrap_cmd.add_argument(
        "--json",
        action="store_true",
        help="Also dump the full bootstrap report as JSON after the human-readable summary.",
    )
    bootstrap_cmd.set_defaults(func=bootstrap_handler)

    verdict_cmd = sub.add_parser(
        "verdict",
        help="Re-score an existing run against current verdict rules and regenerate artifacts.",
    )
    verdict_cmd.add_argument("run_dir", help="Path to a runs/<timestamp>/ directory containing run.sqlite.")
    verdict_cmd.set_defaults(func=rescore_verdict)

    rescore_cmd = sub.add_parser(
        "rescore",
        help="Replay an existing run through the current verdict engine without modifying source artifacts.",
    )
    rescore_cmd.add_argument("run_dir", nargs="?", help="Path to a runs/<timestamp>/ directory containing run.sqlite.")
    rescore_cmd.add_argument("--all", action="store_true", help="Replay all runs under RUNS_DIR and emit a summary report.")
    rescore_cmd.set_defaults(func=replay_rescore)

    baseline_cmd = sub.add_parser(
        "baseline",
        help="Record or manage known-good per-bench baseline files.",
    )
    baseline_sub = baseline_cmd.add_subparsers(dest="baseline_command", required=True)
    baseline_record_cmd = baseline_sub.add_parser(
        "record",
        help="Capture a baseline JSON from a PASS run.",
    )
    baseline_record_cmd.add_argument("--profile", default="system-full-v1", help="Profile name to search when --run-dir is not given.")
    baseline_record_cmd.add_argument("--run-dir", help="Specific runs/<timestamp>/ directory to use as the baseline source.")
    baseline_record_cmd.set_defaults(func=baseline_record)

    batch_cmd = sub.add_parser(
        "batch",
        help="Scan historical runs for a PSU lot/date code and emit a markdown summary.",
    )
    batch_cmd.add_argument("lot_code", help="PSU lot/date code to scan under RUNS_DIR.")
    batch_cmd.add_argument("--output", help="Write the report to a specific markdown path.")
    batch_cmd.add_argument("--print", action="store_true", help="Also print the generated markdown report.")
    batch_cmd.set_defaults(func=batch_summary)

    about_cmd = sub.add_parser("about", help="Print full PS-FU identity and methodology statement.")
    about_cmd.set_defaults(func=_about_handler)

    selftest_cmd = sub.add_parser(
        "selftest",
        help="Run an NVMe extended (or short) self-test on a device and report the result.",
    )
    selftest_cmd.add_argument("device", help="Block device path, e.g. /dev/nvme0n1")
    selftest_cmd.add_argument(
        "--short",
        action="store_true",
        help="Run a short self-test (1-2 min) instead of extended (10-30 min).",
    )
    selftest_cmd.add_argument(
        "--out-dir",
        help="Directory to write the self-test log JSON (default: current directory).",
    )
    selftest_cmd.set_defaults(func=_selftest_handler)

    return parser


def _about_handler(_: argparse.Namespace) -> int:
    from psfu.config import APP_NAME, APP_FULL_NAME, APP_TAGLINE, APP_ABOUT
    print("{} — {}".format(APP_NAME, APP_FULL_NAME))
    print(APP_TAGLINE)
    print()
    print(APP_ABOUT)
    return 0


def demo_run(args: argparse.Namespace) -> int:
    """Write a synthetic run under RUNS_DIR and print its verdict."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = RUNS_DIR / "{}_demo-{}".format(stamp, args.scenario)
    result = build_synthetic_run(run_dir, args.scenario, seed=args.seed)
    print("SYNTHETIC DATA -- no hardware was exercised.")
    print("Run dir:  {}".format(run_dir))
    print("Verdict:  {}".format(result["status"]))
    print("Scope:    {}".format(result["claim_scope"]))
    for name, sub in result["subsystems"].items():
        print("  {:<8} {}".format(name, sub["status"]))
    for reason in result["reasons"]:
        print("- {}".format(reason))
    print("Summary:  {}".format(run_dir / "summary.md"))
    return 0


def _configure_stdio() -> None:
    """Reports contain non-ASCII (arrows, dashes); don't crash on cp1252 consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def main(argv: Optional[List[str]] = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_path_overrides(
        data_dir=args.data_dir,
        runs_dir=args.runs_dir,
        baselines_dir=args.baselines_dir,
        tools_dir=args.tools_dir,
        profiles_dir=args.profiles_dir,
    )
    if getattr(args, "func", None) is replay_rescore and not getattr(args, "all", False) and not getattr(args, "run_dir", None):
        parser.error("rescore requires RUN_DIR unless --all is passed")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
