"""Top-level runtime utilities: profile loading, preflight, bootstrap handler,
and run-dir creation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from psfu.common import command_exists, run_capture, shell_join, slugify, utc_now
from psfu.config import APP_NAME, BASELINES_DIR, PROFILES_DIR, RUNS_DIR
from psfu.env import ensure_runtime_dependencies
from psfu.hardware import (
    board_identity,
    fio_binary_path,
    fio_repo_dir,
    gpu_burn_usable_binary,
    nvcc_version_tuple,
    nvme_cli_binary_path,
    nvme_cli_repo_dir,
    nvidia_gpu_compute_caps,
    smartctl_binary_path,
    smartmontools_repo_dir,
    stress_ng_binary_path,
    stress_ng_repo_dir,
)
from psfu.models import Profile
from psfu.storage.devices import discover_storage_devices, serialize_storage_inventory


def load_profile(name: str) -> Profile:
    raw = Path(name).expanduser()

    # Accept explicit paths as well as logical profile names.
    direct_candidates = []
    if raw.suffix == ".json":
        direct_candidates.append(raw)
    else:
        direct_candidates.append(raw.with_suffix(".json"))

    if raw.is_absolute() or len(raw.parts) > 1:
        candidates = direct_candidates
    else:
        # Production profiles live directly in profiles/. Dev / TUI-preview profiles
        # live in profiles/dev/ so they don't clutter operator tab-completion but
        # remain reachable by the same --profile name.
        candidates = [
            PROFILES_DIR / "{}.json".format(name),
            PROFILES_DIR / "dev" / "{}.json".format(name),
        ]

    for profile_path in candidates:
        if profile_path.exists():
            return Profile(json.loads(profile_path.read_text(encoding="utf-8")))
    raise SystemExit("Profile not found: {} (searched: {})".format(
        name, ", ".join(str(p) for p in candidates)
    ))


def get_preflight() -> Dict[str, object]:
    checks = []
    for label in ["stress-ng", "fio", "sensors", "nvidia-smi", "journalctl", "lsblk", "lspci", "smartctl", "nvme"]:
        if label == "stress-ng":
            resolved = stress_ng_binary_path()
            checks.append(
                {
                    "name": label,
                    "available": resolved is not None,
                    "path": resolved,
                    "vendored": bool(resolved and resolved.startswith(str(stress_ng_repo_dir()))),
                }
            )
            continue
        if label == "fio":
            resolved = fio_binary_path()
            checks.append(
                {
                    "name": label,
                    "available": resolved is not None,
                    "path": resolved,
                    "vendored": bool(resolved and resolved.startswith(str(fio_repo_dir()))),
                }
            )
            continue
        if label == "nvme":
            resolved = nvme_cli_binary_path()
            checks.append(
                {
                    "name": label,
                    "available": resolved is not None,
                    "path": resolved,
                    "vendored": bool(resolved and resolved.startswith(str(nvme_cli_repo_dir()))),
                }
            )
            continue
        if label == "smartctl":
            resolved = smartctl_binary_path()
            checks.append(
                {
                    "name": label,
                    "available": resolved is not None,
                    "path": resolved,
                    "vendored": bool(resolved and resolved.startswith(str(smartmontools_repo_dir()))),
                }
            )
            continue
        checks.append(
            {
                "name": label,
                "available": command_exists(label),
                "path": shutil.which(label),
            }
        )

    # Corsair PSU direct telemetry (iCue USB-header → USB-C cable + corsair_psu
    # kernel module). Not a command-existence check — walks /sys/class/hwmon
    # for a device whose ``name`` file reports ``corsairpsu``/``corsair-psu``.
    # Absence is a normal state (non-Corsair PSU or cable unplugged); presence
    # means rail_12v/rail_5v/rail_33v + psu_* columns will populate live.
    corsair_device: Optional[str] = None
    corsair_features: List[str] = []
    try:
        hwmon = Path("/sys/class/hwmon")
        if hwmon.exists():
            for device in hwmon.iterdir():
                name_file = device / "name"
                if not name_file.exists():
                    continue
                try:
                    if name_file.read_text().strip() not in ("corsairpsu", "corsair-psu"):
                        continue
                except Exception:
                    continue
                corsair_device = str(device)
                for tag, fname in (
                    ("Vin",        "in0_input"),
                    ("12V",        "in1_input"),
                    ("5V",         "in2_input"),
                    ("3.3V",       "in3_input"),
                    ("I-per-rail", "curr2_input"),
                    ("Pin",        "power1_input"),
                    ("P-per-rail", "power2_input"),
                    ("Temp",       "temp1_input"),
                    ("Fan",        "fan1_input"),
                ):
                    if (device / fname).exists():
                        corsair_features.append(tag)
                break
    except Exception:
        pass
    checks.append(
        {
            "name": "corsair-psu",
            "available": corsair_device is not None,
            "path": corsair_device,
            "features": corsair_features,
        }
    )

    return {"checks": checks}


def current_gpu_name() -> str:
    if not command_exists("nvidia-smi"):
        return "no-gpu"
    code, stdout, _ = run_capture(["nvidia-smi", "-L"], timeout=10)
    if code != 0 or not stdout:
        return "no-gpu"
    first = stdout.splitlines()[0].strip()
    if ": " in first:
        first = first.split(": ", 1)[1]
    if " (UUID:" in first:
        first = first.split(" (UUID:", 1)[0]
    return first or "no-gpu"


def current_bench_slug() -> str:
    identity = board_identity()
    board = identity.get("name") or "unknown-board"
    gpu = current_gpu_name()
    return slugify("{} {}".format(board, gpu).strip())


def load_bench_baseline(board: str, gpu: str) -> Optional[Dict[str, Any]]:
    """Load a per-bench Gauge R&R baseline for the given board/GPU tuple."""
    path = BASELINES_DIR / "{}.json".format(slugify("{} {}".format(board, gpu).strip()))
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def create_run_dir(serial: str, *, category: Optional[Path] = None) -> Path:
    root = RUNS_DIR / category if category is not None else RUNS_DIR
    root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = root / "{}_{}".format(stamp, slugify(serial))
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: List[sqlite3.Row], columns: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[column] for column in columns])


def get_hardware_snapshot() -> Dict[str, object]:
    """Capture a DUT snapshot via lscpu / lsblk / lspci / nvidia-smi / nvme /
    smartctl. Missing tools get recorded as `available: False` rather than
    breaking the snapshot, so the artifact is always produced."""
    # Hostname redaction — this snapshot flows into verdict.json and the RMA
    # bundle, which leaves the bench. We alias the hostname to a stable
    # pseudonym (`bench-<hash8>`) so a vendor can still tell two runs are on
    # the same bench without learning the internal bench-naming convention.
    # uname is reassembled WITHOUT the node field for the same reason. The
    # internal bench identity stays in the run row's `bench_name` column,
    # which is operator-entered and under the operator's control.
    raw_host = platform.node()
    host_hash = hashlib.sha256(raw_host.encode("utf-8")).hexdigest()[:8]
    u = platform.uname()
    snapshot: Dict[str, object] = {
        "captured_at": utc_now(),
        "hostname": "bench-{}".format(host_hash),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "board": board_identity(),
        # uname tuple = (system, node, release, version, machine, processor);
        # skip index 1 (node) to avoid duplicating the hostname leak.
        "uname": " ".join([u.system, u.release, u.version, u.machine, u.processor]),
        "commands": {},
    }
    commands = {
        "lscpu": ["lscpu"],
        "lsblk": ["lsblk", "-o", "NAME,MODEL,SIZE,TYPE,TRAN,ROTA,MOUNTPOINT"],
        "lspci_storage": ["lspci", "-nn"],
        # `uname -m` instead of `uname -a`: -a would include the hostname
        # (nodename) which we are redacting elsewhere in this snapshot.
        "uname": ["uname", "-m"],
        "nvidia_smi_L": ["nvidia-smi", "-L"],
        "nvme_list": [nvme_cli_binary_path() or "nvme", "list", "-o", "json"],
        "smartctl_scan": [smartctl_binary_path() or "smartctl", "--scan-open"],
    }
    for name, args in commands.items():
        if not shutil.which(args[0]):
            snapshot["commands"][name] = {"available": False}
            continue
        code, stdout, stderr = run_capture(args)
        snapshot["commands"][name] = {
            "available": True,
            "exit_code": code,
            "stdout": stdout,
            "stderr": stderr,
            "command": shell_join(args),
        }
    # Final scrub: any stdout/stderr that contains the raw hostname gets
    # replaced with the redacted alias. Belt-and-suspenders — catches
    # hostname leaks from lscpu / lspci / smartctl output that we don't
    # enumerate here.
    if raw_host:
        alias = "bench-{}".format(host_hash)
        for entry in snapshot["commands"].values():
            if not isinstance(entry, dict):
                continue
            for key in ("stdout", "stderr"):
                val = entry.get(key)
                if isinstance(val, str) and raw_host in val:
                    entry[key] = val.replace(raw_host, alias)
    snapshot["storage_inventory"] = serialize_storage_inventory(discover_storage_devices())
    return snapshot


def print_subsystem_summary(subsystems: Dict[str, Dict[str, object]]) -> None:
    if not subsystems:
        return
    print("Subsystems:")
    for name in ["power", "thermal", "memory", "storage", "pcie"]:
        data = subsystems.get(name, {})
        status = data.get("status", "unknown")
        reasons = data.get("reasons", [])
        detail = reasons[0] if reasons else "none"
        print("  {:<8} {:<5} {}".format(name, status, detail))


def preflight_handler(_: argparse.Namespace) -> int:
    print(json.dumps(get_preflight(), indent=2))
    return 0


def bootstrap_handler(args: argparse.Namespace) -> int:
    auto_install = not args.no_auto_install
    print("{} bootstrap".format(APP_NAME))
    print("  auto-install: {}".format("on" if auto_install else "off"))
    print("  board: {}".format(board_identity()))
    nvcc = nvcc_version_tuple()
    if nvcc:
        print("  nvcc: {}.{}".format(nvcc[0], nvcc[1]))
    else:
        print("  nvcc: not found (PATH missing /usr/local/cuda-12.8/bin?)")
    caps = nvidia_gpu_compute_caps()
    if caps:
        print("  GPU compute caps: {}".format(", ".join(str(cap) for cap in caps)))
    else:
        print("  GPU compute caps: none visible (nvidia-smi missing or failed)")
    print("")

    report = ensure_runtime_dependencies(auto_install=auto_install)

    if report.get("actions"):
        print("Bootstrap actions:")
        for action in report["actions"]:
            print("- {}".format(action))
        print("")

    print("Steps:")
    for step in report.get("steps", []):
        name = step.get("name", "?")
        status = step.get("status", "?")
        detail = step.get("detail", "")
        print("  [{}] {}: {}".format(status, name, detail))
    print("")

    readiness = report.get("telemetry_readiness", {})
    level = str(readiness.get("level", "unknown")).upper()
    print("Telemetry readiness: {}".format(level))
    for message in readiness.get("messages", []):
        print("- {}".format(message))

    gpu_burn = gpu_burn_usable_binary()
    if gpu_burn:
        print("")
        print("gpu_burn usable at: {}".format(gpu_burn))

    if args.json:
        print("")
        print(json.dumps(report, indent=2, default=str))

    if level == "LIMITED":
        print("")
        print("FAIL: readiness is LIMITED — DUT is not ready for a PSU triage run.")
        return 2
    if level == "PARTIAL":
        print("")
        print("WARN: readiness is PARTIAL — run will produce narrower claims (see steps above).")
    return 0
