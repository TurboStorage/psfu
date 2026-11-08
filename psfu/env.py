"""Runtime-environment bootstrap and readiness assessment.

Walks through the critical tool chain (stress-ng, gpu-burn, nvidia-smi, sensors)
with optional auto-install, produces a readiness report that the run setup and
the `bootstrap` subcommand both consume, and classifies the result into one of
three readiness levels consumed by the verdict / claim-scope logic.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from psfu.common import (
    TURBOSTAT_SUDO_BIN_CANDIDATES,
    command_exists,
    is_linux,
    turbostat_sudo_path,
    utc_now,
)
from psfu.bootstrap.build_tools import (
    ensure_fio,
    ensure_gpu_burn,
    ensure_nvme_cli,
    ensure_smartmontools,
    ensure_stress_ng,
    ensure_trx50_sensor_stack,
)
from psfu.hardware import (
    add_step,
    board_identity,
    fio_binary_path,
    fio_repo_dir,
    gpu_burn_binary_path,
    gpu_burn_usable_binary,
    is_trx50_ai_top,
    nvme_cli_binary_path,
    nvme_cli_repo_dir,
    ipmi_have_named_rails,
    sensors_have_named_rails,
    smartctl_binary_path,
    smartmontools_repo_dir,
    stress_ng_binary_path,
    stress_ng_repo_dir,
)


def _find_corsair_psu_hwmon() -> Dict[str, object]:
    """Look for a corsair-psu hwmon device and summarise what it exposes.

    Returns a dict with ``present`` (bool), ``device`` (path string or None),
    ``model`` (string pulled from ``/sys/.../name`` or None), and a list of
    ``features`` seen (voltages/currents/powers/temps/fan). Used for the
    preflight step so the operator knows before the run whether the iCue cable is
    seated and ``corsair_psu`` is loaded.
    """
    info: Dict[str, object] = {
        "present": False,
        "device": None,
        "model": None,
        "features": [],
    }
    hwmon = Path("/sys/class/hwmon")
    if not hwmon.exists():
        return info
    try:
        for device in hwmon.iterdir():
            name_file = device / "name"
            if not name_file.exists():
                continue
            try:
                name = name_file.read_text().strip()
            except Exception:
                continue
            if name not in ("corsairpsu", "corsair-psu"):
                continue
            info["present"] = True
            info["device"] = str(device)
            info["model"] = name
            features: List[str] = []
            if (device / "in0_input").exists(): features.append("Vin")
            if (device / "in1_input").exists(): features.append("12V")
            if (device / "in2_input").exists(): features.append("5V")
            if (device / "in3_input").exists(): features.append("3.3V")
            if (device / "curr2_input").exists(): features.append("I-per-rail")
            if (device / "power1_input").exists(): features.append("Pin")
            if (device / "power2_input").exists(): features.append("P-per-rail")
            if (device / "temp1_input").exists(): features.append("Temp")
            if (device / "fan1_input").exists(): features.append("Fan")
            info["features"] = features
            break
    except Exception:
        pass
    return info


def telemetry_readiness(report: Dict[str, object]) -> Dict[str, object]:
    steps = {step["name"]: step for step in report.get("steps", [])}
    cpu_ready = steps.get("stress-ng", {}).get("status") == "ready"
    gpu_load_status = steps.get("gpu-burn", {}).get("status", "missing")
    gpu_load_ready = gpu_load_status == "ready"
    gpu_telem_ready = steps.get("nvidia-smi", {}).get("status") == "ready"
    board_status = steps.get("board-telemetry", {}).get("status", "missing")

    if cpu_ready and gpu_load_ready and gpu_telem_ready and board_status == "ready":
        level = "ready"
    elif cpu_ready and (gpu_load_ready or gpu_telem_ready or board_status in ("ready", "partial")):
        level = "partial"
    else:
        level = "limited"

    notes = [
        "CPU stress {}".format("ready" if cpu_ready else "missing"),
        "GPU load {}".format(gpu_load_status),
        "GPU telemetry {}".format("ready" if gpu_telem_ready else "missing"),
        "board telemetry {}".format(board_status),
    ]
    return {"level": level, "messages": notes}


def _dep_progress(label: str) -> float:
    """Print a dependency-check sub-step label; return monotonic time for elapsed calc."""
    sys.stdout.write("\r\033[K[prep]   {} {}...".format("\u2502", label))
    sys.stdout.flush()
    return time.monotonic()


def _dep_done(t0: float) -> None:
    elapsed = time.monotonic() - t0
    if elapsed >= 0.5:
        sys.stdout.write(" ({:.1f}s)".format(elapsed))
    sys.stdout.write("\n")
    sys.stdout.flush()


_SUDOERS_FILE = Path("/etc/sudoers.d/psfu-turbostat")
# Dot-prefixed sibling in the same directory: sudo's #includedir skips files
# whose names contain a dot, so this staging path is never parsed as a live
# rule even between write and mv. Same filesystem → mv is atomic rename.
_SUDOERS_STAGING = Path("/etc/sudoers.d/.psfu-turbostat.staging")
_MSR_MODULES_FILE = Path("/etc/modules-load.d/msr.conf")

# Absolute paths baked into the sudoers NOPASSWD rule. Hardcoded, not
# PATH-resolved: an attacker-influenced PATH that flips these to a
# different binary would turn the sudoers rule into a privilege escalation.
# If no candidate exists on disk, we refuse to write the rule.
# turbostat: see TURBOSTAT_SUDO_BIN_CANDIDATES in psfu.common (secure_path
# order, so a hand-built /usr/local/bin/turbostat is the one named). modprobe
# lives in /usr/sbin on modern distros and /sbin on legacy layouts.
_MODPROBE_SUDO_BIN_CANDIDATES = ("/usr/sbin/modprobe", "/sbin/modprobe")


def _msr_loaded() -> bool:
    try:
        with open("/proc/modules") as f:
            return any(line.startswith("msr ") for line in f)
    except OSError:
        return False


def _turbostat_probe() -> bool:
    """Return True if turbostat can actually run right now (sudo + msr)."""
    path = turbostat_sudo_path()
    if not path:
        return False
    try:
        r = subprocess.run(
            ["sudo", "-n", path, "--show", "PkgWatt", "sleep", "0.1"],
            capture_output=True, timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


def ensure_turbostat_sudo(actions: List[object]) -> str:
    """Make turbostat work non-interactively on any fresh system.

    Does a real probe every time — no file-exists shortcuts.  Fixes whatever
    is broken: missing msr module, missing sudoers rule, missing persistence.
    Prompts for password at most once (first run on a fresh box).

    Returns: "ready", "installed", or "unavailable".
    """
    if not is_linux():
        return "unavailable"

    # turbostat_path and modprobe_path are HARDCODED absolute paths that will
    # appear in the sudoers NOPASSWD rule. We deliberately do not fall back to
    # shutil.which() here — PATH is attacker-influenceable on a shared bench.
    turbostat_path = turbostat_sudo_path()
    modprobe_path = next(
        (p for p in _MODPROBE_SUDO_BIN_CANDIDATES if os.path.exists(p)),
        "",
    )
    if not turbostat_path:
        actions.append(
            "turbostat not found as a root-owned file at {} — CPU MHz/Busy%/PkgWatt "
            "unavailable (refusing to write sudoers rule against a PATH-resolved "
            "or user-writable path)".format(
                " or ".join(TURBOSTAT_SUDO_BIN_CANDIDATES)
            )
        )
        return "unavailable"

    needed_setup = False

    # ── 1. msr kernel module ─────────────────────────────────────────────
    if not _msr_loaded():
        # Try loading without password first (might already have sudo rule).
        subprocess.run(
            ["sudo", "-n", "modprobe", "msr"],
            capture_output=True, timeout=5,
        )
        if not _msr_loaded():
            needed_setup = True

    # ── 2. sudoers rule ──────────────────────────────────────────────────
    has_sudoers = _SUDOERS_FILE.exists()

    # ── 3. real probe — does turbostat actually produce output? ───────────
    if _msr_loaded() and (has_sudoers or _turbostat_probe()):
        if _turbostat_probe():
            # Persist msr if not already.
            if not _MSR_MODULES_FILE.exists():
                try:
                    subprocess.run(
                        ["sudo", "-n", "tee", str(_MSR_MODULES_FILE)],
                        input=b"msr\n", stdout=subprocess.DEVNULL, timeout=5,
                    )
                except Exception:
                    pass
            return "ready"
        # sudoers file exists but probe fails — stale or incomplete, redo it.
        needed_setup = True

    needed_setup = True  # If we got here, something needs fixing.

    # ── 4. interactive setup (one password prompt) ───────────────────────
    user = os.environ.get("USER", os.environ.get("LOGNAME", ""))
    if not user:
        actions.append("turbostat: cannot determine current user for sudoers rule")
        return "unavailable"
    # Reject anything that could produce malformed or privilege-escalating
    # sudoers syntax. POSIX usernames are [a-zA-Z0-9._-]; anything else
    # (newline, space, comma, '=') means the env var was tampered with.
    if not re.match(r"^[a-zA-Z0-9._-]+$", user):
        actions.append("turbostat: refusing sudoers rule for unsafe username {!r}".format(user))
        return "unavailable"

    # Refuse to write the rule if the modprobe absolute path is missing —
    # we would otherwise embed an empty string or a PATH-resolved location.
    if not modprobe_path:
        actions.append(
            "turbostat: modprobe not found at {} — refusing to write "
            "sudoers rule with non-absolute path".format(
                " or ".join(_MODPROBE_SUDO_BIN_CANDIDATES)
            )
        )
        return "unavailable"

    # Exact argument forms only (see collectors/turbostat.py and
    # _turbostat_probe). A bare path would let `sudo turbostat <anything>` run
    # as root, including turbostat's trailing "command" form. Commas in
    # sudoers arguments must be escaped.
    ts_args = [
        "--show {} -i 1.0".format(cols.replace(",", "\\,"))
        for cols in (
            "Core,CPU,Avg_MHz,Busy%,CorWatt,PkgWatt,CPU%c6",
            "Core,CPU,Avg_MHz,Busy%,CorWatt,PkgWatt",
        )
    ] + ["--show PkgWatt sleep 0.1"]
    rule = "{user} ALL=(ALL) NOPASSWD: {ts}, {modprobe} msr".format(
        user=user,
        ts=", ".join("{} {}".format(turbostat_path, a) for a in ts_args),
        modprobe=modprobe_path,
    )
    print()
    print("=" * 60)
    print("  TURBOSTAT SETUP (one-time, survives reboots)")
    print()
    print("  turbostat reads CPU MSRs → MHz, Busy%, PkgWatt.")
    print("  PS-FU needs two passwordless sudo rules:")
    print("    1. turbostat  (read MSRs)")
    print("    2. modprobe msr  (load kernel module)")
    print()
    print("  Rule: {}".format(rule))
    print("  File: {}".format(_SUDOERS_FILE))
    print()
    print("  Enter your password once. Never prompted again.")
    print("=" * 60)
    sys.stdout.flush()

    try:
        # Cache sudo credentials interactively.
        auth = subprocess.run(["sudo", "-v"], timeout=60)
        if auth.returncode != 0:
            actions.append("turbostat: sudo authentication failed")
            return "unavailable"

        # Load msr module now that we have sudo.
        if not _msr_loaded():
            subprocess.run(["sudo", "modprobe", "msr"], capture_output=True, timeout=5)

        # Persist msr across reboots.
        if not _MSR_MODULES_FILE.exists():
            subprocess.run(
                ["sudo", "tee", str(_MSR_MODULES_FILE)],
                input=b"msr\n", stdout=subprocess.DEVNULL, timeout=5,
            )

        # Write sudoers rule to a DOT-PREFIXED staging path in the same
        # directory. sudo's #includedir parser skips files with a dot in the
        # name, so an unvalidated rule is never live. Same-filesystem mv
        # (below) is an atomic rename. Do NOT write directly to _SUDOERS_FILE
        # — the rule is active the instant tee returns, and a bad syntax
        # there would lock sudo entirely before visudo -cf could catch it.
        result = subprocess.run(
            ["sudo", "tee", str(_SUDOERS_STAGING)],
            input="{}\n".format(rule).encode(),
            stdout=subprocess.DEVNULL, timeout=10,
        )
        if result.returncode != 0:
            actions.append("turbostat: sudo tee failed (rc={})".format(result.returncode))
            return "unavailable"

        # sudoers files must be mode 0440 to be accepted at runtime. Set this
        # before validation so visudo sees the final state.
        subprocess.run(
            ["sudo", "chmod", "0440", str(_SUDOERS_STAGING)],
            capture_output=True, timeout=5,
        )

        # Validate the staging file. Bad syntax never reached an active path.
        check = subprocess.run(
            ["sudo", "visudo", "-cf", str(_SUDOERS_STAGING)],
            capture_output=True, timeout=10,
        )
        if check.returncode != 0:
            subprocess.run(
                ["sudo", "rm", "-f", str(_SUDOERS_STAGING)],
                capture_output=True, timeout=5,
            )
            actions.append("turbostat sudoers rule failed validation — staging removed, live file untouched")
            return "unavailable"

        # Atomic promote. mv within the same filesystem is a rename(2) call,
        # so sudo sees either the old file or the new one — never partial.
        promote = subprocess.run(
            ["sudo", "mv", str(_SUDOERS_STAGING), str(_SUDOERS_FILE)],
            capture_output=True, timeout=5,
        )
        if promote.returncode != 0:
            subprocess.run(
                ["sudo", "rm", "-f", str(_SUDOERS_STAGING)],
                capture_output=True, timeout=5,
            )
            actions.append("turbostat: sudoers promote failed (rc={})".format(promote.returncode))
            return "unavailable"

        # Final verification — actually run turbostat end-to-end.
        if _turbostat_probe():
            actions.append("turbostat configured (sudoers + msr module + persistence)")
            return "installed"
        else:
            actions.append("turbostat: setup completed but probe still fails")
            return "unavailable"

    except subprocess.TimeoutExpired:
        actions.append("turbostat: sudo prompt timed out")
        return "unavailable"
    except Exception as exc:
        actions.append("turbostat: setup failed ({})".format(exc))
        return "unavailable"


def _pip_install(package: str, actions: List[object]) -> bool:
    """Install a Python package via pip. Returns True on success."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", package],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0:
            actions.append("pip installed {}".format(package))
            return True
        actions.append("pip install {} failed (rc={})".format(package, result.returncode))
        return False
    except Exception as exc:
        actions.append("pip install {} failed ({})".format(package, exc))
        return False


def _apt_install(package: str, actions: List[object]) -> bool:
    """Install a system package via apt. Requires sudo. Returns True on success."""
    # Check if sudo -n works (passwordless) before attempting.
    try:
        check = subprocess.run(
            ["sudo", "-n", "true"], capture_output=True, timeout=5,
        )
        if check.returncode != 0:
            actions.append("skipped apt install {} (no passwordless sudo)".format(package))
            return False
        result = subprocess.run(
            ["sudo", "-n", "apt-get", "install", "-y", "-qq", package],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0:
            actions.append("apt installed {}".format(package))
            return True
        actions.append("apt install {} failed (rc={})".format(package, result.returncode))
        return False
    except Exception as exc:
        actions.append("apt install {} failed ({})".format(package, exc))
        return False


def _ensure_pynvml(actions: List[object]) -> bool:
    """Install pynvml if not importable. Returns True if available after attempt."""
    try:
        import pynvml as _  # noqa: F401
        return True
    except ImportError:
        pass
    # Pinned version: typosquat defense on unpinned pip installs. Bump
    # deliberately when NVML adds features we need; validate the pin still
    # installs cleanly against the PS-FU import path before landing the bump.
    if _pip_install("pynvml==11.5.3", actions):
        # Reload the flag in the telemetry module so this session picks it up.
        try:
            import pynvml as _  # noqa: F401, F811
            # Patch the module-level flag in telemetry.py for this process.
            from psfu.collectors import telemetry
            telemetry._HAS_PYNVML = True
            return True
        except ImportError:
            return False
    return False


def ensure_runtime_dependencies(auto_install: bool) -> Dict[str, object]:
    report: Dict[str, object] = {
        "timestamp": utc_now(),
        "platform": sys.platform,
        "board": board_identity(),
        "actions": [],
        "steps": [],
    }
    actions = report["actions"]

    if not is_linux():
        actions.append("non-Linux environment; auto-install skipped")
        add_step(report, "platform", "unsupported", "Non-Linux environment detected.")
        report["telemetry_readiness"] = telemetry_readiness(report)
        return report

    if not auto_install:
        actions.append("auto-install disabled")
        add_step(report, "bootstrap", "skipped", "Auto-install disabled by operator.")

        t0 = _dep_progress("stress-ng")
        stress_path = stress_ng_binary_path()
        if not stress_path:
            # Self-heal: vendored build is local and doesn't need apt sudo for
            # the typical case (toolchain present). Try to build before
            # reporting missing.
            actions.append("stress-ng missing -- attempting auto-build")
            ensure_stress_ng(actions, build=True)
            stress_path = stress_ng_binary_path()
        if stress_path:
            origin = "vendored" if stress_path.startswith(str(stress_ng_repo_dir())) else "system"
            add_step(
                report,
                "stress-ng",
                "ready",
                "CPU stress tool available ({}: {}).".format(origin, stress_path),
            )
        else:
            add_step(report, "stress-ng", "missing", "CPU stress tool not found (auto-build failed -- run `psfu.py bootstrap` for full diagnostics).")
        _dep_done(t0)

        t0 = _dep_progress("gpu-burn")
        gpu_burn_path = gpu_burn_binary_path()
        # Self-heal: missing binary OR present-but-not-validated both warrant a
        # rebuild attempt. The vendored build is local; cap detection + smoke
        # test inside ensure_gpu_burn handle fleet changes (e.g. moving a checkout between an sm_89 dev rig and an sm_120 DUT).
        if not gpu_burn_path or not gpu_burn_usable_binary():
            reason = "missing" if not gpu_burn_path else "not validated on current GPU fleet"
            actions.append("gpu-burn {} -- attempting auto-build".format(reason))
            ensure_gpu_burn(actions, build=True)
            gpu_burn_path = gpu_burn_binary_path()
        if gpu_burn_path and gpu_burn_usable_binary():
            add_step(report, "gpu-burn", "ready", "gpu-burn validated at {}.".format(gpu_burn_path))
        elif gpu_burn_path:
            add_step(
                report,
                "gpu-burn",
                "broken",
                "gpu-burn present at {} but smoke test failed after auto-build. See bootstrap actions for details.".format(gpu_burn_path),
            )
        else:
            add_step(report, "gpu-burn", "missing", "gpu-burn not available after auto-build (run `psfu.py bootstrap` for full diagnostics).")
        _dep_done(t0)

        t0 = _dep_progress("fio")
        fio_path = fio_binary_path()
        if not fio_path:
            actions.append("fio missing -- attempting vendored build")
            ensure_fio(actions, build=True)
            fio_path = fio_binary_path()
        if fio_path:
            origin = "vendored" if fio_path.startswith(str(fio_repo_dir())) else "system"
            add_step(report, "fio", "ready", "fio available ({}: {}).".format(origin, fio_path))
        else:
            add_step(report, "fio", "missing", "fio not available after vendored build attempt.")
        _dep_done(t0)

        t0 = _dep_progress("nvme-cli")
        nvme_path = nvme_cli_binary_path()
        if not nvme_path:
            actions.append("nvme-cli missing -- attempting vendored build")
            ensure_nvme_cli(actions, build=True)
            nvme_path = nvme_cli_binary_path()
        if nvme_path:
            origin = "vendored" if nvme_path.startswith(str(nvme_cli_repo_dir())) else "system"
            add_step(report, "nvme-cli", "ready", "nvme-cli available ({}: {}).".format(origin, nvme_path))
        else:
            add_step(report, "nvme-cli", "missing", "nvme-cli not available after vendored build attempt.")
        _dep_done(t0)

        t0 = _dep_progress("smartmontools")
        smartctl_path = smartctl_binary_path()
        if not smartctl_path:
            actions.append("smartmontools missing -- attempting vendored build")
            ensure_smartmontools(actions, build=True)
            smartctl_path = smartctl_binary_path()
        if smartctl_path:
            origin = "vendored" if smartctl_path.startswith(str(smartmontools_repo_dir())) else "system"
            add_step(report, "smartmontools", "ready", "smartctl available ({}: {}).".format(origin, smartctl_path))
        else:
            add_step(report, "smartmontools", "missing", "smartctl not available after vendored build attempt.")
        _dep_done(t0)

        t0 = _dep_progress("nvidia-smi")
        if command_exists("nvidia-smi"):
            add_step(report, "nvidia-smi", "ready", "NVIDIA telemetry tool available.")
        else:
            add_step(report, "nvidia-smi", "missing", "NVIDIA telemetry tool not found.")
        _dep_done(t0)

        t0 = _dep_progress("board telemetry")
        if command_exists("sensors"):
            if sensors_have_named_rails():
                add_step(report, "board-telemetry", "ready", "Named rails visible via sensors.")
            elif ipmi_have_named_rails():
                add_step(report, "board-telemetry", "ready", "Named rails available via IPMI (server board).")
            else:
                add_step(report, "board-telemetry", "partial", "sensors available, but named 12V/5V rails not found.")
        else:
            add_step(report, "board-telemetry", "missing", "sensors not found.")
        _dep_done(t0)

        # Corsair PSU direct telemetry (iCue USB-header → USB-C cable + corsair_psu
        # kernel driver). When present, gives per-rail V/I/P, input voltage,
        # efficiency, two temps, fan RPM — the unit reports on itself during
        # the test. Absence is normal (non-Corsair PSU, cable unplugged, module
        # not loaded); absence degrades to board sensors + manual multimeter.
        t0 = _dep_progress("corsair-psu")
        corsair_info = _find_corsair_psu_hwmon()
        if corsair_info["present"]:
            detail = "Corsair PSU hwmon present at {} (features: {}).".format(
                corsair_info["device"],
                ", ".join(corsair_info["features"]) or "none",
            )
            add_step(report, "corsair-psu", "ready", detail)
        else:
            add_step(
                report,
                "corsair-psu",
                "missing",
                "No corsair-psu hwmon device — iCue cable unplugged or corsair_psu kernel module not loaded.",
            )
        _dep_done(t0)

        # pynvml
        t0 = _dep_progress("pynvml")
        try:
            import pynvml as _  # noqa: F401
            add_step(report, "pynvml", "ready", "pynvml available (VRAM temp, ECC, PCIe throughput, throttle reasons).")
        except ImportError:
            add_step(report, "pynvml", "missing", "pynvml not found (run bootstrap to auto-install).")
        _dep_done(t0)

        # Memory validation tools
        t0 = _dep_progress("memtester")
        if command_exists("memtester"):
            add_step(report, "memtester", "ready", "memtester available.")
        else:
            add_step(report, "memtester", "missing", "memtester not found (run bootstrap to auto-install).")
        _dep_done(t0)

        t0 = _dep_progress("stressapptest")
        if command_exists("stressapptest"):
            add_step(report, "stressapptest", "ready", "stressapptest available.")
        else:
            add_step(report, "stressapptest", "missing", "stressapptest not found (run bootstrap to auto-install).")
        _dep_done(t0)

        t0 = _dep_progress("EDAC")
        from psfu.collectors.edac import snapshot_edac
        edac_snap = snapshot_edac()
        if edac_snap["available"]:
            add_step(report, "edac", "ready", "EDAC sysfs available.")
        else:
            add_step(report, "edac", "missing", "EDAC sysfs not available.")
        _dep_done(t0)

        t0 = _dep_progress("turbostat")
        ts_status = ensure_turbostat_sudo(actions)
        if ts_status in ("ready", "installed"):
            add_step(report, "turbostat", "ready", "turbostat available with passwordless sudo.")
        else:
            add_step(report, "turbostat", "missing", "turbostat unavailable — CPU MHz/Busy%/PkgWatt will be blank.")
        _dep_done(t0)

        report["telemetry_readiness"] = telemetry_readiness(report)
        return report

    t0 = _dep_progress("stress-ng (ensure + build)")
    ensure_stress_ng(actions, build=auto_install)
    stress_path = stress_ng_binary_path()
    if stress_path:
        origin = "vendored" if stress_path.startswith(str(stress_ng_repo_dir())) else "system"
        add_step(report, "stress-ng", "ready", "CPU stress tool available ({}: {}).".format(origin, stress_path))
    else:
        add_step(report, "stress-ng", "missing", "stress-ng not available after vendored build attempt. See bootstrap actions above.")
    _dep_done(t0)

    t0 = _dep_progress("gpu-burn (ensure + smoke test)")
    ensure_gpu_burn(actions, build=auto_install)
    gpu_burn_path = gpu_burn_binary_path()
    if gpu_burn_path and gpu_burn_usable_binary():
        add_step(report, "gpu-burn", "ready", "gpu-burn validated at {}.".format(gpu_burn_path))
    elif gpu_burn_path:
        add_step(
            report,
            "gpu-burn",
            "broken",
            "gpu-burn built at {} but smoke test did not confirm GPU load. See bootstrap actions for details.".format(gpu_burn_path),
        )
    else:
        add_step(report, "gpu-burn", "missing", "gpu-burn not available after bootstrap.")
    _dep_done(t0)

    t0 = _dep_progress("fio (ensure + build)")
    ensure_fio(actions, build=auto_install)
    fio_path = fio_binary_path()
    if fio_path:
        origin = "vendored" if fio_path.startswith(str(fio_repo_dir())) else "system"
        add_step(report, "fio", "ready", "fio available ({}: {}).".format(origin, fio_path))
    else:
        add_step(report, "fio", "missing", "fio not available after vendored build attempt. See bootstrap actions above.")
    _dep_done(t0)

    t0 = _dep_progress("nvme-cli (ensure + build)")
    ensure_nvme_cli(actions, build=auto_install)
    nvme_path = nvme_cli_binary_path()
    if nvme_path:
        origin = "vendored" if nvme_path.startswith(str(nvme_cli_repo_dir())) else "system"
        add_step(report, "nvme-cli", "ready", "nvme-cli available ({}: {}).".format(origin, nvme_path))
    else:
        add_step(report, "nvme-cli", "missing", "nvme-cli not available after vendored build attempt. See bootstrap actions above.")
    _dep_done(t0)

    t0 = _dep_progress("smartmontools (ensure + build)")
    ensure_smartmontools(actions, build=auto_install)
    smartctl_path = smartctl_binary_path()
    if smartctl_path:
        origin = "vendored" if smartctl_path.startswith(str(smartmontools_repo_dir())) else "system"
        add_step(report, "smartmontools", "ready", "smartctl available ({}: {}).".format(origin, smartctl_path))
    else:
        add_step(report, "smartmontools", "missing", "smartctl not available after vendored build attempt. See bootstrap actions above.")
    _dep_done(t0)

    t0 = _dep_progress("sensor stack")
    ensure_trx50_sensor_stack(actions)
    if command_exists("sensors"):
        if sensors_have_named_rails():
            detail = "Named rails visible via sensors."
            if is_trx50_ai_top():
                detail = "TRX50 AI TOP rails visible via sensors."
            add_step(report, "board-telemetry", "ready", detail)
        elif ipmi_have_named_rails():
            add_step(report, "board-telemetry", "ready", "Named rails available via IPMI (server board).")
        else:
            if is_trx50_ai_top():
                add_step(
                    report,
                    "board-telemetry",
                    "unavailable-confirmed",
                    "TRX50 AI TOP: named 12V/5V rails not exposed after trying it87 fork and nct6687d. CPU/NVMe temps still available; run is software-triage-only for rail claims.",
                )
            else:
                add_step(report, "board-telemetry", "partial", "sensors available, but named 12V/5V rails not found.")
    else:
        add_step(report, "board-telemetry", "missing", "sensors not available after bootstrap.")
    _dep_done(t0)

    # Corsair PSU direct telemetry — see matching block in the fast path above.
    t0 = _dep_progress("corsair-psu")
    corsair_info = _find_corsair_psu_hwmon()
    if corsair_info["present"]:
        detail = "Corsair PSU hwmon present at {} (features: {}).".format(
            corsair_info["device"],
            ", ".join(corsair_info["features"]) or "none",
        )
        add_step(report, "corsair-psu", "ready", detail)
    else:
        add_step(
            report,
            "corsair-psu",
            "missing",
            "No corsair-psu hwmon device — iCue cable unplugged or corsair_psu kernel module not loaded.",
        )
    _dep_done(t0)

    t0 = _dep_progress("nvidia-smi")
    if command_exists("nvidia-smi"):
        add_step(report, "nvidia-smi", "ready", "NVIDIA telemetry tool available.")
    else:
        add_step(report, "nvidia-smi", "missing", "NVIDIA telemetry tool not found.")
    _dep_done(t0)

    # pynvml — GPU VRAM temp, ECC, PCIe throughput, throttle reasons
    t0 = _dep_progress("pynvml")
    pynvml_ok = _ensure_pynvml(actions)
    if pynvml_ok:
        add_step(report, "pynvml", "ready", "pynvml available (VRAM temp, ECC, PCIe throughput, throttle reasons).")
    else:
        add_step(report, "pynvml", "missing", "pynvml not available after pip install attempt.")
    _dep_done(t0)

    # Memory validation tools — auto-install via apt if sudo available
    t0 = _dep_progress("memtester")
    if not command_exists("memtester"):
        _apt_install("memtester", actions)
    if command_exists("memtester"):
        add_step(report, "memtester", "ready", "memtester available (pattern-based DRAM fault testing).")
    else:
        add_step(report, "memtester", "missing", "memtester not found (no sudo or apt install failed).")
    _dep_done(t0)

    t0 = _dep_progress("stressapptest")
    if not command_exists("stressapptest"):
        _apt_install("stressapptest", actions)
    if command_exists("stressapptest"):
        add_step(report, "stressapptest", "ready", "stressapptest available (Google concurrent memory stress).")
    else:
        add_step(report, "stressapptest", "missing", "stressapptest not found (no sudo or apt install failed).")
    _dep_done(t0)

    # EDAC (ECC error counters) — passive, kernel-provided
    t0 = _dep_progress("EDAC")
    from psfu.collectors.edac import snapshot_edac
    edac_snap = snapshot_edac()
    if edac_snap["available"]:
        add_step(report, "edac", "ready", "EDAC sysfs available ({} memory controller(s)).".format(
            len(edac_snap["controllers"])))
    else:
        add_step(report, "edac", "missing", "EDAC sysfs not available (no ECC or no kernel driver).")
    _dep_done(t0)

    # turbostat — needs passwordless sudo for MSR access (CPU MHz, Busy%, PkgWatt)
    t0 = _dep_progress("turbostat")
    ts_status = ensure_turbostat_sudo(actions)
    if ts_status in ("ready", "installed"):
        add_step(report, "turbostat", "ready", "turbostat available with passwordless sudo.")
    else:
        add_step(report, "turbostat", "missing", "turbostat unavailable — CPU MHz/Busy%/PkgWatt will be blank.")
    _dep_done(t0)

    add_step(report, "bootstrap", "enabled", "Auto-install/bootstrap enabled for this run.")
    report["telemetry_readiness"] = telemetry_readiness(report)
    return report
