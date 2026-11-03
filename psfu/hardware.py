"""Bench hardware probing helpers.

Read-only queries against the DUT: board identity, nvidia-smi GPU enumeration,
binary path lookups, PowerLimitGuard, and sensor availability checks. These
functions are Linux-only by design (degrade to no-ops on other platforms) and
do not touch any database or run state.

Build/clone/smoke-test logic lives in ``build_tools.py``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from psfu.common import (
    command_exists,
    is_linux,
    run_capture,
    run_capture_env,
)
from psfu.config import TOOLS_DIR


# ---------------------------------------------------------------------------
# Board + sudo + apt helpers
# ---------------------------------------------------------------------------


def read_first_existing(paths: List[Path]) -> str:
    for path in paths:
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
    return ""


def board_identity() -> Dict[str, str]:
    if not is_linux():
        return {"vendor": "", "name": ""}
    return {
        "vendor": read_first_existing([Path("/sys/class/dmi/id/board_vendor")]),
        "name": read_first_existing([Path("/sys/class/dmi/id/board_name")]),
    }


def is_trx50_ai_top() -> bool:
    identity = board_identity()
    vendor = identity["vendor"].lower()
    name = identity["name"].lower()
    return "gigabyte" in vendor and "trx50 ai top" in name


def sudo_prefix() -> List[str]:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return []
    return ["sudo", "-n"]


def can_run_sudo_noninteractive() -> bool:
    prefix = sudo_prefix()
    if not prefix:
        return True
    code, _, _ = run_capture_env(prefix + ["true"], timeout=10)
    return code == 0


def apt_install(packages: List[str], actions: List[str]) -> bool:
    if not is_linux() or not command_exists("apt-get") or not packages:
        return False
    if not can_run_sudo_noninteractive():
        actions.append("sudo -n unavailable; cannot auto-install {}".format(" ".join(packages)))
        return False
    env = {"DEBIAN_FRONTEND": "noninteractive"}
    code, _, err = run_capture_env(sudo_prefix() + ["apt-get", "update"], timeout=1800, env=env)
    actions.append("apt-get update -> {}".format("ok" if code == 0 else err or code))
    if code != 0:
        return False
    code, _, err = run_capture_env(
        sudo_prefix() + ["apt-get", "install", "-y"] + packages,
        timeout=3600,
        env=env,
    )
    actions.append(
        "apt-get install {} -> {}".format(
            " ".join(packages),
            "ok" if code == 0 else err or code,
        )
    )
    return code == 0


def add_step(report: Dict[str, object], name: str, status: str, detail: str) -> None:
    report.setdefault("steps", []).append(
        {
            "name": name,
            "status": status,
            "detail": detail,
        }
    )


# ---------------------------------------------------------------------------
# NVIDIA helpers
# ---------------------------------------------------------------------------


def nvidia_gpu_compute_caps() -> List[int]:
    """Return compute caps of visible NVIDIA GPUs as packed ints (8.9 -> 89, 12.0 -> 120)."""
    if not command_exists("nvidia-smi"):
        return []
    code, out, _ = run_capture(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        timeout=10,
    )
    if code != 0 or not out:
        return []
    caps: List[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            major, minor = line.split(".", 1)
            caps.append(int(major) * 10 + int(minor.split()[0]))
        except (ValueError, IndexError):
            continue
    return caps


def nvcc_version_tuple() -> Optional[Tuple[int, int]]:
    """Return (major, minor) of nvcc release, or None."""
    if not command_exists("nvcc"):
        return None
    code, out, _ = run_capture(["nvcc", "--version"], timeout=10)
    if code != 0:
        return None
    for token in out.split():
        if token[:1].isdigit() and "." in token:
            try:
                parts = token.rstrip(",").split(".")
                return int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                continue
    for line in out.splitlines():
        if "release" in line.lower():
            idx = line.lower().find("release")
            tail = line[idx + len("release"):].strip()
            try:
                ver = tail.split(",")[0].strip()
                parts = ver.split(".")
                return int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                continue
    return None


def nvcc_supports_compute_cap(cap: int) -> bool:
    """Heuristic: does this nvcc support building for the given compute cap?

    Conservative minimum CUDA toolkit versions per arch:
      sm_80 (A100):       CUDA 11.0
      sm_86 (3090/A40):   CUDA 11.1
      sm_89 (4090/L40):   CUDA 11.8
      sm_90 (H100):       CUDA 12.0
      sm_100 (B200):      CUDA 12.8
      sm_120 (5090/RTX PRO 6000 Blackwell): CUDA 12.8
    """
    ver = nvcc_version_tuple()
    if ver is None:
        return False
    major, minor = ver
    if cap <= 80:
        return (major, minor) >= (11, 0)
    if cap <= 86:
        return (major, minor) >= (11, 1)
    if cap <= 89:
        return (major, minor) >= (11, 8)
    if cap <= 90:
        return (major, minor) >= (12, 0)
    return (major, minor) >= (12, 8)


# ---------------------------------------------------------------------------
# gpu-burn: build + smoke test + per-fleet validation cache
# ---------------------------------------------------------------------------


def gpu_burn_repo_dir() -> Path:
    return TOOLS_DIR / "gpu-burn"


def _ensure_executable(path: Path) -> None:
    """Restore the execute bit (file-transfer tools / cloud-sync clients can strip it)."""
    if path.exists() and not os.access(str(path), os.X_OK):
        try:
            path.chmod(path.stat().st_mode | 0o755)
        except OSError:
            pass


def gpu_burn_binary_for_cap(cap: int) -> Optional[str]:
    """Return path to the per-compute-cap gpu_burn binary if present.

    Binaries are named `gpu_burn-cc<cap>` (e.g., gpu_burn-cc89, gpu_burn-cc120)
    so multiple architectures coexist without overwriting each other when a
    rig's GPU fleet changes.
    """
    local = gpu_burn_repo_dir() / "gpu_burn-cc{}".format(cap)
    if local.exists():
        _ensure_executable(local)
        return str(local)
    return None


def gpu_burn_binary_path() -> Optional[str]:
    """Return path to the gpu_burn binary that matches the current GPU fleet.

    Resolution order:
      1. Per-cap binary `gpu_burn-cc<cap>` matching min(visible compute caps)
      2. Legacy `gpu_burn` (no cap suffix) — back-compat for pre-multi-cap builds
      3. System PATH `gpu_burn`
    Does NOT validate the binary; that's bootstrap's job.
    """
    caps = nvidia_gpu_compute_caps()
    if caps:
        target_cap = min(caps)
        per_cap = gpu_burn_binary_for_cap(target_cap)
        if per_cap:
            return per_cap
    legacy = gpu_burn_repo_dir() / "gpu_burn"
    if legacy.exists():
        _ensure_executable(legacy)
        return str(legacy)
    path = shutil.which("gpu_burn")
    if path:
        return path
    return None


def gpu_burn_usable_binary() -> Optional[str]:
    """Return the gpu_burn path if the binary exists. Validation (smoke test,
    fleet signature) is handled by `bootstrap` — at run time we just check
    the binary is present."""
    return gpu_burn_binary_path()


# ---------------------------------------------------------------------------
# stress-ng: cloned + built into tools/stress-ng/ so the project is portable
# onto any DUT with a C toolchain. Mirrors the gpu-burn pattern (without the
# per-fleet validation cache since stress-ng is pure CPU and arch-agnostic).
# ---------------------------------------------------------------------------


def stress_ng_repo_dir() -> Path:
    return TOOLS_DIR / "stress-ng"


def stress_ng_binary_path() -> Optional[str]:
    """Prefer the vendored build; fall back to system PATH.

    Auto-restores the execute bit if FileZilla stripped it during transfer.
    """
    local = stress_ng_repo_dir() / "stress-ng"
    if local.exists():
        if not os.access(str(local), os.X_OK):
            try:
                local.chmod(local.stat().st_mode | 0o755)
            except OSError:
                pass
        if os.access(str(local), os.X_OK):
            return str(local)
    path = shutil.which("stress-ng")
    if path:
        return path
    return None


def fio_repo_dir() -> Path:
    return TOOLS_DIR / "fio"


def fio_binary_path() -> Optional[str]:
    """Prefer the vendored fio build; fall back to system PATH.

    Vendored source lives at ``tools/fio`` and the built binary is expected at
    ``tools/fio/fio``. If the vendored tree exists but is not built yet, this
    returns the system ``fio`` when available rather than inventing a nested
    path or treating the source tree itself as executable.
    """
    local = fio_repo_dir() / "fio"
    if local.exists():
        if not os.access(str(local), os.X_OK):
            try:
                local.chmod(local.stat().st_mode | 0o755)
            except OSError:
                pass
        if os.access(str(local), os.X_OK):
            return str(local)
    path = shutil.which("fio")
    if path:
        return path
    return None


def nvme_cli_repo_dir() -> Path:
    return TOOLS_DIR / "nvme-cli"


def nvme_cli_binary_path() -> Optional[str]:
    """Prefer vendored nvme build at tools/nvme-cli/builddir/nvme; fall back to system PATH."""
    local = nvme_cli_repo_dir() / "builddir" / "nvme"
    if local.exists() and os.access(str(local), os.X_OK):
        return str(local)
    path = shutil.which("nvme")
    if path:
        return path
    return None


def smartmontools_repo_dir() -> Path:
    return TOOLS_DIR / "smartmontools"


def smartctl_binary_path() -> Optional[str]:
    """Prefer vendored smartctl build at tools/smartmontools/src/smartctl; fall back to system PATH."""
    local = smartmontools_repo_dir() / "src" / "smartctl"
    if local.exists() and os.access(str(local), os.X_OK):
        return str(local)
    path = shutil.which("smartctl")
    if path:
        return path
    return None




# ---------------------------------------------------------------------------
# GPU power limit cap
# ---------------------------------------------------------------------------


def nvidia_gpu_max_power_watts() -> Optional[int]:
    """Highest power-limit ceiling across visible GPUs, as int watts."""
    if not command_exists("nvidia-smi"):
        return None
    code, out, _ = run_capture(
        ["nvidia-smi", "--query-gpu=power.max_limit", "--format=csv,noheader,nounits"],
        timeout=10,
    )
    if code != 0 or not out:
        return None
    caps: List[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            caps.append(int(float(line)))
        except ValueError:
            continue
    if not caps:
        return None
    return max(caps)


def nvidia_gpu_current_power_limits() -> List[Tuple[int, int]]:
    """Return list of (gpu_index, current_power_limit_watts) for each visible GPU."""
    if not command_exists("nvidia-smi"):
        return []
    code, out, _ = run_capture(
        ["nvidia-smi", "--query-gpu=index,power.limit", "--format=csv,noheader,nounits"],
        timeout=10,
    )
    if code != 0 or not out:
        return []
    result: List[Tuple[int, int]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            result.append((int(parts[0]), int(float(parts[1]))))
        except ValueError:
            continue
    return result


def nvidia_gpu_power_limit_ranges() -> Dict[int, Tuple[int, int]]:
    """Return {gpu_index: (min_limit_w, max_limit_w)} from nvidia-smi.

    Consumer cards enforce a floor on ``nvidia-smi -pl`` (RTX 5090 typically
    ~300W, RTX 4090 ~150W). A target below that floor is rejected outright, so
    we query the range up front and clamp rather than let the cap silently fail.
    """
    if not command_exists("nvidia-smi"):
        return {}
    code, out, _ = run_capture(
        ["nvidia-smi", "--query-gpu=index,power.min_limit,power.max_limit", "--format=csv,noheader,nounits"],
        timeout=10,
    )
    if code != 0 or not out:
        return {}
    result: Dict[int, Tuple[int, int]] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            lo = int(float(parts[1]))
            hi = int(float(parts[2]))
            result[idx] = (lo, hi)
        except ValueError:
            continue
    return result


def nvidia_set_power_limit(watts: int, gpu_indices: Optional[List[int]] = None) -> Tuple[bool, str]:
    """Apply nvidia-smi -pl to the given GPUs (or all, if None). Needs root/sudo.
    Returns (ok, detail)."""
    if not command_exists("nvidia-smi"):
        return False, "nvidia-smi missing"
    cmd = sudo_prefix() + ["nvidia-smi", "-pl", str(int(watts))]
    if gpu_indices:
        cmd = sudo_prefix() + ["nvidia-smi", "-i", ",".join(str(i) for i in gpu_indices), "-pl", str(int(watts))]
    code, out, err = run_capture_env(cmd, timeout=30)
    if code != 0:
        return False, (err or out or "rc={}".format(code)).strip()[:200]
    return True, "nvidia-smi -pl {} ok".format(watts)


class PowerLimitGuard:
    """Context manager that applies a capped GPU power limit and restores it on exit.
    No-op when sudo is unavailable, scale == 1.0, or nvidia-smi is missing."""

    def __init__(self, scale: float, actions: List[str]) -> None:
        self.scale = scale
        self.actions = actions
        self.original: List[Tuple[int, int]] = []
        self.applied = False

    def __enter__(self) -> "PowerLimitGuard":
        if self.scale >= 0.999:
            return self
        if not command_exists("nvidia-smi"):
            return self
        if not can_run_sudo_noninteractive():
            self.actions.append(
                "cannot cap GPU power: sudo -n unavailable. Run `sudo -v` before start, "
                "or run `sudo nvidia-smi -pl <watts>` manually before the stress phase."
            )
            return self
        self.original = nvidia_gpu_current_power_limits()
        if not self.original:
            self.actions.append("cannot cap GPU power: nvidia-smi did not report a current power limit")
            return self
        # Consumer GPUs reject `nvidia-smi -pl` below power.min_limit (RTX 5090
        # ~300W, RTX 4090 ~150W). Clamp the requested target to [min_limit, current]
        # per GPU — a silent floor still scales proportionally for higher fractions
        # and caps the lower fractions at whatever the hardware actually allows.
        ranges = nvidia_gpu_power_limit_ranges()
        for idx, current in self.original:
            raw = int(current * self.scale)
            min_limit, _ = ranges.get(idx, (0, 0))
            floor = max(min_limit, 100)
            target = max(floor, raw)
            ok, detail = nvidia_set_power_limit(target, gpu_indices=[idx])
            if ok:
                self.applied = True
                if target > raw:
                    self.actions.append(
                        "GPU {} power limit {} -> {} W (requested {} W, clamped to hw min {} W)".format(
                            idx, current, target, raw, min_limit or floor
                        )
                    )
                else:
                    self.actions.append("GPU {} power limit {} -> {} W ({})".format(idx, current, target, detail))
            else:
                self.actions.append("GPU {} power limit cap FAILED: {}".format(idx, detail))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.applied:
            return
        for idx, original in self.original:
            ok, detail = nvidia_set_power_limit(original, gpu_indices=[idx])
            if ok:
                self.actions.append("GPU {} power limit restored to {} W".format(idx, original))
            else:
                self.actions.append("GPU {} power limit restore FAILED: {}".format(idx, detail))


# ---------------------------------------------------------------------------
# lm-sensors / TRX50 board telemetry stack
# ---------------------------------------------------------------------------


def sensors_have_named_rails() -> bool:
    if not command_exists("sensors"):
        return False
    code, stdout, _ = run_capture(["sensors", "-u"], timeout=10)
    if code != 0:
        return False
    lowered = stdout.lower()
    return "12v" in lowered or "+12v" in lowered or "5v" in lowered or "+5v" in lowered


def ipmi_have_named_rails() -> bool:
    """Return True when ipmitool can supply named 12V/5V rail readings.

    Server boards (dual-socket EPYC, for example) expose voltage rails via the BMC rather
    than a Super I/O chip, so ``sensors_have_named_rails`` returns False even
    though telemetry is fully available. Call sites that report rail readiness
    should check both functions.
    """
    if not command_exists("ipmitool"):
        return False
    code, stdout, _ = run_capture(["ipmitool", "sensor"], timeout=15)
    if code != 0 or not stdout:
        return False
    for line in stdout.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        name  = parts[0].lower()
        units = parts[2].lower()
        if "volt" not in units:
            continue
        status = parts[4].strip().lower() if len(parts) > 4 else ""
        if status in ("ns", "na", "nr", "no reading"):
            continue
        if "12" in name or "5v" in name or name == "p_5v":
            try:
                float(parts[1])
                return True
            except ValueError:
                continue
    return False

