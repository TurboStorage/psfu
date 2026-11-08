"""Build, clone, and smoke-test vendored tools (gpu-burn, stress-ng).

Also handles board telemetry kernel module installation (it87 fork,
nct6687d) for TRX50 AI TOP and similar boards.

These functions are only called from ``bootstrap`` mode
(``env.py::ensure_runtime_dependencies`` with ``auto_install=True``).
Run-time code should use the lightweight check functions in
``hardware.py`` (``gpu_burn_binary_path``, ``stress_ng_binary_path``,
``gpu_burn_usable_binary``) which return instantly.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from psfu.common import (
    command_exists,
    is_linux,
    run_capture,
    run_capture_env,
    utc_now,
)
from psfu.config import TOOLS_DIR
from psfu.hardware import (
    apt_install,
    can_run_sudo_noninteractive,
    fio_binary_path,
    fio_repo_dir,
    gpu_burn_binary_for_cap,
    gpu_burn_binary_path,
    gpu_burn_repo_dir,
    is_trx50_ai_top,
    nvme_cli_binary_path,
    nvme_cli_repo_dir,
    nvidia_gpu_compute_caps,
    nvcc_supports_compute_cap,
    nvcc_version_tuple,
    ipmi_have_named_rails,
    sensors_have_named_rails,
    smartctl_binary_path,
    smartmontools_repo_dir,
    stress_ng_binary_path,
    stress_ng_repo_dir,
    sudo_prefix,
)


# ---------------------------------------------------------------------------
# gpu-burn build/smoke-test constants
# ---------------------------------------------------------------------------

GPU_BURN_SMOKE_DURATION_SEC = 5
GPU_BURN_SMOKE_TIMEOUT_SEC = 60
GPU_BURN_FAIL_MARKERS = (
    "couldn't find compare kernel",
    "no cuda-capable device",
    "named symbol not found",
    "cuda error",
    "cublas error",
    "compute_",  # "unsupported compute_NN" messages
)

# stress-ng vendoring
STRESS_NG_REPO_URL = "https://github.com/ColinIanKing/stress-ng.git"
STRESS_NG_SMOKE_TIMEOUT_SEC = 30
FIO_SMOKE_TIMEOUT_SEC = 15
NVME_CLI_SMOKE_TIMEOUT_SEC = 10
SMARTMONTOOLS_SMOKE_TIMEOUT_SEC = 10

# ─── Pinned upstream refs ────────────────────────────────────────────────
# Supply-chain defense: never build off a moving HEAD. Each tool pins to
# either a release tag (preferred when upstream tags) or a commit SHA.
# After clone, the resolved HEAD SHA is compared to PINNED_SHA and the
# build aborts on mismatch. When upstream publishes a new vetted release:
#   1. update the pin here
#   2. delete tools/<name> so the next bootstrap re-clones at the new pin
#   3. re-run the smoke suite
#
# Leave ``None`` to allow tracking master with a loud warning (first-run
# bootstrap path for a fresh DUT — fill in the resolved SHA afterwards).
GPU_BURN_PINNED_REF: Optional[str] = None     # set to commit SHA
GPU_BURN_PINNED_SHA: Optional[str] = None     # set to same SHA for verify
STRESS_NG_PINNED_REF: Optional[str] = None    # set to release tag e.g. "V0.17.08"
STRESS_NG_PINNED_SHA: Optional[str] = None    # resolved SHA of that tag
# Kernel-module repos are built and installed with root; both are pinned.
# Pins are the upstream HEAD as of 2026-11-07 (taken with git ls-remote, not
# independently reviewed). Bump together with a code review of the diff.
IT87_PINNED_SHA = "bd7a566c8a60fe0308d67e26761e16260368191a"
NCT6687D_PINNED_SHA = "49ac9bced1771e73b9d5e3d50d5167167548efd4"


def _verify_pinned_sha(repo_dir: "Path", expected_sha: Optional[str], tool: str, actions: List[object]) -> bool:
    """Resolve HEAD and compare to the pinned SHA. Returns True if matches
    or no pin is configured (logs a warning in that case). False aborts build."""
    if expected_sha is None:
        actions.append(
            "WARNING: {} pin not set — building against HEAD of master. "
            "After this run succeeds, record the resolved SHA in "
            "build_tools.py to prevent future drift.".format(tool)
        )
        return True
    code, stdout, err = run_capture_env(
        ["git", "rev-parse", "HEAD"], timeout=10, cwd=repo_dir,
    )
    if code != 0:
        actions.append("{}: git rev-parse HEAD failed ({})".format(tool, err or code))
        return False
    actual = stdout.strip()
    if actual != expected_sha:
        actions.append(
            "{}: PINNED SHA MISMATCH — expected {}, got {}. Upstream may have "
            "been compromised, or the pin is stale. Build aborted.".format(
                tool, expected_sha, actual
            )
        )
        return False
    actions.append("{}: verified pinned SHA {}".format(tool, actual[:12]))
    return True


def _clone_pinned(url: str, repo_dir: "Path", sha: str, tool: str, actions: List[object]) -> bool:
    """Clone (if missing) and check out the pinned commit; verify HEAD."""
    if not repo_dir.exists():
        code, _, err = run_capture_env(["git", "clone", url, str(repo_dir)], timeout=1800)
        actions.append("git clone {} -> {}".format(tool, "ok" if code == 0 else err or code))
        if code != 0:
            return False
        code, _, err = run_capture_env(["git", "checkout", "--detach", sha], timeout=60, cwd=repo_dir)
        if code != 0:
            actions.append("{}: git checkout {} failed ({})".format(tool, sha[:12], err or code))
            return False
    return _verify_pinned_sha(repo_dir, sha, tool, actions)


# ---------------------------------------------------------------------------
# gpu-burn: per-fleet validation cache
# ---------------------------------------------------------------------------


def _gpu_fleet_signature() -> str:
    """Stable signature of the currently visible GPU set, used to invalidate cache on fleet change."""
    if not command_exists("nvidia-smi"):
        return "no-nvidia-smi"
    code, out, _ = run_capture(
        ["nvidia-smi", "--query-gpu=uuid,compute_cap", "--format=csv,noheader"],
        timeout=10,
    )
    if code != 0:
        return "query-failed"
    return "|".join(sorted(line.strip() for line in out.splitlines() if line.strip()))


def _gpu_burn_validation_file() -> Path:
    return gpu_burn_repo_dir() / ".gpu_burn_validated.json"


def _read_validation_file() -> Dict[str, object]:
    """Return the full validation cache dict, or {} if missing/corrupt.

    Schema (v2, per-cap):
      {
        "version": 2,
        "per_cap": {
          "89":  {"result": "ok", "binary": ".../gpu_burn-cc89",  "fleet_signature": "...", "detail": "...", "timestamp": "..."},
          "120": {"result": "ok", "binary": ".../gpu_burn-cc120", "fleet_signature": "...", "detail": "...", "timestamp": "..."}
        }
      }

    Legacy v1 schema (single binary, top-level keys: result/binary/fleet_signature/compute/detail)
    is migrated on read into v2 under its compute key.
    """
    path = _gpu_burn_validation_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    if data.get("version") == 2 and isinstance(data.get("per_cap"), dict):
        return data
    # Migrate legacy v1
    if data.get("result") == "ok" and "compute" in data:
        cap_key = str(data["compute"])
        return {"version": 2, "per_cap": {cap_key: data}}
    return {}


def _write_validation_file(data: Dict[str, object]) -> None:
    try:
        _gpu_burn_validation_file().write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def _load_gpu_burn_validation(cap: int) -> Optional[Dict[str, object]]:
    """Return the validation entry for a given compute cap if cached + fleet still matches.

    Keyed on fleet signature (GPU UUIDs + compute caps). Multiple caps can
    coexist in the cache so swapping rigs (sm_89 <-> sm_120) doesn't thrash
    rebuilds.
    """
    data = _read_validation_file()
    per_cap = data.get("per_cap") or {}
    entry = per_cap.get(str(cap))
    if not isinstance(entry, dict):
        return None
    if entry.get("result") != "ok":
        return None
    if entry.get("fleet_signature") != _gpu_fleet_signature():
        return None
    binary = entry.get("binary")
    if isinstance(binary, str) and not Path(binary).exists():
        return None
    return entry


def _save_gpu_burn_validation(binary: str, compute: int, detail: str) -> None:
    data = _read_validation_file()
    if "per_cap" not in data or not isinstance(data.get("per_cap"), dict):
        data = {"version": 2, "per_cap": {}}
    data["version"] = 2
    data["per_cap"][str(compute)] = {  # type: ignore[index]
        "result": "ok",
        "timestamp": utc_now(),
        "binary": binary,
        "fleet_signature": _gpu_fleet_signature(),
        "compute": compute,
        "detail": detail,
    }
    _write_validation_file(data)


def _invalidate_gpu_burn_cap(cap: int) -> None:
    """Drop the cached validation entry for one compute cap (others preserved)."""
    data = _read_validation_file()
    per_cap = data.get("per_cap")
    if isinstance(per_cap, dict) and str(cap) in per_cap:
        del per_cap[str(cap)]
        _write_validation_file(data)


# ---------------------------------------------------------------------------
# gpu-burn: smoke test + ensure (build mode)
# ---------------------------------------------------------------------------


def gpu_burn_smoke_test(binary: str) -> Tuple[bool, str]:
    """Run a short gpu_burn invocation and decide if the binary is usable on this GPU."""
    cwd = Path(binary).parent
    code, out, err = run_capture_env(
        [binary, str(GPU_BURN_SMOKE_DURATION_SEC)],
        timeout=GPU_BURN_SMOKE_TIMEOUT_SEC,
        cwd=cwd,
    )
    combined = ((out or "") + "\n" + (err or "")).lower()
    for marker in GPU_BURN_FAIL_MARKERS:
        if marker in combined:
            snippet = combined.strip().splitlines()[-1] if combined.strip() else ""
            return False, "fail marker '{}' in output: {}".format(marker, snippet[:200])
    if code != 0:
        return False, "gpu_burn exited rc={}: {}".format(code, (err or out or "").strip()[:200])
    if "done" not in combined and "proc'd" not in combined and "gflop" not in combined:
        return False, "gpu_burn exited 0 but produced no success markers: {}".format(combined[:200])
    return True, "smoke test passed ({}s)".format(GPU_BURN_SMOKE_DURATION_SEC)


def ensure_gpu_burn(actions: List[str], *, build: bool = False) -> None:
    """Check that the vendored gpu-burn binary exists for the current GPU fleet.

    Per-cap binaries (`gpu_burn-cc<cap>`) coexist so swapping rigs between
    architectures (sm_89 dev rig <-> sm_120 DUT) does not invalidate the
    other rig's binary on the next build.

    With ``build=False`` (default, used by ``run``): just check the binary
    matching the visible GPUs is present. No clone, no make, no smoke test.

    With ``build=True`` (used by ``bootstrap``): build from source if the
    cap-specific binary is missing or fails fleet validation. Other caps'
    binaries are preserved untouched.
    """
    if not is_linux():
        return

    caps = nvidia_gpu_compute_caps()
    target_cap = min(caps) if caps else 75
    if caps:
        cap_msg = "nvidia-smi compute caps: {} (target compute_{})".format(
            ", ".join(str(c) for c in caps), target_cap,
        )
    else:
        cap_msg = "no NVIDIA GPU visible to nvidia-smi; defaulting target to compute_{}".format(target_cap)

    existing_per_cap = gpu_burn_binary_for_cap(target_cap)

    if not build:
        # --- Run mode: fast check only --------------------------------
        if existing_per_cap:
            actions.append("gpu-burn binary present for compute_{} at {}".format(target_cap, existing_per_cap))
        else:
            legacy = gpu_burn_repo_dir() / "gpu_burn"
            if legacy.exists():
                actions.append(
                    "gpu-burn legacy binary at {} (no per-cap binary for compute_{}). "
                    "Run `psfu.py bootstrap` to migrate.".format(legacy, target_cap)
                )
            else:
                actions.append(
                    "gpu-burn binary not found for compute_{}. "
                    "Run `psfu.py bootstrap` to build it for this GPU.".format(target_cap)
                )
        return

    # --- Bootstrap mode: build if needed ------------------------------
    actions.append(cap_msg)

    # Migrate legacy single-binary build if present.
    repo_dir = gpu_burn_repo_dir()
    legacy_binary = repo_dir / "gpu_burn"
    if legacy_binary.exists() and not existing_per_cap:
        # Try to identify what cap the legacy binary was built for via the v1
        # cache (already migrated by _read_validation_file). If a cache entry
        # still references the legacy path, claim the legacy binary as that cap.
        cache = _read_validation_file()
        per_cap = cache.get("per_cap") or {}
        for cap_key, entry in list(per_cap.items()):
            if isinstance(entry, dict) and entry.get("binary") == str(legacy_binary):
                target_name = repo_dir / "gpu_burn-cc{}".format(cap_key)
                try:
                    legacy_binary.rename(target_name)
                    entry["binary"] = str(target_name)
                    actions.append("migrated legacy gpu_burn -> {} (cached as compute_{})".format(target_name, cap_key))
                except OSError as exc:
                    actions.append("could not migrate legacy gpu_burn: {}".format(exc))
                else:
                    _write_validation_file(cache)
                break
        existing_per_cap = gpu_burn_binary_for_cap(target_cap)

    if existing_per_cap:
        cached = _load_gpu_burn_validation(target_cap)
        if cached is not None:
            actions.append("gpu_burn compute_{} already validated on this GPU fleet ({})".format(
                target_cap, cached.get("detail", "")
            ))
            return
        # Binary exists but no cache -- smoke test before rebuilding
        if caps:
            ok, detail = gpu_burn_smoke_test(existing_per_cap)
            if ok:
                _save_gpu_burn_validation(existing_per_cap, target_cap, detail)
                actions.append("gpu-burn compute_{} smoke test passed -- cached ({})".format(target_cap, detail))
                return
            actions.append("gpu-burn compute_{} smoke test failed: {}; will rebuild".format(target_cap, detail))

    # Install build prerequisites
    if not command_exists("git"):
        apt_install(["git"], actions)
    if not command_exists("make") or not command_exists("gcc"):
        apt_install(["build-essential"], actions)
    if not command_exists("nvcc"):
        actions.append("nvcc not available; attempting distro CUDA toolkit install")
        apt_install(["nvidia-cuda-toolkit"], actions)

    if not command_exists("nvcc"):
        actions.append(
            "nvcc still unavailable; install CUDA Toolkit manually. "
            "For Blackwell (sm_120) you need CUDA 12.8+ from NVIDIA's repo: "
            "https://developer.nvidia.com/cuda-downloads"
        )
        return

    ver = nvcc_version_tuple()
    if ver is not None:
        actions.append("nvcc version detected: {}.{}".format(ver[0], ver[1]))
    if caps and not nvcc_supports_compute_cap(target_cap):
        actions.append(
            "installed nvcc {} is too old to target compute_{}. "
            "For Blackwell (sm_120) install CUDA 12.8+ from https://developer.nvidia.com/cuda-downloads "
            "(or for WSL: target 'wsl-ubuntu' in the NVIDIA repo). "
            "gpu-burn build aborted.".format(
                "{}.{}".format(*ver) if ver else "unknown",
                target_cap,
            )
        )
        return

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    if not repo_dir.exists():
        clone_cmd = ["git", "clone", "https://github.com/wilicc/gpu-burn.git", str(repo_dir)]
        if GPU_BURN_PINNED_REF:
            clone_cmd = [
                "git", "clone", "--branch", GPU_BURN_PINNED_REF, "--depth", "1",
                "https://github.com/wilicc/gpu-burn.git", str(repo_dir),
            ]
        code, _, err = run_capture_env(clone_cmd, timeout=1800)
        actions.append("git clone gpu-burn -> {}".format("ok" if code == 0 else err or code))
        if code != 0:
            return
    if not _verify_pinned_sha(repo_dir, GPU_BURN_PINNED_SHA, "gpu-burn", actions):
        return

    # `make clean` removes the canonical Makefile output (./gpu_burn) but does
    # NOT touch our renamed gpu_burn-cc<N> artifacts -- so other caps persist.
    run_capture_env(["make", "clean"], timeout=120, cwd=repo_dir)
    _invalidate_gpu_burn_cap(target_cap)

    code, stdout, err = run_capture_env(
        ["make", "COMPUTE={}".format(target_cap)],
        timeout=3600,
        cwd=repo_dir,
    )
    actions.append("make gpu-burn COMPUTE={} -> {}".format(
        target_cap,
        "ok" if code == 0 else (err or stdout or code),
    ))
    if code != 0:
        return

    raw_binary = repo_dir / "gpu_burn"
    if not raw_binary.exists():
        actions.append("gpu-burn make reported success but binary missing at {}".format(raw_binary))
        return

    # Rename to per-cap so a future build for a different cap doesn't clobber us.
    target_binary = repo_dir / "gpu_burn-cc{}".format(target_cap)
    try:
        if target_binary.exists():
            target_binary.unlink()
        raw_binary.rename(target_binary)
    except OSError as exc:
        actions.append("could not rename gpu_burn -> {}: {}".format(target_binary, exc))
        return
    actions.append("gpu-burn binary built and renamed to {}".format(target_binary))

    if not caps:
        actions.append("no GPU visible; skipped gpu-burn smoke test")
        return

    ok, detail = gpu_burn_smoke_test(str(target_binary))
    if ok:
        _save_gpu_burn_validation(str(target_binary), target_cap, detail)
        actions.append("gpu-burn compute_{} smoke test -> {}".format(target_cap, detail))
    else:
        actions.append("gpu-burn compute_{} smoke test FAILED: {}".format(target_cap, detail))


# ---------------------------------------------------------------------------
# stress-ng: smoke test + ensure (build mode)
# ---------------------------------------------------------------------------


def stress_ng_smoke_test(binary: str) -> Tuple[bool, str]:
    """Run a 1s stress-ng invocation. Validates the binary is usable on this host.

    Exit code is the primary signal — stress-ng exits 0 on a clean run regardless
    of version. Output format varies significantly between versions (0.13 prints
    "successful run completed", 0.17+ does not), so we only use output to confirm
    the binary actually started, not to validate success.
    """
    code, out, err = run_capture_env(
        [binary, "--cpu", "1", "--timeout", "1", "--metrics-brief"],
        timeout=STRESS_NG_SMOKE_TIMEOUT_SEC,
    )
    if code != 0:
        return False, "stress-ng exited rc={}: {}".format(code, (err or out or "").strip()[:200])
    combined = ((out or "") + "\n" + (err or "")).strip()
    if not combined:
        return False, "stress-ng exited 0 but produced no output at all — binary may be broken"
    return True, "smoke test passed (1s, rc=0)"


def ensure_stress_ng(actions: List[str], *, build: bool = False) -> None:
    """Check that stress-ng is available — vendored binary or system PATH.

    With ``build=False`` (default, used by ``run``): just check the binary
    is present. No clone, no make, no smoke test.

    With ``build=True`` (used by ``bootstrap``): clone and build from source
    if no usable binary exists.
    """
    if not is_linux():
        return

    existing = stress_ng_binary_path()
    if existing:
        origin = "vendored" if existing.startswith(str(stress_ng_repo_dir())) else "system"
        if build:
            # Bootstrap mode: verify it actually works
            ok, detail = stress_ng_smoke_test(existing)
            if ok:
                actions.append("stress-ng usable ({}: {}, {})".format(origin, existing, detail))
                return
            actions.append("stress-ng at {} failed smoke test: {}; will rebuild".format(existing, detail))
        else:
            actions.append("stress-ng available ({}: {})".format(origin, existing))
            return

    if not build:
        actions.append(
            "stress-ng not found (no vendored binary at {}, not on PATH). "
            "Run `psfu.py bootstrap` or `apt install stress-ng`.".format(
                stress_ng_repo_dir() / "stress-ng"
            )
        )
        return

    # --- Bootstrap mode: build from source ----------------------------
    if not command_exists("git"):
        apt_install(["git"], actions)
    if not command_exists("make") or not command_exists("gcc"):
        apt_install(["build-essential"], actions)

    if not command_exists("git") or not command_exists("make") or not command_exists("gcc"):
        actions.append(
            "stress-ng build prerequisites missing (need git, make, gcc). "
            "Install them manually on the DUT (apt install git build-essential)."
        )
        return

    repo_dir = stress_ng_repo_dir()
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    if not repo_dir.exists():
        clone_cmd = ["git", "clone", "--depth", "1", STRESS_NG_REPO_URL, str(repo_dir)]
        if STRESS_NG_PINNED_REF:
            clone_cmd = [
                "git", "clone", "--branch", STRESS_NG_PINNED_REF, "--depth", "1",
                STRESS_NG_REPO_URL, str(repo_dir),
            ]
        code, _, err = run_capture_env(clone_cmd, timeout=1800)
        actions.append("git clone stress-ng -> {}".format("ok" if code == 0 else err or code))
        if code != 0:
            return
    if not _verify_pinned_sha(repo_dir, STRESS_NG_PINNED_SHA, "stress-ng", actions):
        return

    code, stdout, err = run_capture_env(
        ["make", "-j"],
        timeout=3600,
        cwd=repo_dir,
    )
    actions.append("make stress-ng -> {}".format(
        "ok" if code == 0 else (err or stdout or code)
    ))
    if code != 0:
        actions.append(
            "stress-ng build failed. Upstream has optional deps (libaio-dev, "
            "libcap-dev, libbsd-dev etc); core build should still work without "
            "them. Check the make output above for which header it's missing."
        )
        return

    binary = repo_dir / "stress-ng"
    if not binary.exists():
        actions.append("stress-ng make reported success but binary missing at {}".format(binary))
        return
    actions.append("stress-ng binary present at {}".format(binary))

    ok, detail = stress_ng_smoke_test(str(binary))
    if ok:
        actions.append("stress-ng smoke test -> {}".format(detail))
    else:
        actions.append("stress-ng smoke test FAILED: {}".format(detail))


# ---------------------------------------------------------------------------
# fio: vendored source tree in tools/fio/, built binary at tools/fio/fio.
# S3 uses fio for read-only NVMe jobs; unlike stress-ng/gpu-burn we do not
# clone here because the operator already syncs the vendored source tree.
# ---------------------------------------------------------------------------


def fio_smoke_test(binary: str) -> Tuple[bool, str]:
    """Validate that fio starts and reports a version string."""
    code, out, err = run_capture_env([binary, "--version"], timeout=FIO_SMOKE_TIMEOUT_SEC)
    combined = ((out or "") + "\n" + (err or "")).strip()
    if code != 0:
        return False, "fio exited rc={}: {}".format(code, combined[:200])
    if "fio-" not in combined.lower() and "fio " not in combined.lower():
        return False, "fio returned rc=0 but version output looked wrong: {}".format(combined[:200])
    return True, "smoke test passed ({})".format(combined.splitlines()[0][:120] if combined else "version")


def ensure_fio(actions: List[str], *, build: bool = False) -> None:
    """Check that fio is available — vendored build or system PATH.

    With ``build=False``: report presence only.
    With ``build=True``: try to build the vendored source tree at ``tools/fio``.
    """
    if not is_linux():
        return

    existing = fio_binary_path()
    if existing:
        origin = "vendored" if existing.startswith(str(fio_repo_dir())) else "system"
        if build:
            ok, detail = fio_smoke_test(existing)
            if ok:
                actions.append("fio usable ({}: {}, {})".format(origin, existing, detail))
                return
            actions.append("fio at {} failed smoke test: {}; will rebuild vendored tree".format(existing, detail))
        else:
            actions.append("fio available ({}: {})".format(origin, existing))
            return

    if not build:
        actions.append(
            "fio not found (no vendored binary at {}, not on PATH). "
            "Build tools/fio or install fio on the DUT.".format(fio_repo_dir() / "fio")
        )
        return

    if not command_exists("make") or not command_exists("gcc"):
        apt_install(["build-essential"], actions)
    if not command_exists("make") or not command_exists("gcc"):
        actions.append(
            "fio build prerequisites missing (need make, gcc). "
            "Install them manually on the DUT (apt install build-essential)."
        )
        return

    repo_dir = fio_repo_dir()
    if not repo_dir.exists():
        actions.append("fio source tree missing at {} — vendored source not present".format(repo_dir))
        return

    code, stdout, err = run_capture_env(
        ["make", "-j"],
        timeout=3600,
        cwd=repo_dir,
    )
    actions.append("make fio -> {}".format("ok" if code == 0 else (err or stdout or code)))
    if code != 0:
        actions.append(
            "fio build failed. Check the make output above for missing headers "
            "or distro-specific build deps."
        )
        return

    binary = repo_dir / "fio"
    if not binary.exists():
        actions.append("fio make reported success but binary missing at {}".format(binary))
        return
    actions.append("fio binary present at {}".format(binary))

    ok, detail = fio_smoke_test(str(binary))
    if ok:
        actions.append("fio smoke test -> {}".format(detail))
    else:
        actions.append("fio smoke test FAILED: {}".format(detail))


# ---------------------------------------------------------------------------
# nvme-cli: vendored source tree in tools/nvme-cli/, built binary at
# tools/nvme-cli/builddir/nvme.  meson build; fabrics disabled to avoid
# rdma/ibverbs deps that are not present on most bench DUTs.
# ---------------------------------------------------------------------------


def nvme_cli_smoke_test(binary: str) -> Tuple[bool, str]:
    """Run nvme --version and verify it produces a recognisable version line."""
    code, out, err = run_capture_env([binary, "--version"], timeout=NVME_CLI_SMOKE_TIMEOUT_SEC)
    combined = ((out or "") + "\n" + (err or "")).strip()
    if code != 0:
        return False, "nvme exited rc={}: {}".format(code, combined[:200])
    if "nvme" not in combined.lower():
        return False, "nvme returned rc=0 but version output looked wrong: {}".format(combined[:200])
    return True, "smoke test passed ({})".format(combined.splitlines()[0][:120] if combined else "version")


def ensure_nvme_cli(actions: List[str], *, build: bool = False) -> None:
    """Check that nvme-cli is available — vendored build or system PATH.

    With ``build=False`` (default, used by ``run``): just check the binary
    is present. No build, no smoke test.

    With ``build=True`` (used by ``bootstrap``): build from the vendored
    source tree at ``tools/nvme-cli`` using meson + ninja if no usable
    binary exists.
    """
    if not is_linux():
        return

    existing = nvme_cli_binary_path()
    if existing:
        origin = "vendored" if existing.startswith(str(nvme_cli_repo_dir())) else "system"
        if build:
            ok, detail = nvme_cli_smoke_test(existing)
            if ok:
                actions.append("nvme-cli usable ({}: {}, {})".format(origin, existing, detail))
                return
            actions.append("nvme-cli at {} failed smoke test: {}; will rebuild".format(existing, detail))
        else:
            actions.append("nvme-cli available ({}: {})".format(origin, existing))
            return

    if not build:
        actions.append(
            "nvme-cli not found (no vendored binary at {}, not on PATH). "
            "Run `psfu.py bootstrap` or `apt install nvme-cli`.".format(
                nvme_cli_repo_dir() / "builddir" / "nvme"
            )
        )
        return

    # --- Bootstrap mode: build from vendored source tree -----------------
    if not command_exists("meson"):
        apt_install(["meson"], actions)
    if not command_exists("ninja"):
        apt_install(["ninja-build"], actions)
    if not command_exists("meson") or not command_exists("ninja"):
        actions.append(
            "nvme-cli build prerequisites missing (need meson, ninja). "
            "Install them manually: apt install meson ninja-build"
        )
        return

    repo_dir = nvme_cli_repo_dir()
    if not repo_dir.exists():
        actions.append("nvme-cli source tree missing at {} — vendored source not present".format(repo_dir))
        return

    build_dir = repo_dir / "builddir"
    code, stdout, err = run_capture_env(
        ["meson", "setup", str(build_dir), "--wipe", "-Dfabrics=disabled"],
        timeout=300,
        cwd=repo_dir,
    )
    actions.append("meson setup nvme-cli -> {}".format("ok" if code == 0 else (err or stdout or code)))
    if code != 0:
        return

    code, stdout, err = run_capture_env(
        ["ninja", "-C", str(build_dir)],
        timeout=600,
    )
    actions.append("ninja nvme-cli -> {}".format("ok" if code == 0 else (err or stdout or code)))
    if code != 0:
        return

    binary = build_dir / "nvme"
    if not binary.exists():
        actions.append("nvme-cli ninja reported success but binary missing at {}".format(binary))
        return
    actions.append("nvme-cli binary present at {}".format(binary))

    ok, detail = nvme_cli_smoke_test(str(binary))
    if ok:
        actions.append("nvme-cli smoke test -> {}".format(detail))
    else:
        actions.append("nvme-cli smoke test FAILED: {}".format(detail))


# ---------------------------------------------------------------------------
# smartmontools: vendored source tree in tools/smartmontools/, built binary
# at tools/smartmontools/src/smartctl.  autotools build.
# ---------------------------------------------------------------------------


def smartctl_smoke_test(binary: str) -> Tuple[bool, str]:
    """Run smartctl --version and verify it produces a recognisable version line."""
    code, out, err = run_capture_env([binary, "--version"], timeout=SMARTMONTOOLS_SMOKE_TIMEOUT_SEC)
    combined = ((out or "") + "\n" + (err or "")).strip()
    if code not in (0, 1):
        return False, "smartctl exited rc={}: {}".format(code, combined[:200])
    if "smartctl" not in combined.lower():
        return False, "smartctl returned unexpected output: {}".format(combined[:200])
    return True, "smoke test passed ({})".format(combined.splitlines()[0][:120] if combined else "version")


def ensure_smartmontools(actions: List[str], *, build: bool = False) -> None:
    """Check that smartmontools (smartctl) is available — vendored build or system PATH.

    With ``build=False`` (default, used by ``run``): just check the binary
    is present. No build, no smoke test.

    With ``build=True`` (used by ``bootstrap``): build from the vendored
    source tree at ``tools/smartmontools`` using autotools if no usable
    binary exists.
    """
    if not is_linux():
        return

    existing = smartctl_binary_path()
    if existing:
        origin = "vendored" if existing.startswith(str(smartmontools_repo_dir())) else "system"
        if build:
            ok, detail = smartctl_smoke_test(existing)
            if ok:
                actions.append("smartmontools usable ({}: {}, {})".format(origin, existing, detail))
                return
            actions.append("smartmontools at {} failed smoke test: {}; will rebuild".format(existing, detail))
        else:
            actions.append("smartmontools available ({}: {})".format(origin, existing))
            return

    if not build:
        actions.append(
            "smartmontools not found (no vendored binary at {}, not on PATH). "
            "Run `psfu.py bootstrap` or `apt install smartmontools`.".format(
                smartmontools_repo_dir() / "src" / "smartctl"
            )
        )
        return

    # --- Bootstrap mode: build from vendored source tree -----------------
    if not command_exists("make") or not command_exists("gcc"):
        apt_install(["build-essential"], actions)
    if not command_exists("autoreconf"):
        apt_install(["autoconf", "automake", "libtool"], actions)

    if not command_exists("make") or not command_exists("gcc"):
        actions.append(
            "smartmontools build prerequisites missing (need make, gcc). "
            "Install them manually: apt install build-essential"
        )
        return

    repo_dir = smartmontools_repo_dir()
    if not repo_dir.exists():
        actions.append("smartmontools source tree missing at {} — vendored source not present".format(repo_dir))
        return

    code, stdout, err = run_capture_env(["./autogen.sh"], timeout=120, cwd=repo_dir)
    actions.append("autogen.sh smartmontools -> {}".format("ok" if code == 0 else (err or stdout or code)))
    if code != 0:
        return

    code, stdout, err = run_capture_env(
        ["./configure", "--without-selinux", "--without-systemd", "--without-libcap-ng"],
        timeout=180,
        cwd=repo_dir,
    )
    actions.append("configure smartmontools -> {}".format("ok" if code == 0 else (err or stdout or code)))
    if code != 0:
        return

    code, stdout, err = run_capture_env(["make", "-j"], timeout=600, cwd=repo_dir)
    actions.append("make smartmontools -> {}".format("ok" if code == 0 else (err or stdout or code)))
    if code != 0:
        return

    binary = repo_dir / "src" / "smartctl"
    if not binary.exists():
        actions.append("smartmontools make reported success but binary missing at {}".format(binary))
        return
    actions.append("smartmontools binary present at {}".format(binary))

    ok, detail = smartctl_smoke_test(str(binary))
    if ok:
        actions.append("smartmontools smoke test -> {}".format(detail))
    else:
        actions.append("smartmontools smoke test FAILED: {}".format(detail))


# ---------------------------------------------------------------------------
# lm-sensors / TRX50 board telemetry kernel module stack
# ---------------------------------------------------------------------------


def _try_it87_fork(actions: List[str]) -> bool:
    """Install and load frankcrawford/it87 fork with force_id=0x8689 for TRX50 AI TOP."""
    if not can_run_sudo_noninteractive():
        actions.append("sudo -n unavailable; skipped it87-dkms install")
        return False

    repo_dir = TOOLS_DIR / "it87"
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    apt_install(
        ["git", "build-essential", "dkms", "dh-dkms", "linux-headers-{}".format(platform.release())],
        actions,
    )
    if not _clone_pinned("https://github.com/frankcrawford/it87.git", repo_dir, IT87_PINNED_SHA, "it87", actions):
        return False

    # Build as the user; sudo is only for the install step below.
    code, _, err = run_capture_env(["make"], timeout=600, cwd=repo_dir)
    actions.append("make it87 (as user) -> {}".format("ok" if code == 0 else err or code))
    if code != 0:
        return False

    code, _, err = run_capture_env(sudo_prefix() + ["make", "dkms"], timeout=3600, cwd=repo_dir)
    actions.append("make dkms it87 -> {}".format("ok" if code == 0 else err or code))
    if code != 0:
        code, _, err = run_capture_env(sudo_prefix() + ["insmod", "./it87.ko", "force_id=0x8689"], timeout=60, cwd=repo_dir)
        actions.append("insmod it87.ko force_id=0x8689 -> {}".format("ok" if code == 0 else err or code))
        if code != 0:
            return False
    else:
        run_capture_env(sudo_prefix() + ["modprobe", "-r", "it87"], timeout=30)
        code, _, err = run_capture_env(
            sudo_prefix() + ["modprobe", "it87", "force_id=0x8689", "ignore_resource_conflict=1"],
            timeout=60,
        )
        actions.append("modprobe it87 force_id=0x8689 -> {}".format("ok" if code == 0 else err or code))
        if code != 0:
            return False

    run_capture_env(["sensors"], timeout=10)
    return sensors_have_named_rails()


def _try_nct6687d(actions: List[str]) -> bool:
    """Try the Fred78290 nct6687d out-of-tree module. Known broken on current
    TRX50 AI TOP firmware (EC Invalid address 0xFFFF), kept as secondary fallback."""
    if not can_run_sudo_noninteractive():
        actions.append("sudo -n unavailable; skipped nct6687d dkms install")
        return False

    repo_dir = TOOLS_DIR / "nct6687d"
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    if not _clone_pinned("https://github.com/Fred78290/nct6687d.git", repo_dir, NCT6687D_PINNED_SHA, "nct6687d", actions):
        return False
    # Build as the user first; sudo only installs.
    code, _, err = run_capture_env(["make"], timeout=600, cwd=repo_dir)
    actions.append("make nct6687d (as user) -> {}".format("ok" if code == 0 else err or code))
    if code != 0:
        return False
    code, _, err = run_capture_env(sudo_prefix() + ["make", "dkms_install"], timeout=3600, cwd=repo_dir)
    actions.append("make dkms_install nct6687d -> {}".format("ok" if code == 0 else err or code))
    if code != 0:
        return False
    code, _, err = run_capture_env(sudo_prefix() + ["modprobe", "nct6687", "force=1"], timeout=120)
    actions.append("modprobe nct6687 force=1 -> {}".format("ok" if code == 0 else err or code))
    if code != 0:
        return False
    run_capture_env(["sensors"], timeout=10)
    return sensors_have_named_rails()


_BOARD_TELEMETRY_CACHE = Path(__file__).resolve().parent.parent / ".board-telemetry-status"


def ensure_trx50_sensor_stack(actions: List[str]) -> None:
    """Install and configure lm-sensors + kernel modules for board rail telemetry.

    On TRX50 AI TOP boards, tries frankcrawford/it87 fork first, then nct6687d.
    Caches the result so failed kernel module builds aren't retried on every run.
    """
    if not is_linux():
        return
    if ipmi_have_named_rails():
        actions.append("IPMI exposing named rails; skipped sensors-detect")
        return
    if not command_exists("sensors"):
        apt_install(["lm-sensors"], actions)
    if sensors_have_named_rails():
        actions.append("sensors already exposing named rails; skipped sensors-detect")
    elif command_exists("sensors-detect") and can_run_sudo_noninteractive():
        code, _, err = run_capture_env(sudo_prefix() + ["sensors-detect", "--auto"], timeout=900)
        actions.append("sensors-detect --auto -> {}".format("ok" if code == 0 else err or code))
    elif command_exists("sensors-detect"):
        actions.append("sudo -n unavailable; skipped sensors-detect --auto")
    if not is_trx50_ai_top():
        return

    if sensors_have_named_rails():
        actions.append("TRX50 AI TOP rails already visible via sensors")
        return

    # Check cache — don't re-try kernel module builds that already failed
    if _BOARD_TELEMETRY_CACHE.exists():
        cached = _BOARD_TELEMETRY_CACHE.read_text().strip()
        if cached == "unavailable-confirmed":
            actions.append(
                "TRX50 AI TOP: board telemetry previously confirmed unavailable (cached). "
                "Delete .board-telemetry-status to retry kernel module builds."
            )
            return

    actions.append("TRX50 AI TOP detected; trying frankcrawford/it87 fork (force_id=0x8689)")
    if _try_it87_fork(actions):
        actions.append("TRX50 AI TOP rails visible via it87 fork")
        return

    actions.append("it87 fork did not expose named rails; trying nct6687d as fallback")
    if _try_nct6687d(actions):
        actions.append("TRX50 AI TOP rails visible via nct6687d")
        return

    # Cache the failure so we don't retry on every run
    try:
        _BOARD_TELEMETRY_CACHE.write_text("unavailable-confirmed\n")
    except OSError:
        pass

    actions.append(
        "TRX50 AI TOP: neither it87 fork nor nct6687d exposed named rails. "
        "Board telemetry will be marked 'unavailable-confirmed'. "
        "CPU temps via k10temp and GPU telemetry via nvidia-smi still work; "
        "the run can proceed but cannot make claims about 12V/5V rails."
    )
