from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from psfu.common import command_exists
from psfu.config import APP_ROOT
from psfu.hardware import (
    fio_binary_path,
    gpu_burn_usable_binary,
    stress_ng_binary_path,
)
from psfu.models import Phase


FIO_JOBS_DIR = APP_ROOT / "psfu" / "storage" / "fio_jobs"


@dataclass(frozen=True)
class FioSpec:
    """Frozen bundle describing how to run fio for one phase / device.

    Built by :func:`build_phase_commands` when ``storage_mode=stress``
    and both a vendored/PATH fio binary and an operator-selected NVMe
    DUT are available. Consumed in :mod:`psfu.runners.execute` to build
    a :class:`FioDutRunner`. Mixed-type storage "command" is avoided;
    storage axis returns either a legacy list[str] (filesystem sha256
    fallback) or a FioSpec — never both, never a union."""

    job_file: Path
    device: str
    phase_name: str
    run_dir: Path
    destructive: bool = False


def _select_fio_job_for_phase_destructive(phase: Phase) -> Path:
    """Pick the destructive jobfile for this phase by name pattern.

    Phase name contains "speed-verify" / "profile-verify" → high-throughput
    sequential CRC32c verify tuned for profiling sustained write speed and
    cliff/GC behaviour.
    Phase name contains "verify" → write+CRC32c verify job (data-integrity check).
    Phase name contains "mixed" → 70/30 random R/W QD32 (controller stability).
    All others (full-write, warmup, etc.) → sequential full-capacity write (SLC flush).
    """
    name = phase.name.lower()
    if "speed-verify" in name or "profile-verify" in name:
        return FIO_JOBS_DIR / "seq_write_verify_2m_nj16_qd8.fio"
    if "verify" in name:
        return FIO_JOBS_DIR / "seq_write_verify.fio"
    if "mixed" in name:
        return FIO_JOBS_DIR / "rand_rw_mixed_70_30_qd32.fio"
    return FIO_JOBS_DIR / "seq_write_full_capacity.fio"


def _select_fio_job_for_phase(phase: Phase) -> Path:
    """Pick the read-only jobfile for this phase.

    S4 ships two read-only jobfiles — ``rand_read_qd1_latency.fio`` for
    short / latency-sensitive phases and ``seq_read_full_capacity.fio``
    for long sequential soak. The default for ``storage_mode=stress``
    is the sequential job; shorter phases fall back to the latency job
    so an 8-second transient pulse doesn't try to stream a full-capacity
    read. The cutoff is deliberately conservative — 60 seconds is enough
    for the sequential job to show bandwidth behaviour, below that the
    QD1 latency job gives more signal per wall-second.
    """
    if phase.duration_sec < 60:
        return FIO_JOBS_DIR / "rand_read_qd1_latency.fio"
    return FIO_JOBS_DIR / "seq_read_full_capacity.fio"


DEFAULT_MEM_STRESS_PCT = 70
PULSE_ON_SEC: int = 2
PULSE_OFF_SEC: int = 2
RAMP_STEPS: int = 5

# Valid mem_tool values.  "stress-ng" is the legacy default.
VALID_MEM_TOOLS = ("stress-ng", "memtester", "stressapptest")


def default_storage_targets() -> List[Path]:
    targets: List[Path] = []
    for candidate in [Path("/mnt"), Path("/media"), Path("/srv")]:
        if candidate.exists():
            targets.append(candidate)
    if not targets:
        targets.append(APP_ROOT)
    return targets


def resolve_storage_targets(paths: Optional[List[str]]) -> List[Path]:
    raw = [Path(p).expanduser() for p in (paths or [])]
    resolved = [p for p in raw if p.exists() and os.access(p, os.R_OK)]
    if resolved:
        return resolved
    defaults = default_storage_targets()
    return [p for p in defaults if p.exists() and os.access(p, os.R_OK)]


def build_storage_read_cmd(duration_sec: int, storage_targets: List[Path]) -> Optional[List[str]]:
    if not storage_targets:
        return None
    cmd = [sys.executable, str(APP_ROOT / "psfu" / "runners" / "workers" / "storage_read.py"), "--timeout", str(max(1, duration_sec))]
    for target in storage_targets:
        cmd.extend(["--path", str(target)])
    return cmd


def cpu_worker_count(load_fraction: float) -> int:
    """Number of stress-ng CPU workers for a given load fraction (0.0–1.0)."""
    online = os.cpu_count() or 1
    return max(1, min(online, int(round(online * load_fraction))))


def build_ramp_cpu_cmd(
    stress_ng_path: str,
    duration_sec: int,
    load_fraction: float,
) -> List[str]:
    workers = cpu_worker_count(load_fraction)
    return [stress_ng_path, "--cpu", str(workers), "--timeout", str(max(1, duration_sec))]


def mem_target_pct(phase: Phase, load_fraction: float) -> int:
    base_pct = phase.mem_target_pct if phase.mem_target_pct is not None else DEFAULT_MEM_STRESS_PCT
    scaled_pct = int(round(base_pct * max(0.0, load_fraction)))
    return max(10, min(90, scaled_pct))


def available_ram_mb() -> int:
    """Return available RAM in MB from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return 0


def build_mem_cmd(stress_ng_path: str, duration_sec: int, target_pct: int) -> List[str]:
    return [
        stress_ng_path,
        "--vm", "2",
        "--vm-bytes", "{}%".format(target_pct),
        "--vm-keep",
        "--vm-populate",
        "--verify",
        "--timeout", str(max(1, duration_sec)),
    ]


def build_memtester_cmd(duration_sec: int, target_pct: int) -> Optional[List[str]]:
    """Build a parallel memtester command via the wrapper script.

    memtester is single-threaded, so on large-RAM systems (256GB+) a
    single instance takes 60-90+ minutes per March pass.  The wrapper
    splits the allocation across N workers (auto-scaled: 1 per 32GB,
    min 2, capped at CPU count).  Each worker runs memtester on its
    own chunk — SIGSTOP/SIGCONT from the pulse engine freezes all
    children because they share the process group.

    Iteration count is estimated from duration and a per-chunk time
    model: ~5 min per 32GB chunk per pass.
    """
    if _find_tool("memtester") is None:
        return None
    ram = available_ram_mb()
    if ram <= 0:
        return None
    size_mb = max(64, int(ram * target_pct / 100))
    # Auto worker count: 1 per 32GB, min 2, capped at CPU count
    workers = max(2, min(os.cpu_count() or 4, size_mb // (32 * 1024)))
    chunk_mb = size_mb // workers
    # Estimate ~5 min per 32GB chunk per pass
    time_per_pass_sec = max(300, (chunk_mb / (32 * 1024)) * 300)
    iters = max(1, int(duration_sec / time_per_pass_sec))
    wrapper = str(APP_ROOT / "psfu" / "runners" / "workers" / "memtester_parallel.py")
    return [
        sys.executable, wrapper,
        "--size-mb", str(size_mb),
        "--workers", str(workers),
        "--iters", str(iters),
    ]


def build_stressapptest_cmd(duration_sec: int, target_pct: int) -> Optional[List[str]]:
    """Build a stressapptest command.  Google's tool for production memory
    qualification: multi-threaded copy + CRC verify, designed to find
    marginal DIMMs that pass pattern tests but fail under thermal/bus
    contention.

    -s <seconds>  -M <MB>  -W (write CRC verify)  --force_error_exit
    """
    path = _find_tool("stressapptest")
    if path is None:
        return None
    ram = available_ram_mb()
    if ram <= 0:
        return None
    size_mb = max(64, int(ram * target_pct / 100))
    return [
        path,
        "-s", str(max(1, duration_sec)),
        "-M", str(size_mb),
        "-W",
        "--force_error_exit",
    ]


def _find_tool(name: str) -> Optional[str]:
    """Locate a tool on PATH, return absolute path or None."""
    if command_exists(name):
        import shutil
        return shutil.which(name)
    return None


def build_ramp_mem_cmd(
    stress_ng_path: str,
    phase: Phase,
    duration_sec: int,
    load_fraction: float,
) -> List[str]:
    return build_mem_cmd(stress_ng_path, duration_sec, mem_target_pct(phase, load_fraction))


def build_phase_commands(
    phase: Phase,
    dry_run: bool,
    *,
    storage_paths: Optional[List[str]] = None,
    storage_dut_paths: Optional[List[str]] = None,
    run_dir: Optional[Path] = None,
    cpu_load_fraction: float = 1.0,
    gpu_load_fraction: float = 1.0,
    dut_enabled: Optional[dict] = None,
    destructive: bool = False,
) -> Tuple[
    Optional[List[str]],
    Optional[List[str]],
    Optional[str],
    Optional[List[str]],
    Optional[List[str]],
    Optional["FioSpec"],
]:
    if dry_run:
        return None, None, None, None, None, None

    enabled = {
        "cpu": True,
        "gpu": True,
        "mem": True,
        "storage": True,
    }
    if dut_enabled:
        enabled.update({key: bool(value) for key, value in dut_enabled.items() if key in enabled})

    cpu_cmd = None
    gpu_cmd = None
    gpu_cwd = None
    mem_cmd = None
    storage_cmd: Optional[List[str]] = None
    fio_spec: Optional[FioSpec] = None
    stress_ng_path = stress_ng_binary_path()

    cpu_mode = phase.cpu_mode if enabled["cpu"] else "idle"
    gpu_mode = phase.gpu_mode if enabled["gpu"] else "idle"
    mem_mode = phase.mem_mode if enabled["mem"] else "idle"
    storage_mode = phase.storage_mode if enabled["storage"] else "idle"

    effective_cpu = "stress" if cpu_mode == "pulse" else cpu_mode
    if effective_cpu == "stress" and stress_ng_path:
        workers = cpu_worker_count(cpu_load_fraction)
        cpu_cmd = [stress_ng_path, "--cpu", str(workers)] + list(phase.cpu_stress_flags) + ["--timeout", str(phase.duration_sec)]

    effective_mem = "stress" if mem_mode == "pulse" else mem_mode
    if effective_mem == "stress":
        pct = mem_target_pct(phase, phase.mem_load_fraction)
        tool = getattr(phase, "mem_tool", "stress-ng")
        if tool == "memtester":
            mem_cmd = build_memtester_cmd(phase.duration_sec, pct)
        elif tool == "stressapptest":
            mem_cmd = build_stressapptest_cmd(phase.duration_sec, pct)
        if mem_cmd is None and stress_ng_path:
            # Fallback to stress-ng if preferred tool unavailable
            mem_cmd = build_mem_cmd(stress_ng_path, phase.duration_sec, pct)

    effective_storage = "stress" if storage_mode == "pulse" else storage_mode
    if effective_storage == "stress":
        # Prefer fio against an operator-selected NVMe DUT. The fallback
        # to the sha256 filesystem reader stays live when fio isn't built
        # or no NVMe DUT was picked in the lobby — per PLAN.md S4 "fall
        # back to storage_read.py when fio unavailable or no NVMe DUT".
        fio_bin = fio_binary_path()
        nvme_duts = list(storage_dut_paths or [])
        if fio_bin and nvme_duts and run_dir is not None:
            job_file = (
                _select_fio_job_for_phase_destructive(phase)
                if destructive
                else _select_fio_job_for_phase(phase)
            )
            if job_file.exists():
                # S4 ships one DUT per FioSpec; the S4 exit criteria says
                # "one bench, one DUT" (PLAN.md §Scope constraints, P3).
                # Multi-DUT batch-runner is its own future commit.
                fio_spec = FioSpec(
                    job_file=job_file,
                    device=nvme_duts[0],
                    phase_name=phase.name,
                    run_dir=run_dir,
                    destructive=destructive,
                )
        if fio_spec is None:
            storage_cmd = build_storage_read_cmd(phase.duration_sec, resolve_storage_targets(storage_paths))

    gpu_burn = gpu_burn_usable_binary()
    effective_gpu = gpu_mode
    if effective_gpu in ("pulse", "ramp"):
        effective_gpu = "stress"

    if effective_gpu == "stress" and gpu_burn:
        gpu_cmd = [gpu_burn] + list(phase.gpu_burn_flags) + [str(phase.duration_sec)]
        gpu_cwd = str(Path(gpu_burn).parent)
    elif effective_gpu == "probe" and command_exists("nvidia-smi"):
        gpu_cmd = ["nvidia-smi", "dmon", "-s", "puct", "-d", "1"]

    return cpu_cmd, gpu_cmd, gpu_cwd, mem_cmd, storage_cmd, fio_spec
