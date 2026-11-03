from pathlib import Path

# App Metadata
APP_NAME = "PS-FU"
APP_FULL_NAME = "Platform Stress & Failure Utility"
APP_TAGLINE = "The art of breaking things on purpose."
APP_DESCRIPTION = (
    "Stress and telemetry harness for PC hardware: drives load, logs sensors "
    "and kernel hardware errors, and scores each run PASS / WARN / FAIL."
)
APP_ABOUT = """PS-FU is a stress and telemetry harness for validating PC hardware (CPU, GPU,
RAM, PSU, storage). It drives load with stress-ng, gpu-burn and fio, samples
sensors at 1 Hz into SQLite, watches the kernel log for hardware errors, and
scores each run PASS / WARN / FAIL per subsystem with an evidence bundle.
It is a bench tool, not a benchmark, and its scoring protocol is pre-1.0 and
has not been independently validated. Software sensors are not a multimeter:
runs without a manual reading are labelled software-only."""
APP_VERSION = "0.1.0"
PROTOCOL_VERSION = "0.22"

# Paths
#
# Resolution order for every location (first match wins):
#   1. CLI flag (``--data-dir`` / ``--runs-dir`` / ...), applied by
#      :func:`apply_path_overrides` before any command runs
#   2. Environment variable (``PSFU_HOME`` for the data root, or the more
#      specific ``PSFU_RUNS_DIR`` / ``PSFU_BASELINES_DIR`` / ``PSFU_TOOLS_DIR``
#      / ``PSFU_PROFILES_DIR``)
#   3. Default: ``~/psfu-data`` for all mutable data; bundled profiles ship
#      inside the package.
import os
import sys

# psfu/config.py -> psfu/ -> repository root (or site-packages when installed)
APP_ROOT = Path(__file__).resolve().parent.parent

ENV_HOME = "PSFU_HOME"
ENV_RUNS = "PSFU_RUNS_DIR"
ENV_BASELINES = "PSFU_BASELINES_DIR"
ENV_TOOLS = "PSFU_TOOLS_DIR"
ENV_PROFILES = "PSFU_PROFILES_DIR"


def _env_path(name: str) -> "Path | None":
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else None


def default_data_home() -> Path:
    """Root for runs/, baselines/, tools/ and the logs. ``$PSFU_HOME`` or ``~/psfu-data``."""
    return _env_path(ENV_HOME) or (Path.home() / "psfu-data")


_HOME = default_data_home()
PROFILES_DIR = _env_path(ENV_PROFILES) or (Path(__file__).resolve().parent / "profiles")
RUNS_DIR = _env_path(ENV_RUNS) or (_HOME / "runs")
TOOLS_DIR = _env_path(ENV_TOOLS) or (_HOME / "tools")
BASELINES_DIR = _env_path(ENV_BASELINES) or (_HOME / "baselines")

FM_LOG_FILENAME = "Failure Mode Log.md"
BATCH_TRACKER_FILENAME = "batch-tracker.md"


def fm_log_path() -> Path:
    """Failure-mode log; lives in the data root."""
    return _HOME / FM_LOG_FILENAME


def batch_tracker_path() -> Path:
    """Per-lot batch tracker; lives in the data root."""
    return _HOME / BATCH_TRACKER_FILENAME


FM_LOG_PATH = fm_log_path()


def apply_path_overrides(
    *,
    data_dir: "Path | str | None" = None,
    runs_dir: "Path | str | None" = None,
    baselines_dir: "Path | str | None" = None,
    tools_dir: "Path | str | None" = None,
    profiles_dir: "Path | str | None" = None,
) -> None:
    """Apply CLI-supplied path overrides.

    Several modules bind ``RUNS_DIR`` & co. at import time
    (``from psfu.config import RUNS_DIR``), so updating this module alone
    is not enough: every already-imported ``psfu.*`` module still holding
    the old value is re-pointed too. Call once, early in ``main``.
    """
    global _HOME, PROFILES_DIR, RUNS_DIR, TOOLS_DIR, BASELINES_DIR, FM_LOG_PATH
    old = {
        "PROFILES_DIR": PROFILES_DIR, "RUNS_DIR": RUNS_DIR,
        "TOOLS_DIR": TOOLS_DIR, "BASELINES_DIR": BASELINES_DIR,
        "FM_LOG_PATH": FM_LOG_PATH,
    }
    env_specific = {
        "RUNS_DIR": _env_path(ENV_RUNS), "TOOLS_DIR": _env_path(ENV_TOOLS),
        "BASELINES_DIR": _env_path(ENV_BASELINES),
    }
    if data_dir is not None:
        _HOME = Path(data_dir).expanduser()
        # A data dir moves everything not pinned by a more specific setting.
        RUNS_DIR = env_specific["RUNS_DIR"] or _HOME / "runs"
        TOOLS_DIR = env_specific["TOOLS_DIR"] or _HOME / "tools"
        BASELINES_DIR = env_specific["BASELINES_DIR"] or _HOME / "baselines"
    if runs_dir is not None:
        RUNS_DIR = Path(runs_dir).expanduser()
    if baselines_dir is not None:
        BASELINES_DIR = Path(baselines_dir).expanduser()
    if tools_dir is not None:
        TOOLS_DIR = Path(tools_dir).expanduser()
    if profiles_dir is not None:
        PROFILES_DIR = Path(profiles_dir).expanduser()
    FM_LOG_PATH = fm_log_path()
    new = {
        "PROFILES_DIR": PROFILES_DIR, "RUNS_DIR": RUNS_DIR,
        "TOOLS_DIR": TOOLS_DIR, "BASELINES_DIR": BASELINES_DIR,
        "FM_LOG_PATH": FM_LOG_PATH,
    }
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("psfu") or mod is None:
            continue
        for attr, old_val in old.items():
            if getattr(mod, attr, None) == old_val and new[attr] != old_val:
                setattr(mod, attr, new[attr])
