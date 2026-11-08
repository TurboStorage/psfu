import datetime as dt
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import subprocess
import os


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    clean = "".join(ch if ch.isalnum() else "-" for ch in value.strip())
    compact = "-".join(part for part in clean.split("-") if part)
    return compact.lower() or "unknown"


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def is_linux() -> bool:
    return sys.platform.startswith("linux")


# Where turbostat may live, in the order of sudo's default secure_path
# (/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin). A hand-built turbostat
# in /usr/local/bin shadows the distro one for `sudo turbostat`, so the sudoers
# rule, the setup probe and the collector must all name the same absolute path.
# Hardcoded, never PATH-resolved: a PATH an attacker can influence must not
# pick the binary that the NOPASSWD rule names.
TURBOSTAT_SUDO_BIN_CANDIDATES = (
    "/usr/local/sbin/turbostat",
    "/usr/local/bin/turbostat",
    "/usr/sbin/turbostat",
    "/usr/bin/turbostat",
)


def _root_owned_not_writable(path: str) -> bool:
    """True if path and every parent folder are owned by root and not
    writable by group or others (a user-writable binary in a NOPASSWD rule
    would be a root shell)."""
    p = Path(path)
    for item in [p] + list(p.parents):
        try:
            st = os.stat(item)
        except OSError:
            return False
        if st.st_uid != 0 or st.st_mode & 0o022:
            return False
    return True


def turbostat_sudo_path() -> str:
    """Absolute path of the turbostat that `sudo turbostat` runs, or "" if
    none of the candidates exists as a safe, root-owned file."""
    for cand in TURBOSTAT_SUDO_BIN_CANDIDATES:
        if os.path.isfile(cand):
            return cand if _root_owned_not_writable(cand) else ""
    return ""


def shell_join(args: List[str]) -> str:
    return " ".join(args)


def run_capture(args: List[str], timeout: int = 10) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            args,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return 1, "", str(exc)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def run_capture_env(
    args: List[str],
    timeout: int = 10,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        proc = subprocess.run(
            args,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
        )
    except Exception as exc:
        return 1, "", str(exc)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
