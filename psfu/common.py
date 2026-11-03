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
