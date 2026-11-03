"""NVMe extended self-test — trigger + poll + result.

Runs ``nvme device-self-test -s 0x2 <dev>`` (extended self-test) then polls
``nvme self-test-log <dev> -o json`` at 10-second intervals until done or
timeout. Stores the final self-test log JSON as an artifact.

Self-test types:
  0x1 — short   (1-2 minutes, surface read check only)
  0x2 — extended (10-30 minutes, full media scan + error log validation)

Exit conditions:
  Completion code 0x0 — passed
  Completion code 0x1-0xF — vendor-defined failure or abort
  Controller not present / nvme-cli not available — returns error immediately

Usage:
  psfu.py selftest /dev/nvme0n1 [--short] [--out-dir PATH]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from psfu.common import run_capture
from psfu.hardware import nvme_cli_binary_path


# Poll every 10 s; extended self-test can take up to 30 minutes on large drives.
_POLL_INTERVAL_SEC = 10
_EXTENDED_TIMEOUT_SEC = 1800  # 30 min
_SHORT_TIMEOUT_SEC = 180       # 3 min

# Completion code 0 = pass; anything else = fail/abort.
_COMPLETION_CODE_PASS = 0x0

# Result codes from this module.
SELFTEST_PASSED = "passed"
SELFTEST_FAILED = "failed"
SELFTEST_TIMEOUT = "timeout"
SELFTEST_ABORTED = "aborted"
SELFTEST_ERROR = "error"


def _trigger_self_test(nvme_bin: str, device: str, *, short: bool = False) -> Tuple[bool, str]:
    """Send the self-test start command. Returns (success, message)."""
    test_type = "0x1" if short else "0x2"
    code, stdout, stderr = run_capture(
        [nvme_bin, "device-self-test", "-s", test_type, device],
        timeout=15,
    )
    if code != 0:
        return False, "trigger failed rc={} stderr={}".format(code, stderr.strip()[:200])
    return True, stdout.strip() or "self-test started"


def _fetch_self_test_log(nvme_bin: str, device: str) -> Optional[dict]:
    """Fetch ``nvme self-test-log <dev> -o json``. Returns parsed dict or None."""
    code, stdout, _ = run_capture(
        [nvme_bin, "self-test-log", device, "-o", "json"],
        timeout=10,
    )
    if code != 0 or not stdout:
        return None
    try:
        data = json.loads(stdout)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _poll_completion(nvme_bin: str, device: str, *, timeout_sec: int, verbose: bool = True) -> Tuple[str, Optional[dict]]:
    """Poll self-test-log until the current operation completes or times out.

    Returns (result_code, final_log_dict).
    """
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        log = _fetch_self_test_log(nvme_bin, device)
        if log is None:
            if verbose:
                print("  [self-test] could not fetch log, retrying...")
            time.sleep(_POLL_INTERVAL_SEC)
            continue

        # Current Device Self-Test Operation field (bits 3:0 of byte 0).
        # 0x0 = no operation in progress (done or never started).
        current_op = log.get("current_operation", {})
        op_code = current_op.get("value", 0) if isinstance(current_op, dict) else 0

        # Completion percent.
        completion = log.get("current_completion", 0)
        if isinstance(completion, dict):
            completion = completion.get("value", 0)

        if op_code == 0:
            # No operation in progress — check the most recent result entry.
            results = log.get("table", [])
            if not results:
                return SELFTEST_ERROR, log
            latest = results[0]
            code_field = latest.get("self_test_result", {})
            result_code = code_field.get("value", 0xFF) if isinstance(code_field, dict) else code_field
            if result_code == _COMPLETION_CODE_PASS:
                return SELFTEST_PASSED, log
            if result_code == 0xF:
                return SELFTEST_ABORTED, log
            return SELFTEST_FAILED, log

        if verbose:
            print("  [self-test] progress {}%  (op=0x{:x})".format(completion, op_code), end="\r")
            sys.stdout.flush()

        time.sleep(_POLL_INTERVAL_SEC)

    return SELFTEST_TIMEOUT, None


def run_selftest(
    device: str,
    *,
    short: bool = False,
    out_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[str, Optional[dict]]:
    """Trigger and poll an NVMe self-test.

    Returns ``(result_code, log_dict)`` where result_code is one of the
    module-level constants (SELFTEST_PASSED, SELFTEST_FAILED, …).
    If ``out_dir`` is given, the final log is written there as
    ``nvme-selftest-<dev_name>.json``.
    """
    nvme_bin = nvme_cli_binary_path()
    if not nvme_bin:
        return SELFTEST_ERROR, None

    test_label = "short" if short else "extended"
    if verbose:
        print("[self-test] starting {} self-test on {}".format(test_label, device))

    ok, msg = _trigger_self_test(nvme_bin, device, short=short)
    if not ok:
        if verbose:
            print("[self-test] ERROR: {}".format(msg))
        return SELFTEST_ERROR, None
    if verbose:
        print("[self-test] {}".format(msg))

    timeout = _SHORT_TIMEOUT_SEC if short else _EXTENDED_TIMEOUT_SEC
    result, log = _poll_completion(nvme_bin, device, timeout_sec=timeout, verbose=verbose)

    if verbose:
        print()  # clear the progress line
        print("[self-test] result: {}".format(result.upper()))

    if out_dir is not None and log is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        dev_name = device.lstrip("/").replace("/", "-")
        path = out_dir / "nvme-selftest-{}.json".format(dev_name)
        path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if verbose:
            print("[self-test] log written to {}".format(path))

    return result, log
