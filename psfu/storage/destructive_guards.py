"""Destructive-mode safety guards for S5.

Four independent guards that ALL must pass before any write fio job is
allowed to start. Any guard returning a non-None error string aborts the
run before ``prepare_run`` creates any state.

Guards:
  1. check_destructive_flag   — ``--destructive`` was explicitly passed
  2. check_profile_destructive — profile JSON has ``"destructive": true``
  3. confirm_serial_reentry   — operator re-types the run serial to confirm
  4. check_device_not_mounted  — DUT is not mounted (root FS or any partition)

Rule: treat these as a checklist, not a hierarchy. All must pass.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


def check_destructive_flag(destructive_arg: bool) -> Optional[str]:
    """Guard 1: ``--destructive`` must be explicitly set on the CLI."""
    if not destructive_arg:
        return (
            "--destructive flag not set. Destructive mode requires explicit "
            "opt-in: psfu run --destructive --profile storage-destructive-v1 ..."
        )
    return None


def check_profile_destructive(profile_destructive: bool) -> Optional[str]:
    """Guard 2: Profile must opt in via ``\"destructive\": true``."""
    if not profile_destructive:
        return (
            "Profile does not opt in to destructive mode. "
            "Add \"destructive\": true to the profile JSON, or use a "
            "non-destructive profile (e.g. storage-v1)."
        )
    return None


def confirm_serial_reentry(serial: str, *, _input_fn=input) -> Optional[str]:
    """Guard 3: Operator re-types the run serial as a destruction confirmation.

    ``_input_fn`` is injectable for tests (default: ``input``).
    Strips whitespace from both values before comparison.
    """
    serial = (serial or "").strip()
    if not serial:
        return (
            "Serial is blank. Provide a non-empty --psu-serial to enable "
            "destructive mode serial re-entry confirmation."
        )
    print("")
    print("=" * 72)
    print("  DESTRUCTIVE MODE — all data on the DUT will be overwritten.")
    print("  Re-enter the DUT serial to confirm:")
    print("=" * 72)
    try:
        entered = _input_fn("  Serial: ").strip()
    except (EOFError, KeyboardInterrupt):
        return "Destructive mode confirmation cancelled."
    if entered != serial:
        return (
            "Serial mismatch: expected {!r}, got {!r}. "
            "Destructive run aborted.".format(serial, entered)
        )
    print("  Confirmed. Proceeding with destructive run.")
    print("")
    return None


def _read_mount_devices() -> List[str]:
    """Parse /proc/mounts and return a list of source device paths."""
    try:
        text = Path("/proc/mounts").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    devices: List[str] = []
    for line in text.splitlines():
        parts = line.split()
        if parts:
            devices.append(parts[0])
    return devices


def check_device_not_mounted(device: str) -> Optional[str]:
    """Guard 4: DUT block device must not be mounted.

    Checks /proc/mounts for the exact device path and for partition variants
    (e.g. ``/dev/nvme0n1`` also blocks if ``/dev/nvme0n1p1`` is mounted).
    Also resolves symlinks on both sides before comparing.
    """
    device = device.strip()
    try:
        real_device = str(Path(device).resolve())
    except OSError:
        real_device = device

    mounted = _read_mount_devices()
    for mount_dev in mounted:
        try:
            real_mount_dev = str(Path(mount_dev).resolve())
        except OSError:
            real_mount_dev = mount_dev

        if real_mount_dev == real_device:
            return (
                "Device {} is mounted. Unmount all partitions before running "
                "a destructive test.".format(device)
            )
        if real_mount_dev.startswith(real_device + "p") or real_mount_dev.startswith(real_device + "n"):
            return (
                "Partition {} of {} is mounted. Unmount all partitions before "
                "running a destructive test.".format(mount_dev, device)
            )
    return None


def run_all_guards(
    *,
    destructive_flag: bool,
    profile_destructive: bool,
    serial: str,
    devices: List[str],
    _input_fn=input,
) -> Optional[str]:
    """Run all four guards in sequence. Returns the first error or None.

    Call this before ``prepare_run`` so no run state is created on failure.
    """
    err = check_destructive_flag(destructive_flag)
    if err:
        return err

    err = check_profile_destructive(profile_destructive)
    if err:
        return err

    for device in devices:
        err = check_device_not_mounted(device)
        if err:
            return err

    err = confirm_serial_reentry(serial, _input_fn=_input_fn)
    if err:
        return err

    return None
