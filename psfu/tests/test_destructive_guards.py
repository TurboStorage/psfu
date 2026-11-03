"""Tests for S5 destructive-mode safety guards.

Four independent guards must ALL pass before any destructive fio job starts.
Tests cover each guard in isolation, run_all_guards composition, and the
mount-before-serial ordering rule.
"""

import pytest
from psfu.storage.destructive_guards import (
    check_destructive_flag,
    check_device_not_mounted,
    check_profile_destructive,
    confirm_serial_reentry,
    run_all_guards,
)


# ---------------------------------------------------------------------------
# Guard 1: check_destructive_flag
# ---------------------------------------------------------------------------


def test_flag_guard_fails_when_false():
    assert check_destructive_flag(False) is not None


def test_flag_guard_passes_when_true():
    assert check_destructive_flag(True) is None


# ---------------------------------------------------------------------------
# Guard 2: check_profile_destructive
# ---------------------------------------------------------------------------


def test_profile_guard_fails_when_false():
    assert check_profile_destructive(False) is not None


def test_profile_guard_passes_when_true():
    assert check_profile_destructive(True) is None


# ---------------------------------------------------------------------------
# Guard 3: confirm_serial_reentry
# ---------------------------------------------------------------------------


def test_serial_reentry_passes_on_exact_match():
    err = confirm_serial_reentry("SN12345", _input_fn=lambda _: "SN12345")
    assert err is None


def test_serial_reentry_fails_on_mismatch():
    err = confirm_serial_reentry("SN12345", _input_fn=lambda _: "WRONG")
    assert err is not None
    assert "mismatch" in err.lower()


def test_serial_reentry_strips_whitespace():
    err = confirm_serial_reentry("  SN-ABC  ", _input_fn=lambda _: "SN-ABC  ")
    assert err is None


def test_serial_reentry_fails_on_blank_serial():
    err = confirm_serial_reentry("", _input_fn=lambda _: "anything")
    assert err is not None
    assert "blank" in err.lower() or "serial" in err.lower()


def test_serial_reentry_fails_on_eof():
    def _raise(_prompt: str) -> str:
        raise EOFError

    err = confirm_serial_reentry("SN12345", _input_fn=_raise)
    assert err is not None
    assert "cancel" in err.lower()


def test_serial_reentry_fails_on_keyboard_interrupt():
    def _raise(_prompt: str) -> str:
        raise KeyboardInterrupt

    err = confirm_serial_reentry("SN12345", _input_fn=_raise)
    assert err is not None


# ---------------------------------------------------------------------------
# Guard 4: check_device_not_mounted
# ---------------------------------------------------------------------------


def test_mount_guard_passes_when_device_not_in_proc_mounts(monkeypatch):
    monkeypatch.setattr(
        "psfu.storage.destructive_guards._read_mount_devices",
        lambda: ["/dev/sda1", "/dev/sdb2"],
    )
    assert check_device_not_mounted("/dev/nvme0n1") is None


def test_mount_guard_fails_when_exact_device_mounted(monkeypatch):
    monkeypatch.setattr(
        "psfu.storage.destructive_guards._read_mount_devices",
        lambda: ["/dev/nvme0n1"],
    )
    err = check_device_not_mounted("/dev/nvme0n1")
    assert err is not None
    assert "mounted" in err.lower()


def test_mount_guard_fails_when_partition_mounted(monkeypatch):
    monkeypatch.setattr(
        "psfu.storage.destructive_guards._read_mount_devices",
        lambda: ["/dev/nvme0n1p1"],
    )
    err = check_device_not_mounted("/dev/nvme0n1")
    assert err is not None
    assert "partition" in err.lower() or "mounted" in err.lower()


def test_mount_guard_fails_when_nvme_namespace_partition_mounted(monkeypatch):
    monkeypatch.setattr(
        "psfu.storage.destructive_guards._read_mount_devices",
        lambda: ["/dev/nvme0n1n1"],
    )
    err = check_device_not_mounted("/dev/nvme0n1")
    assert err is not None


# ---------------------------------------------------------------------------
# run_all_guards composition
# ---------------------------------------------------------------------------


def test_run_all_guards_passes_all_four(monkeypatch):
    monkeypatch.setattr(
        "psfu.storage.destructive_guards._read_mount_devices",
        lambda: [],
    )
    err = run_all_guards(
        destructive_flag=True,
        profile_destructive=True,
        serial="SN-TEST",
        devices=["/dev/nvme0n1"],
        _input_fn=lambda _: "SN-TEST",
    )
    assert err is None


def test_run_all_guards_stops_at_guard_1(monkeypatch):
    monkeypatch.setattr(
        "psfu.storage.destructive_guards._read_mount_devices",
        lambda: [],
    )
    called = []
    err = run_all_guards(
        destructive_flag=False,
        profile_destructive=True,
        serial="SN-TEST",
        devices=["/dev/nvme0n1"],
        _input_fn=lambda _: called.append(True) or "SN-TEST",
    )
    assert err is not None
    assert not called, "serial prompt must not fire when guard 1 blocks"


def test_run_all_guards_checks_mount_before_serial(monkeypatch):
    """Mount guard (non-interactive) must run before serial prompt (interactive)."""
    monkeypatch.setattr(
        "psfu.storage.destructive_guards._read_mount_devices",
        lambda: ["/dev/nvme0n1"],
    )
    prompted = []
    err = run_all_guards(
        destructive_flag=True,
        profile_destructive=True,
        serial="SN-TEST",
        devices=["/dev/nvme0n1"],
        _input_fn=lambda _: prompted.append(True) or "SN-TEST",
    )
    assert err is not None
    assert not prompted, "serial prompt must not fire when device is mounted"


def test_run_all_guards_no_devices_still_checks_serial(monkeypatch):
    monkeypatch.setattr(
        "psfu.storage.destructive_guards._read_mount_devices",
        lambda: [],
    )
    err = run_all_guards(
        destructive_flag=True,
        profile_destructive=True,
        serial="SN-XYZ",
        devices=[],
        _input_fn=lambda _: "SN-XYZ",
    )
    assert err is None
