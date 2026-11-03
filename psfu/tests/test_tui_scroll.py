"""Smoke tests for the TUI scrollable pad refactor.

Curses initialisation doesn't work in the pytest environment (no tty), so
these tests exercise the state machine directly rather than spinning up a
real curses session. The live render paths are covered by the end-to-end
smoke in ``profiles/smoke-v1``.
"""
from __future__ import annotations

import curses
from typing import Any, List
from unittest import mock

import pytest

from psfu.storage.devices import StorageDUT
from psfu.ui.tui import TuiDashboard


class _StubProfile:
    """Minimal profile stub — TuiDashboard only touches .name, .phases,
    .available_duts(). Real Profile imports work too but this keeps the
    test focused on UI state."""
    name = "test-profile"

    class _Phase:
        name = "idle"
        duration_sec = 10

    phases = (_Phase(),)

    def available_duts(self):
        return {"cpu": True, "gpu": False, "mem": False, "storage": False}


class _StubStorageProfile(_StubProfile):
    name = "storage-v1"

    def available_duts(self):
        return {"cpu": False, "gpu": False, "mem": False, "storage": True}


class _StubStdscr:
    """Just enough curses window API to let TuiDashboard.__init__ run."""
    def keypad(self, _flag): pass
    def nodelay(self, _flag): pass


@pytest.fixture
def dashboard(monkeypatch):
    """Build a TuiDashboard with all curses side-effects stubbed out."""
    # Neutralise the global curses calls in __init__
    monkeypatch.setattr(curses, "curs_set", lambda _v: None)
    monkeypatch.setattr(curses, "cbreak", lambda: None)
    monkeypatch.setattr(curses, "noecho", lambda: None)
    monkeypatch.setattr(curses, "start_color", lambda: None, raising=False)
    monkeypatch.setattr(curses, "use_default_colors", lambda: None, raising=False)
    monkeypatch.setattr(curses, "init_pair", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(curses, "can_change_color", lambda: False, raising=False)
    # Minimise color-setup side effects
    if not hasattr(curses, "COLORS"):
        monkeypatch.setattr(curses, "COLORS", 0, raising=False)
    else:
        monkeypatch.setattr(curses, "COLORS", 0)

    return TuiDashboard(
        stdscr=_StubStdscr(),
        profile=_StubProfile(),
        psu_serial="SN-TEST",
        app_name="PS-FU",
    )


@pytest.fixture
def storage_dashboard(monkeypatch):
    monkeypatch.setattr(curses, "curs_set", lambda _v: None)
    monkeypatch.setattr(curses, "cbreak", lambda: None)
    monkeypatch.setattr(curses, "noecho", lambda: None)
    monkeypatch.setattr(curses, "start_color", lambda: None, raising=False)
    monkeypatch.setattr(curses, "use_default_colors", lambda: None, raising=False)
    monkeypatch.setattr(curses, "init_pair", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(curses, "can_change_color", lambda: False, raising=False)
    if not hasattr(curses, "COLORS"):
        monkeypatch.setattr(curses, "COLORS", 0, raising=False)
    else:
        monkeypatch.setattr(curses, "COLORS", 0)
    monkeypatch.setattr(
        "psfu.ui.tui.discover_storage_devices",
        lambda: [
            StorageDUT("/dev/nvme0n1", "nvme0n1", "SN850X", "WD", "nvme", False, [], "unknown", "nvme"),
            StorageDUT("/dev/sda", "sda", "MX500", "Crucial", "sata", False, [], "unknown", "sata"),
        ],
    )
    return TuiDashboard(
        stdscr=_StubStdscr(),
        profile=_StubStorageProfile(),
        psu_serial="SN-TEST",
        app_name="PS-FU",
    )


class TestScrollState:
    def test_scroll_state_initialised(self, dashboard):
        assert dashboard._scroll_y == 0
        assert dashboard._content_height == 0
        assert dashboard._pad is None
        assert dashboard._render_target is not None

    def test_pgdown_increments_scroll_in_live_mode(self, dashboard):
        dashboard.awaiting_start = False  # live mode
        dashboard._handle_key(curses.KEY_NPAGE)
        assert dashboard._scroll_y == 10

    def test_pgup_decrements_scroll(self, dashboard):
        dashboard.awaiting_start = False
        dashboard._scroll_y = 25
        dashboard._handle_key(curses.KEY_PPAGE)
        assert dashboard._scroll_y == 15

    def test_pgup_clamps_to_zero(self, dashboard):
        dashboard.awaiting_start = False
        dashboard._scroll_y = 3
        dashboard._handle_key(curses.KEY_PPAGE)
        assert dashboard._scroll_y == 0

    def test_home_resets_scroll(self, dashboard):
        dashboard.awaiting_start = False
        dashboard._scroll_y = 42
        dashboard._handle_key(curses.KEY_HOME)
        assert dashboard._scroll_y == 0

    def test_end_jumps_to_content_height(self, dashboard):
        dashboard.awaiting_start = False
        dashboard._content_height = 150
        dashboard._handle_key(curses.KEY_END)
        assert dashboard._scroll_y == 150

    def test_arrow_down_scrolls_in_live_only(self, dashboard):
        # Lobby mode: KEY_DOWN is field navigation, not scroll
        dashboard.awaiting_start = True
        dashboard._scroll_y = 0
        dashboard._handle_key(curses.KEY_DOWN)
        assert dashboard._scroll_y == 0  # unchanged

        # Live mode: KEY_DOWN scrolls
        dashboard.awaiting_start = False
        dashboard._handle_key(curses.KEY_DOWN)
        assert dashboard._scroll_y == 1

    def test_pgdn_works_in_lobby_mode(self, dashboard):
        """Lobby screen can exceed viewport on small terminals — operator
        must be able to scroll to reach the START footer."""
        dashboard.awaiting_start = True
        dashboard._handle_key(curses.KEY_NPAGE)
        assert dashboard._scroll_y == 10

    def test_scroll_suppressed_during_field_edit(self, dashboard):
        """While editing a text field, all keys go into the edit buffer —
        scroll must not intercept PgDn mid-type."""
        dashboard.awaiting_start = True
        dashboard._field_editing = True
        dashboard._scroll_y = 0
        dashboard._handle_key(curses.KEY_NPAGE)
        assert dashboard._scroll_y == 0  # unchanged

    def test_scroll_suppressed_while_picker_open(self, dashboard):
        """Modal picker claims the keys until it closes."""
        dashboard.awaiting_start = True
        dashboard._picker_open = True
        dashboard._scroll_y = 5
        dashboard._handle_key(curses.KEY_NPAGE)
        assert dashboard._scroll_y == 5  # unchanged

    def test_scroll_suppressed_while_storage_picker_open(self, storage_dashboard):
        storage_dashboard.awaiting_start = True
        storage_dashboard._storage_picker_open = True
        storage_dashboard._scroll_y = 5
        storage_dashboard._handle_key(curses.KEY_NPAGE)
        assert storage_dashboard._scroll_y == 5


class _FakePad:
    """Stand-in for curses.newpad() — curses.newpad itself requires initscr."""
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols


class TestPadLifecycle:
    def test_pad_reallocation_on_width_change(self, dashboard, monkeypatch):
        """Terminal resize must rebuild the pad so horizontal layout matches."""
        monkeypatch.setattr(curses, "newpad", lambda r, c: _FakePad(r, c))
        # First allocation
        dashboard._ensure_pad(24, 120)
        first_pad = dashboard._pad
        assert first_pad is not None
        assert dashboard._pad_cols == 120

        # Same size — must reuse
        dashboard._ensure_pad(24, 120)
        assert dashboard._pad is first_pad

        # Width change — must reallocate
        dashboard._ensure_pad(24, 80)
        assert dashboard._pad is not first_pad
        assert dashboard._pad_cols == 80

    def test_pad_oversized_for_content(self, dashboard, monkeypatch):
        """Pad must be tall enough to hold max realistic content."""
        monkeypatch.setattr(curses, "newpad", lambda r, c: _FakePad(r, c))
        dashboard._ensure_pad(24, 120)
        assert dashboard._pad_rows >= 200


class TestKeyResizeUnchanged:
    def test_key_resize_still_clears_screen(self, dashboard):
        """The ghost-cell fix on resize must survive the refactor."""
        stdscr_clear = mock.Mock()
        stdscr_clearok = mock.Mock()
        dashboard.stdscr = mock.Mock()
        dashboard.stdscr.clear = stdscr_clear
        dashboard.stdscr.clearok = stdscr_clearok
        dashboard._handle_key(curses.KEY_RESIZE)
        stdscr_clear.assert_called_once()
        stdscr_clearok.assert_called_once_with(True)


class TestStorageLobby:
    def test_enter_on_storage_field_opens_storage_picker(self, storage_dashboard):
        storage_dashboard.awaiting_start = True
        storage_dashboard._field_cursor = 3  # storage_duts
        storage_dashboard._handle_key(10)
        assert storage_dashboard._storage_picker_open is True

    def test_storage_picker_toggle_and_commit_sets_lobby_value(self, storage_dashboard):
        storage_dashboard.awaiting_start = True
        storage_dashboard._field_cursor = 3
        storage_dashboard._handle_key(10)
        storage_dashboard._handle_key(ord(" "))
        storage_dashboard._handle_key(curses.KEY_DOWN)
        storage_dashboard._handle_key(ord(" "))
        storage_dashboard._handle_key(10)
        assert storage_dashboard._lobby_values["storage_duts"] == "/dev/nvme0n1,/dev/sda"

    def test_storage_profile_start_requires_storage_dut(self, storage_dashboard):
        storage_dashboard.awaiting_start = True
        storage_dashboard._lobby_values["psu_serial"] = "SN-TEST"
        storage_dashboard._handle_key(ord(" "))
        assert storage_dashboard.awaiting_start is True
        assert storage_dashboard.start_hint == "Select at least one storage DUT."

    def test_storage_profile_start_succeeds_with_storage_dut(self, storage_dashboard):
        storage_dashboard.awaiting_start = True
        storage_dashboard._lobby_values["psu_serial"] = "SN-TEST"
        storage_dashboard._lobby_values["storage_duts"] = "/dev/nvme0n1"
        storage_dashboard._handle_key(ord(" "))
        assert storage_dashboard.awaiting_start is False
