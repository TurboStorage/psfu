"""turbostat_sudo_path: the sudoers rule, probe and collector share one path."""

import os
import stat

from psfu import common


class _St:
    def __init__(self, uid=0, mode=0o755):
        self.st_uid = uid
        self.st_mode = stat.S_IFREG | mode


def _fake(monkeypatch, files, stats=None):
    stats = stats or {}
    monkeypatch.setattr(common.os.path, "isfile", lambda p: p in files)
    monkeypatch.setattr(common.os, "stat", lambda p: stats.get(str(p), _St()))


def test_prefers_usr_local_bin_over_usr_bin(monkeypatch):
    # sudo's secure_path puts /usr/local/bin before /usr/bin
    _fake(monkeypatch, {"/usr/local/bin/turbostat", "/usr/bin/turbostat"})
    assert common.turbostat_sudo_path() == "/usr/local/bin/turbostat"


def test_falls_back_to_distro_path(monkeypatch):
    _fake(monkeypatch, {"/usr/sbin/turbostat"})
    assert common.turbostat_sudo_path() == "/usr/sbin/turbostat"


def test_none_found(monkeypatch):
    _fake(monkeypatch, set())
    assert common.turbostat_sudo_path() == ""


def test_refuses_user_owned_binary(monkeypatch):
    _fake(monkeypatch, {"/usr/local/bin/turbostat", "/usr/bin/turbostat"},
          {"/usr/local/bin/turbostat": _St(uid=1000)})
    # never skips to /usr/bin: sudo would still run the shadowing binary
    assert common.turbostat_sudo_path() == ""


def test_refuses_group_writable_folder(monkeypatch):
    _fake(monkeypatch, {"/usr/local/bin/turbostat"},
          {"/usr/local/bin": _St(mode=0o775)})
    assert common.turbostat_sudo_path() == ""
