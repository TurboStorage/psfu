from __future__ import annotations

import json
from pathlib import Path

import pytest

from psfu.runtime import get_hardware_snapshot
from psfu.storage.devices import (
    StorageDUT,
    capture_nvme_artifacts,
    discover_storage_devices,
    parse_storage_dut_paths,
    validate_storage_dut_paths,
)


def _make_block_tree(tmp_path: Path, name: str, *, model: str = "", vendor: str = "", rotational: str = "0") -> Path:
    dev = tmp_path / name
    (dev / "device").mkdir(parents=True)
    (dev / "queue").mkdir(parents=True)
    (dev / "device" / "model").write_text(model, encoding="utf-8")
    (dev / "device" / "vendor").write_text(vendor, encoding="utf-8")
    (dev / "queue" / "rotational").write_text(rotational, encoding="utf-8")
    return dev


def test_discover_nvme_controllers(monkeypatch, tmp_path: Path) -> None:
    nvme = _make_block_tree(tmp_path, "nvme0n1", model="SN850X", vendor="WD")
    monkeypatch.setattr("psfu.storage.devices._mount_map", lambda: {"nvme0n1": []})
    devices = discover_storage_devices(tmp_path)
    assert len(devices) == 1
    assert devices[0].path == "/dev/nvme0n1"
    assert devices[0].transport == "nvme"
    assert devices[0].kind == "nvme"
    assert devices[0].model == "SN850X"
    assert nvme.exists()


def test_discover_sata_drives(monkeypatch, tmp_path: Path) -> None:
    dev = _make_block_tree(tmp_path, "sda", model="MX500", vendor="Crucial", rotational="0")
    subsystem = dev / "device" / "subsystem"
    target = tmp_path / "subsys" / "ata"
    target.mkdir(parents=True)
    try:
        subsystem.symlink_to(target)
    except OSError:
        pytest.skip("creating symlinks needs elevated privilege on this OS")
    monkeypatch.setattr("psfu.storage.devices._mount_map", lambda: {"sda": ["/data"]})
    devices = discover_storage_devices(tmp_path)
    assert len(devices) == 1
    assert devices[0].kind == "sata"
    assert devices[0].mountpoints == ["/data"]
    assert devices[0].role == "mounted"


def test_os_drive_flag(monkeypatch, tmp_path: Path) -> None:
    _make_block_tree(tmp_path, "nvme0n1", model="OSDisk", vendor="Samsung")
    monkeypatch.setattr("psfu.storage.devices._mount_map", lambda: {"nvme0n1": ["/"]})
    devices = discover_storage_devices(tmp_path)
    assert devices[0].role == "os_drive"


def test_role_unknown_when_unmounted(monkeypatch, tmp_path: Path) -> None:
    _make_block_tree(tmp_path, "sdb", model="Cold", vendor="Seagate", rotational="1")
    monkeypatch.setattr("psfu.storage.devices._mount_map", lambda: {})
    devices = discover_storage_devices(tmp_path)
    assert devices[0].role == "unknown"
    assert devices[0].rotational is True


def test_parse_storage_dut_paths_dedupes_and_trims() -> None:
    assert parse_storage_dut_paths(" /dev/nvme0n1 , /dev/nvme1n1, /dev/nvme0n1 ") == [
        "/dev/nvme0n1",
        "/dev/nvme1n1",
    ]


def test_capture_nvme_artifacts_writes_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("psfu.storage.devices.nvme_cli_binary_path", lambda: "/usr/bin/nvme")
    monkeypatch.setattr(
        "psfu.storage.devices.run_capture",
        lambda cmd, timeout=15: (0, json.dumps({"ok": cmd[1]}), ""),
    )
    capture_nvme_artifacts(tmp_path, ["/dev/nvme0n1"], stage="start")
    path = tmp_path / "storage" / "nvme" / "nvme0n1" / "start-id-ctrl.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["payload"]["ok"] == "id-ctrl"


def test_validate_storage_dut_paths_filters_unknown_inventory() -> None:
    inventory = [
        StorageDUT("/dev/nvme0n1", "nvme0n1", "SN850X", "WD", "nvme", False, [], "unknown", "nvme"),
        StorageDUT("/dev/sda", "sda", "MX500", "Crucial", "sata", False, [], "unknown", "sata"),
    ]
    assert validate_storage_dut_paths(
        ["/dev/nvme0n1", "/dev/unknown0"],
        inventory,
    ) == ["/dev/nvme0n1"]


def test_hardware_snapshot_includes_storage_inventory(monkeypatch) -> None:
    monkeypatch.setattr("psfu.runtime.command_exists", lambda _: False)
    monkeypatch.setattr(
        "psfu.runtime.discover_storage_devices",
        lambda: [StorageDUT("/dev/nvme0n1", "nvme0n1", "SN850X", "WD", "nvme", False, [], "unknown", "nvme")]
    )
    snapshot = get_hardware_snapshot()
    assert snapshot["storage_inventory"][0]["path"] == "/dev/nvme0n1"
