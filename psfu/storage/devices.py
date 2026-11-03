from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from psfu.common import run_capture
from psfu.hardware import nvme_cli_binary_path


_SYS_BLOCK = Path("/sys/block")


@dataclass(frozen=True)
class StorageDUT:
    path: str
    kernel_name: str
    model: str
    vendor: str
    transport: str
    rotational: bool
    mountpoints: List[str]
    role: str
    kind: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def discover_storage_devices(sys_block: Path = _SYS_BLOCK) -> List[StorageDUT]:
    devices: List[StorageDUT] = []
    mounts = _mount_map()
    if not sys_block.exists():
        return devices

    for dev in sorted(sys_block.iterdir(), key=lambda p: p.name):
        name = dev.name
        if not _is_storage_block_device(name):
            continue
        mountpoints = mounts.get(name, [])
        devices.append(
            StorageDUT(
                path="/dev/{}".format(name),
                kernel_name=name,
                model=_read_text(dev / "device" / "model"),
                vendor=_read_text(dev / "device" / "vendor"),
                transport=_transport_for(dev, name),
                rotational=_read_text(dev / "queue" / "rotational") == "1",
                mountpoints=mountpoints,
                role="os_drive" if "/" in mountpoints else ("mounted" if mountpoints else "unknown"),
                kind="nvme" if name.startswith("nvme") else ("sata" if name.startswith("sd") else "block"),
            )
        )
    return devices


def parse_storage_dut_paths(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    values: List[str] = []
    seen = set()
    for part in str(raw).split(","):
        item = part.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        values.append(item)
    return values


def validate_storage_dut_paths(chosen: List[str], inventory: List[StorageDUT]) -> List[str]:
    """Return only DUT paths present in the discovered inventory.

    If inventory discovery returned nothing, preserve the operator-provided
    list as-is so CLI flows still work on hosts where discovery is degraded.
    """
    if not inventory:
        return list(chosen)
    known = {device.path for device in inventory}
    return [path for path in chosen if path in known]


def serialize_storage_inventory(devices: List[StorageDUT]) -> List[Dict[str, Any]]:
    return [device.to_dict() for device in devices]


def capture_nvme_artifacts(run_dir: Path, device_paths: List[str], *, stage: str) -> None:
    nvme_bin = nvme_cli_binary_path()
    if not nvme_bin:
        return
    base_dir = run_dir / "storage" / "nvme"
    base_dir.mkdir(parents=True, exist_ok=True)
    for device in device_paths:
        if not str(device).startswith("/dev/nvme"):
            continue
        dev_name = Path(device).name
        out_dir = base_dir / dev_name
        out_dir.mkdir(parents=True, exist_ok=True)
        _capture_nvme_cmd(out_dir / "{}-id-ctrl.json".format(stage), [nvme_bin, "id-ctrl", device, "-o", "json"])
        _capture_nvme_cmd(out_dir / "{}-id-ns.json".format(stage), [nvme_bin, "id-ns", device, "-o", "json"])
        _capture_nvme_cmd(out_dir / "{}-error-log.json".format(stage), [nvme_bin, "error-log", device, "-o", "json"])
        _capture_nvme_cmd(out_dir / "{}-persistent-event-log.json".format(stage), [nvme_bin, "persistent-event-log", device, "-o", "json"])


def _capture_nvme_cmd(path: Path, cmd: List[str]) -> None:
    code, stdout, stderr = run_capture(cmd, timeout=15)
    payload: Dict[str, Any] = {
        "command": cmd,
        "exit_code": code,
        "stderr": stderr,
    }
    if stdout:
        try:
            payload["payload"] = json.loads(stdout)
        except Exception:
            payload["stdout"] = stdout
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_storage_block_device(name: str) -> bool:
    if name.startswith("loop") or name.startswith("ram"):
        return False
    return name.startswith("nvme") or name.startswith("sd")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _transport_for(dev: Path, kernel_name: str) -> str:
    if kernel_name.startswith("nvme"):
        return "nvme"
    try:
        link = (dev / "device" / "subsystem").resolve()
        if link.name:
            return link.name
    except Exception:
        pass
    return "unknown"


def _mount_map() -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    try:
        data = Path("/proc/mounts").read_text(encoding="utf-8").splitlines()
    except Exception:
        return mapping
    for line in data:
        parts = line.split()
        if len(parts) < 2:
            continue
        src, mountpoint = parts[0], parts[1]
        if not src.startswith("/dev/"):
            continue
        name = Path(src).name
        base = _base_device_name(name)
        mapping.setdefault(base, [])
        if mountpoint not in mapping[base]:
            mapping[base].append(mountpoint)
    return mapping


def _base_device_name(name: str) -> str:
    if name.startswith("nvme") and "p" in name:
        return name.split("p", 1)[0]
    if name.startswith("sd") and len(name) > 3 and name[-1].isdigit():
        return name.rstrip("0123456789")
    return name
