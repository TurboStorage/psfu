"""EDAC (Error Detection and Correction) sysfs counter snapshots.

On ECC-equipped systems the kernel's EDAC subsystem tracks correctable (CE)
and uncorrectable (UE) memory errors per memory controller and per chip-select
row (which maps roughly to a DIMM slot).  A CE is not "fine because ECC
caught it" — it means the DIMM is degrading.  Any CE delta during a stress
run is at minimum a WARN.

Sysfs tree (when available):

    /sys/devices/system/edac/mc/
    ├── mc0/
    │   ├── ce_count
    │   ├── ue_count
    │   ├── csrow0/
    │   │   ├── ce_count
    │   │   └── ue_count
    │   ├── csrow1/ ...
    ├── mc1/ ...

On systems without ECC (or where the kernel driver doesn't expose EDAC),
the snapshot returns ``{"available": False}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

EDAC_MC_ROOT = Path("/sys/devices/system/edac/mc")


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _snapshot_csrows(mc_dir: Path) -> Dict[str, Dict[str, Optional[int]]]:
    rows: Dict[str, Dict[str, Optional[int]]] = {}
    for child in sorted(mc_dir.iterdir()):
        if child.is_dir() and child.name.startswith("csrow"):
            rows[child.name] = {
                "ce_count": _read_int(child / "ce_count"),
                "ue_count": _read_int(child / "ue_count"),
            }
    return rows


def snapshot_edac() -> Dict[str, Any]:
    """Take a point-in-time snapshot of all EDAC counters.

    Returns a dict like::

        {
            "available": True,
            "controllers": {
                "mc0": {
                    "ce_count": 0,
                    "ue_count": 0,
                    "csrows": {
                        "csrow0": {"ce_count": 0, "ue_count": 0},
                        ...
                    }
                },
                ...
            },
            "total_ce": 0,
            "total_ue": 0,
        }
    """
    if not EDAC_MC_ROOT.is_dir():
        return {"available": False, "controllers": {}, "total_ce": 0, "total_ue": 0}

    controllers: Dict[str, Dict[str, Any]] = {}
    total_ce = 0
    total_ue = 0
    found_any = False

    for child in sorted(EDAC_MC_ROOT.iterdir()):
        if not child.is_dir() or not child.name.startswith("mc"):
            continue
        ce = _read_int(child / "ce_count")
        ue = _read_int(child / "ue_count")
        if ce is None and ue is None:
            continue
        found_any = True
        ce = ce or 0
        ue = ue or 0
        total_ce += ce
        total_ue += ue
        controllers[child.name] = {
            "ce_count": ce,
            "ue_count": ue,
            "csrows": _snapshot_csrows(child),
        }

    return {
        "available": found_any,
        "controllers": controllers,
        "total_ce": total_ce,
        "total_ue": total_ue,
    }


def edac_delta(
    start: Dict[str, Any],
    end: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute the difference between two EDAC snapshots.

    Returns::

        {
            "available": True,
            "delta_ce": 0,
            "delta_ue": 0,
            "per_controller": {
                "mc0": {"delta_ce": 0, "delta_ue": 0, "csrows": {...}},
                ...
            },
        }
    """
    if not start.get("available") or not end.get("available"):
        return {"available": False, "delta_ce": 0, "delta_ue": 0, "per_controller": {}}

    delta_ce = end.get("total_ce", 0) - start.get("total_ce", 0)
    delta_ue = end.get("total_ue", 0) - start.get("total_ue", 0)

    per_mc: Dict[str, Dict[str, Any]] = {}
    for mc_name in end.get("controllers", {}):
        e = end["controllers"][mc_name]
        s = start.get("controllers", {}).get(mc_name, {"ce_count": 0, "ue_count": 0, "csrows": {}})
        mc_dce = e["ce_count"] - s.get("ce_count", 0)
        mc_due = e["ue_count"] - s.get("ue_count", 0)

        csrow_deltas: Dict[str, Dict[str, int]] = {}
        for row_name, row_end in e.get("csrows", {}).items():
            row_start = s.get("csrows", {}).get(row_name, {"ce_count": 0, "ue_count": 0})
            csrow_deltas[row_name] = {
                "delta_ce": (row_end.get("ce_count") or 0) - (row_start.get("ce_count") or 0),
                "delta_ue": (row_end.get("ue_count") or 0) - (row_start.get("ue_count") or 0),
            }

        per_mc[mc_name] = {
            "delta_ce": mc_dce,
            "delta_ue": mc_due,
            "csrows": csrow_deltas,
        }

    return {
        "available": True,
        "delta_ce": delta_ce,
        "delta_ue": delta_ue,
        "per_controller": per_mc,
    }


def edac_events_from_delta(
    delta: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Convert an EDAC delta into verdict-ready event dicts.

    Returns a list of ``{"severity": ..., "kind": ..., "message": ...}``
    entries suitable for ``db.add_event()``."""
    if not delta.get("available"):
        return []

    events: List[Dict[str, str]] = []

    if delta["delta_ue"] > 0:
        events.append({
            "severity": "crit",
            "kind": "edac_ue",
            "message": (
                "EDAC uncorrectable memory error(s) detected during run: "
                "{} UE across all controllers. "
                "DIMM is producing errors that ECC cannot correct — RMA.".format(
                    delta["delta_ue"]
                )
            ),
        })

    if delta["delta_ce"] > 0:
        # Identify which slot(s)
        slots: List[str] = []
        for mc_name, mc_data in delta.get("per_controller", {}).items():
            for row_name, row_data in mc_data.get("csrows", {}).items():
                if row_data.get("delta_ce", 0) > 0:
                    slots.append("{}/{}(+{})".format(mc_name, row_name, row_data["delta_ce"]))
        slot_detail = " [{}]".format(", ".join(slots)) if slots else ""
        events.append({
            "severity": "warn",
            "kind": "edac_ce",
            "message": (
                "EDAC correctable memory error(s) detected during run: "
                "{} CE total.{} "
                "DIMM is degrading — ECC is masking errors now, "
                "but this rate will increase. Flag for monitoring or RMA.".format(
                    delta["delta_ce"], slot_detail
                )
            ),
        })

    return events
