import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any


# Profile JSON can specify extra flags passed to stress-ng and gpu-burn. Both
# tools accept options that break evidence integrity — stress-ng's
# ``--temp-path`` writes anywhere, ``--exec`` runs arbitrary binaries;
# gpu-burn's ``-c`` loads a compare file from an arbitrary path. An
# explicit allowlist is the only honest defense: reject unknown flags at
# profile load, before any worker is spawned. Extend deliberately when a
# new profile actually needs a flag.
_GPU_BURN_FLAG_SPEC: Dict[str, str] = {
    "-tc":   "bool",   # use tensor cores
    "-d":    "bool",   # use doubles instead of singles
    "-stts": "bool",   # tear-down timing summary
    "-m":    "int",    # memory in MB (or -m N%)
    "-i":    "int",    # GPU index
}

_STRESS_NG_FLAG_SPEC: Dict[str, str] = {
    "--cpu-method": "ident",  # alphanumeric method name (fft, matrixprod, ...)
    "--cache":      "int",    # cache level
    "--vm-bytes":   "size",   # e.g. "80%" or "4G"
    "--metrics":    "bool",
    "--cpu-load":   "int",    # 0-100 pct
}

_IDENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SIZE_RE = re.compile(r"^\d+(?:\.\d+)?[KkMmGgTt%]?B?$")


def _validate_flag_list(flags: List[object], spec: Dict[str, str], tool: str) -> List[str]:
    """Validate a flag list against ``spec``. Returns the canonical list (all
    strings) or raises ValueError. Rules:
      - only flags in ``spec`` are permitted
      - "bool" flags stand alone
      - "int"/"ident"/"size" flags must be followed by a validated value
    """
    out: List[str] = []
    i = 0
    while i < len(flags):
        raw = flags[i]
        if not isinstance(raw, str):
            raise ValueError("{}: non-string flag at index {}: {!r}".format(tool, i, raw))
        flag = raw
        kind = spec.get(flag)
        if kind is None:
            raise ValueError(
                "{}: flag {!r} not in allowlist {}".format(tool, flag, sorted(spec))
            )
        if kind == "bool":
            out.append(flag)
            i += 1
            continue
        # value-taking flag
        if i + 1 >= len(flags):
            raise ValueError("{}: flag {!r} expects a value".format(tool, flag))
        val_raw = flags[i + 1]
        if not isinstance(val_raw, (str, int, float)):
            raise ValueError("{}: flag {!r} value must be scalar, got {!r}".format(tool, flag, val_raw))
        val = str(val_raw)
        if kind == "int":
            # permit "80%" too for -m
            if not (val.isdigit() or _SIZE_RE.match(val)):
                raise ValueError("{}: flag {!r} expects integer, got {!r}".format(tool, flag, val))
        elif kind == "ident":
            if not _IDENT_RE.match(val):
                raise ValueError("{}: flag {!r} expects identifier, got {!r}".format(tool, flag, val))
        elif kind == "size":
            if not _SIZE_RE.match(val):
                raise ValueError("{}: flag {!r} expects size literal, got {!r}".format(tool, flag, val))
        out.append(flag)
        out.append(val)
        i += 2
    return out


def _validate_gpu_burn_flags(flags: Optional[List[object]]) -> Optional[List[str]]:
    if flags is None:
        return None
    return _validate_flag_list(list(flags), _GPU_BURN_FLAG_SPEC, "gpu_burn_flags")


def _validate_cpu_stress_flags(flags: Optional[List[object]]) -> Optional[List[str]]:
    if flags is None:
        return None
    return _validate_flag_list(list(flags), _STRESS_NG_FLAG_SPEC, "cpu_stress_flags")


class _Frozen:
    """Mixin that rejects attribute mutation after ``__init__`` completes.
    Callers treating Phase/Profile as typed models silently mutated them;
    this turns the violation into a loud AttributeError at the point of
    the bad write.

    Internal init assignments use ``object.__setattr__`` to bypass the guard.
    Flag completion by setting ``_frozen = True`` at the end of ``__init__``.
    """
    __slots__ = ()

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(
                "{} is frozen; cannot assign to {!r} after construction. "
                "Build a new instance instead of mutating.".format(
                    type(self).__name__, name
                )
            )
        object.__setattr__(self, name, value)


class Phase(_Frozen):
    def __init__(
        self,
        name: str,
        duration_sec: int,
        cpu_mode: str = "idle",
        gpu_mode: str = "idle",
        mem_mode: str = "idle",
        storage_mode: str = "idle",
        notes: str = "",
        cpu_pulse_on_sec: Optional[float] = None,
        cpu_pulse_off_sec: Optional[float] = None,
        gpu_pulse_on_sec: Optional[float] = None,
        gpu_pulse_off_sec: Optional[float] = None,
        mem_pulse_on_sec: Optional[float] = None,
        mem_pulse_off_sec: Optional[float] = None,
        storage_pulse_on_sec: Optional[float] = None,
        storage_pulse_off_sec: Optional[float] = None,
        cpu_load_fraction: float = 1.0,
        gpu_load_fraction: float = 1.0,
        mem_load_fraction: float = 1.0,
        mem_target_pct: Optional[int] = None,
        mem_tool: str = "stress-ng",
        gpu_burn_flags: Optional[List[str]] = None,
        cpu_stress_flags: Optional[List[str]] = None,
        pulse_invert: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.duration_sec = duration_sec
        self.cpu_mode = cpu_mode
        self.gpu_mode = gpu_mode
        self.mem_mode = mem_mode
        self.storage_mode = storage_mode
        self.notes = notes
        self.cpu_pulse_on_sec = cpu_pulse_on_sec
        self.cpu_pulse_off_sec = cpu_pulse_off_sec
        self.gpu_pulse_on_sec = gpu_pulse_on_sec
        self.gpu_pulse_off_sec = gpu_pulse_off_sec
        self.mem_pulse_on_sec = mem_pulse_on_sec
        self.mem_pulse_off_sec = mem_pulse_off_sec
        self.storage_pulse_on_sec = storage_pulse_on_sec
        self.storage_pulse_off_sec = storage_pulse_off_sec
        self.cpu_load_fraction = cpu_load_fraction
        self.gpu_load_fraction = gpu_load_fraction
        self.mem_load_fraction = mem_load_fraction
        self.mem_target_pct = mem_target_pct
        self.mem_tool = mem_tool
        # Tuples so elements themselves can't be mutated after construction.
        self.gpu_burn_flags = tuple(gpu_burn_flags) if gpu_burn_flags is not None else ("-tc",)
        self.cpu_stress_flags = tuple(cpu_stress_flags) if cpu_stress_flags is not None else ()
        self.pulse_invert = frozenset(pulse_invert) if pulse_invert else frozenset()
        self._frozen = True

    def replace(self, **overrides: object) -> "Phase":
        """Return a new Phase with ``overrides`` applied. Frozen-model safe
        substitute for attribute mutation."""
        fields: Dict[str, object] = {
            "name": self.name,
            "duration_sec": self.duration_sec,
            "cpu_mode": self.cpu_mode,
            "gpu_mode": self.gpu_mode,
            "mem_mode": self.mem_mode,
            "storage_mode": self.storage_mode,
            "notes": self.notes,
            "cpu_pulse_on_sec": self.cpu_pulse_on_sec,
            "cpu_pulse_off_sec": self.cpu_pulse_off_sec,
            "gpu_pulse_on_sec": self.gpu_pulse_on_sec,
            "gpu_pulse_off_sec": self.gpu_pulse_off_sec,
            "mem_pulse_on_sec": self.mem_pulse_on_sec,
            "mem_pulse_off_sec": self.mem_pulse_off_sec,
            "storage_pulse_on_sec": self.storage_pulse_on_sec,
            "storage_pulse_off_sec": self.storage_pulse_off_sec,
            "cpu_load_fraction": self.cpu_load_fraction,
            "gpu_load_fraction": self.gpu_load_fraction,
            "mem_load_fraction": self.mem_load_fraction,
            "mem_target_pct": self.mem_target_pct,
            "mem_tool": self.mem_tool,
            "gpu_burn_flags": list(self.gpu_burn_flags),
            "cpu_stress_flags": list(self.cpu_stress_flags),
            "pulse_invert": list(self.pulse_invert),
        }
        unknown = set(overrides) - set(fields)
        if unknown:
            raise TypeError("Phase.replace: unknown fields {}".format(sorted(unknown)))
        fields.update(overrides)
        return Phase(**fields)  # type: ignore[arg-type]


class Profile(_Frozen):
    def __init__(self, payload: Dict[str, object]) -> None:
        self.name = str(payload.get("name", "unknown"))
        self.version = str(payload.get("version", "0.0.0"))
        self.description = str(payload.get("description", ""))
        self.destructive = bool(payload.get("destructive", False))
        # Legacy fields — ignored. Load is controlled by per-phase cpu_load_fraction / gpu_load_fraction.
        # Kept only so old profile JSON files don't fail to load.
        # ``phases`` is a tuple — callers that need to filter must build a new
        # local list (see ``execute_phases`` ``active_phases``).
        self.phases = tuple(
            Phase(
                name=str(item["name"]),
                duration_sec=int(item["duration_sec"]),
                cpu_mode=str(item.get("cpu_mode", "idle")),
                gpu_mode=str(item.get("gpu_mode", "idle")),
                mem_mode=str(item.get("mem_mode", "idle")),
                storage_mode=str(item.get("storage_mode", "idle")),
                notes=str(item.get("notes", "")),
                cpu_pulse_on_sec=item.get("cpu_pulse_on_sec"),
                cpu_pulse_off_sec=item.get("cpu_pulse_off_sec"),
                gpu_pulse_on_sec=item.get("gpu_pulse_on_sec"),
                gpu_pulse_off_sec=item.get("gpu_pulse_off_sec"),
                mem_pulse_on_sec=item.get("mem_pulse_on_sec"),
                mem_pulse_off_sec=item.get("mem_pulse_off_sec"),
                storage_pulse_on_sec=item.get("storage_pulse_on_sec"),
                storage_pulse_off_sec=item.get("storage_pulse_off_sec"),
                cpu_load_fraction=float(item.get("cpu_load_fraction", 1.0)),
                gpu_load_fraction=float(item.get("gpu_load_fraction", 1.0)),
                mem_load_fraction=float(item.get("mem_load_fraction", 1.0)),
                mem_target_pct=(
                    int(item["mem_target_pct"])
                    if item.get("mem_target_pct") is not None
                    else None
                ),
                mem_tool=str(item.get("mem_tool", "stress-ng")),
                gpu_burn_flags=_validate_gpu_burn_flags(item.get("gpu_burn_flags")),
                cpu_stress_flags=_validate_cpu_stress_flags(item.get("cpu_stress_flags")),
                pulse_invert=item.get("pulse_invert"),
            )
            for item in payload["phases"]
        )
        self._frozen = True

    def with_scaled_durations(self, scale: float) -> "Profile":
        """Return a new Profile with every phase's duration_sec multiplied by
        ``scale`` (floor 1s). Returns self if scale is a no-op. Rebuilds phases
        as new Phase instances — the original profile is untouched because it
        may be referenced elsewhere (orchestrator, reports)."""
        if scale <= 0 or abs(scale - 1.0) < 0.001:
            return self
        new_profile = Profile.__new__(Profile)
        object.__setattr__(new_profile, "name", self.name)
        object.__setattr__(new_profile, "version", self.version)
        object.__setattr__(new_profile, "description", self.description)
        object.__setattr__(new_profile, "destructive", self.destructive)
        object.__setattr__(
            new_profile,
            "phases",
            tuple(
                p.replace(duration_sec=max(1, int(p.duration_sec * scale)))
                for p in self.phases
            ),
        )
        object.__setattr__(new_profile, "_frozen", True)
        return new_profile

    def available_duts(self) -> Dict[str, bool]:
        """Return which DUT types have at least one non-idle phase.

        CPU is always available — it's the host, telemetry runs regardless
        of whether the profile drives stress load onto it.
        """
        has = {"cpu": True, "gpu": False, "mem": False, "storage": False}
        for p in self.phases:
            if p.gpu_mode != "idle":
                has["gpu"] = True
            if p.mem_mode != "idle":
                has["mem"] = True
            if p.storage_mode != "idle":
                has["storage"] = True
        return has
