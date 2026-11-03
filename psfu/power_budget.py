"""Bench power headroom check.

Prevents the undersized-PSU class of failure: a PSU-under-test being asked to power
a stress bench whose sustained + transient demand exceeds its rating.

The operator supplies the PSU rated wattage. This module detects the bench
CPU TDP and GPU TGP, applies a transient multiplier for GPUs documented to
spike (Ada/Blackwell consumer cards), adds fixed overhead, and compares.

Called from ``prepare_run`` in lifecycle.py BEFORE any stress phase runs.
Warn-only: an undersized PSU is reported and recorded, never blocked;
``--allow-undersized-psu`` is accepted for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from psfu.common import command_exists, run_capture
from psfu.hardware import nvidia_gpu_max_power_watts


# Fixed bench overhead: motherboard VRM losses, DIMMs, drives, fans, pumps,
# AIC cards. 150W is conservative for a HEDT / workstation bench.
BENCH_OVERHEAD_WATTS: int = 150

# GPUs with documented sub-millisecond transient spikes well above TGP.
# Multiplier is applied to the reported max power limit. Values come from
# published ATX 3.0/3.1 spec (2x TGP for ≥100us) and Igor's Lab / NVIDIA
# transient measurements. Conservative — favors blocking over a false pass.
TRANSIENT_GPU_KEYWORDS: dict[str, float] = {
    "rtx 5090": 1.9,
    "rtx 5080": 1.7,
    "rtx 4090": 1.8,
    "rtx 4080": 1.6,
    "rtx 3090": 1.7,
    "rtx 3080": 1.5,
}

# Conservative CPU TDP fallback when detection fails. Threadripper/EPYC
# territory — intentionally high so a missing detection errors toward
# blocking rather than waving through.
FALLBACK_CPU_TDP_WATTS: int = 350

# Threadripper / EPYC / high-core-count CPUs we want to treat as ≥280W even
# if cpuinfo/lscpu does not expose a TDP field. Keyed by lowercased model
# substring. Extend when a new SKU shows up on a bench.
KNOWN_CPU_TDP: dict[str, int] = {
    "threadripper 7980x":    350,
    "threadripper 7970x":    350,
    "threadripper 7960x":    350,
    "threadripper pro 7995": 350,
    "threadripper pro 7985": 350,
    "threadripper pro 7975": 350,
    "threadripper pro 7965": 350,
    "threadripper pro 7955": 350,
    "threadripper pro 7945": 350,
    "epyc 9654": 360,
    "epyc 9554": 360,
    "epyc 9474": 360,
    "epyc 9454": 290,
    "xeon w9":   350,
    "xeon w7":   270,
    "xeon w5":   270,
}


@dataclass(frozen=True)
class PowerBudget:
    """Result of the bench power budget calculation."""
    cpu_tdp_watts: int
    gpu_max_watts: int
    gpu_transient_multiplier: float
    overhead_watts: int
    sustained_required: int
    transient_required: int
    psu_rated_watts: int
    cpu_source: str          # "known_model" | "lscpu" | "fallback"
    gpu_model: Optional[str]

    def is_adequate(self) -> bool:
        """PSU must cover sustained load. Transient headroom is advisory."""
        return self.psu_rated_watts >= self.sustained_required

    def has_transient_headroom(self) -> bool:
        """PSU has margin for documented GPU transient spikes."""
        return self.psu_rated_watts >= self.transient_required

    def shortfall_watts(self) -> int:
        """How many watts short of sustained (0 if adequate)."""
        return max(0, self.sustained_required - self.psu_rated_watts)


def detect_gpu_model() -> Optional[str]:
    """Primary GPU product name via nvidia-smi. None if no NVIDIA GPU visible."""
    if not command_exists("nvidia-smi"):
        return None
    code, out, _ = run_capture(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        timeout=10,
    )
    if code != 0 or not out:
        return None
    for line in out.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def gpu_transient_multiplier_for(model: Optional[str]) -> float:
    """Transient spike multiplier for a GPU model name. Defaults to 1.3 for
    unknown NVIDIA cards (small transient headroom) and 1.0 for None/iGPU."""
    if not model:
        return 1.0
    lowered = model.lower()
    for keyword, multiplier in TRANSIENT_GPU_KEYWORDS.items():
        if keyword in lowered:
            return multiplier
    return 1.3


def detect_cpu_tdp_watts() -> tuple[int, str]:
    """Best-effort CPU TDP detection. Returns (watts, source).

    Order: match lscpu model string against KNOWN_CPU_TDP; fall back to a
    conservative constant. Never raises — on any failure, returns the
    fallback.
    """
    if not command_exists("lscpu"):
        return FALLBACK_CPU_TDP_WATTS, "fallback"
    code, out, _ = run_capture(["lscpu"], timeout=5)
    if code != 0 or not out:
        return FALLBACK_CPU_TDP_WATTS, "fallback"
    model_line = ""
    for line in out.splitlines():
        if line.lower().startswith("model name"):
            model_line = line.split(":", 1)[1].strip().lower() if ":" in line else ""
            break
    if not model_line:
        return FALLBACK_CPU_TDP_WATTS, "fallback"
    for keyword, tdp in KNOWN_CPU_TDP.items():
        if keyword in model_line:
            return tdp, "known_model"
    return FALLBACK_CPU_TDP_WATTS, "fallback"


def compute_power_budget(psu_rated_watts: int) -> PowerBudget:
    """Compute required bench wattage and compare against PSU-under-test rating.

    Does NOT raise on zero / unknown wattage — callers check
    ``psu_rated_watts`` and decide whether to block. The check is advisory
    at profile level and enforced at prepare_run level.
    """
    cpu_tdp, cpu_source = detect_cpu_tdp_watts()
    gpu_max = nvidia_gpu_max_power_watts() or 0
    gpu_model = detect_gpu_model()
    gpu_mult = gpu_transient_multiplier_for(gpu_model)

    sustained = cpu_tdp + gpu_max + BENCH_OVERHEAD_WATTS
    transient = cpu_tdp + int(round(gpu_max * gpu_mult)) + BENCH_OVERHEAD_WATTS

    return PowerBudget(
        cpu_tdp_watts=cpu_tdp,
        gpu_max_watts=gpu_max,
        gpu_transient_multiplier=gpu_mult,
        overhead_watts=BENCH_OVERHEAD_WATTS,
        sustained_required=sustained,
        transient_required=transient,
        psu_rated_watts=psu_rated_watts,
        cpu_source=cpu_source,
        gpu_model=gpu_model,
    )


def format_budget_report(budget: PowerBudget) -> str:
    """Human-readable one-block report for the preflight output."""
    lines = [
        "Bench power budget:",
        "  CPU TDP:         {}W ({})".format(budget.cpu_tdp_watts, budget.cpu_source),
        "  GPU max power:   {}W ({})".format(
            budget.gpu_max_watts, budget.gpu_model or "none detected"
        ),
        "  Transient x:     {:.1f}".format(budget.gpu_transient_multiplier),
        "  Fixed overhead:  {}W".format(budget.overhead_watts),
        "  Sustained need:  {}W".format(budget.sustained_required),
        "  Transient need:  {}W".format(budget.transient_required),
        "  PSU rated:       {}W".format(budget.psu_rated_watts or 0),
    ]
    if budget.psu_rated_watts == 0:
        lines.append("  Verdict:         UNKNOWN (PSU wattage not entered)")
    elif not budget.is_adequate():
        lines.append(
            "  Verdict:         INSUFFICIENT — short by {}W sustained".format(
                budget.shortfall_watts()
            )
        )
    elif not budget.has_transient_headroom():
        lines.append("  Verdict:         MARGINAL — covers sustained, no transient margin")
    else:
        lines.append("  Verdict:         ADEQUATE")
    return "\n".join(lines)
