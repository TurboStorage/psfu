"""Pure-string Unicode rendering helpers (sparklines, progress bars, time fmt).

No curses dependency — safe to import from non-TUI code paths (markdown
reports, CLI subcommands, Windows hosts where _curses is unavailable).
"""

from __future__ import annotations

from typing import List, Optional, Tuple


SPARK_BLOCKS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
BAR_FULL = "\u2588"
BAR_EMPTY = "\u2591"


def render_sparkline(
    values: List[Optional[float]],
    width: int,
    abs_range: Optional[Tuple[float, float]] = None,
) -> str:
    """Render a list of numeric samples as a width-N Unicode sparkline.

    If *abs_range* is provided as (lo, hi), bars are scaled against that
    fixed range so the height has consistent meaning across redraws. Without
    it, bars are scaled relative to the min/max of the current data window
    (useful for post-hoc charts where the full run context is available).
    """
    if width <= 0:
        return ""
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return " " * width
    if len(values) >= width:
        step = len(values) / float(width)
        sampled = [values[min(len(values) - 1, int(i * step))] for i in range(width)]
    else:
        sampled = [None] * (width - len(values)) + list(values)

    if abs_range is not None:
        r_lo, r_hi = abs_range
        span = r_hi - r_lo if r_hi > r_lo else 1.0
    else:
        r_lo = min(cleaned)
        r_hi = max(cleaned)
        span = r_hi - r_lo if r_hi > r_lo else 1.0

    out: List[str] = []
    for v in sampled:
        if v is None:
            out.append(" ")
            continue
        idx = int(round((max(r_lo, min(r_hi, v)) - r_lo) / span * 7))
        idx = max(0, min(7, idx))
        out.append(SPARK_BLOCKS[idx])
    return "".join(out)


def render_progress_bar(current: float, total: float, width: int) -> str:
    """Render a Unicode progress bar. Clamped to [0, width]."""
    if width <= 0:
        return ""
    if total <= 0:
        return BAR_EMPTY * width
    ratio = max(0.0, min(1.0, float(current) / float(total)))
    filled = int(round(ratio * width))
    return BAR_FULL * filled + BAR_EMPTY * (width - filled)


def fmt_mmss(seconds: float) -> str:
    secs = max(0, int(seconds))
    return "{:02d}:{:02d}".format(secs // 60, secs % 60)


def resample(values: List[Optional[float]], width: int) -> List[Optional[float]]:
    """Downsample or left-pad a value list to exactly *width* entries."""
    if width <= 0:
        return []
    if len(values) >= width:
        step = len(values) / float(width)
        return [values[min(len(values) - 1, int(i * step))] for i in range(width)]
    return [None] * (width - len(values)) + list(values)
