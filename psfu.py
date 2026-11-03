#!/usr/bin/env python3
"""PS-FU — Platform Stress & Failure Utility

The art of breaking things on purpose.

PS-FU is a stress and telemetry harness for validating PC hardware (CPU, GPU,
RAM, PSU, storage). It drives load, logs sensors at 1 Hz, watches the kernel
log for hardware errors, and scores each run PASS / WARN / FAIL per subsystem.

Thin entrypoint shim. All logic lives under ``psfu/``. Operators run::

    python3 psfu.py run --profile system-full-v1
"""

from __future__ import annotations

import sys

from psfu.cli import main


if __name__ == "__main__":
    sys.exit(main())
