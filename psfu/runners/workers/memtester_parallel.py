#!/usr/bin/env python3
"""Parallel memtester wrapper.

Splits the target allocation across N workers so March-pattern tests
complete in reasonable time on large-RAM systems (256GB+).  Each worker
runs memtester on its own chunk — the OS gives each one a distinct
physical page range, so the combined coverage approaches the full
allocation.

This script is invoked as a subprocess by the orchestrator.  It inherits
the process group from StressWorker (via setsid), so SIGSTOP/SIGCONT
from the pulse engine freezes/resumes all children automatically.

Usage:
    python3 memtester_parallel.py --size-mb 200000 --workers 8 --iters 1

Exit code 0 = all workers passed.  Non-zero = at least one worker failed
(memtester exits non-zero on any detected error).
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
from typing import List


def find_memtester() -> str:
    path = shutil.which("memtester")
    if path is None:
        print("memtester not found on PATH", file=sys.stderr)
        sys.exit(1)
    return path


def run(args: argparse.Namespace) -> int:
    memtester = find_memtester()
    workers = max(1, args.workers)
    chunk_mb = max(64, args.size_mb // workers)
    iters = max(1, args.iters)

    procs: List[subprocess.Popen] = []
    for i in range(workers):
        try:
            p = subprocess.Popen(
                [memtester, "{}M".format(chunk_mb), str(iters)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            procs.append(p)
        except Exception as exc:
            print("worker {} failed to start: {}".format(i, exc), file=sys.stderr)

    if not procs:
        print("no workers started", file=sys.stderr)
        return 1

    # Wait for all — if we get SIGTERM the whole group dies (setsid),
    # so this is just the happy path.
    failed = 0
    for i, p in enumerate(procs):
        rc = p.wait()
        if rc != 0:
            failed += 1

    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel memtester wrapper")
    parser.add_argument("--size-mb", type=int, required=True,
                        help="Total RAM to test in MB (split across workers)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel memtester instances (0 = auto)")
    parser.add_argument("--iters", type=int, default=1,
                        help="Iterations per worker")
    args = parser.parse_args()

    if args.workers <= 0:
        # Auto: 1 worker per 32GB, minimum 2, capped at CPU count
        args.workers = max(2, min(os.cpu_count() or 4, args.size_mb // (32 * 1024)))

    sys.exit(run(args))


if __name__ == "__main__":
    main()
