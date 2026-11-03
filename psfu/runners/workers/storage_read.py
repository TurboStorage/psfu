#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List


CHUNK_SIZE = 4 * 1024 * 1024
# Resolved at import time so symlink-laundered paths can't slip past. Using
# string-equality against raw dirpaths (previous behaviour) allowed a caller
# to pass ``/tmp/../etc`` or a symlink like ``/opt/shadow -> /etc/shadow`` and
# still get reads under the blocked roots. ``Path.is_relative_to`` on the
# resolved form is the honest containment check.
_SKIP_DIR_LITERALS = (
    "/proc",
    "/sys",
    "/dev",
    "/run",
    "/tmp",
    "/var/tmp",
    "/etc",          # added — never hash config/credentials
    "/boot",         # added — firmware/kernel images
    "/root",
)


def _resolved_skip_dirs() -> List[Path]:
    out: List[Path] = []
    for raw in _SKIP_DIR_LITERALS:
        try:
            out.append(Path(raw).resolve())
        except Exception:
            continue
    return out


SKIP_DIRS = _resolved_skip_dirs()


def _is_under_skip(resolved: Path) -> bool:
    for skip in SKIP_DIRS:
        try:
            if resolved == skip or resolved.is_relative_to(skip):
                return True
        except AttributeError:
            # is_relative_to added in 3.9; fallback for older interpreters.
            try:
                resolved.relative_to(skip)
                return True
            except ValueError:
                continue
    return False


def iter_files(paths: List[Path]) -> Iterable[Path]:
    for root in paths:
        try:
            root = root.resolve()
        except Exception:
            continue
        if not root.exists():
            continue
        if _is_under_skip(root):
            continue
        if root.is_file():
            yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            try:
                dp_resolved = Path(dirpath).resolve()
            except Exception:
                dirnames[:] = []
                continue
            if _is_under_skip(dp_resolved):
                dirnames[:] = []
                continue
            # Also prune subdirs that would descend into a skipped tree.
            pruned: List[str] = []
            for name in dirnames:
                try:
                    cand = (Path(dirpath) / name).resolve()
                except Exception:
                    continue
                if _is_under_skip(cand):
                    continue
                pruned.append(name)
            dirnames[:] = pruned
            for filename in filenames:
                path = Path(dirpath) / filename
                try:
                    resolved = path.resolve()
                except Exception:
                    continue
                if _is_under_skip(resolved):
                    continue
                try:
                    if resolved.is_file():
                        yield resolved
                except Exception:
                    continue


def hash_file(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    digest.digest()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PS-FU read-only storage validation worker")
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--path", action="append", dest="paths", required=True)
    args = parser.parse_args(argv)

    targets = [Path(p) for p in args.paths]
    files = list(iter_files(targets))
    if not files:
        print("no readable files found", file=sys.stderr)
        return 2

    deadline = time.monotonic() + max(1, args.timeout)
    idx = 0
    while time.monotonic() < deadline:
        path = files[idx % len(files)]
        try:
            hash_file(path)
        except Exception:
            return 3
        idx += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
