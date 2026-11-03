"""Shared test helpers for the PS-FU suite.

Ensures ``psfu`` is importable when pytest is invoked from the repo root, and
provides a ``make_run_db`` factory that returns a pre-populated
``psfu.persistence.db.Database`` with a minimal-but-valid run row, so each test
can just add the samples/events it cares about.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Make the PS-FU source tree importable when running pytest from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from psfu.common import utc_now  # noqa: E402
from psfu.config import PROTOCOL_VERSION  # noqa: E402
from psfu.persistence.db import Database  # noqa: E402


from psfu.synthetic import base_run_metadata as _base_run_metadata  # noqa: E402
from psfu.synthetic import base_sample as _base_sample  # noqa: E402


def make_run_db(
    tmp_path: Path,
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> Database:
    """Return a fresh on-disk Database with one inserted run row.

    ``overrides`` is merged into the default run metadata so tests can set
    exactly the fields their verdict-path cares about (known_good_swap_result,
    post_result, meter_available, etc.).
    """
    db_path = tmp_path / "run.sqlite"
    db = Database(db_path)
    meta = _base_run_metadata()
    if overrides:
        meta.update(overrides)
    db.insert_run(meta)
    return db


__all__ = ["_base_sample", "make_run_db"]
