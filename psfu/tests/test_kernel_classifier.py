"""S6 (v0.17) — kernel pattern classifier fixtures.

Guards the three NVMe patterns added in 0.17:

  nvme_io_timeout        — \\bnvme\\b anchored — must not match SATA/SCSI
  nvme_controller_reset  — \\bnvme\\b anchored — must not match SATA/SCSI
  nvme_aer_error         — \\bnvme\\b anchored — must not match SATA/SCSI

The `\\bnvme\\b` gate is the only thing preventing SATA/SCSI controller
reset / timeout noise from firing a storage FAIL. Fixtures under
`psfu/tests/fixtures/kernel/` cover:

  * One canonical dmesg line per kind (positive match exactly once)
  * `sata_noise.txt` — SATA timeout + SCSI controller reset + generic
    "async event" lines. MUST match zero of the three kinds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from psfu.collectors.kernel import KERNEL_WATCH_COMPILED

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "kernel"


def _match_kinds(text: str) -> list[str]:
    """Return the sorted set of kinds any line of `text` matches."""
    hits: set[str] = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        for compiled, _severity, kind in KERNEL_WATCH_COMPILED:
            if compiled.search(line):
                hits.add(kind)
    return sorted(hits)


@pytest.mark.parametrize(
    "fixture,expected_kind",
    [
        ("nvme_io_timeout.txt", "nvme_io_timeout"),
        ("nvme_controller_reset.txt", "nvme_controller_reset"),
        ("nvme_aer_error.txt", "nvme_aer_error"),
    ],
)
def test_nvme_fixture_matches_expected_kind(fixture: str, expected_kind: str) -> None:
    text = (FIXTURE_DIR / fixture).read_text(encoding="utf-8")
    hits = _match_kinds(text)
    assert expected_kind in hits, (
        "Fixture {} did not match expected kind {!r}. Hits: {}".format(
            fixture, expected_kind, hits
        )
    )


def test_sata_noise_does_not_match_any_nvme_kind() -> None:
    """The `\\bnvme\\b` word-boundary gate in the three NVMe patterns is the
    only thing keeping SATA/SCSI controller resets + timeouts from firing a
    storage FAIL. If this regresses (e.g. someone drops the `\\b` or widens
    the pattern), every bench with a spinning rust boot drive will FAIL
    spuriously under load."""
    text = (FIXTURE_DIR / "sata_noise.txt").read_text(encoding="utf-8")
    hits = _match_kinds(text)
    nvme_hits = [k for k in hits if k.startswith("nvme_")]
    assert not nvme_hits, (
        "SATA/SCSI noise matched NVMe kind(s): {}. The \\bnvme\\b anchor "
        "regressed — S6 storage FAIL will now trigger on boot-drive ATA "
        "timeouts.".format(nvme_hits)
    )
