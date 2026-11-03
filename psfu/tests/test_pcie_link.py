from __future__ import annotations

from psfu.collectors.pcie_link import (
    PcieLinkCollector,
    _parse_lspci_link_state,
)


def test_parse_lnksta_gen5_x4() -> None:
    payload = _parse_lspci_link_state(
        """
        LnkCap: Port #0, Speed 32GT/s, Width x4, ASPM not supported
        LnkSta: Speed 32GT/s (ok), Width x4 (ok)
        """
    )
    assert payload is not None
    assert payload["cap_gen"] == 5
    assert payload["cap_width"] == 4
    assert payload["current_gen"] == 5
    assert payload["current_width"] == 4


def test_parse_lnksta_downshifted() -> None:
    payload = _parse_lspci_link_state(
        """
        LnkCap: Port #0, Speed 32GT/s, Width x4, ASPM not supported
        LnkSta: Speed 8GT/s (downgraded), Width x2 (downgraded)
        """
    )
    assert payload is not None
    assert payload["cap_gen"] == 5
    assert payload["current_gen"] == 3
    assert payload["cap_width"] == 4
    assert payload["current_width"] == 2


def test_parse_handles_missing_lnksta() -> None:
    payload = _parse_lspci_link_state("Capabilities: [90] Express Endpoint, MSI 00")
    assert payload is None


def test_snapshot_pins_first(monkeypatch) -> None:
    monkeypatch.setattr("psfu.collectors.pcie_link.command_exists", lambda name: name == "lspci")
    monkeypatch.setattr(
        "psfu.collectors.pcie_link._device_bdf",
        lambda device: {"/dev/nvme0n1": "0000:01:00.0", "/dev/nvme1n1": "0000:02:00.0"}[device],
    )

    def _fake_run_capture(cmd, timeout=5):
        if cmd[2] == "0000:01:00.0":
            return 0, "LnkCap: Speed 32GT/s, Width x4\nLnkSta: Speed 32GT/s, Width x4", ""
        return 0, "LnkCap: Speed 32GT/s, Width x4\nLnkSta: Speed 16GT/s, Width x2", ""

    monkeypatch.setattr("psfu.collectors.pcie_link.run_capture", _fake_run_capture)
    collector = PcieLinkCollector(["/dev/nvme0n1", "/dev/nvme1n1"])
    assert collector.start() is None
    snap = collector.snapshot()
    cap = collector.first_capability()
    collector.stop()
    assert snap["nvme_pcie_link_gen"] == 5
    assert snap["nvme_pcie_link_width"] == 4
    assert cap["max_gen"] == 5
    assert cap["max_width"] == 4
