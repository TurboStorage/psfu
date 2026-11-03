"""S4 phase_runners tests: fio-backed storage runner lifecycle and pulse paths."""
from __future__ import annotations

from pathlib import Path
from typing import List

from psfu.runners.phase_runners import FioDutRunner, FioPulseRunner, WorkerLike
from psfu.runners.workers.fio_runner import FioRunner


def test_fio_dut_runner_prepare_emits_warning_on_start_failure(tmp_path) -> None:
    """If the FioRunner refuses to start (missing jobfile, missing binary,
    or non-readonly without --destructive), prepare() must surface a
    ``storage_worker_start`` warn event rather than silently swallow it."""
    missing_job = tmp_path / "does-not-exist.fio"
    fio = FioRunner(
        missing_job,
        device="/dev/nvme0n1",
        phase_name="steady",
        run_dir=tmp_path,
        binary="/usr/bin/fio",
    )

    events: List[dict] = []

    def emit(severity, phase, kind, message, *, tui_text=None):  # type: ignore[no-untyped-def]
        events.append(
            {
                "severity": severity,
                "phase": phase,
                "kind": kind,
                "message": message,
                "tui_text": tui_text,
            }
        )

    runner = FioDutRunner(fio, emit_event=emit, phase_name="steady")
    runner.prepare(0.0)

    # The worker is still the FioRunner we handed in (not replaced).
    assert runner.worker is fio
    # Exactly one storage_worker_start warn event was emitted.
    warns = [e for e in events if e["kind"] == "storage_worker_start"]
    assert len(warns) == 1
    assert warns[0]["severity"] == "warn"
    assert warns[0]["phase"] == "steady"


def test_fio_dut_runner_label_is_storage(tmp_path) -> None:
    fio = FioRunner(
        tmp_path / "job.fio",
        device="/dev/nvme0n1",
        phase_name="steady",
        run_dir=tmp_path,
        binary="/usr/bin/fio",
    )
    runner = FioDutRunner(fio, emit_event=lambda *a, **k: None, phase_name="steady")
    assert runner.label == "storage"


def test_fio_dut_runner_current_cmd_identifies_jobfile_and_device(tmp_path) -> None:
    """``current_cmd`` isn't the exact fio argv (that lives inside
    FioRunner.build_command), but it must include enough to tell
    ``check_worker_exit`` which device and jobfile blew up."""
    job = tmp_path / "some.fio"
    fio = FioRunner(
        job,
        device="/dev/nvme0n1",
        phase_name="steady",
        run_dir=tmp_path,
        binary="/usr/bin/fio",
    )
    runner = FioDutRunner(fio, emit_event=lambda *a, **k: None, phase_name="steady")
    cmd = runner.current_cmd()
    assert cmd is not None
    assert str(job) in cmd
    assert any("/dev/nvme0n1" in token for token in cmd)


def test_worker_like_protocol_is_satisfied_by_fio_runner(tmp_path) -> None:
    """FioRunner must duck-type into the WorkerLike Protocol; otherwise the
    DutRunner.worker attribute widening is a lie. Uses runtime isinstance()
    against a runtime-checkable Protocol — Python's typing module doesn't
    check this by default, so we exercise each attribute explicitly."""
    fio = FioRunner(
        tmp_path / "job.fio",
        device="/dev/nvme0n1",
        phase_name="steady",
        run_dir=tmp_path,
        binary="/usr/bin/fio",
    )
    # Every WorkerLike attribute must be callable on a real FioRunner.
    for name in ("start", "pause", "resume", "stop", "state"):
        assert callable(getattr(fio, name)), (
            "FioRunner is missing WorkerLike method {}".format(name)
        )


def test_fio_dut_runner_teardown_stops_worker(tmp_path) -> None:
    """DutRunner.teardown() must call worker.stop() — FioDutRunner inherits
    this and the test pins it, because fio needs stop() to parse its JSON
    output (``_parse_output`` in ``FioRunner.stop``)."""

    class _StubFio:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False
            self.paused = False
            self.resumed = False
            self.job_file = Path("/tmp/job.fio")
            self.device = "/dev/nvme0n1"
            self.summary_rows: list = []

        def start(self):
            self.started = True
            return None

        def pause(self):
            self.paused = True

        def resume(self):
            self.resumed = True

        def stop(self):
            self.stopped = True

        def state(self) -> str:
            return "storage:run"

    stub = _StubFio()
    runner = FioDutRunner(stub, emit_event=lambda *a, **k: None, phase_name="steady")  # type: ignore[arg-type]
    runner.prepare(0.0)
    assert stub.started is True
    runner.pause()
    assert stub.paused is True
    runner.resume(0.5)
    assert stub.resumed is True
    runner.teardown()
    assert stub.stopped is True


class _StatusStub:
    def __init__(self) -> None:
        self.extra = ""
        self.notes: List[str] = []

    def set_phase_extra(self, text: str) -> None:
        self.extra = text

    def scrollback_note(self, text: str) -> None:
        self.notes.append(text)


def test_fio_pulse_runner_toggles_pause_resume_and_emits_pulse_cycle(tmp_path) -> None:
    class _StubFio:
        def __init__(self) -> None:
            self.started = False
            self.paused = False
            self.resumed = False
            self.pause_count = 0
            self.resume_count = 0
            self.stopped = False
            self.job_file = Path("/tmp/job.fio")
            self.device = "/dev/nvme0n1"
            self.run_dir = tmp_path
            self.summary_rows: list = []
            self.interval_rows: list = []
            self.output_dir = tmp_path / "storage" / "fio" / "burst" / "job-nvme0n1"
            self.output_dir.mkdir(parents=True, exist_ok=True)

        def start(self):
            self.started = True
            return None

        def pause(self):
            self.paused = True
            self.pause_count += 1

        def resume(self):
            self.resumed = True
            self.resume_count += 1

        def stop(self):
            self.stopped = True

        def state(self) -> str:
            return "storage:run"

    events: List[dict] = []

    def emit(severity, phase, kind, message, *, tui_text=None):  # type: ignore[no-untyped-def]
        events.append({"severity": severity, "phase": phase, "kind": kind, "message": message})

    status = _StatusStub()
    fio = _StubFio()
    runner = FioPulseRunner(  # type: ignore[arg-type]
        fio,
        on_sec=2.0,
        off_sec=2.0,
        inverted=False,
        emit_event=emit,
        status=status,
        phase_name="burst",
    )
    runner.prepare(0.0)
    assert fio.started is True
    runner.tick(2.0, 2.0)
    assert fio.pause_count == 1
    assert events[-1]["kind"] == "pulse_cycle"
    runner.tick(4.0, 4.0)
    assert fio.resume_count == 1
    assert events[-1]["kind"] == "pulse_cycle"
    runner.teardown()
    assert fio.stopped is True
