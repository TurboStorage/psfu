import os
import signal
import subprocess
from typing import List, Optional


class StressWorker:
    def __init__(self, command: Optional[List[str]], cwd: Optional[str] = None, label: Optional[str] = None) -> None:
        self.command = command
        self.cwd = cwd
        self.process = None  # type: ignore
        self.label = label or (command[0] if command else "idle")
        self._paused: bool = False

    def start(self) -> Optional[str]:
        if not self.command:
            return None
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                # Start in a new process group so we can SIGSTOP/SIGCONT the whole group (all threads)
                preexec_fn=os.setsid,
                cwd=self.cwd,
            )
        except Exception as exc:
            return str(exc)
        self._paused = False
        return None

    def pause(self) -> None:
        """SIGSTOP the process group — instant OS-level freeze.

        Used by the transient pulse phase to cut load without the startup
        latency that respawning introduces. The process stays in memory;
        resume() brings it back within one scheduler tick.
        No-op if the process is not running or already paused.
        """
        if not self.process or self.process.poll() is not None or self._paused:
            return
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGSTOP)
            self._paused = True
        except (ProcessLookupError, OSError):
            # Process/group exited between the poll() check and the kill.
            # Any other exception (e.g. PermissionError from a misconfigured
            # process group) is a real bug and must propagate.
            pass

    def resume(self) -> None:
        """SIGCONT the process group — resumes from SIGSTOP instantly.

        No-op if the process is not running or not paused.
        """
        if not self.process or self.process.poll() is not None or not self._paused:
            return
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGCONT)
            self._paused = False
        except (ProcessLookupError, OSError):
            # Process/group exited while paused. Other exceptions must propagate.
            pass

    def state(self) -> str:
        if not self.command:
            return "{}:off".format(self.label)
        if not self.process:
            return "{}:err".format(self.label)
        code = self.process.poll()
        if code is None:
            return "{}:pausd".format(self.label) if self._paused else "{}:run".format(self.label)
        return "{}:rc={}".format(self.label, code)

    def stop(self) -> None:
        if not self.process:
            return
        if self.process.poll() is not None:
            return
        # If paused, resume first so SIGTERM is delivered — a SIGSTOP'd process
        # will not respond to SIGTERM until it's running again.
        if self._paused:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGCONT)
            except (ProcessLookupError, OSError):
                pass
            self._paused = False
        # SIGTERM the whole process group (gpu-burn spawns children).
        pgid = None
        try:
            pgid = os.getpgid(self.process.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            # Process group gone — try the process directly.
            try:
                self.process.terminate()
            except (ProcessLookupError, OSError):
                return
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # SIGTERM didn't work — SIGKILL the group.
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    self.process.kill()
            except (ProcessLookupError, OSError):
                pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # SIGKILL did not reap. Process is almost certainly in D-state
                # (uninterruptible kernel wait on I/O); no user-space recovery
                # possible. Best-effort WNOHANG sweep to at least avoid leaving
                # a zombie if the kernel reaped after our wait gave up.
                self._reap_nohang()

    def _reap_nohang(self) -> None:
        """Non-blocking waitpid sweep. Safe to call repeatedly; no-op when the
        process has already been reaped or is not running. Prevents the class
        of zombie-accumulation observed on long pulse-heavy runs where SIGKILL
        timing races with the kernel's own reaping."""
        if not self.process:
            return
        try:
            pid, _status = os.waitpid(self.process.pid, os.WNOHANG)
            if pid != 0:
                self.process.returncode = _status
        except (ChildProcessError, OSError):
            # Already reaped by subprocess.Popen or never started properly.
            return
