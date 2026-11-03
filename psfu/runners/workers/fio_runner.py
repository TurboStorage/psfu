from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from psfu.hardware import fio_binary_path
from psfu.storage.cliff import extract_fio_interval_row


def _job_is_readonly(job_file: Path) -> bool:
    try:
        lines = job_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    normalized = [line.strip().lower() for line in lines if line.strip() and not line.strip().startswith(("#", ";"))]
    if any(line == "readonly" or line == "readonly=1" for line in normalized):
        return True
    return False


def _pct(metrics: Dict[str, Any], key: str) -> Optional[float]:
    percentile_sources = [metrics]
    for nested_key in ("lat_ns", "clat_ns", "clat_us", "lat_us"):
        nested = metrics.get(nested_key)
        if isinstance(nested, dict):
            percentile_sources.append(nested)
    candidates = [
        key,
        "{}00000".format(key),
        "{:.6f}".format(float(key)),
    ]
    for source in percentile_sources:
        percentiles = source.get("percentile")
        if not isinstance(percentiles, dict):
            continue
        for candidate in candidates:
            value = percentiles.get(candidate)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _max_or_none(values: List[Optional[float]]) -> Optional[float]:
    cleaned = [float(value) for value in values if isinstance(value, (int, float))]
    return max(cleaned) if cleaned else None


def extract_interval_display_metrics(payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        return {}
    read_p99_ns: List[Optional[float]] = []
    write_p99_ns: List[Optional[float]] = []
    read_p99_999_ns: List[Optional[float]] = []
    write_p99_999_ns: List[Optional[float]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        read = job.get("read") if isinstance(job.get("read"), dict) else {}
        write = job.get("write") if isinstance(job.get("write"), dict) else {}
        read_p99_ns.append(_pct(read, "99.000000"))
        write_p99_ns.append(_pct(write, "99.000000"))
        read_p99_999_ns.append(_pct(read, "99.999000"))
        write_p99_999_ns.append(_pct(write, "99.999000"))
    return {
        "fio_read_lat_p99_us": (_max_or_none(read_p99_ns) or 0.0) / 1000.0 if _max_or_none(read_p99_ns) is not None else None,
        "fio_write_lat_p99_us": (_max_or_none(write_p99_ns) or 0.0) / 1000.0 if _max_or_none(write_p99_ns) is not None else None,
        "fio_read_lat_p99_999_us": (_max_or_none(read_p99_999_ns) or 0.0) / 1000.0 if _max_or_none(read_p99_999_ns) is not None else None,
        "fio_write_lat_p99_999_us": (_max_or_none(write_p99_999_ns) or 0.0) / 1000.0 if _max_or_none(write_p99_999_ns) is not None else None,
    }


def parse_json_plus_output(payload: Dict[str, Any], *, phase_name: str, device: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for job in payload.get("jobs", []) or []:
        if not isinstance(job, dict):
            continue
        read = job.get("read") if isinstance(job.get("read"), dict) else {}
        write = job.get("write") if isinstance(job.get("write"), dict) else {}
        verify = job.get("verify") if isinstance(job.get("verify"), dict) else {}
        row = {
            "phase_name": phase_name,
            "job_name": str(job.get("jobname") or "fio"),
            "device": device,
            "read_bw_kbps": read.get("bw"),
            "read_iops": read.get("iops"),
            "read_lat_mean_ns": (read.get("lat_ns") or {}).get("mean") if isinstance(read.get("lat_ns"), dict) else None,
            "read_lat_stddev_ns": (read.get("lat_ns") or {}).get("stddev") if isinstance(read.get("lat_ns"), dict) else None,
            "read_lat_p50_ns": _pct(read, "50.000000"),
            "read_lat_p99_ns": _pct(read, "99.000000"),
            "read_lat_p99_9_ns": _pct(read, "99.900000"),
            "read_lat_p99_99_ns": _pct(read, "99.990000"),
            "read_lat_p99_999_ns": _pct(read, "99.999000"),
            "read_lat_max_ns": (read.get("lat_ns") or {}).get("max") if isinstance(read.get("lat_ns"), dict) else None,
            "write_bw_kbps": write.get("bw"),
            "write_iops": write.get("iops"),
            "write_lat_mean_ns": (write.get("lat_ns") or {}).get("mean") if isinstance(write.get("lat_ns"), dict) else None,
            "write_lat_stddev_ns": (write.get("lat_ns") or {}).get("stddev") if isinstance(write.get("lat_ns"), dict) else None,
            "write_lat_p50_ns": _pct(write, "50.000000"),
            "write_lat_p99_ns": _pct(write, "99.000000"),
            "write_lat_p99_9_ns": _pct(write, "99.900000"),
            "write_lat_p99_99_ns": _pct(write, "99.990000"),
            "write_lat_p99_999_ns": _pct(write, "99.999000"),
            "write_lat_max_ns": (write.get("lat_ns") or {}).get("max") if isinstance(write.get("lat_ns"), dict) else None,
            "verify_mismatches": verify.get("errors", 0) if isinstance(verify.get("errors", 0), (int, float)) else 0,
            "io_errors": job.get("error", 0),
            "runtime_ms": job.get("job_runtime"),
        }
        rows.append(row)
    return rows


class FioRunner:
    def __init__(
        self,
        job_file: Path,
        *,
        device: str,
        phase_name: str,
        run_dir: Path,
        destructive: bool = False,
        binary: Optional[str] = None,
        label: str = "storage",
    ) -> None:
        self.job_file = Path(job_file)
        self.device = device
        self.phase_name = phase_name
        self.run_dir = Path(run_dir)
        self.destructive = destructive
        self.binary = binary
        self.label = label
        self.process: Optional[subprocess.Popen[Any]] = None
        self._paused = False
        self.summary_rows: List[Dict[str, Any]] = []
        safe_name = Path(device).name
        self.output_dir = self.run_dir / "storage" / "fio" / self.phase_name / "{}-{}".format(self.job_file.stem, safe_name)
        self.output_json = self.output_dir / "fio.json"
        self.stderr_path = self.output_dir / "fio.stderr"
        # S6 — per-interval stanza parsing via fio `--status-interval=1`.
        # fio streams complete JSON+ snapshots to stdout once per second; the
        # final summary is still written to `self.output_json` via `--output=`
        # so `_parse_output()` at stop() is unchanged. The reader thread owns
        # `_latest_interval` and `_reader_buf`; callers read via the locked
        # `latest_interval()` / `ios_progress_snapshot()` helpers.
        self._lock = threading.Lock()
        self._latest_interval: Optional[Dict[str, Any]] = None
        self.interval_rows: List[Dict[str, Any]] = []
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_buf = ""
        self.interval_json = self.output_dir / "fio-intervals.json"

    def build_command(self) -> List[str]:
        binary = self.binary or fio_binary_path()
        if not binary:
            raise RuntimeError("fio binary not found")
        if not self.job_file.exists():
            raise RuntimeError("fio job file missing: {}".format(self.job_file))
        if not self.destructive and not _job_is_readonly(self.job_file):
            raise RuntimeError("refusing to run non-readonly fio job without destructive mode")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return [
            binary,
            "--filename={}".format(self.device),
            "--output-format=json+",
            "--output={}".format(self.output_json),
            "--status-interval=1",
            "--eta=never",
            str(self.job_file),
        ]

    def start(self) -> Optional[str]:
        try:
            cmd = self.build_command()
        except Exception as exc:
            return str(exc)
        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=self.stderr_path.open("w", encoding="utf-8"),
                preexec_fn=os.setsid,
                cwd=str(Path(cmd[0]).parent),
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            return str(exc)
        self._paused = False
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="fio-stanza-reader", daemon=True
        )
        self._reader_thread.start()
        return None

    def _reader_loop(self) -> None:
        """Consume `--status-interval=1` stanzas from fio stdout.

        fio emits a complete JSON+ object once per second (plus a final
        summary stanza at exit). Stanzas can span many lines, so we stream
        into a buffer and use ``json.JSONDecoder.raw_decode`` to extract
        objects as they complete. Latest successfully-parsed stanza is
        stashed under ``self._lock`` for the per-tick health check.
        """
        if not self.process or not self.process.stdout:
            return
        decoder = json.JSONDecoder()
        try:
            for chunk in self.process.stdout:
                self._reader_buf += chunk
                while True:
                    stripped = self._reader_buf.lstrip()
                    if not stripped:
                        self._reader_buf = ""
                        break
                    try:
                        obj, end = decoder.raw_decode(stripped)
                    except ValueError:
                        # Incomplete stanza — wait for more input.
                        self._reader_buf = stripped
                        break
                    if isinstance(obj, dict):
                        interval_row = extract_fio_interval_row(obj)
                        with self._lock:
                            self._latest_interval = obj
                            if interval_row is not None:
                                self.interval_rows.append(interval_row)
                    self._reader_buf = stripped[end:]
        except Exception:
            # Pipe closed, decode drift, or reader-thread interrupt. Leave
            # whatever was captured in `_latest_interval` intact so the health
            # check's last-known progress snapshot is preserved.
            pass

    def latest_interval(self) -> Optional[Dict[str, Any]]:
        """Return the most recently parsed fio stanza (or None). Shallow
        copy not required — callers read nested fields read-only."""
        with self._lock:
            return self._latest_interval

    def ios_progress_snapshot(self) -> Optional[Tuple[int, int]]:
        """Sum ``total_ios`` across all jobs in the latest stanza.

        Returns ``(read_ios, write_ios)`` as cumulative integers. ``None``
        if no stanza has been parsed yet (fio hasn't emitted its first
        status interval) or the stanza lacks ``jobs``. Health check uses
        this to detect controller hangs: two consecutive snapshots with
        the same ``(read, write)`` tuple mean no IO progress for the
        interval between them.
        """
        payload = self.latest_interval()
        if not payload:
            return None
        jobs = payload.get("jobs")
        if not isinstance(jobs, list):
            return None
        total_read = 0
        total_write = 0
        for job in jobs:
            if not isinstance(job, dict):
                continue
            read = job.get("read")
            write = job.get("write")
            if isinstance(read, dict):
                try:
                    total_read += int(read.get("total_ios") or 0)
                except (TypeError, ValueError):
                    pass
            if isinstance(write, dict):
                try:
                    total_write += int(write.get("total_ios") or 0)
                except (TypeError, ValueError):
                    pass
        return total_read, total_write

    def pause(self) -> None:
        if not self.process or self.process.poll() is not None or self._paused:
            return
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGSTOP)
            self._paused = True
        except (ProcessLookupError, OSError):
            pass

    def resume(self) -> None:
        if not self.process or self.process.poll() is not None or not self._paused:
            return
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGCONT)
            self._paused = False
        except (ProcessLookupError, OSError):
            pass

    def state(self) -> str:
        if not self.process:
            return "{}:err".format(self.label)
        code = self.process.poll()
        if code is None:
            return "{}:pausd".format(self.label) if self._paused else "{}:run".format(self.label)
        return "{}:rc={}".format(self.label, code)

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            if self._paused:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGCONT)
                except (ProcessLookupError, OSError):
                    pass
                self._paused = False
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        # Join the reader thread so no stanzas land after stop() returns.
        # Thread is daemon so a 2 s timeout is a safety net, not a contract.
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2)
        self._write_interval_output()
        self._parse_output()

    def _parse_output(self) -> None:
        if not self.output_json.exists():
            return
        try:
            payload = json.loads(self.output_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        self.summary_rows = parse_json_plus_output(payload, phase_name=self.phase_name, device=self.device)

    def _write_interval_output(self) -> None:
        if not self.interval_rows:
            return
        payload = {
            "phase_name": self.phase_name,
            "device": self.device,
            "job_file": str(self.job_file),
            "interval_sec": 1,
            "rows": self.interval_rows,
        }
        try:
            self.interval_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except OSError:
            pass
