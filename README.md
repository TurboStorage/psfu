# PS-FU: Platform Stress & Failure Utility

A Python stress and telemetry harness for validating PC hardware (CPU, GPU, RAM, storage, PSU). It drives load, logs sensors at 1 Hz into SQLite, watches the kernel log for hardware errors, and scores the run PASS / WARN / FAIL per subsystem. Every run leaves a hashed evidence bundle you can attach to a manufacturer return claim.

It is a bench tool, not a benchmark. The question it answers is "did this part misbehave under load, and what can I defensibly say about it?"

- About 22k lines of standard-library Python (no required dependencies) plus about 8.6k lines of tests
- Linux is the target for real runs. Demo mode, scoring, reports and the test suite also run on Windows.
- Status: alpha (`0.1.0`). Scoring protocol `0.22` is pre-1.0 and has not been validated against an independent lab. See [Limitations](#limitations).

## What it does

1. Runs a profile (a JSON list of phases) against CPU, GPU, RAM, storage or the whole system, using `stress-ng`, `gpu-burn`, `memtester`-style workers and `fio`.
2. Samples about 140 columns at 1 Hz: NVML, `sensors`, turbostat/RAPL, `/proc`, EDAC counters, PCIe AER counters, NVMe SMART, OCP vendor logs, Corsair PSU telemetry, NUT UPS state.
3. Watches the kernel log for MCE, AER, NVMe timeouts/resets, Xid, thermal, brownout and similar patterns, table-driven in `collectors/kernel.py`.
4. Scores the run against subsystem rules (power, thermal, memory, storage, pcie) and reports what scope of claim the evidence supports.
5. Writes an evidence bundle: `run.sqlite`, `samples.csv`, `events.csv`, `verdict.json`, `summary.md`, charts and a `bundle.sha256` over the key files.

Notable analysis:

- **NVMe write-cliff detection.** 10 s windows over fio bandwidth intervals, cliff ratio vs peak, changepoint counting, optional per-model baseline.
- **12V rail statistical process control.** Western Electric run-rules and CUSUM catch slow sag that stays under the fixed 4% / 8% drift bands.
- **NVMe OCP health logs** (0xC0 counters, 0xC3 latency monitor) and ATA SMART deltas, taken at start and end of a run.
- **Claim scoping.** Without a multimeter reading the verdict is labelled `software-only`, and the conclusion text refuses to claim rail compliance.

## Install

```bash
git clone <repo-url> psfu && cd psfu
python -m pip install -e .            # standard library only
python -m pip install -e ".[gpu,charts]"   # optional: pynvml, matplotlib
```

Python 3.10 or newer. On Windows the TUI needs `windows-curses` (installed automatically as a dependency there).

Real stress runs also need external tools. PS-FU does not vendor them; `bootstrap` builds or locates them under the tools directory:

| Tool | Used for | Notes |
|---|---|---|
| `stress-ng` | CPU / memory load | built from source by `psfu bootstrap` |
| `fio` | storage load | built from source |
| `nvme-cli`, `smartmontools` | NVMe / SATA health | built or system packages |
| `gpu-burn` | GPU load | GPL-2.0; cloned and built locally, needs CUDA. `scripts/install-cuda.sh` installs CUDA 12.8 on Ubuntu 24.04 |
| `turbostat`, `lm-sensors`, `ipmitool` | CPU / board sensors | system packages |

```bash
python psfu.py preflight    # show what is available
python psfu.py bootstrap    # build what is missing (Linux, needs a compiler)
```

## Quick start

No hardware needed:

```bash
python psfu.py demo --scenario warn    # pass | warn | fail
python psfu.py list
python psfu.py chart ~/psfu-data/runs/<run-folder>
```

On a bench machine:

```bash
python psfu.py run --profile smoke-v1 --dry-run --no-tui   # 30 s collector check
python psfu.py run --profile system-full-v1 --psu-wattage 1000 --psu-serial <serial>
python psfu.py verdict <run-folder>     # re-score under current rules
python psfu.py rescore --all            # replay every stored run, report deltas
```

### Where output goes

Precedence is flag, then environment variable, then default.

| Location | Flag | Env var | Default |
|---|---|---|---|
| Data root (parent of the rest, plus logs) | `--data-dir` | `PSFU_HOME` | `~/psfu-data` |
| Run bundles | `--runs-dir` | `PSFU_RUNS_DIR` | `<data root>/runs` |
| Baselines | `--baselines-dir` | `PSFU_BASELINES_DIR` | `<data root>/baselines` |
| Built tools | `--tools-dir` | `PSFU_TOOLS_DIR` | `<data root>/tools` |
| Profiles | `--profiles-dir` | `PSFU_PROFILES_DIR` | bundled `psfu/profiles/` |

Global flags go before the subcommand: `psfu --data-dir /srv/bench run --profile psu-v1`.

## Sample output

Everything below comes from `psfu demo`, which generates a seeded synthetic run (fake sensor data, no hardware). The full sample bundle is in [`docs/sample-output/`](docs/sample-output/).

```text
$ python psfu.py demo --scenario fail
SYNTHETIC DATA -- no hardware was exercised.
Verdict:  FAIL
Scope:    software-only
  power    FAIL
  thermal  PASS
  memory   PASS
  storage  FAIL
  pcie     FAIL
- [power] 12V drift exceeded 8% from soak baseline (12.10V → 10.89V).
- [storage] NVMe I/O timeout during run — controller failed to service a queued command within the kernel timeout window. (1 event(s)): nvme nvme0: I/O 123 QID 4 timeout, aborting (synthetic)
- [pcie] PCIe AER corrected errors exceeded 3 in phase 'full-load' (5 in that phase, 5 total). Persistent bus instability — test with known-good PSU.
```

`psfu chart` on the `warn` scenario (excerpt; full text in [`docs/sample-output/chart.txt`](docs/sample-output/chart.txt)):

```text
Verdict: WARN  (claim scope: software-only)

Phases:
  soak           [  30s]  ████████████████
  full-load      [  60s]  ████████████████████████████████
  steady         [  60s]  ████████████████████████████████
  transient      [  30s]  ████████████████
  cooldown       [  30s]  ████████████████

  CPU °C  ▁▁▁▁▁▁▁▁▁▃▃▃▃▄▄▄▅▅▅▆▆▆▆▆▇████████▇████████▇███████▇█▁▁▂▁▁▁▁▁  min 51.3  max 84.6
  12V     █████████▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅▅▅▅▅▅▅▅▅▆▅▅▅▆▆▅▅▆▅▅▅▆▅▅▅▅████████  min 11.54  max 12.11
  5V      ▅▄▅▅▄▃▅▇▅▅▄▅▆█▅▅▇▃▄▇▅▅▄▆▅▇▆▅▅▅▇▄▄▃▂▄▃▅▆▆▅▄▇▃▃▄▅▃▄▅▃▄▄▁▂▅▄▅▄▅  min 5.01  max 5.03

Verdict: WARN
  - [power] 12V drift reached warning band (12.10V → 11.54V).
```

A run directory looks like this (real runs add `last_gasp.jsonl`, EDAC and storage-health snapshots, and `hardware.json` with the hostname aliased):

```text
<runs-dir>/2026-01-01_120000_<serial>/
├── run.sqlite          full 1 Hz telemetry and events
├── samples.csv  events.csv
├── verdict.json        overall + per-subsystem status, reasons, evidence, claim scope
├── summary.md          human-readable report (see docs/sample-output/summary.md)
├── rma-summary.md      manufacturer-facing claim text
├── fm-entry.md         draft failure-mode-log entry
├── chart_rails.png  chart_power_temps.png
└── bundle.sha256       sha256 of the files above; detects post-run edits
```

Paths inside the bundle are relative, so bundles can be shared without leaking local directory names.

## Profiles

Shipped in `psfu/profiles/`. Durations are the sum of phase lengths.

| Profile | Duration | Purpose |
|---|---|---|
| `system-full-v1` | ~50 min | baseline, full-load, steady, transient pulse, ramp, cooldown |
| `system-quick-v1` | ~12 min | fast screen |
| `psu-v1` / `psu-v1-stepped` | ~30 / ~25 min | PSU load: sustained, CPU/GPU rail swap, isolation, transient, stepped ramps |
| `cpu-v1` | ~50 min | ALU, FP, cache isolation |
| `gpu-v1`, `gpu-rma-v2` | ~34 / ~32 min | GPU isolation; GPU-only cycling |
| `ram-v1` | ~80 min | pattern tests with EDAC monitoring |
| `storage-v1` | ~38 min | non-destructive read/burst workload |
| `storage-cliff-v1` | ~33 min | **destructive** sequential write for pSLC cliff characterization |
| `storage-destructive-v1` | ~38 min | **destructive** full write, write+verify, mixed R/W |
| `storage-speed-verify-v1` | ~24 h | **destructive** capacity-based write+verify |
| `smoke-v1` | 30 s | collector smoke test |

Destructive profiles write to raw block devices. Four guards must all pass: `--destructive` on the command line, `"destructive": true` in the profile, the operator retyping the run serial, and the target device not being mounted. Read `psfu/storage/destructive_guards.py` before pointing them at anything.

## Architecture

```text
profiles/*.json → cli.py → lifecycle.prepare_run → execute.execute_phases
                                                     │  emit_event(kind, severity, phase, message)
        collectors/ (telemetry, turbostat, edac,     ├─ workers/ (stress-ng, gpu-burn, fio; SIGSTOP/SIGCONT pulsing)
        nvme_*, storage_*, kernel watcher, ups)      └─ operator_io (TuiIO | PlainIO)
                                    ↓
                       persistence/db.py  (SQLite: run / sample / event, versioned migrations)
                                    ↓
                       verdict/rules.py   (VerdictEngine.evaluate → status, reasons, claim scope, subsystems)
                                    ↓
                       reports/exporter.py (verdict.json, summary.md, CSVs, bundle.sha256)
```

Design choices worth knowing:

- **Events are the only path to a verdict.** Collectors and workers emit events; `verdict/rules.py` consumes them. The kernel-pattern table auto-routes new patterns to FAIL/WARN loops, and a test checks every kind has a consumer.
- **The verdict engine is replayable.** Scoring reads only the SQLite file, so `psfu rescore` can re-run any stored run through current rules and report drift. A `PROTOCOL_VERSION` bump accompanies every rule change.
- **Rules key off numeric payloads, not emitter-supplied severity strings.**
- **Sensors degrade to `NULL`, not zero.** A missing sensor lowers a coverage-confidence score instead of passing silently.
- More detail in [ARCHITECTURE.md](ARCHITECTURE.md).

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

510 tests pass on Windows and Linux; one symlink test skips on Windows without the symlink privilege. Hardware access is stubbed, so the suite exercises parsing, scoring and report logic, not real devices. Golden tests run against the seeded generator in `psfu/synthetic.py`.

## Limitations

Please read these before trusting a verdict.

- **Not independently validated.** The scoring protocol is `0.22`, pre-1.0. Thresholds (for example the 4% / 8% 12V bands) are engineering judgement tuned on a small number of benches, not derived from a standard.
- **Software sensors are not a multimeter.** 12V readings come from board sensors or a PSU's own telemetry. Ripple, transient response, absolute rail compliance and OCP trip points are out of scope; a run without a manual reading is labelled `software-only`.
- **Linux only for real runs.** Collectors read `/proc`, `/sys`, `dmesg`/journal and vendor CLIs. On Windows you get demo mode, re-scoring, reports and tests.
- **Hardware coverage is narrow.** It was developed against a small set of platforms (one Threadripper + GeForce bench, one Ryzen + GeForce dev machine, a few NVMe and SATA drives). Board-specific sensor work (it87 / nct6687d bring-up) is specific to one motherboard family.
- **NVIDIA-only GPU path** (NVML, `gpu-burn`). No AMD or Intel GPU support.
- **Destructive storage profiles are dangerous.** Guards exist and are tested, but they are guards on a tool that overwrites drives.
- **Large modules.** `ui/tui.py` and `collectors/telemetry.py` are each several thousand lines and only partly covered by tests. `execute.py` is a long orchestration loop.
- **No unattended mode.** An operator is expected to be present.
- **Power budget gate is heuristic.** It compares PSU nameplate to detected CPU TDP + GPU TGP + overhead and warns (and records it in the run) when the PSU looks undersized, but never blocks the run; `--allow-undersized-psu` is accepted for compatibility.

## Extending

| Change | Where |
|---|---|
| New kernel fault pattern | `collectors/kernel.py` `KERNEL_WATCH_PATTERNS` and a reason in `verdict/rules.py` |
| New verdict rule | `verdict/rules.py`, a test in `test_verdict.py`, bump `PROTOCOL_VERSION` |
| New telemetry column | collector in `collectors/telemetry.py` plus a migration in `persistence/db.py` |
| New stress mode | `DutRunner` subclass in `runners/phase_runners.py` plus a profile |

## License

MIT. See [LICENSE](LICENSE). External tools that PS-FU invokes (`stress-ng`, `fio`, `gpu-burn`, `nvme-cli`, `smartmontools`) carry their own licenses and are not distributed with this repository.
