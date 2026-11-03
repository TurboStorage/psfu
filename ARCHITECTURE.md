# PS-FU Architecture — One-Page Map

For a new contributor to get productive in an hour. User-facing docs are in `README.md`; the rest is in the code.

---

## Data flow

```
                              ┌───────────────────┐
                              │    profiles/*.json│
                              └─────────┬─────────┘
                                        │
                                        ▼
          ┌─────────────────────────────────────────────────────────┐
          │                       cli.py                            │
          │   argparse → run_test → prepare_run → execute_phases    │
          └───────────────────┬─────────────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                 runners/execute.py  (execute_phases)                │
 │                                                                     │
 │    emit_event(kind=…, severity=…, phase=…, message=…)               │
 │             │                                                       │
 │   ┌─────────┼─────────────────────────────────────────────┐         │
 │   │         │                                             │         │
 │   ▼         ▼                                             ▼         │
 │ collectors/  workers/ (CPU/GPU/mem/storage stress)   kernel watcher │
 │   telemetry.py             |                          (kernel.py)   │
 │   turbostat.py             ▼                                        │
 │   storage_health.py  StressWorker (SIGSTOP/SIGCONT)                 │
 │   edac.py                                                           │
 └─────────────────────────────┬───────────────────────────────────────┘
                               │  emit_event fans out to:
                               ▼
               ┌───────────────────────────────────┐
               │       persistence/db.py           │
               │   run.sqlite  (run / sample /     │
               │                event tables)      │
               └───────────────┬───────────────────┘
                               │
                               ▼
                  ┌──────────────────────────────┐
                  │      verdict/rules.py        │
                  │  VerdictEngine.evaluate(db)  │
                  └───────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────────┐
                   │      reports/exporter.py        │
                   │   verdict.json + summary.md +   │
                   │   events.csv + samples.csv +    │
                   │   fm-entry.md + bundle.sha256   │
                   └─────────────────────────────────┘
```

The beating heart of PS-FU is `emit_event`. It is a closure defined inside `execute_phases` that fans out to two places: `db.add_event` (evidence record) and `dashboard.log_event` (live TUI). Everything a collector, worker, or health check wants to communicate goes through it. Every `kind` it receives must have a consumer in `verdict/rules.py` (design rule: no event without a consumer).

---

## The 5 files you must read (in order)

1. **`psfu/cli.py`** — entrypoint + subcommands. Shows the command surface (`run`, `report`, `chart`, `verdict`, `recover`, `pdf`, `preflight`) and the `_load_run_db` containment check.
2. **`psfu/runners/execute.py`** — the live phase loop. Owns `emit_event`, the collector tick, worker lifecycle delegation, the kernel watcher, the last-gasp file, and the finally block that stops everything. kept closed for new features — new work goes in a new file (I/O: `operator_io.py`; phase modes: `DutRunner` subclass in `phase_runners.py`).
3. **`psfu/verdict/rules.py`** — the verdict engine. `VerdictEngine.evaluate(db)` reads samples + events and produces `(verdict, reasons, claim_scope, subsystems)`. All silent-wrong-answer risk lives here. Kernel watcher kinds now drive a generic loop via `KERNEL_WATCH_PATTERNS` in `collectors/kernel.py`. FAIL/WARN dispatch keys off `kind` + numeric payload, never an emitter-supplied `severity` string (the `thermal_drift` rule was rewritten to key off its numeric payload for exactly this reason).
4. **`psfu/persistence/db.py`** — SQLite writer. Three tables (`run`, `sample`, `event`) + `schema_version`. EXCLUSIVE locking + `check_same_thread=False` + internal `self._lock` for thread safety. `_MIGRATIONS` list + `CURRENT_SCHEMA_VERSION` constant drive explicit migrations; `fetch_run` asserts version. Must call `.close()` at end of run to release the file lock.
5. **`psfu/collectors/kernel.py`** — kernel log watcher. `KERNEL_WATCH_PATTERNS` is a 5-tuple table (pattern, severity, kind, subsystem, default_verdict). Adding a new fault pattern here auto-scores in `verdict/rules.py` via the generic FAIL/WARN loops.

Everything else is a leaf (telemetry readers, markdown renderers, TUI, worker subclasses, etc.). Read those only when the bug is obviously in that subtree.

---

## Key invariants (don't break these)

- **Every emitted event kind has a verdict consumer.** Enforced by `test_kernel_watcher_kinds_have_verdict_consumers` for kernel kinds; manually for others.
- **`PROTOCOL_VERSION` bumps on any rule change** (see `config.py`). Failure-mode log entries and evidence bundles stamp this version.
- **Models are frozen.** `Phase` and `Profile` use a `__setattr__` guard — you build a new instance, you never mutate.
- **A manual multimeter reading is the reference method.** PS-FU augments it. `defensible_conclusion` in `ui/operator.py` narrows claim scope when meter readings are absent.
- **Run dir is contained under `RUNS_DIR`.** `cli._load_run_db` refuses to operate elsewhere (prevents arbitrary-file read/write via crafted `--run-dir`).
- **Evidence bundle is hashed.** `bundle.sha256` in every run dir. FM log entry records the hash prefix — post-hoc tampering is detectable.

---

## Test suite shape

- **Unit scope, real DB.** Each test spins a temp SQLite file via `make_run_db`. No mocks of the DB.
- **Two focus areas:** `persistence/db.py` (silent data loss) and `verdict/rules.py` (silent wrong answers).
- **Golden tests use synthetic runs.** `psfu/synthetic.py` generates seeded PASS/WARN/FAIL runs; `test_synthetic.py` pins their verdicts.
- **Cross-platform.** The suite runs on Linux and Windows (Windows needs `windows-curses` for the TUI tests; one symlink test skips without privilege).
- **Run the suite:** `python -m pytest`

---

## Where to add new work

| Change | Where |
|---|---|
| New kernel fault pattern | `collectors/kernel.py` `KERNEL_WATCH_PATTERNS` (5-tuple) + reason in `verdict/rules.py` `_KERNEL_KIND_REASONS` |
| New verdict rule | New block in `verdict/rules.py` + test in `test_verdict.py` + bump `PROTOCOL_VERSION` |
| New per-tick health check | `runners/health_checks.py` (not `execute.py`) |
| New stress mode | New `DutRunner` subclass in `runners/phase_runners.py` + profile that uses it, same commit |
| New telemetry field | `collectors/telemetry.py` collect method + DB migration in `persistence/db.py._migrate` |
| New CLI subcommand | `cli.py` (after the containment check pattern) |

---

See `CHANGELOG.md` for the current protocol and schema versions.
