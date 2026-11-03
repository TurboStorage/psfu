# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/). Versions follow SemVer for the package; the scoring `PROTOCOL_VERSION` (currently `0.22`) and SQLite `CURRENT_SCHEMA_VERSION` (currently `10`) are tracked separately in `psfu/config.py` and `psfu/persistence/db.py`.

## [0.1.0] - unreleased (first public staging)

First open-source cut of an internal bench tool. Scoring rules, collectors and reports are unchanged from the internal version except where listed under Changed and Fixed.

### Added
- `psfu demo --scenario {pass,warn,fail}`: deterministic synthetic runs (`psfu/synthetic.py`) that exercise the real verdict engine and exporter with no hardware.
- Configurable output locations: `--data-dir`, `--runs-dir`, `--baselines-dir`, `--tools-dir`, `--profiles-dir`, and `PSFU_HOME`, `PSFU_RUNS_DIR`, `PSFU_BASELINES_DIR`, `PSFU_TOOLS_DIR`, `PSFU_PROFILES_DIR`. Default data root is `~/psfu-data`.
- `pyproject.toml` packaging (console script `psfu`, bundled profiles and fio jobs), `python -m psfu`.
- Windows support for the operator-input layer (`msvcrt` polling when `termios` is unavailable).
- Tests for synthetic golden verdicts and for path configuration.
- `scripts/install-cuda.sh` (CUDA 12.8 installer for Ubuntu 24.04 / WSL) replaces the vendored tools tree.

### Changed
- `OperatorIO` is now an abstract base class; its seven abstract methods are declared with `@abstractmethod` instead of raising `NotImplementedError` at call time.
- Evidence bundles no longer embed absolute local paths: `verdict.json`, `summary.md`, `fm-entry.md` and `rma-summary.md` use paths relative to the run directory.
- Profiles moved into the package (`psfu/profiles/`).
- Batch tracker file is now `batch-tracker.md` in the data root; the failure-mode log lives in the data root instead of a parent folder.
- CLI output is forced to UTF-8 so reports print on Windows code-page consoles.

### Fixed
- `--allow-undersized-psu` was parsed but ignored: an undersized PSU only produced a warning. The power-budget gate is warn-only by design: it warns and records the shortfall but never blocks the run.
- NameError (`PULSE_ON_SEC` / `PULSE_OFF_SEC`) in the fio pulse path of `execute.py`.
- Windows failures: `termios`/`select` imports at module load, `/tmp` path assertion, colon-named sysfs test directories, symlink creation without privilege (that test now skips with a reason).
- Stale NVMe SMART temperature-time tests that stubbed the wrong lookup.

### Removed
- Recorded real-hardware golden fixtures (replaced by synthetic ones), internal planning and operations documents, vendored third-party tool sources and build outputs.
