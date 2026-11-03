"""Output-path configuration: flag > env var > default."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import psfu.config as config
from psfu import cli


@pytest.fixture(autouse=True)
def _restore_config():
    """apply_path_overrides mutates module globals; put them back afterwards."""
    orig = dict(
        data_dir=config._HOME, runs_dir=config.RUNS_DIR,
        baselines_dir=config.BASELINES_DIR, tools_dir=config.TOOLS_DIR,
        profiles_dir=config.PROFILES_DIR,
    )
    yield
    config.apply_path_overrides(**orig)


def test_default_home_is_under_user_home(monkeypatch) -> None:
    monkeypatch.delenv(config.ENV_HOME, raising=False)
    assert config.default_data_home() == Path.home() / "psfu-data"


def test_env_var_moves_default_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(config.ENV_HOME, str(tmp_path))
    assert config.default_data_home() == tmp_path


def test_specific_env_var_beats_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(config.ENV_HOME, str(tmp_path / "home"))
    monkeypatch.setenv(config.ENV_RUNS, str(tmp_path / "elsewhere"))
    reloaded = importlib.reload(config)
    assert reloaded.RUNS_DIR == tmp_path / "elsewhere"
    assert reloaded.BASELINES_DIR == tmp_path / "home" / "baselines"


def test_data_dir_flag_rebinds_importing_modules(tmp_path: Path) -> None:
    config.apply_path_overrides(data_dir=tmp_path)
    assert config.RUNS_DIR == tmp_path / "runs"
    assert config.FM_LOG_PATH == tmp_path / config.FM_LOG_FILENAME
    # modules that did ``from psfu.config import RUNS_DIR`` follow along
    assert cli.RUNS_DIR == tmp_path / "runs"


def test_runs_dir_flag_beats_data_dir(tmp_path: Path) -> None:
    config.apply_path_overrides(data_dir=tmp_path / "d", runs_dir=tmp_path / "r")
    assert config.RUNS_DIR == tmp_path / "r"
    assert config.BASELINES_DIR == tmp_path / "d" / "baselines"


def test_demo_command_writes_under_runs_dir(tmp_path: Path, capsys) -> None:
    rc = cli.main(["--data-dir", str(tmp_path), "demo", "--scenario", "pass"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Verdict:  PASS" in out
    assert any((tmp_path / "runs").glob("*_demo-pass/run.sqlite"))
