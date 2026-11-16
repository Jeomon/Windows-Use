import json
import logging

import pytest
from typer.testing import CliRunner

from windows_use.cli import __main__ as cli_main
from windows_use.cli import config as cli_config


@pytest.fixture
def isolated_logging_config(tmp_path, monkeypatch):
    config_dir = tmp_path / ".windows-use"
    config_file = config_dir / "logging_config.json"
    monkeypatch.setattr(cli_config, "WINDOWS_USE_DIR", config_dir)
    monkeypatch.setattr(cli_config, "LOGGING_CONFIG_FILE", config_file)
    monkeypatch.setattr(cli_main, "LOGGING_CONFIG_FILE", config_file)
    return config_file


def test_default_logging_level_is_info(isolated_logging_config):
    assert cli_config.get_logging_config() == {"level": "INFO"}
    assert cli_config.get_log_level() == "INFO"


def test_save_logging_config_accepts_verbose_alias(isolated_logging_config):
    saved = cli_config.save_logging_config("verbose")
    assert saved == {"level": "DEBUG"}
    assert json.loads(isolated_logging_config.read_text(encoding="utf-8")) == {"level": "DEBUG"}
    assert cli_config.get_log_level() == "DEBUG"


def test_save_logging_config_rejects_invalid_level(isolated_logging_config):
    with pytest.raises(ValueError, match="Invalid log level"):
        cli_config.save_logging_config("loud")


def test_invalid_file_falls_back_to_info(isolated_logging_config):
    isolated_logging_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_logging_config.write_text("{not json", encoding="utf-8")
    assert cli_config.get_log_level() == "INFO"


def test_apply_logging_config_updates_loggers(isolated_logging_config):
    cli_config.save_logging_config("debug")
    numeric = cli_config.apply_logging_config()
    assert numeric == logging.DEBUG
    assert logging.getLogger("windows_use").level == logging.DEBUG
    assert logging.getLogger("windows_use.speech.service").level == logging.DEBUG

    cli_config.apply_logging_config("INFO")
    assert logging.getLogger("windows_use").level == logging.INFO


def test_cli_logging_command_persists_and_shows_level(isolated_logging_config):
    runner = CliRunner()

    res_show = runner.invoke(cli_main.app, ["logging"])
    assert res_show.exit_code == 0
    assert "INFO" in res_show.stdout

    res_set = runner.invoke(cli_main.app, ["logging", "verbose"])
    assert res_set.exit_code == 0
    assert "DEBUG" in res_set.stdout
    assert cli_config.get_log_level() == "DEBUG"

    res_bad = runner.invoke(cli_main.app, ["logging", "bogus"])
    assert res_bad.exit_code == 1
