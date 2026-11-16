import os

from windows_use.cli import config as cli_config


def test_load_env_vars_precedence(monkeypatch):
    monkeypatch.setenv("WU_TEST_PROCESS", "from_process")
    monkeypatch.delenv("WU_TEST_USER", raising=False)
    monkeypatch.delenv("WU_TEST_SYSTEM", raising=False)
    monkeypatch.delenv("WU_TEST_OVERRIDE", raising=False)

    monkeypatch.setattr(cli_config, "load_dotenv", lambda: os.environ.setdefault("WU_TEST_DOTENV", "from_dotenv"))

    def fake_read(root_name, _subkey):
        if root_name == "HKEY_CURRENT_USER":
            return {
                "WU_TEST_PROCESS": "from_user_should_not_win",
                "WU_TEST_USER": "from_user",
                "WU_TEST_OVERRIDE": "from_user",
            }
        if root_name == "HKEY_LOCAL_MACHINE":
            return {
                "WU_TEST_SYSTEM": "from_system",
                "WU_TEST_OVERRIDE": "from_system_should_not_win",
            }
        return {}

    monkeypatch.setattr(cli_config, "_read_windows_registry_env", fake_read)

    cli_config.load_env_vars()

    assert os.environ["WU_TEST_PROCESS"] == "from_process"
    assert os.environ["WU_TEST_DOTENV"] == "from_dotenv"
    assert os.environ["WU_TEST_USER"] == "from_user"
    assert os.environ["WU_TEST_SYSTEM"] == "from_system"
    assert os.environ["WU_TEST_OVERRIDE"] == "from_user"


def test_get_env_var_refreshes_from_registry(monkeypatch):
    monkeypatch.delenv("WU_TEST_LAZY_KEY", raising=False)
    monkeypatch.setattr(cli_config, "load_dotenv", lambda: False)
    monkeypatch.setattr(
        cli_config,
        "_read_windows_registry_env",
        lambda root, _sub: {"WU_TEST_LAZY_KEY": "secret-from-user"} if root == "HKEY_CURRENT_USER" else {},
    )

    assert cli_config.get_env_var("WU_TEST_LAZY_KEY") == "secret-from-user"
    assert cli_config.get_env_var("WU_TEST_MISSING", "fallback") == "fallback"
