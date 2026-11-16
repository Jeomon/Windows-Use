from unittest.mock import patch

from windows_use.cli.registry import (
    get_models,
    get_provider_display,
    get_providers,
    provider_requires_api_key,
)
from windows_use.providers.opper import ChatOpper
from windows_use.providers.opper.llm import OPPER_BASE_URL


@patch("windows_use.providers.openai.llm.AsyncOpenAI")
@patch("windows_use.providers.openai.llm.OpenAI")
def test_opper_defaults(mock_openai, mock_async_openai, monkeypatch):
    monkeypatch.setenv("OPPER_API_KEY", "opper-key")
    monkeypatch.delenv("OPPER_API_BASE", raising=False)

    llm = ChatOpper()

    assert llm.provider == "opper"
    assert llm.model_name == "claude-sonnet-4-6"
    assert llm.base_url == OPPER_BASE_URL
    mock_openai.assert_called_once_with(
        api_key="opper-key",
        base_url=OPPER_BASE_URL,
        timeout=600.0,
        max_retries=2,
    )
    mock_async_openai.assert_called_once_with(
        api_key="opper-key",
        base_url=OPPER_BASE_URL,
        timeout=600.0,
        max_retries=2,
    )


def test_opper_is_available_in_cli_registry():
    assert ("Opper", "opper") in get_providers()
    assert ("Claude Sonnet 4.6 (recommended)", "claude-sonnet-4-6") in get_models("opper")
    assert get_provider_display("opper") == "Opper"
    assert provider_requires_api_key("opper") is True
