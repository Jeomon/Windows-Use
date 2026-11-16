from unittest.mock import patch

from windows_use.cli.registry import (
    get_models,
    get_provider_display,
    get_providers,
    provider_requires_api_key,
)
from windows_use.providers.cheapaimodel import ChatCheapAIModel
from windows_use.providers.cheapaimodel.llm import CHEAPAIMODEL_BASE_URL


@patch("windows_use.providers.openai.llm.AsyncOpenAI")
@patch("windows_use.providers.openai.llm.OpenAI")
def test_cheapaimodel_defaults(mock_openai, mock_async_openai, monkeypatch):
    monkeypatch.setenv("CHEAPAIMODEL_API_KEY", "cam-key")
    monkeypatch.delenv("CHEAPAIMODEL_API_BASE", raising=False)

    llm = ChatCheapAIModel()

    assert llm.provider == "cheapaimodel"
    assert llm.model_name == "deepseek-v4-1-flash"
    assert llm.base_url == CHEAPAIMODEL_BASE_URL
    mock_openai.assert_called_once_with(
        api_key="cam-key",
        base_url=CHEAPAIMODEL_BASE_URL,
        timeout=600.0,
        max_retries=2,
    )
    mock_async_openai.assert_called_once_with(
        api_key="cam-key",
        base_url=CHEAPAIMODEL_BASE_URL,
        timeout=600.0,
        max_retries=2,
    )


def test_cheapaimodel_is_available_in_cli_registry():
    assert ("cheapaimodel", "cheapaimodel") in get_providers()
    assert ("DeepSeek V4.1 Flash (recommended)", "deepseek-v4-1-flash") in get_models(
        "cheapaimodel"
    )
    assert get_provider_display("cheapaimodel") == "cheapaimodel"
    assert provider_requires_api_key("cheapaimodel") is True
