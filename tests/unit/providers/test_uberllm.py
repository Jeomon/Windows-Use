from unittest.mock import patch

from windows_use.cli.registry import (
    get_models,
    get_provider_display,
    get_providers,
    provider_requires_api_key,
)
from windows_use.providers.uberllm import ChatUberLLM
from windows_use.providers.uberllm.llm import UBERLLM_BASE_URL


@patch("windows_use.providers.openai.llm.AsyncOpenAI")
@patch("windows_use.providers.openai.llm.OpenAI")
def test_uberllm_defaults(mock_openai, mock_async_openai, monkeypatch):
    monkeypatch.setenv("UBERLLM_API_KEY", "ullm-key")
    monkeypatch.delenv("UBERLLM_API_BASE", raising=False)

    llm = ChatUberLLM()

    assert llm.provider == "uberllm"
    assert llm.model_name == "openai/gpt-5.4-mini"
    assert llm.base_url == UBERLLM_BASE_URL
    mock_openai.assert_called_once_with(
        api_key="ullm-key",
        base_url=UBERLLM_BASE_URL,
        timeout=600.0,
        max_retries=2,
    )
    mock_async_openai.assert_called_once_with(
        api_key="ullm-key",
        base_url=UBERLLM_BASE_URL,
        timeout=600.0,
        max_retries=2,
    )


def test_uberllm_is_available_in_cli_registry():
    assert ("UberLLM", "uberllm") in get_providers()
    assert ("GPT-5.4 mini (recommended)", "openai/gpt-5.4-mini") in get_models("uberllm")
    assert get_provider_display("uberllm") == "UberLLM"
    assert provider_requires_api_key("uberllm") is True
