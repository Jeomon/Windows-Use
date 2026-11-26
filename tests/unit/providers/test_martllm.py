from unittest.mock import patch

from windows_use.cli.registry import (
    get_models,
    get_provider_display,
    get_providers,
    provider_requires_api_key,
)
from windows_use.providers.martllm import ChatMartLLM
from windows_use.providers.martllm.llm import MARTLLM_BASE_URL


@patch("windows_use.providers.openai.llm.AsyncOpenAI")
@patch("windows_use.providers.openai.llm.OpenAI")
def test_martllm_defaults(mock_openai, mock_async_openai, monkeypatch):
    monkeypatch.setenv("MARTLLM_API_KEY", "ullm-key")
    monkeypatch.delenv("MARTLLM_API_BASE", raising=False)

    llm = ChatMartLLM()

    assert llm.provider == "martllm"
    assert llm.model_name == "openai/gpt-5.4-mini"
    assert llm.base_url == MARTLLM_BASE_URL
    mock_openai.assert_called_once_with(
        api_key="ullm-key",
        base_url=MARTLLM_BASE_URL,
        timeout=600.0,
        max_retries=2,
    )
    mock_async_openai.assert_called_once_with(
        api_key="ullm-key",
        base_url=MARTLLM_BASE_URL,
        timeout=600.0,
        max_retries=2,
    )


def test_martllm_is_available_in_cli_registry():
    assert ("MartLLM", "martllm") in get_providers()
    assert ("GPT-5.4 mini (recommended)", "openai/gpt-5.4-mini") in get_models("martllm")
    assert get_provider_display("martllm") == "MartLLM"
    assert provider_requires_api_key("martllm") is True
