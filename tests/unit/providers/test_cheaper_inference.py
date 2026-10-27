from unittest.mock import patch

from windows_use.cli.registry import (
    get_models,
    get_provider_display,
    get_providers,
    provider_requires_api_key,
)
from windows_use.providers.cheaper_inference import ChatCheaperInference
from windows_use.providers.cheaper_inference.llm import CHEAPER_INFERENCE_BASE_URL


@patch("windows_use.providers.openai.llm.AsyncOpenAI")
@patch("windows_use.providers.openai.llm.OpenAI")
def test_cheaper_inference_defaults(mock_openai, mock_async_openai, monkeypatch):
    monkeypatch.setenv("CHEAPER_INFERENCE_API_KEY", "ci-key")
    monkeypatch.delenv("CHEAPER_INFERENCE_API_BASE", raising=False)

    llm = ChatCheaperInference()

    assert llm.provider == "cheaper_inference"
    assert llm.model_name == "gpt-5.4-mini"
    assert llm.base_url == CHEAPER_INFERENCE_BASE_URL
    mock_openai.assert_called_once_with(
        api_key="ci-key",
        base_url=CHEAPER_INFERENCE_BASE_URL,
        timeout=600.0,
        max_retries=2,
    )
    mock_async_openai.assert_called_once_with(
        api_key="ci-key",
        base_url=CHEAPER_INFERENCE_BASE_URL,
        timeout=600.0,
        max_retries=2,
    )


def test_cheaper_inference_is_available_in_cli_registry():
    assert ("Cheaper Inference", "cheaper_inference") in get_providers()
    assert ("GPT-5.4 mini (recommended)", "gpt-5.4-mini") in get_models("cheaper_inference")
    assert get_provider_display("cheaper_inference") == "Cheaper Inference"
    assert provider_requires_api_key("cheaper_inference") is True
