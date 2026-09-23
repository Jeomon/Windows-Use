from unittest.mock import patch

from windows_use.cli.registry import (
    get_models,
    get_provider_display,
    get_providers,
    provider_requires_api_key,
)
from windows_use.providers.atlas_cloud import ChatAtlasCloud
from windows_use.providers.atlas_cloud.llm import ATLAS_CLOUD_BASE_URL


@patch("windows_use.providers.openai.llm.AsyncOpenAI")
@patch("windows_use.providers.openai.llm.OpenAI")
def test_atlas_cloud_defaults(mock_openai, mock_async_openai, monkeypatch):
    monkeypatch.setenv("ATLAS_CLOUD_API_KEY", "atlas-key")

    llm = ChatAtlasCloud()

    assert llm.provider == "atlas_cloud"
    assert llm.model_name == "openai/gpt-4.1-mini"
    assert llm.base_url == ATLAS_CLOUD_BASE_URL
    mock_openai.assert_called_once_with(
        api_key="atlas-key",
        base_url=ATLAS_CLOUD_BASE_URL,
        timeout=600.0,
        max_retries=2,
    )
    mock_async_openai.assert_called_once_with(
        api_key="atlas-key",
        base_url=ATLAS_CLOUD_BASE_URL,
        timeout=600.0,
        max_retries=2,
    )


def test_atlas_cloud_is_available_in_cli_registry():
    assert ("Atlas Cloud", "atlas_cloud") in get_providers()
    assert get_models("atlas_cloud") == [("OpenAI GPT-4.1 mini", "openai/gpt-4.1-mini")]
    assert get_provider_display("atlas_cloud") == "Atlas Cloud"
    assert provider_requires_api_key("atlas_cloud") is True
