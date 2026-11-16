from unittest.mock import patch

from windows_use.providers.open_router import ChatOpenRouter
from windows_use.providers.open_router.llm import OPENROUTER_BASE_URL
from windows_use.providers.perplexity import ChatPerplexity


@patch("windows_use.providers.open_router.llm.AsyncOpenAI")
@patch("windows_use.providers.open_router.llm.OpenAI")
def test_open_router_uses_default_base_url_when_none_passed(mock_openai, mock_async_openai, monkeypatch):
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    llm = ChatOpenRouter(model="openai/gpt-4o-mini", api_key="or-key", base_url=None)

    assert llm.base_url == OPENROUTER_BASE_URL
    mock_openai.assert_called_once()
    assert mock_openai.call_args.kwargs["base_url"] == OPENROUTER_BASE_URL
    assert mock_async_openai.call_args.kwargs["base_url"] == OPENROUTER_BASE_URL


@patch("windows_use.providers.open_router.llm.AsyncOpenAI")
@patch("windows_use.providers.open_router.llm.OpenAI")
def test_open_router_uses_explicit_base_url_when_passed(mock_openai, mock_async_openai):
    custom = "https://custom.openrouter.example/v1"
    llm = ChatOpenRouter(model="openai/gpt-4o-mini", api_key="or-key", base_url=custom)

    assert llm.base_url == custom
    assert mock_openai.call_args.kwargs["base_url"] == custom
    assert mock_async_openai.call_args.kwargs["base_url"] == custom


def test_perplexity_uses_default_and_explicit_base_url(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_BASE_URL", raising=False)
    llm_default = ChatPerplexity(model="sonar-pro", api_key="pplx-key", base_url=None)
    assert llm_default.base_url == "https://api.perplexity.ai"

    llm_custom = ChatPerplexity(
        model="sonar-pro",
        api_key="pplx-key",
        base_url="https://custom.perplexity.example/",
    )
    assert llm_custom.base_url == "https://custom.perplexity.example"
