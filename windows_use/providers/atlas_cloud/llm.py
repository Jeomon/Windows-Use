"""Atlas Cloud LLM provider via its OpenAI-compatible API."""

import os

from windows_use.providers.openai.llm import ChatOpenAI

ATLAS_CLOUD_BASE_URL = "https://api.atlascloud.ai/v1"


class ChatAtlasCloud(ChatOpenAI):
    """Chat model served through Atlas Cloud's OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = "openai/gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        temperature: float | None = None,
        **kwargs,
    ):
        api_key = api_key or os.environ.get("ATLAS_CLOUD_API_KEY")
        base_url = base_url or os.environ.get("ATLAS_CLOUD_API_BASE") or ATLAS_CLOUD_BASE_URL
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
            **kwargs,
        )

    @property
    def provider(self) -> str:
        return "atlas_cloud"
