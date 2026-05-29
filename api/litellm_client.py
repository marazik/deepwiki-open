import os
from openai import AsyncOpenAI, OpenAI

from api.openai_client import OpenAIClient


class LiteLLMClient(OpenAIClient):
    """
    LiteLLM OpenAI-compatible client.

    LiteLLM exposes an OpenAI-compatible API surface, so we can
    reuse almost all OpenAIClient behavior while overriding only
    the client initialization.

    Expected environment variables:

    LITELLM_BASE_URL=http://litellm:4000
    LITELLM_API_KEY=sk-1234

    Example model names:
        openai/gpt-4o
        anthropic/claude-3-5-sonnet
        gemini/gemini-2.5-pro
        ollama/llama3
    """

    def init_sync_client(self):
        """
        Initialize synchronous LiteLLM OpenAI-compatible client.
        """

        api_key = os.getenv("LITELLM_API_KEY", "dummy")

        base_url = os.getenv(
            "LITELLM_BASE_URL",
            "http://localhost:4000"
        )

        return OpenAI(
            api_key=api_key,
            base_url=f"{base_url}/v1",
        )

    def init_async_client(self):
        """
        Initialize asynchronous LiteLLM OpenAI-compatible client.
        """

        api_key = os.getenv("LITELLM_API_KEY", "dummy")

        base_url = os.getenv(
            "LITELLM_BASE_URL",
            "http://localhost:4000"
        )

        return AsyncOpenAI(
            api_key=api_key,
            base_url=f"{base_url}/v1",
        )
