"""Provider factory. Reads MODEL_PROVIDER from config and returns the
matching adapter instance. Imports the provider module lazily so a user
running ollama-only doesn't need httpx-level plumbing loaded at startup
(and vice versa)."""

import os

from .base import Provider, ProviderError, StreamChunk, ToolCall, Usage


def get_provider() -> Provider:
    from config import MODEL_PROVIDER

    if MODEL_PROVIDER == "ollama":
        from config import OLLAMA_BASE_URL, OLLAMA_MODEL
        from .ollama_adapter import OllamaProvider

        return OllamaProvider(OLLAMA_MODEL, OLLAMA_BASE_URL)

    if MODEL_PROVIDER == "openai-compat":
        from config import OPENAI_COMPAT_BASE_URL, OPENAI_COMPAT_MODEL
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            model=OPENAI_COMPAT_MODEL,
            base_url=OPENAI_COMPAT_BASE_URL,
            api_key=os.environ.get("OPENAI_COMPAT_API_KEY", ""),
        )

    raise ValueError(
        f"unknown MODEL_PROVIDER: {MODEL_PROVIDER!r} "
        "(expected 'ollama' or 'openai-compat')"
    )


__all__ = [
    "Provider",
    "ProviderError",
    "StreamChunk",
    "ToolCall",
    "Usage",
    "get_provider",
]
