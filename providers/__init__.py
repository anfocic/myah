"""Provider factory + active-provider registry.

`get_provider()` builds the startup adapter from config env vars.
`build_provider(name, model)` builds an arbitrary adapter for runtime swaps.
`get_active_provider()` / `set_active_provider()` are the mutable slot the
REPL reads from — startup writes once, /model writes again on switch.
Imports the adapter modules lazily so an ollama-only user doesn't pay for
the httpx stack at import (and vice versa)."""

import os

from .base import Provider, ProviderError, StreamChunk, ToolCall, Usage


def build_provider(name: str, model: str) -> Provider:
    """Construct an adapter by provider name. Base URLs + API key still come
    from config/env — only the model name changes per call."""
    if name == "ollama":
        from config import OLLAMA_BASE_URL
        from .ollama_adapter import OllamaProvider

        return OllamaProvider(model, OLLAMA_BASE_URL)

    if name == "openai-compat":
        from config import OPENAI_COMPAT_BASE_URL
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            model=model,
            base_url=OPENAI_COMPAT_BASE_URL,
            api_key=os.environ.get("OPENAI_COMPAT_API_KEY", ""),
        )

    raise ValueError(
        f"unknown provider: {name!r} (expected 'ollama' or 'openai-compat')"
    )


def get_provider() -> Provider:
    """Build the startup provider from env/config. Called once at import
    time; after that use `get_active_provider()` to read the current one."""
    from config import MODEL_PROVIDER

    if MODEL_PROVIDER == "ollama":
        from config import OLLAMA_MODEL
        return build_provider("ollama", OLLAMA_MODEL)

    if MODEL_PROVIDER == "openai-compat":
        from config import OPENAI_COMPAT_MODEL
        return build_provider("openai-compat", OPENAI_COMPAT_MODEL)

    raise ValueError(
        f"unknown MODEL_PROVIDER: {MODEL_PROVIDER!r} "
        "(expected 'ollama' or 'openai-compat')"
    )


_active: Provider | None = None


def get_active_provider() -> Provider:
    """Return the live provider the REPL is currently talking to. Lazy-inits
    from config on first access so imports stay cheap."""
    global _active
    if _active is None:
        _active = get_provider()
    return _active


def set_active_provider(p: Provider) -> None:
    """Swap the live provider. Called by /model after a successful switch."""
    global _active
    _active = p


def list_ollama_models() -> list[str]:
    """Ask the Ollama daemon which models are pulled locally. Returns an
    empty list on any failure (daemon down, not installed, etc.) — the
    caller prints a hint rather than crashing."""
    try:
        import ollama
        from config import OLLAMA_BASE_URL
        client = ollama.Client(host=OLLAMA_BASE_URL) if OLLAMA_BASE_URL else ollama
        resp = client.list()
        # ollama 0.3+: resp is a ListResponse dataclass with .models[*].model
        # Older clients return a dict with "models" → list of {"name": ...}.
        models = getattr(resp, "models", None) or resp.get("models", []) or []
        out = []
        for m in models:
            name = getattr(m, "model", None) or m.get("model") or m.get("name")
            if name:
                out.append(name)
        return out
    except Exception:
        return []


__all__ = [
    "Provider",
    "ProviderError",
    "StreamChunk",
    "ToolCall",
    "Usage",
    "build_provider",
    "get_provider",
    "get_active_provider",
    "set_active_provider",
    "list_ollama_models",
]
