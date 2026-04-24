"""Provider factory + active-provider registry.

`get_provider()` builds the startup adapter from config env vars.
`build_provider(name, model)` builds an arbitrary adapter for runtime swaps.
`get_active_provider()` / `set_active_provider()` are the mutable slot the
REPL reads from — startup writes once, /model writes again on switch.
Imports the adapter modules lazily so an ollama-only user doesn't pay for
the httpx stack at import (and vice versa)."""

import os

from env import load_dotenv

from .base import Provider, ProviderError, StreamChunk, ToolCall, Usage

# Default base URLs for the hosted providers. Overridable via env so
# proxies (LiteLLM, OpenRouter pretending to be OpenAI, an Anthropic-on-
# Bedrock gateway) can redirect traffic without a code change.
_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"

# Reasonable defaults for `/model <provider-name>` with no model specified
# and for the startup path when MIA_PROVIDER selects one of these without
# a matching *_MODEL env var set.
_DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-4-6",
    "deepseek": "deepseek-chat",
    "google": "gemma-4-e4b",
}


def build_provider(name: str, model: str) -> Provider:
    """Construct an adapter by provider name. Base URLs + API key still come
    from config/env — only the model name changes per call.

    Supported names: `ollama`, `openai-compat`, `openai`, `anthropic`,
    `deepseek`. `openai-compat` is the generic adapter (local llama.cpp /
    LM Studio / vLLM / OpenRouter); `openai` and `deepseek` are presets
    that reuse the same adapter but default to the first-party hosts and
    each provider's dedicated API-key env var. `anthropic` has its own
    native adapter because the Messages API differs in message shape,
    streaming events, and tool schema."""
    load_dotenv()

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

    if name == "openai":
        # First-party OpenAI. Reuses the openai-compat adapter because the
        # Chat Completions wire format is identical — the only difference
        # from the generic compat path is a hardcoded base URL and a
        # dedicated API-key env var. If/when we want Responses API or
        # reasoning_effort plumbing, that motivates a separate adapter.
        from .openai_compat import OpenAICompatProvider

        provider = OpenAICompatProvider(
            model=model,
            base_url=os.environ.get("OPENAI_BASE_URL", _OPENAI_BASE_URL),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
        provider.name = "openai"
        return provider

    if name == "anthropic":
        from .anthropic_adapter import AnthropicProvider

        return AnthropicProvider(
            model=model,
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            base_url=os.environ.get("ANTHROPIC_BASE_URL", _ANTHROPIC_BASE_URL),
        )

    if name == "deepseek":
        # DeepSeek exposes an OpenAI-compatible endpoint; same adapter,
        # different host + API-key env var. Split into a dedicated adapter
        # if/when DeepSeek-specific features (FIM, reasoning-mode toggles
        # for `deepseek-reasoner`) need bespoke handling.
        from .openai_compat import OpenAICompatProvider

        provider = OpenAICompatProvider(
            model=model,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", _DEEPSEEK_BASE_URL),
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        )
        provider.name = "deepseek"
        return provider

    if name == "google":
        # Google exposes an OpenAI-compatible endpoint; same adapter,
        # different host + API-key env var. Split into a dedicated adapter
        # if/when Gemini-specific features need bespoke handling.
        from .openai_compat import OpenAICompatProvider

        provider = OpenAICompatProvider(
            model=model,
            base_url=os.environ.get(
                "GOOGLE_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            ),
            api_key=os.environ.get("GOOGLE_API_KEY", ""),
        )
        provider.name = "google"
        return provider

    raise ValueError(
        f"unknown provider: {name!r} "
        "(expected one of: ollama, openai-compat, openai, anthropic, deepseek, google)"
    )


SUPPORTED_PROVIDERS = frozenset(
    {"ollama", "openai-compat", "openai", "anthropic", "deepseek", "google"}
)


def get_provider() -> Provider:
    """Build the startup provider from env/config. Called once at import
    time; after that use `get_active_provider()` to read the current one.

    Each hosted provider reads a `<NAME>_MODEL` env var (e.g.
    `OPENAI_MODEL`) for the model name, falling back to a sensible
    default from `_DEFAULT_MODELS` if unset."""
    from config import MODEL_PROVIDER

    if MODEL_PROVIDER == "ollama":
        from config import OLLAMA_MODEL

        return build_provider("ollama", OLLAMA_MODEL)

    if MODEL_PROVIDER == "openai-compat":
        from config import OPENAI_COMPAT_MODEL

        return build_provider("openai-compat", OPENAI_COMPAT_MODEL)

    if MODEL_PROVIDER == "openai":
        return build_provider(
            "openai",
            os.environ.get("OPENAI_MODEL", _DEFAULT_MODELS["openai"]),
        )

    if MODEL_PROVIDER == "anthropic":
        return build_provider(
            "anthropic",
            os.environ.get("ANTHROPIC_MODEL", _DEFAULT_MODELS["anthropic"]),
        )

    if MODEL_PROVIDER == "deepseek":
        return build_provider(
            "deepseek",
            os.environ.get("DEEPSEEK_MODEL", _DEFAULT_MODELS["deepseek"]),
        )

    if MODEL_PROVIDER == "google":
        return build_provider(
            "google",
            os.environ.get("GOOGLE_MODEL", _DEFAULT_MODELS["google"]),
        )

    raise ValueError(
        f"unknown MODEL_PROVIDER: {MODEL_PROVIDER!r} "
        f"(expected one of: {', '.join(sorted(SUPPORTED_PROVIDERS))})"
    )


_active: Provider | None = None


def get_active_provider() -> Provider:
    """Return the live provider the REPL is currently talking to. Lazy-inits
    from config on first access so imports stay cheap. Routes through
    `set_active_provider` so the single-model-resident invariant is enforced
    on startup too, not only on /model swaps."""
    global _active
    if _active is None:
        set_active_provider(get_provider())
    assert _active is not None  # set_active_provider just assigned it
    return _active


def set_active_provider(p: Provider) -> None:
    """Swap the live provider. Called by /model after a successful switch,
    by the eval runner when a task pins a provider, and by the lazy-init
    path on first access.

    After installing `p`, ask it to evict every other model resident on the
    same backend. The harness's single-model invariant exists because two
    large models concurrently loaded can OOM a local GPU; the provider's
    `ensure_exclusive` knows how to reach its own backend's unload path
    (ollama daemon, LM Studio `lms` CLI, etc.). The method is optional, so
    hosted providers (Anthropic, OpenAI, DeepSeek) simply don't expose it
    and the hook is a no-op for them.
    """
    global _active
    _active = p
    ensure = getattr(p, "ensure_exclusive", None)
    if callable(ensure):
        try:
            ensure()
        except Exception:
            # Best-effort: the user's session must not fail to start just
            # because model eviction on the backend hiccuped. Worst case
            # they're in the pre-fix state with two models resident, which
            # is exactly what they were in before this call.
            pass


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
    "SUPPORTED_PROVIDERS",
    "ToolCall",
    "Usage",
    "build_provider",
    "get_provider",
    "get_active_provider",
    "set_active_provider",
    "list_ollama_models",
]
