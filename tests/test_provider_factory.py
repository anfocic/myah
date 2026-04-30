"""Factory tests — the new `openai`, `anthropic`, `deepseek` presets wire
up the right adapter class, base URL, and API-key env var.

Everything is exercised through `build_provider` so this doubles as a
regression net against someone adding a sixth provider and breaking the
dispatch. Real HTTP is never made — we just inspect attributes on the
constructed adapter."""
import os
from unittest.mock import patch

import pytest

from providers import (
    SUPPORTED_PROVIDERS,
    ProviderError,
    build_provider,
)
from providers.anthropic_adapter import AnthropicProvider
from providers.openai_compat import OpenAICompatProvider


def test_supported_providers_matches_factory_branches():
    """If someone adds a provider to build_provider but forgets to update
    SUPPORTED_PROVIDERS (or vice versa), the `/model` command's colon
    parser silently stops recognizing that prefix. This test locks them
    together."""
    assert SUPPORTED_PROVIDERS == {
        "ollama", "openai-compat", "openai", "anthropic", "deepseek", "google", "opencode",
    }


def test_build_openai_uses_first_party_base_and_api_key():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
        p = build_provider("openai", "gpt-4.1-mini")
    assert isinstance(p, OpenAICompatProvider)
    assert p.name == "openai"
    assert p.model == "gpt-4.1-mini"
    assert p._base == "https://api.openai.com/v1"
    assert p._headers.get("Authorization") == "Bearer sk-test"


def test_build_deepseek_uses_deepseek_host_and_api_key():
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "ds-test"}, clear=False):
        p = build_provider("deepseek", "deepseek-chat")
    assert isinstance(p, OpenAICompatProvider)
    assert p.name == "deepseek"
    assert p.model == "deepseek-chat"
    assert p._base == "https://api.deepseek.com/v1"
    assert p._headers.get("Authorization") == "Bearer ds-test"


def test_build_google_uses_gemini_host_and_api_key():
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "g-test"}, clear=False):
        p = build_provider("google", "gemini-2.5-flash")
    assert isinstance(p, OpenAICompatProvider)
    assert p.name == "google"
    assert p.model == "gemini-2.5-flash"
    assert p._base == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert p._headers.get("Authorization") == "Bearer g-test"


def test_build_opencode_uses_opencode_host_and_api_key():
    with patch.dict(os.environ, {"OPENCODE_API_KEY": "oc-test"}, clear=False):
        p = build_provider("opencode", "opencode/default")
    assert isinstance(p, OpenAICompatProvider)
    assert p.name == "opencode"
    assert p.model == "opencode/default"
    assert p._base == "https://api.opencode.dev/v1"
    assert p._headers.get("Authorization") == "Bearer oc-test"


def test_build_anthropic_requires_api_key():
    """Anthropic is the only provider where the factory fails closed on
    missing auth — their API responds with a 401 we'd otherwise only see
    on the first turn's network round-trip, and the failure message would
    be opaque. OpenAI-compat gives a pass because many local servers
    don't check auth."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
        with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
            build_provider("anthropic", "claude-sonnet-4-6")


def test_build_anthropic_native_adapter_with_key():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False):
        p = build_provider("anthropic", "claude-sonnet-4-6")
    assert isinstance(p, AnthropicProvider)
    assert p.name == "anthropic"
    assert p.model == "claude-sonnet-4-6"
    assert p._base == "https://api.anthropic.com/v1"
    # Anthropic uses `x-api-key` header, not `Authorization: Bearer`.
    assert p._headers.get("x-api-key") == "sk-ant-test"
    assert p._headers.get("anthropic-version") == "2023-06-01"


def test_build_unknown_provider_lists_supported_ones():
    """The ValueError message names every supported provider so a typo
    (`/model opnai:gpt-4`) gives the user the full menu without them
    needing to grep the source."""
    with pytest.raises(ValueError, match="ollama.*anthropic|anthropic.*ollama"):
        build_provider("bogus", "anything")


def test_set_active_provider_calls_ensure_exclusive():
    """The harness enforces a one-model-resident invariant on each provider
    swap: two large local models loaded at once will OOM a typical GPU. The
    contract is that `set_active_provider` asks the new provider to evict
    everything else on its backend via `ensure_exclusive`."""
    from providers import get_active_provider, set_active_provider

    calls = []

    class ExclusiveProvider:
        name = "fake-exclusive"
        model = "fake-exclusive-v1"

        def ensure_exclusive(self):
            calls.append("ensure_exclusive")

        def stream_chat(self, messages, tools, num_ctx):
            yield from ()

        def chat(self, messages, num_ctx):
            raise NotImplementedError

        def count_tokens(self, messages, tools=None):
            return 0

    original = get_active_provider()
    try:
        set_active_provider(ExclusiveProvider())
        assert calls == ["ensure_exclusive"]
    finally:
        set_active_provider(original)


def test_set_active_provider_swallows_ensure_exclusive_errors():
    """Eviction failures must not crash the REPL — the worst case is that
    the user stays in the pre-swap state with both models resident, exactly
    where they were before the call. A raised exception here would break
    /model and startup for a best-effort-cleanup reason."""
    from providers import get_active_provider, set_active_provider

    class BadProvider:
        name = "fake-bad"
        model = "fake-bad-v1"

        def ensure_exclusive(self):
            raise RuntimeError("simulated backend hiccup")

        def stream_chat(self, messages, tools, num_ctx):
            yield from ()

        def chat(self, messages, num_ctx):
            raise NotImplementedError

        def count_tokens(self, messages, tools=None):
            return 0

    original = get_active_provider()
    try:
        # Must not raise.
        set_active_provider(BadProvider())
    finally:
        set_active_provider(original)


def test_openai_compat_ensure_exclusive_noop_for_non_lm_studio(monkeypatch):
    """Generic OpenAI-compat endpoints (vLLM, llama.cpp, OpenRouter, first-
    party OpenAI) have no portable unload API, so the eviction hook must
    silently do nothing for them. Only LM Studio (port 1234 on localhost by
    convention, or explicit opt-in via env) gets the `lms unload` treatment."""
    invocations = []

    def fake_run(*args, **kwargs):
        invocations.append(args)
        raise AssertionError("subprocess.run should not be invoked for non-LM-Studio base URLs")

    monkeypatch.setattr("providers.openai_compat.subprocess.run", fake_run)

    p = OpenAICompatProvider(model="gpt-4.1-mini", base_url="https://api.openai.com/v1")
    p.ensure_exclusive()  # must not raise or call subprocess
    assert invocations == []
