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
        "ollama", "openai-compat", "openai", "anthropic", "deepseek",
    }


def test_build_openai_uses_first_party_base_and_api_key():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
        p = build_provider("openai", "gpt-4.1-mini")
    assert isinstance(p, OpenAICompatProvider)
    assert p.name == "openai-compat"  # adapter class-level name
    assert p.model == "gpt-4.1-mini"
    assert p._base == "https://api.openai.com/v1"
    assert p._headers.get("Authorization") == "Bearer sk-test"


def test_build_deepseek_uses_deepseek_host_and_api_key():
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "ds-test"}, clear=False):
        p = build_provider("deepseek", "deepseek-chat")
    assert isinstance(p, OpenAICompatProvider)
    assert p.model == "deepseek-chat"
    assert p._base == "https://api.deepseek.com/v1"
    assert p._headers.get("Authorization") == "Bearer ds-test"


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
