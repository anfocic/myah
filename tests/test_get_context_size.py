"""Unit tests for `config.get_context_size()`.

The function is consulted on every prompt render, every microcompact
check, every /context call. It must:
  - return the active provider's window when one is set
  - fall back to NUM_CTX when the provider lookup raises a known
    "no provider yet" / "missing attr" error
  - NOT swallow generic exceptions — a real provider bug should
    propagate so we hear about it instead of silently mis-sizing.
"""
import pytest

import config
import providers


class _FakeProvider:
    name = "fake"
    model = "fake-model"
    context_size = 12345

    def stream_chat(self, messages, tools, num_ctx):  # pragma: no cover
        yield from ()

    def chat(self, messages, num_ctx):  # pragma: no cover
        raise NotImplementedError

    def count_tokens(self, messages, tools=None):  # pragma: no cover
        return 0


@pytest.fixture
def restore_active_provider():
    """Save and restore the global active provider slot so tests don't
    leak state into each other."""
    original = providers._active
    yield
    providers._active = original


def test_returns_active_provider_context_size(restore_active_provider):
    providers.set_active_provider(_FakeProvider())
    assert config.get_context_size() == 12345


def test_falls_back_to_num_ctx_when_no_provider(restore_active_provider, monkeypatch):
    """If `get_active_provider` raises RuntimeError (e.g. lazy-init can't
    construct one because env is unset in a test), we degrade to the
    config-file default rather than crashing render code."""
    def _raise():
        raise RuntimeError("no provider configured")

    monkeypatch.setattr(providers, "get_active_provider", _raise)
    assert config.get_context_size() == config.NUM_CTX


def test_falls_back_when_provider_lacks_attr(restore_active_provider):
    """A test stub (or an adapter someone added without updating the
    Protocol) might not carry `context_size`. The fallback keeps the UI
    rendering instead of throwing on every keystroke."""

    class _NoCtxSize:
        name = "broken"
        model = "broken-v1"
        # no context_size

        def stream_chat(self, messages, tools, num_ctx):  # pragma: no cover
            yield from ()

        def chat(self, messages, num_ctx):  # pragma: no cover
            raise NotImplementedError

        def count_tokens(self, messages, tools=None):  # pragma: no cover
            return 0

    providers.set_active_provider(_NoCtxSize())
    assert config.get_context_size() == config.NUM_CTX


def test_does_not_swallow_arbitrary_exceptions(restore_active_provider, monkeypatch):
    """The fallback list is intentionally narrow. A ValueError from inside
    the provider lookup (e.g. malformed config) should surface, not be
    masked as a NUM_CTX read."""
    def _raise():
        raise ValueError("simulated bad config")

    monkeypatch.setattr(providers, "get_active_provider", _raise)
    with pytest.raises(ValueError, match="simulated bad config"):
        config.get_context_size()
