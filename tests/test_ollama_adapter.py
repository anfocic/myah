"""Ollama adapter — currently only exercises count_tokens since the
streaming path is covered end-to-end by tests/test_integration.py via a
scripted fake provider.

Strategy: monkeypatch the `ollama` module's `Client.chat` with a fake that
returns a simple object exposing `prompt_eval_count`. The adapter reads
exactly that attribute (via getattr), so we don't need to simulate the
full pydantic response shape."""
from types import SimpleNamespace

import pytest

from providers import ProviderError
from providers.ollama_adapter import OllamaProvider


def _fake_response(prompt_eval_count: int | None) -> SimpleNamespace:
    # The real Ollama SDK returns a pydantic model; the adapter only reads
    # `prompt_eval_count`, so a bare object with that attribute is enough.
    # `.message` is unused by count_tokens but supplied so the adapter stays
    # compatible if it ever reaches further into the response.
    return SimpleNamespace(
        prompt_eval_count=prompt_eval_count,
        eval_count=1,
        message=SimpleNamespace(content="x", tool_calls=None),
    )


@pytest.fixture
def provider(monkeypatch):
    # Stub the ollama module's Client constructor so __init__ doesn't try
    # to reach a real daemon. `OllamaProvider` only needs `_client.chat`.
    import ollama

    class FakeClient:
        def __init__(self, host=None):
            self.host = host
            self.calls: list[dict] = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return self._next

        _next = _fake_response(42)

    monkeypatch.setattr(ollama, "Client", FakeClient)
    p = OllamaProvider(model="qwen2.5:7b-instruct", base_url="http://127.0.0.1:0")
    return p


def test_count_tokens_returns_prompt_eval_count(provider):
    provider._client._next = _fake_response(77)
    n = provider.count_tokens(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ],
    )
    assert n == 77
    # The trick: num_predict=1 so the server tokenizes the prompt, emits one
    # token, then stops. Anything else and we'd pay full inference latency.
    call = provider._client.calls[-1]
    assert call["options"]["num_predict"] == 1
    assert call["tools"] is None


def test_count_tokens_passes_tools_through(provider):
    provider._client._next = _fake_response(100)
    tools = [{"type": "function", "function": {"name": "read_file", "description": "d", "parameters": {}}}]
    provider.count_tokens(messages=[{"role": "user", "content": "hi"}], tools=tools)
    call = provider._client.calls[-1]
    assert call["tools"] == tools


def test_count_tokens_raises_when_daemon_unreachable(provider):
    def boom(**_kwargs):
        raise ConnectionError("refused")
    provider._client.chat = boom
    with pytest.raises(ProviderError, match="ollama unreachable"):
        provider.count_tokens([{"role": "user", "content": "hi"}])


def test_count_tokens_raises_when_prompt_eval_count_missing(provider):
    # Some older Ollama builds omit prompt_eval_count when num_predict=0.
    # Our adapter raises ProviderError so commands.py can fall back to char/4.
    provider._client._next = _fake_response(None)
    with pytest.raises(ProviderError, match="prompt_eval_count"):
        provider.count_tokens([{"role": "user", "content": "hi"}])


def test_count_tokens_strips_internal_tool_calls(provider):
    """Assistant messages in replay history carry our internal `tool_calls`
    shape (`{name, arguments}`); Ollama's pydantic validator rejects that
    and expects `{function: {name, arguments}}`. The adapter's existing
    _strip_internal_tool_calls helper handles it for chat()/stream_chat();
    count_tokens uses the same path."""
    provider._client._next = _fake_response(10)
    provider.count_tokens(messages=[
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "read_file", "arguments": {"path": "x.py"}}],
        },
    ])
    sent = provider._client.calls[-1]["messages"]
    assistant = next(m for m in sent if m["role"] == "assistant")
    assert "tool_calls" not in assistant
