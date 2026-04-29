"""Token counting: tiktoken path, provider delegation, fallback chain,
and extractive summary fallback in context management."""
from unittest.mock import MagicMock

from agent.tokens import (
    PER_MESSAGE_TOKENS,
    PER_REPLY_TOKENS,
    _estimate_tokens_char4,
    count_message_tokens,
    count_tokens,
    count_tool_schema_tokens,
    estimate_tokens,
)


def test_char4_matches_basic():
    """Pure text should land close to the old char/4 heuristic."""
    text = "hello world " * 100  # 1200 chars
    messages = [{"role": "user", "content": text}]
    # char/4 = 300
    assert _estimate_tokens_char4(messages) == 300


def test_count_message_tokens_includes_overhead():
    """A message costs more than just its content — role framing adds
    PER_MESSAGE_TOKENS."""
    msg = {"role": "user", "content": "hello"}
    raw = len("hello") // 4  # old heuristic: ~1
    exact = count_message_tokens(msg)
    assert exact > raw
    # Overhead should be present
    assert exact >= PER_MESSAGE_TOKENS + 1


def test_count_message_tokens_with_tool_calls():
    """Assistant messages carrying tool_calls pay extra for the JSON."""
    msg = {
        "role": "assistant",
        "content": "ok",
        "tool_calls": [
            {"name": "read_file", "arguments": {"path": "foo.py"}},
        ],
    }
    without_tool_calls = count_message_tokens({"role": "assistant", "content": "ok"})
    with_tool_calls = count_message_tokens(msg)
    assert with_tool_calls > without_tool_calls


def test_count_tool_schema_tokens():
    """Tool schema definitions consume real tokens."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert count_tool_schema_tokens(tools) > 0
    assert count_tool_schema_tokens(None) == 0
    assert count_tool_schema_tokens([]) == 0


def test_count_tokens_uses_provider_when_available():
    """If provider.count_tokens succeeds, its value wins."""
    provider = MagicMock()
    provider.count_tokens.return_value = 12345
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "x"}}]
    assert count_tokens(messages, tools=tools, provider=provider) == 12345
    provider.count_tokens.assert_called_once_with(messages, tools)


def test_count_tokens_falls_back_to_tiktoken_on_provider_error():
    """ProviderError bubbles out of provider.count_tokens; the local
    tiktoken path should take over and still return a sensible int."""
    provider = MagicMock()
    provider.count_tokens.side_effect = RuntimeError("network down")
    messages = [{"role": "user", "content": "hello world"}]
    result = count_tokens(messages, tools=None, provider=provider)
    assert isinstance(result, int)
    assert result > 0


def test_count_tokens_without_provider_uses_tiktoken():
    """No provider passed → local tiktoken path."""
    messages = [{"role": "user", "content": "test"}]
    result = count_tokens(messages)
    assert isinstance(result, int)
    assert result > 0


def test_estimate_tokens_backward_compatible():
    """estimate_tokens is still importable and returns an int."""
    messages = [{"role": "user", "content": "foo bar baz"}]
    result = estimate_tokens(messages)
    assert isinstance(result, int)
    assert result > 0


def test_count_tokens_tools_add_to_total():
    """Passing tools should increase the token count vs messages alone."""
    messages = [{"role": "user", "content": "do something"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "do_thing",
                "description": "Does a thing",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    without_tools = count_tokens(messages, tools=None)
    with_tools = count_tokens(messages, tools=tools)
    assert with_tools > without_tools


def test_count_tokens_reply_overhead_present():
    """The per-reply overhead is baked into count_tokens so the total
    matches the OpenAI chat-format convention."""
    messages = []
    result = count_tokens(messages)
    # Even with zero messages we pay PER_REPLY_TOKENS
    assert result == PER_REPLY_TOKENS
