"""Anthropic adapter tests — message translation, tool schema translation,
SSE parsing.

Network-free. The SSE parser takes an iterator of string lines, so we
just feed it hand-written event sequences that mirror what the
Anthropic API emits. Message translation is a pure function."""
import json

import httpx
import pytest

from providers import ProviderError, ToolCall
from providers.anthropic_adapter import (
    AnthropicProvider,
    _parse_sse,
    _translate_messages,
    _translate_tools,
)

# ---------- Message translation ------------------------------------------------


def test_system_messages_lift_to_top_level_field():
    system, out = _translate_messages([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ])
    assert system == "You are helpful."
    assert out == [{"role": "user", "content": "hi"}]


def test_multiple_system_messages_are_concatenated():
    """Summaries injected by the §7 summarize-dropped path show up as
    extra system-role messages mid-history. Anthropic only takes a single
    `system` string, so we concat."""
    system, _out = _translate_messages([
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "Earlier: user asked about X."},
        {"role": "user", "content": "q2"},
    ])
    assert system is not None
    assert "persona" in system
    assert "Earlier: user asked about X." in system
    # Blank line separator so the two system chunks don't bleed together.
    assert "\n\n" in system


def test_assistant_tool_call_becomes_typed_content_blocks():
    """Mia's internal shape: {role: assistant, content: "", tool_calls: [...]}
    Anthropic shape: {role: assistant, content: [{type: text...},
                                                  {type: tool_use, ...}]}"""
    _system, out = _translate_messages([
        {"role": "user", "content": "read it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "read_file", "arguments": {"path": "x.py"}}],
        },
        {"role": "tool", "content": "contents of x.py"},
        {"role": "assistant", "content": "The file says hello."},
    ])

    # Assistant-with-tool-call becomes a content-blocks list.
    assistant_msg = out[1]
    assert assistant_msg["role"] == "assistant"
    assert isinstance(assistant_msg["content"], list)
    # No leading text block when the assistant's text was empty.
    assert len(assistant_msg["content"]) == 1
    tool_use = assistant_msg["content"][0]
    assert tool_use["type"] == "tool_use"
    assert tool_use["name"] == "read_file"
    assert tool_use["input"] == {"path": "x.py"}
    assert tool_use["id"].startswith("toolu_")

    # Tool result becomes a user message with tool_result block, id pairing.
    tool_msg = out[2]
    assert tool_msg["role"] == "user"
    assert tool_msg["content"][0]["type"] == "tool_result"
    assert tool_msg["content"][0]["tool_use_id"] == tool_use["id"]
    assert tool_msg["content"][0]["content"] == "contents of x.py"


def test_parallel_tool_results_coalesce_into_one_user_turn():
    """Anthropic rejects adjacent user turns that both carry only
    tool_result blocks — they must live in a single turn. Our internal
    representation has them as separate `role: tool` messages."""
    _system, out = _translate_messages([
        {"role": "user", "content": "scan"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"name": "glob", "arguments": {"pattern": "*.py"}},
                {"name": "grep", "arguments": {"pattern": "TODO"}},
            ],
        },
        {"role": "tool", "content": "a.py\nb.py"},
        {"role": "tool", "content": "b.py:3: TODO"},
    ])

    # out is: [user "scan", assistant [tool_use, tool_use], user [result, result]]
    assert len(out) == 3
    last = out[-1]
    assert last["role"] == "user"
    assert len(last["content"]) == 2
    assert all(b["type"] == "tool_result" for b in last["content"])
    # IDs pair up in order.
    assert last["content"][0]["tool_use_id"] == "toolu_0"
    assert last["content"][1]["tool_use_id"] == "toolu_1"


def test_plain_text_messages_pass_through_with_string_content():
    _system, out = _translate_messages([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
    ])
    assert out == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
    ]


def test_orphan_tool_message_gets_synthesized_id():
    """Shouldn't happen under run_agent's contract, but a truncated
    session file could deliver a tool message without a preceding
    tool_use. Defensive code synthesizes an id so the payload stays
    structurally valid rather than crashing at send time."""
    _system, out = _translate_messages([
        {"role": "tool", "content": "orphan result"},
    ])
    assert out[0]["role"] == "user"
    assert out[0]["content"][0]["type"] == "tool_result"
    assert out[0]["content"][0]["tool_use_id"].startswith("toolu_")


# ---------- Tool schema translation --------------------------------------------


def test_tool_schema_translation():
    openai_tool = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
    out = _translate_tools([openai_tool])
    assert out == [{
        "name": "read_file",
        "description": "Read a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }]


# ---------- SSE parsing --------------------------------------------------------


def _sse_lines(events: list[tuple[str, dict]]):
    """Format (event_name, data_dict) pairs into the `event:\\ndata:` SSE
    shape Anthropic emits, yielding one line at a time."""
    for name, data in events:
        yield f"event: {name}"
        yield f"data: {json.dumps(data)}"
        yield ""  # blank line delimiter between events (matches real SSE)


def _collect(parser_iter):
    return list(parser_iter)


def test_sse_text_deltas_stream_as_content_chunks():
    lines = _sse_lines([
        ("message_start", {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 42}},
        }),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello "},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "world!"},
        }),
        ("content_block_stop", {
            "type": "content_block_stop",
            "index": 0,
        }),
        ("message_delta", {
            "type": "message_delta",
            "usage": {"output_tokens": 7},
        }),
        ("message_stop", {"type": "message_stop"}),
    ])

    chunks = _collect(_parse_sse(lines))

    content_deltas = [c.content_delta for c in chunks if c.content_delta]
    assert content_deltas == ["Hello ", "world!"]

    # Last chunk is done + usage.
    assert chunks[-1].done
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.prompt_tokens == 42
    assert chunks[-1].usage.completion_tokens == 7


def test_sse_tool_use_buffers_and_emits_on_block_stop():
    """Tool arguments arrive as partial-JSON fragments across multiple
    input_json_delta events. The adapter must buffer them and emit a
    single completed ToolCall on content_block_stop — the model sees
    one tool call, not N partial ones."""
    lines = _sse_lines([
        ("message_start", {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 10}},
        }),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "read_file",
                "input": {},
            },
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"pa'},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": 'th": "x.py"}'},
        }),
        ("content_block_stop", {
            "type": "content_block_stop",
            "index": 0,
        }),
        ("message_delta", {
            "type": "message_delta",
            "usage": {"output_tokens": 20},
        }),
        ("message_stop", {"type": "message_stop"}),
    ])

    chunks = _collect(_parse_sse(lines))

    tool_call_chunks = [c for c in chunks if c.tool_calls]
    assert len(tool_call_chunks) == 1
    tc = tool_call_chunks[0].tool_calls[0]
    assert isinstance(tc, ToolCall)
    assert tc.name == "read_file"
    assert tc.arguments == {"path": "x.py"}

    # And the done chunk still carries usage.
    assert chunks[-1].done
    assert chunks[-1].usage.completion_tokens == 20


def test_sse_ping_and_unknown_events_are_ignored():
    """Anthropic sends periodic `ping` events to keep the connection alive.
    Anything the parser doesn't recognize must be silently ignored — future
    event types must not crash an older client."""
    lines = _sse_lines([
        ("message_start", {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 1}},
        }),
        ("ping", {"type": "ping"}),
        ("something_new", {"type": "something_new", "payload": "ignore me"}),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "ok"},
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_stop", {"type": "message_stop"}),
    ])

    chunks = _collect(_parse_sse(lines))
    deltas = [c.content_delta for c in chunks if c.content_delta]
    assert deltas == ["ok"]
    assert chunks[-1].done


def test_sse_malformed_json_raises_provider_error():
    """Bad JSON from upstream becomes ProviderError — one catch clause in
    the agent loop handles all transport failures."""
    def bad_lines():
        yield "event: message_start"
        yield "data: {not valid json"

    with pytest.raises(ProviderError, match="bad SSE JSON"):
        list(_parse_sse(bad_lines()))


# ---------- Round-trip sanity ---------------------------------------------------


def test_translate_then_id_pairing_handles_mixed_history():
    """A realistic-ish conversation: system + two turns, one with a tool
    call. Verify that tool_use_id and tool_result.tool_use_id round-trip
    correctly across a longer history."""
    system, out = _translate_messages([
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "turn 1 reply"},
        {"role": "user", "content": "turn 2"},
        {
            "role": "assistant",
            "content": "investigating...",
            "tool_calls": [{"name": "grep", "arguments": {"pattern": "x"}}],
        },
        {"role": "tool", "content": "match: a.py:3"},
        {"role": "assistant", "content": "found it in a.py"},
    ])
    assert system == "persona"
    # out: user(t1), assistant(t1), user(t2), assistant(+tool_use),
    #      user(tool_result), assistant(final)
    assert len(out) == 6

    # Assistant with the tool call carries a text block + a tool_use block.
    mixed = out[3]
    assert mixed["role"] == "assistant"
    assert len(mixed["content"]) == 2
    assert mixed["content"][0] == {"type": "text", "text": "investigating..."}
    tool_use = mixed["content"][1]
    assert tool_use["type"] == "tool_use"

    # Following user message is the tool_result with matching id.
    result_msg = out[4]
    assert result_msg["role"] == "user"
    assert result_msg["content"][0]["tool_use_id"] == tool_use["id"]


# ---------- count_tokens -------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code: int, payload: dict, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload)

    def json(self) -> dict:
        return self._payload


def _install_fake_post(monkeypatch, *, status: int = 200, payload: dict | None = None, capture: dict | None = None):
    def fake_post(self, url, json=None, headers=None):
        if capture is not None:
            capture["url"] = url
            capture["json"] = json
            capture["headers"] = headers
        return _FakeResponse(status, payload or {}, text=json_text(payload))
    monkeypatch.setattr("httpx.Client.post", fake_post, raising=True)


def json_text(payload):
    return "" if payload is None else _stringify(payload)


def _stringify(payload: dict) -> str:
    import json as _json
    return _json.dumps(payload)


def test_count_tokens_calls_count_endpoint_and_returns_input_tokens(monkeypatch):
    capture: dict = {}
    _install_fake_post(monkeypatch, payload={"input_tokens": 1234}, capture=capture)

    p = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-test")
    n = p.count_tokens(
        messages=[
            {"role": "system", "content": "persona"},
            {"role": "user", "content": "hi"},
        ],
        tools=[{"type": "function", "function": {"name": "read_file", "description": "d", "parameters": {"type": "object", "properties": {}}}}],
    )

    assert n == 1234
    assert capture["url"].endswith("/messages/count_tokens")
    assert capture["json"]["model"] == "claude-sonnet-4-6"
    assert capture["json"]["system"] == "persona"
    assert capture["json"]["messages"] == [{"role": "user", "content": "hi"}]
    # Tools get translated from OpenAI shape to Anthropic shape
    assert capture["json"]["tools"][0]["name"] == "read_file"
    assert "max_tokens" not in capture["json"]  # count endpoint doesn't take it


def test_count_tokens_raises_on_http_error(monkeypatch):
    _install_fake_post(monkeypatch, status=401, payload={"error": "unauthorized"})
    p = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-test")
    with pytest.raises(ProviderError, match="HTTP 401"):
        p.count_tokens([{"role": "user", "content": "hi"}])


def test_count_tokens_raises_on_missing_field(monkeypatch):
    _install_fake_post(monkeypatch, payload={"something_else": 7})
    p = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-test")
    with pytest.raises(ProviderError, match="missing input_tokens"):
        p.count_tokens([{"role": "user", "content": "hi"}])


def test_count_tokens_raises_on_connect_error(monkeypatch):
    def boom(self, url, json=None, headers=None):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr("httpx.Client.post", boom, raising=True)

    p = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-test")
    with pytest.raises(ProviderError, match="unreachable"):
        p.count_tokens([{"role": "user", "content": "hi"}])
