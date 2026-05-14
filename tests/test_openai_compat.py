"""Message translation from internal shape → OpenAI-compat shape.

The interesting property is the tool_call_id contract: every id emitted on
an assistant's tool_calls must have a matching tool message, or OpenAI
rejects the payload. That's the orphan-flush behavior exercised below."""
import json

import pytest

from providers import ProviderError
from providers.openai_compat import OpenAICompatProvider, _parse_sse, _translate_messages


def test_basic_roundtrip_tool_ids_match():
    """Normal flow: assistant with 2 tool_calls → 2 tool results → ids align."""
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "grep", "arguments": {"pattern": "x"}},
            {"name": "read_file", "arguments": {"path": "a.py"}},
        ]},
        {"role": "tool", "content": "grep result"},
        {"role": "tool", "content": "read result"},
    ]
    out = _translate_messages(msgs)

    refs = _referenced_ids(out)
    seen = _tool_message_ids(out)
    assert refs == set(seen)
    assert len(seen) == 2
    assert len(seen) == len(set(seen))


def test_orphan_tool_calls_are_flushed_with_stubs():
    """If the previous assistant has an unanswered tool_call and a new
    assistant arrives, the old ids must still have matching tool messages.
    Without the flush, OpenAI would reject the payload."""
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "grep", "arguments": {"pattern": "x"}},
            {"name": "read_file", "arguments": {"path": "a.py"}},
        ]},
        {"role": "tool", "content": "grep result"},
        # second tool result missing — call_1 is orphaned
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "glob", "arguments": {"pattern": "*.py"}},
        ]},
        {"role": "tool", "content": "glob result"},
    ]
    out = _translate_messages(msgs)

    refs = _referenced_ids(out)
    seen = _tool_message_ids(out)
    assert refs == set(seen)  # every id referenced by an assistant has a tool msg
    assert len(seen) == len(set(seen))  # no duplicate tool_call_ids


def test_tool_message_with_no_preceding_assistant_gets_fresh_id():
    """Edge case: a resumed session could have a tool message with no
    pending assistant (e.g. history got clipped). Adapter should synthesize
    a fresh id rather than crash."""
    msgs = [
        {"role": "tool", "content": "orphan"},
    ]
    out = _translate_messages(msgs)
    assert out[0]["role"] == "tool"
    assert out[0]["tool_call_id"].startswith("call_")


def test_translate_strips_reasoning_content_by_default():
    """Old reasoning text is ephemeral — replaying it back to most models
    (qwen3 in particular) confuses them. The default wire output must NOT
    carry reasoning_content on any role.

    Covers both the assistant-with-tool_calls branch (its own dedicated
    code path) and the catch-all branch (system/user/plain assistant).
    """
    msgs = [
        {"role": "system", "content": "sys", "reasoning_content": "leak1"},
        {"role": "user", "content": "u", "reasoning_content": "leak2"},
        {
            "role": "assistant",
            "content": "a",
            "reasoning_content": "leak3",
            "tool_calls": [{"name": "grep", "arguments": {"x": 1}}],
        },
        {"role": "tool", "content": "t"},
        {"role": "assistant", "content": "final", "reasoning_content": "leak4"},
    ]
    out = _translate_messages(msgs)
    for m in out:
        assert "reasoning_content" not in m, f"leaked on {m.get('role')}: {m}"


def test_translate_preserves_reasoning_content_when_opted_in():
    """Moonshot AI / kimi-k2.6 returns 400 if an assistant message with
    tool_calls lacks reasoning_content in thinking mode. With
    include_reasoning=True, the key must be present on that branch even
    when the source message had no reasoning_content (empty-string
    default), and it must round-trip when present."""
    msgs = [
        {
            "role": "assistant",
            "content": "a",
            "tool_calls": [{"name": "grep", "arguments": {}}],
            # no reasoning_content set — must still emit empty string
        },
        {"role": "tool", "content": "t"},
        {
            "role": "assistant",
            "content": "b",
            "tool_calls": [{"name": "glob", "arguments": {}}],
            "reasoning_content": "step 1",
        },
        {"role": "tool", "content": "t2"},
        {"role": "user", "content": "u", "reasoning_content": "u-reasoning"},
    ]
    out = _translate_messages(msgs, include_reasoning=True)
    # First assistant: empty-string default for kimi's 400 guard.
    asst_msgs = [m for m in out if m["role"] == "assistant"]
    assert asst_msgs[0]["reasoning_content"] == ""
    # Second assistant: round-tripped value.
    assert asst_msgs[1]["reasoning_content"] == "step 1"
    # User message: also passed through when opted in.
    user_msg = next(m for m in out if m["role"] == "user")
    assert user_msg.get("reasoning_content") == "u-reasoning"


def test_sse_reasoning_content_becomes_reasoning_delta():
    """LM Studio (qwen3) and DeepSeek-R1 route chain-of-thought to a
    separate `reasoning_content` delta. Parsing it into `reasoning_delta`
    is what keeps the loop from silently discarding the model's actual
    output — for reasoning-mode qwen3, `content` is often "" while the
    reasoning holds hundreds of tokens.
    """
    lines = iter([
        "data: " + json.dumps({
            "choices": [{"delta": {"reasoning_content": "step one: "}}]
        }),
        "data: " + json.dumps({
            "choices": [{"delta": {"reasoning_content": "analyze."}}]
        }),
        "data: " + json.dumps({
            "choices": [{"delta": {"content": "Done."}}]
        }),
        "data: [DONE]",
    ])
    chunks = list(_parse_sse(lines))
    reasoning = "".join(c.reasoning_delta for c in chunks if c.reasoning_delta)
    content = "".join(c.content_delta for c in chunks if c.content_delta)
    assert reasoning == "step one: analyze."
    assert content == "Done."
    # Reasoning and content must arrive on distinct chunks so the loop
    # can style them separately — if they were bundled we'd smear the
    # dim reasoning stream over the markdown renderer's final reply.
    assert all(not (c.reasoning_delta and c.content_delta) for c in chunks)


def test_sse_openrouter_reasoning_key_becomes_reasoning_delta():
    """OpenRouter-style proxies (OpenCode, which fronts Moonshot/kimi)
    stream chain-of-thought on `delta.reasoning`, not `reasoning_content`.
    Missing it left `turn.reasoning` empty, so kimi's thinking-mode
    tool-call messages replayed with `reasoning_content: ""` and Moonshot
    rejected them with a 400.
    """
    lines = iter([
        "data: " + json.dumps({
            "choices": [{"delta": {"reasoning": "let me think: "}}]
        }),
        "data: " + json.dumps({
            "choices": [{"delta": {"reasoning": "call the tool."}}]
        }),
        "data: " + json.dumps({
            "choices": [{"delta": {"content": "Answer."}}]
        }),
        "data: [DONE]",
    ])
    chunks = list(_parse_sse(lines))
    reasoning = "".join(c.reasoning_delta for c in chunks if c.reasoning_delta)
    content = "".join(c.content_delta for c in chunks if c.content_delta)
    assert reasoning == "let me think: call the tool."
    assert content == "Answer."


def test_bad_streamed_tool_call_json_raises_provider_error():
    lines = iter([
        "data: " + json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "name": "read_file",
                            "arguments": "not-json",
                        },
                    }]
                }
            }]
        }),
        "data: " + json.dumps({
            "choices": [{
                "delta": {},
                "finish_reason": "tool_calls",
            }]
        }),
    ])

    with pytest.raises(ProviderError, match="tool-call JSON"):
        list(_parse_sse(lines))


def _referenced_ids(translated: list[dict]) -> set[str]:
    ids: set[str] = set()
    for m in translated:
        for tc in m.get("tool_calls") or []:
            ids.add(tc["id"])
    return ids


def _tool_message_ids(translated: list[dict]) -> list[str]:
    return [m["tool_call_id"] for m in translated if m.get("role") == "tool"]


# ---------- count_tokens -------------------------------------------------------

def _provider() -> OpenAICompatProvider:
    # base_url is never hit — count_tokens is pure tiktoken, no network.
    return OpenAICompatProvider(
        model="gpt-4o-mini", base_url="http://unused", context_size=128_000,
    )


def test_count_tokens_positive_and_monotonic():
    """Empty prompt has a small floor (role bytes + per-msg overhead +
    reply primer). Adding content can only increase the count."""
    p = _provider()
    floor = p.count_tokens([{"role": "user", "content": ""}])
    small = p.count_tokens([{"role": "user", "content": "hi"}])
    big = p.count_tokens([{"role": "user", "content": "hi " * 200}])
    assert 0 < floor < small < big


def test_count_tokens_includes_system_and_assistant_framing():
    """Each message adds its own framing overhead, not just content."""
    p = _provider()
    one = p.count_tokens([{"role": "user", "content": "x"}])
    three = p.count_tokens([
        {"role": "system", "content": "x"},
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "x"},
    ])
    # Same content tokens (3x "x") but 3 messages → at least 2 extra framings.
    assert three >= one + 2 * 3


def test_count_tokens_counts_tools_schema():
    """Tool schema JSON is folded into the prompt server-side; count_tokens
    must include it or /profile's tools row would be zero."""
    p = _provider()
    base = p.count_tokens([{"role": "user", "content": "hi"}], tools=None)
    with_tools = p.count_tokens(
        [{"role": "user", "content": "hi"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read a file from disk",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }],
    )
    assert with_tools > base
