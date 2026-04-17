"""Message translation from internal shape → OpenAI-compat shape.

The interesting property is the tool_call_id contract: every id emitted on
an assistant's tool_calls must have a matching tool message, or OpenAI
rejects the payload. That's the orphan-flush behavior exercised below."""
from providers.openai_compat import _translate_messages


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


def _referenced_ids(translated: list[dict]) -> set[str]:
    ids: set[str] = set()
    for m in translated:
        for tc in m.get("tool_calls") or []:
            ids.add(tc["id"])
    return ids


def _tool_message_ids(translated: list[dict]) -> list[str]:
    return [m["tool_call_id"] for m in translated if m.get("role") == "tool"]
