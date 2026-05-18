"""Context compaction: manual /compact (compact_history) + intra-turn
microcompact. The summarization path isn't tested here because it hits
the provider — see test_apply_summary_shape for the shape-only check."""
from unittest.mock import MagicMock, patch

from agent import (
    COMPACT_KEEP_LAST,
    ELIDED_PREFIX,
    compact_history,
    microcompact,
    summarize_dropped,
    trim_history,
)
from providers import ProviderError


def _turn(i: int) -> list[dict]:
    return [
        {"role": "user", "content": f"u{i}"},
        {"role": "assistant", "content": f"a{i}"},
    ]


def test_compact_history_drops_oldest_pairs():
    history = _turn(0) + _turn(1) + _turn(2)
    new_history, dropped = compact_history(history, keep_last=2)
    assert len(new_history) == 4
    assert len(dropped) == 2
    assert dropped[0]["content"] == "u0"
    assert new_history[0]["content"] == "u1"


def test_compact_history_noop_when_short():
    history = _turn(0) + _turn(1)
    new_history, dropped = compact_history(history, keep_last=2)
    assert new_history is history  # reference-equal = signal "no work done"
    assert dropped == []


def test_compact_history_default_matches_constant():
    # Sanity: the default keep_last tracks the exported constant so /compact's
    # no-arg call and compact_history's default stay in sync.
    history = _turn(0) + _turn(1) + _turn(2)
    _, dropped_default = compact_history(history)
    _, dropped_explicit = compact_history(history, keep_last=COMPACT_KEEP_LAST)
    assert len(dropped_default) == len(dropped_explicit)


def test_microcompact_elides_older_tool_results():
    messages = [{"role": "system", "content": "sys"}]
    for i in range(5):
        messages.append({"role": "tool", "content": f"payload-{i}-{'x' * 100}"})

    n = microcompact(messages, keep_recent=2)
    assert n == 3  # 5 tool msgs, keep last 2, elide 3

    elided = [m for m in messages[1:] if m["content"].startswith(ELIDED_PREFIX)]
    assert len(elided) == 3
    # Last two intact — content still reflects the original payload prefix
    assert messages[-1]["content"].startswith("payload-4-")
    assert messages[-2]["content"].startswith("payload-3-")


def test_microcompact_is_idempotent():
    messages = [{"role": "tool", "content": "x" * 200} for _ in range(5)]
    microcompact(messages, keep_recent=2)
    assert microcompact(messages, keep_recent=2) == 0


def test_microcompact_noop_when_few_tool_results():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "tool", "content": "r1"},
    ]
    assert microcompact(messages, keep_recent=3) == 0


def test_trim_history_reserves_completion_tokens():
    """trim_history must reduce the target by RESERVED_COMPLETION_TOKENS,
    making it more aggressive than a raw target*num_ctx budget."""
    history = _turn(0) + _turn(1)
    num_ctx = 4000
    ctx_used = 3500  # above 0.8 * 4000 = 3200, so trimming fires
    # target_tokens = 0.5 * 4000 - 1024 = 976
    # Mock counts so the full history exceeds 976 but one turn does not.
    with patch("agent.context.count_tokens", side_effect=[1200, 500]) as mock_count:
        new_history, dropped = trim_history(list(history), ctx_used, num_ctx)
    assert mock_count.call_count == 2
    assert len(dropped) == 2
    assert len(new_history) == 2


def test_trim_history_passes_tools_and_model_name_to_count():
    """The inner count must include tool schemas and use the live
    model_name — otherwise the loop undercounts vs the gate-check value
    and stops trimming before the budget is actually met."""
    history = _turn(0) + _turn(1)
    fake_tools = [{"type": "function", "function": {"name": "x"}}]
    with patch("agent.context.count_tokens", return_value=100) as mock_count:
        trim_history(
            list(history), ctx_used=10_000, num_ctx=4000,
            tools=fake_tools, model_name="gpt-4o",
        )
    # First call inside the while loop received both kwargs.
    args, kwargs = mock_count.call_args
    assert kwargs.get("tools") is fake_tools
    assert kwargs.get("model_name") == "gpt-4o"


def test_summarize_dropped_extractive_fallback():
    """When the provider LLM call fails, summarize_dropped falls back to
    a locally-built extractive summary built from user messages only —
    history never persists tool_calls, so the fallback can't surface
    tool names."""
    dropped = [
        {"role": "user", "content": "Read the auth module and fix the bug."},
        {"role": "assistant", "content": "Done."},
    ]
    with patch("agent.context.get_active_provider") as mock_get:
        mock_provider = MagicMock()
        mock_provider.chat.side_effect = ProviderError("down")
        mock_get.return_value = mock_provider
        result = summarize_dropped(dropped)
    assert result
    assert "auth module" in result


def test_summarize_dropped_handles_list_content_user_message():
    """User messages with image attachments arrive as list-of-blocks
    content. The extractive fallback must read the text block(s) — a
    naive `.split()` on the list shape would crash and leave summary
    callers without a fallback."""
    dropped = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this screenshot of the bug"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "AAAA",
                    },
                },
            ],
        },
        {"role": "assistant", "content": "noted"},
    ]
    with patch("agent.context.get_active_provider") as mock_get:
        mock_provider = MagicMock()
        mock_provider.chat.side_effect = ProviderError("down")
        mock_get.return_value = mock_provider
        result = summarize_dropped(dropped)
    assert "screenshot of the bug" in result
