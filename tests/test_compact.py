"""Context compaction: manual /compact (compact_history) + intra-turn
microcompact. The summarization path isn't tested here because it hits
the provider — see test_apply_summary_shape for the shape-only check."""
from agent import (
    COMPACT_KEEP_LAST,
    ELIDED_PREFIX,
    compact_history,
    microcompact,
)


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
