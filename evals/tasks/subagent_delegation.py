"""Subagent-usage discipline: when a task is self-contained and
investigation-heavy, delegate it rather than burning the parent's
context on reads.

Fixture is five small modules under `lib/`. The prompt asks the model
to produce a one-line summary per file — a perfect subagent job: the
work fits in a fresh context, the parent only needs the final summary,
and the `spawn_subagent` tool description explicitly calls out
"find/summarize" investigations as the target use case.

The prompt leans on the right framing ("delegate", "keep our
conversation focused") without naming the tool — a model reading the
schema should connect the dots. Strict check: the trace must contain
at least one `spawn_subagent` call; content must mention every module
name so we know the subagent did a real pass (not an empty bounce).
"""

TASK = {
    "id": "subagent_delegation",
    "prompt": (
        "I want a quick architectural overview of the Python modules in "
        "`lib/` — one line per file describing what each one does. "
        "Please delegate this investigation to a subagent so we keep "
        "our conversation focused, and return only the final summary."
    ),
    "setup": {"fs": "subagent_delegation"},
    "provider": None,
    "plan_mode": False,
    "permission": "allow_all",
    "limits": {"max_tool_calls": 6, "wall_timeout_s": 180},
    "checks": [
        {"type": "tool_trace", "must_call": ["spawn_subagent"]},
        # Every module named so we know the subagent enumerated the dir
        # rather than fabricating a plausible-looking summary.
        {"type": "content_substr", "value": "auth"},
        {"type": "content_substr", "value": "cache"},
        {"type": "content_substr", "value": "config"},
        {"type": "content_substr", "value": "logger"},
        {"type": "content_substr", "value": "parser"},
    ],
}
