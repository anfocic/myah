"""Summarize a pre-staged diff into a Conventional Commits message.

Fixture ships `DIFF.txt` — a realistic unified diff that switches an
auth module from naive `datetime.utcnow()` to timezone-aware `datetime.now(
timezone.utc)` and adds an expired-session-cleanup branch in `validate()`,
plus a matching test. Any of `fix`/`refactor`/`feat` is a defensible type;
the check accepts all Conventional Commits type prefixes.

The `session` substring check is case-insensitive because the model may
naturally capitalize it ("Session module", "Expire sessions...").

Low max_tool_calls because this is a read+summarize task — more than one
read_file is usually thrashing.
"""

TASK = {
    "id": "commit_msg_from_diff",
    "prompt": (
        "Read `DIFF.txt` in the current directory. Write a Conventional "
        "Commits message that summarizes the change. Output only the "
        "commit message, no prose or code fences."
    ),
    "setup": {"fs": "commit_msg_from_diff"},
    "provider": None,
    "plan_mode": False,
    "permission": "allow_all",
    "limits": {"max_tool_calls": 4, "wall_timeout_s": 60},
    "checks": [
        {"type": "tool_trace", "must_call": ["read_file"]},
        {
            "type": "content_regex",
            "pattern": r"^(feat|fix|refactor|docs|test|chore|perf|build|ci|style)(\(.+?\))?!?:",
        },
        {"type": "content_substr", "value": "session", "ignorecase": True},
    ],
}
