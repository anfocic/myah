"""Seed task: find a literal across the repo via grep.

No fixture — runs in the actual repo cwd. The check is that the model
used grep (not bash with grep, not a shotgun read_file) and that the
answer names the two files the symbol really appears in."""

TASK = {
    "id": "find_string",
    "prompt": (
        "Find every occurrence of the exact identifier "
        "`TOOL_RESULT_MAX_BYTES` in this codebase and list the files "
        "that contain it. Use the grep tool."
    ),
    "setup": {"fs": None},
    "provider": None,
    "plan_mode": False,
    "permission": "allow_all",
    "limits": {"max_tool_calls": 6, "wall_timeout_s": 90},
    "checks": [
        {
            "type": "tool_trace",
            "must_call": ["grep"],
            "must_not_call": ["write_file", "edit_file", "bash"],
        },
        {"type": "content_regex", "pattern": r"config\.py"},
        {"type": "content_regex", "pattern": r"agent/tokens\.py"},
    ],
}
