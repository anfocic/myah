"""Seed task: rename a function via edit_file in a fixture repo.

Fixture is copied to a tempdir and cwd is switched there for the run,
so the model's edits can't touch the real codebase. Grading reads the
resulting sample.py to confirm the rename landed."""

TASK = {
    "id": "edit_rename",
    "prompt": (
        "The file sample.py in the current directory defines a function "
        "named `old_name`. Rename it to `new_name`. Update any references."
    ),
    "setup": {"fs": "edit_rename"},
    "provider": None,
    "plan_mode": False,
    "permission": "allow_all",
    "limits": {"max_tool_calls": 8, "wall_timeout_s": 90},
    "checks": [
        {
            "type": "tool_trace",
            "must_call": ["edit_file"],
            "must_not_call": ["bash"],
        },
        {
            "type": "fs_file_contains",
            "path": "sample.py",
            "pattern": r"^def new_name\(",
        },
        {
            "type": "fs_file_contains",
            "path": "sample.py",
            "pattern": r"^def old_name\(",
            "negate": True,
        },
    ],
}
