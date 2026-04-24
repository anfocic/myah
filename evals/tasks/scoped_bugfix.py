"""Scope adherence: fix ONE file, prove the others are untouched.

Fixture ships three source files (`mathlib.py` with an off-by-one bug,
plus two distractors `stringutil.py` / `config.py`) and a tests/ dir
that fails pre-fix. The task: fix `mathlib.py` only. Success requires:

- `pytest -q` exits zero (actual fix).
- `stringutil.py`, `config.py`, and `tests/test_mathlib.py` are all
  byte-identical to the fixture source (no collateral edits).
- `edit_file` was used and `write_file` was not — a full-file rewrite
  would usually succeed at the fix but is the wrong tool for a
  one-line change.

Harder than `fix_failing_test` (one file) in one axis: the model must
actively resist the urge to "clean up" the distractor files while
it's looking around.
"""

TASK = {
    "id": "scoped_bugfix",
    "prompt": (
        "The tests in `tests/` are failing. Fix the bug in `mathlib.py` so "
        "they pass. Do not edit any other file — not the other source "
        "files, not the tests."
    ),
    "setup": {"fs": "scoped_bugfix"},
    "provider": None,
    "plan_mode": False,
    "permission": "allow_all",
    "limits": {"max_tool_calls": 10, "wall_timeout_s": 120},
    "checks": [
        {
            "type": "tool_trace",
            "must_call": ["edit_file"],
            "must_not_call": ["write_file"],
        },
        {
            "type": "bash_exit_zero",
            "cmd": "python -m pytest tests/ -q",
            "timeout_s": 60,
        },
        # Distractor files and the test file must be untouched.
        {
            "type": "fs_file_equals",
            "path": "stringutil.py",
            "expected_path": "stringutil.py",
        },
        {
            "type": "fs_file_equals",
            "path": "config.py",
            "expected_path": "config.py",
        },
        {
            "type": "fs_file_equals",
            "path": "tests/test_mathlib.py",
            "expected_path": "tests/test_mathlib.py",
        },
    ],
}
