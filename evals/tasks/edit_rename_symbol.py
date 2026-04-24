"""Multi-file rename: change `compute_area` to `calculate_area` everywhere.

Fixture has three files that reference the symbol: `shapes.py` (definition),
`calculator.py` (importer + caller), and `tests/test_shapes.py` (import
+ assertions). A successful run updates all three *and* keeps pytest
green — if the model forgets the test file, the import fails and pytest
exits nonzero, catching a half-done rename that grep alone wouldn't.

Why this exercises the right capabilities:
- Use `grep` to enumerate every reference site before editing (not
  `read_file`-ing each file blindly).
- Use `edit_file` per site with enough surrounding context to stay
  surgical; `write_file` would be wasteful and error-prone.
- Post-edit verification via the test suite — the agent should
  implicitly trust tests as the ground truth.

Stricter than `fix_failing_test`: that task touches one file; this one
requires coordinated edits across three, and a missing site fails the
test run.
"""

TASK = {
    "id": "edit_rename_symbol",
    "prompt": (
        "Rename the function `compute_area` to `calculate_area` everywhere "
        "in this codebase — the definition and every call site. After the "
        "rename, the tests in `tests/` must still pass."
    ),
    "setup": {"fs": "edit_rename"},
    "provider": None,
    "plan_mode": False,
    "permission": "allow_all",
    "limits": {"max_tool_calls": 12, "wall_timeout_s": 120},
    "checks": [
        {
            "type": "tool_trace",
            "must_call": ["grep", "edit_file"],
            "must_not_call": ["write_file"],
        },
        # Old name fully removed.
        {
            "type": "fs_grep_count",
            "path": "shapes.py",
            "pattern": r"\bcompute_area\b",
            "expected": 0,
        },
        {
            "type": "fs_grep_count",
            "path": "calculator.py",
            "pattern": r"\bcompute_area\b",
            "expected": 0,
        },
        {
            "type": "fs_grep_count",
            "path": "tests/test_shapes.py",
            "pattern": r"\bcompute_area\b",
            "expected": 0,
        },
        # New name in each file at least once.
        {
            "type": "fs_grep_count",
            "path": "shapes.py",
            "pattern": r"\bcalculate_area\b",
            "expected": 1,
            "op": "ge",
        },
        {
            "type": "fs_grep_count",
            "path": "calculator.py",
            "pattern": r"\bcalculate_area\b",
            "expected": 1,
            "op": "ge",
        },
        {
            "type": "fs_grep_count",
            "path": "tests/test_shapes.py",
            "pattern": r"\bcalculate_area\b",
            "expected": 1,
            "op": "ge",
        },
        # Final ground truth: the rename is consistent if pytest is green.
        {
            "type": "bash_exit_zero",
            "cmd": "python -m pytest tests/ -q",
            "timeout_s": 60,
        },
    ],
}
