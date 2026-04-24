"""Multi-turn task: fix a bug, then add a follow-up test.

Turn 1 — classic red → green. Fixture ships `calc.divide` returning
`a + b`; tests pin the correct quotient behavior. Model edits
`calc.py` so pytest passes.

Turn 2 — model picks up right where it left off (same history, same
cwd, same conversation) and is asked to extend the test file with a
divide-by-zero case. Python's `/` raises `ZeroDivisionError` natively,
so this is purely a test-writing exercise, not another source fix.

What this actually exercises vs single-turn tasks:
- History / context carryover: turn 2's prompt only makes sense if
  the model still knows what `divide` is from turn 1.
- Incremental edit discipline: turn 2 must add *to* the test file
  without breaking the tests that passed after turn 1.
- Conversational continuation — the fs_grep_count checks all fire
  after both turns, so the model has to get both right.
"""

TASK = {
    "id": "multi_turn_fix",
    "turns": [
        (
            "The tests in `tests/` are failing. Read the code and the tests, "
            "then fix the bug in `calc.py` so pytest passes. Do not modify "
            "the tests."
        ),
        (
            "Good. Now add a test case to `tests/test_calc.py` that asserts "
            "`divide(1, 0)` raises `ZeroDivisionError`. Keep the existing "
            "tests intact."
        ),
    ],
    "setup": {"fs": "multi_turn_fix"},
    "provider": None,
    "plan_mode": False,
    "permission": "allow_all",
    "limits": {"max_tool_calls": 14, "wall_timeout_s": 240},
    "checks": [
        # Both turns actually used edit_file — no single-shot rewrites.
        {"type": "tool_trace", "must_call": ["edit_file"]},
        # After both turns: pytest green (includes the new test).
        {
            "type": "bash_exit_zero",
            "cmd": "python -m pytest tests/ -q",
            "timeout_s": 60,
        },
        # The new divide-by-zero assertion landed in the test file.
        {
            "type": "fs_grep_count",
            "path": "tests/test_calc.py",
            "pattern": r"ZeroDivisionError",
            "expected": 1,
            "op": "ge",
        },
        {
            "type": "fs_grep_count",
            "path": "tests/test_calc.py",
            "pattern": r"divide\s*\(\s*1\s*,\s*0\s*\)",
            "expected": 1,
            "op": "ge",
        },
        # The three original test functions still live in the file —
        # guards against a model that deletes-and-rewrites in turn 2.
        {
            "type": "fs_grep_count",
            "path": "tests/test_calc.py",
            "pattern": r"def test_divide_integer_result",
            "expected": 1,
        },
        {
            "type": "fs_grep_count",
            "path": "tests/test_calc.py",
            "pattern": r"def test_multiply_unaffected",
            "expected": 1,
        },
    ],
}
