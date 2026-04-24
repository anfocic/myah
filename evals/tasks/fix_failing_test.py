"""Classic red->green: buggy impl, failing test, model must fix the impl.

Fixture ships an off-by-one in `sum_range` (`range(start, end)` excludes
the upper bound). Tests assert the inclusive semantics from the docstring.
Model passes iff `pytest` exits 0 AND the test file is byte-identical to
its source — so it cannot "fix" by weakening the tests.
"""

TASK = {
    "id": "fix_failing_test",
    "prompt": (
        "The tests in the `tests/` directory are failing. Read them, fix "
        "the implementation in `calc.py` so they pass. Do not edit the "
        "tests themselves."
    ),
    "setup": {"fs": "fix_failing_test"},
    "provider": None,
    "plan_mode": False,
    "permission": "allow_all",
    "limits": {"max_tool_calls": 10, "wall_timeout_s": 240},
    "checks": [
        {
            "type": "bash_exit_zero",
            "cmd": "python -m pytest tests/ -q",
            "timeout_s": 60,
        },
        {
            "type": "fs_file_equals",
            "path": "tests/test_calc.py",
            "expected_path": "tests/test_calc.py",
        },
    ],
}
