"""TDD: satisfy a full provided test suite by implementing a stub.

Fixture ships `stringutil.slugify` as `raise NotImplementedError` plus a
tests/ dir with 8 cases that pin lowercasing, run-collapsing, edge
stripping, empty input, all-separator input, non-ASCII stripping, and
digit preservation. The docstring spells out the contract.

Model passes iff pytest exits 0 AND the stub's NotImplementedError is
gone — guards against the model deleting the function entirely or
returning a hardcoded constant that happens to pass.
"""

TASK = {
    "id": "tdd_new_fn",
    "prompt": (
        "Implement `slugify` in `stringutil.py` so that the tests in "
        "`tests/` all pass. The docstring describes the required "
        "behavior. Do not edit the tests."
    ),
    "setup": {"fs": "tdd_new_fn"},
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
            "type": "fs_grep_count",
            "path": "stringutil.py",
            "pattern": r"NotImplementedError",
            "expected": 0,
        },
    ],
}
