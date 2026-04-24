"""Pagination discipline: read_file must use offset near line 250, not
the whole 300-line file.

Fixture: `long.py` has 300 numbered filler lines with a unique token at
line 250. The model is asked for the content of line 250. Pulling the
whole file into context works but burns tokens and signals the agent
doesn't know how to paginate; using `grep` on the unique token, or
`read_file` with `offset` in the 200s, is the right approach.

The efficient-access check inspects the captured tool trace — any
`read_file` that reads a small window near the target, OR any use of
`grep`, counts as efficient. A `read_file` without offset that pulls
the whole file fails the check even if the model eventually answers
correctly.
"""
from __future__ import annotations


def _efficient_access(bundle: dict) -> tuple[bool, str]:
    """Pass iff the model used pagination or grep rather than a full read.

    Efficient strategies accepted:
    - `read_file` with `offset >= 200` (targeting the relevant window)
    - `read_file` with `limit <= 30` (small window even if offset=1)
    - any `grep` call (direct line-number lookup via content-mode regex)

    Inefficient strategies rejected:
    - `read_file` with default or missing offset AND a `limit` large
      enough to span the whole file
    """
    read_file_calls = [e for e in bundle["trace"] if e["name"] == "read_file"]
    grep_calls = [e for e in bundle["trace"] if e["name"] == "grep"]

    if grep_calls:
        return True, ""

    if not read_file_calls:
        return False, "no read_file or grep calls — model did not access the file"

    for e in read_file_calls:
        args = e.get("args") or {}
        offset = int(args.get("offset") or 1)
        limit = args.get("limit")
        limit = int(limit) if limit is not None else None
        if offset >= 200:
            return True, ""
        if limit is not None and limit <= 30:
            return True, ""

    return False, (
        "read_file called without pagination — model pulled the whole "
        "300-line file instead of targeting line 250. "
        f"calls: {[e.get('args') for e in read_file_calls]}"
    )


TASK = {
    "id": "pagination_read",
    "prompt": (
        "In `long.py`, what is the content of line 250? Answer with the "
        "exact content of that line and nothing else."
    ),
    "setup": {"fs": "pagination_read"},
    "provider": None,
    "plan_mode": False,
    "permission": "allow_all",
    "limits": {"max_tool_calls": 4, "wall_timeout_s": 60},
    "checks": [
        {"type": "content_substr", "value": "THE_ANSWER_IS_mia_agent_harness_42"},
        _efficient_access,
    ],
}
