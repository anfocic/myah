"""Glob discipline: resolve a bare filename before reading it.

Fixture hides `widget.py` inside `src/utils/` so guessing the path
(`read_file("widget.py")`, `read_file("src/widget.py")`) fails. The
right move is `glob("widget.py")` → get the resolved path → then
`read_file` on that path. `grep` also works as a resolver; what we're
punishing is wasted `read_file` calls on guessed paths before
resolution.

The trace check inspects call order: the first file-access tool used
must be `glob` or `grep`, not a speculative `read_file`.
"""
from __future__ import annotations


def _resolves_before_reading(bundle: dict) -> tuple[bool, str]:
    """Walk the trace in order; the first call in {glob, grep, read_file}
    must be a resolver (glob or grep), not read_file."""
    for e in bundle["trace"]:
        name = e["name"]
        if name in ("glob", "grep"):
            return True, ""
        if name == "read_file":
            path = (e.get("args") or {}).get("path", "?")
            return False, (
                f"read_file({path!r}) called before any glob/grep — model "
                "guessed at the path instead of resolving the bare filename"
            )
    return False, "model never attempted to access widget.py"


TASK = {
    "id": "glob_resolve_bare_name",
    "prompt": (
        "Read `widget.py` in this project and list every function it "
        "defines. Report just the function names."
    ),
    "setup": {"fs": "glob_resolve_bare_name"},
    "provider": None,
    "plan_mode": False,
    "permission": "allow_all",
    "limits": {"max_tool_calls": 6, "wall_timeout_s": 60},
    "checks": [
        _resolves_before_reading,
        {"type": "tool_trace", "must_call": ["read_file"]},
        {"type": "content_substr", "value": "render_widget"},
        {"type": "content_substr", "value": "parse_widget"},
        {"type": "content_substr", "value": "widget_names"},
    ],
}
