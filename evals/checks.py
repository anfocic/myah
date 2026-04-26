"""Check dispatch for the eval runner.

Each check function takes `(check, bundle)` and returns `(passed, why)`.
`bundle` is the post-run snapshot the runner assembles:

    {
        "content":    str,                       # final assistant message
        "trace":      list[dict],                # one entry per tool call
        "stats":      dict,                      # from run_agent's 4th return
        "ctx_used":   int,                       # from run_agent's 3rd return
        "cwd":        pathlib.Path,              # task's working dir
        "fixture_dir": pathlib.Path | None,      # source of fs fixtures
    }

A check dict needs a "type" that matches a key in CHECKS below. A bare
callable is also accepted by `dispatch`, for tasks that want custom
Python logic without registering a new type.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _tool_trace(check: dict, bundle: dict) -> tuple[bool, str]:
    called = [e["name"] for e in bundle["trace"]]
    must_call = check.get("must_call", [])
    must_not_call = check.get("must_not_call", [])
    missing = [n for n in must_call if n not in called]
    forbidden = [n for n in must_not_call if n in called]
    if missing:
        return False, f"expected calls not made: {missing} (called: {called})"
    if forbidden:
        return False, f"forbidden calls made: {forbidden}"
    limit = check.get("call_count_max")
    if limit is not None and len(called) > limit:
        return False, f"too many tool calls: {len(called)} > {limit}"
    return True, ""


def _content_regex(check: dict, bundle: dict) -> tuple[bool, str]:
    pattern = check["pattern"]
    # MULTILINE so `^`/`$` anchor to line boundaries. Models often prefix
    # their reply with a blank line or a short lead-in ("Here is the commit
    # message:\n\nfeat(...): ..."); a strict start-of-string `^` would fail
    # those even though the line itself matches. `_fs_file_contains` already
    # uses MULTILINE for the same reason.
    flags = re.MULTILINE | (re.IGNORECASE if check.get("ignorecase") else 0)
    found = re.search(pattern, bundle["content"], flags) is not None
    if check.get("negate"):
        return (not found, "" if not found else f"pattern unexpectedly matched: {pattern!r}")
    return (found, "" if found else f"pattern not found: {pattern!r}")


def _content_substr(check: dict, bundle: dict) -> tuple[bool, str]:
    needle = check["value"]
    content = bundle["content"]
    if check.get("ignorecase"):
        found = needle.lower() in content.lower()
    else:
        found = needle in content
    if check.get("negate"):
        return (not found, "" if not found else f"substring unexpectedly present: {needle!r}")
    return (found, "" if found else f"substring not found: {needle!r}")


def _resolve_path(check: dict, bundle: dict) -> Path:
    return Path(bundle["cwd"]) / check["path"]


def _normalize_python_cmd(cmd: str) -> str:
    """Run eval Python checks with the interpreter running Myah.

    The eval tasks say `python -m pytest ...` because that is the command a
    human would type. Inside a REPL launched from another environment, plain
    `python` (or `python3`) can resolve to a global interpreter with no
    pytest installed. Normalize either prefix to Myah's own interpreter so
    checks see the same environment the harness runs under.
    """
    quoted = shlex.quote(sys.executable)
    for prefix in ("python3", "python"):
        if cmd == prefix:
            return quoted
        if cmd.startswith(prefix + " "):
            return f"{quoted} {cmd.removeprefix(prefix + ' ')}"
    return cmd


def _fs_file_equals(check: dict, bundle: dict) -> tuple[bool, str]:
    target = _resolve_path(check, bundle)
    if not target.exists():
        return False, f"file not found: {target}"
    actual = target.read_bytes()
    if "expected" in check:
        expected = (
            check["expected"].encode() if isinstance(check["expected"], str) else check["expected"]
        )
    elif "expected_path" in check:
        src = Path(check["expected_path"])
        if not src.is_absolute() and bundle.get("fixture_dir"):
            src = bundle["fixture_dir"] / src
        expected = src.read_bytes()
    else:
        return False, "fs_file_equals needs `expected` or `expected_path`"
    if actual == expected:
        return True, ""
    return False, f"file contents differ: {target}"


def _fs_file_contains(check: dict, bundle: dict) -> tuple[bool, str]:
    target = _resolve_path(check, bundle)
    if not target.exists():
        return False, f"file not found: {target}"
    text = target.read_text(errors="replace")
    flags = re.MULTILINE | (re.IGNORECASE if check.get("ignorecase") else 0)
    found = re.search(check["pattern"], text, flags) is not None
    if check.get("negate"):
        return (
            not found,
            "" if not found else f"pattern unexpectedly matched in {target}: {check['pattern']!r}",
        )
    return (found, "" if found else f"pattern not found in {target}: {check['pattern']!r}")


def _bash_exit_zero(check: dict, bundle: dict) -> tuple[bool, str]:
    # Executable ground truth. Run a shell command in the task's cwd
    # (or a subdir) and pass iff exit == 0. Use for pytest, py_compile,
    # ruff — anything that already exits nonzero on failure.
    cmd = _normalize_python_cmd(check["cmd"])
    cwd = Path(bundle["cwd"])
    cwd_rel = check.get("cwd_rel")
    if cwd_rel:
        cwd = cwd / cwd_rel
    timeout_s = check.get("timeout_s", 30)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"cmd timed out after {timeout_s}s: {cmd!r}"
    if proc.returncode == 0:
        return True, ""
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
    return False, f"exit {proc.returncode}: {cmd!r} | {' | '.join(tail)}"


def _fs_grep_count(check: dict, bundle: dict) -> tuple[bool, str]:
    # Count regex matches in a file, compare against `expected` using op.
    # Stronger than fs_file_contains when "zero old references left" or
    # "new symbol used in N places" is the actual contract.
    target = _resolve_path(check, bundle)
    if not target.exists():
        return False, f"file not found: {target}"
    text = target.read_text(errors="replace")
    flags = re.MULTILINE | (re.IGNORECASE if check.get("ignorecase") else 0)
    count = len(re.findall(check["pattern"], text, flags))
    expected = check["expected"]
    op = check.get("op", "eq")
    if op not in ("eq", "ge", "le"):
        return False, f"fs_grep_count: unknown op {op!r} (use eq/ge/le)"
    ok = (
        (op == "eq" and count == expected)
        or (op == "ge" and count >= expected)
        or (op == "le" and count <= expected)
    )
    if ok:
        return True, ""
    return (
        False,
        f"grep count mismatch in {target}: {check['pattern']!r} found {count}, "
        f"expected {op} {expected}",
    )


def _python(check: dict, bundle: dict) -> tuple[bool, str]:
    # Escape hatch: task supplies a `fn` callable that returns bool or
    # (bool, why). Keeps bespoke logic inline with the task instead of
    # forcing a new check type for every one-off.
    fn: Callable[[dict], Any] = check["fn"]
    result = fn(bundle)
    if isinstance(result, tuple):
        ok, why = result
        return bool(ok), str(why)
    return bool(result), "" if result else "python check returned falsy"


CHECKS: dict[str, Callable[[dict, dict], tuple[bool, str]]] = {
    "tool_trace": _tool_trace,
    "content_regex": _content_regex,
    "content_substr": _content_substr,
    "fs_file_equals": _fs_file_equals,
    "fs_file_contains": _fs_file_contains,
    "fs_grep_count": _fs_grep_count,
    "bash_exit_zero": _bash_exit_zero,
    "python": _python,
}


def dispatch(check: dict | Callable, bundle: dict) -> tuple[bool, str]:
    if callable(check):
        return _python({"fn": check}, bundle)
    kind = check.get("type")
    if not isinstance(kind, str):
        return False, f"unknown check type: {kind!r}"
    fn = CHECKS.get(kind)
    if fn is None:
        return False, f"unknown check type: {kind!r}"
    try:
        return fn(check, bundle)
    except Exception as e:
        return False, f"{kind} check raised: {type(e).__name__}: {e}"
