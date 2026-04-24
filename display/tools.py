"""Tool event presentation for the REPL."""

from __future__ import annotations

import re

from display.previews import (
    _CAT_N_LINE,
    _parse_web_results,
    build_unified_diff,
    render_diff,
    render_file_preview,
    render_web_search_results,
)
from repl.console import console

_SALIENT_ARG_KEYS = ("path", "command", "pattern", "query")
_EXIT_RE = re.compile(r"^exit:\s*(-?\d+)\s*$", re.MULTILINE)


def _args_preview(args: dict) -> str:
    """One-line arg summary. Prefers the salient key per tool."""
    if not args:
        return ""
    for k in _SALIENT_ARG_KEYS:
        if k in args:
            v = str(args[k])
            return v if len(v) <= 70 else v[:67] + "..."
    first = next(iter(args.values()), "")
    s = str(first)
    return s if len(s) <= 70 else s[:67] + "..."


def _duration_label(meta: dict | None) -> str:
    duration_s = (meta or {}).get("duration_s")
    if duration_s is None:
        return ""
    ms = int(round(float(duration_s) * 1000))
    if ms < 1000:
        return f"{ms}ms"
    return f"{float(duration_s):.1f}s"


def _diff_line_counts(old: str, new: str) -> tuple[int, int]:
    added = 0
    removed = 0
    for line in build_unified_diff("(preview)", old, new, context=0).splitlines():
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def _count_data_lines(result: str) -> int:
    return sum(1 for line in result.splitlines() if line.strip() and not line.startswith("..."))


def _read_file_range(result: str) -> tuple[int, int] | None:
    first_line: int | None = None
    last_line: int | None = None
    for line in result.splitlines():
        m = _CAT_N_LINE.match(line)
        if not m:
            continue
        try:
            lineno = int(line.split("\t", 1)[0].strip())
        except ValueError:
            continue
        if first_line is None:
            first_line = lineno
        last_line = lineno
    if first_line is None or last_line is None:
        return None
    return first_line, last_line


def _exit_code(result: str) -> int | None:
    m = _EXIT_RE.search(result)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _first_useful_line(result: str) -> str:
    for line in result.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "[stderr]" or stripped.startswith("exit:"):
            continue
        return stripped if len(stripped) <= 80 else stripped[:77] + "..."
    return "(empty)"


def _result_summary(name: str, args: dict, result: str) -> str:
    if name == "read_file":
        path = str(args.get("path", ""))
        line_range = _read_file_range(result)
        if line_range:
            start, end = line_range
            suffix = f":{start}" if start == end else f":{start}-{end}"
            return f"{path}{suffix}"
    elif name == "edit_file":
        path = str(args.get("path", ""))
        old = str(args.get("old_string", ""))
        new = str(args.get("new_string", ""))
        added, removed = _diff_line_counts(old, new)
        return f"+{added}/-{removed} lines · {path}"
    elif name == "write_file":
        path = str(args.get("path", ""))
        content = str(args.get("content", ""))
        n_bytes = len(content.encode("utf-8"))
        n_lines = content.count("\n") + (1 if content else 0)
        line_label = "line" if n_lines == 1 else "lines"
        return f"{path} · {n_bytes} bytes · {n_lines} {line_label}"
    elif name == "grep":
        if result.strip() == "No matches.":
            return f"0 hits · {args.get('pattern', '')}"
        hits = _count_data_lines(result)
        hit_label = "hit" if hits == 1 else "hits"
        return f"{hits} {hit_label} · {args.get('pattern', '')}"
    elif name == "glob":
        if result.startswith("No files matching "):
            return f"0 paths · {args.get('pattern', '')}"
        hits = _count_data_lines(result)
        path_label = "path" if hits == 1 else "paths"
        return f"{hits} {path_label} · {args.get('pattern', '')}"
    elif name == "bash":
        exit_code = _exit_code(result)
        if exit_code is not None:
            return f"exit {exit_code} · {_first_useful_line(result)}"
    elif name == "web_search":
        hits = len(_parse_web_results(result))
        result_label = "result" if hits == 1 else "results"
        return f"{hits} {result_label} · {args.get('query', '')}"

    lines = result.splitlines()
    n = len(lines)
    if n <= 1:
        return _first_useful_line(result)
    return f"{n} lines · {_first_useful_line(result)}"


def on_tool_start(name: str, args: dict, meta: dict | None = None) -> None:
    preview = _args_preview(args)
    if preview:
        console.print(f"[dim]●[/dim] [bold]{name}[/bold] [dim]{preview}[/dim]")
    else:
        console.print(f"[dim]●[/dim] [bold]{name}[/bold]")


def on_tool_end(name: str, args: dict, result: str, ok: bool, meta: dict | None = None) -> None:
    duration = _duration_label(meta)
    duration_tail = f" [dim]· {duration}[/dim]" if duration else ""
    if not ok:
        if result.startswith("User denied"):
            console.print(f"  [dim]└[/dim] [red]denied[/red]{duration_tail}")
        elif result.startswith("Plan mode:"):
            console.print(f"  [dim]└[/dim] [yellow]blocked (plan mode)[/yellow]{duration_tail}")
        elif result.startswith("Tool raised:"):
            first = result.split("\n", 1)[0]
            console.print(f"  [dim]└[/dim] [red]{first}[/red]{duration_tail}")
        else:
            first = result.splitlines()[0] if result else "(empty)"
            console.print(f"  [dim]└ {first}[/dim]{duration_tail}")
        return
    summary = _result_summary(name, args, result)
    console.print(f"  [dim]└ {summary}[/dim]{duration_tail}")

    if name == "edit_file" and args.get("old_string") is not None:
        render_diff(
            console,
            str(args.get("path", "")),
            str(args.get("old_string", "")),
            str(args.get("new_string", "")),
        )
    elif name == "read_file" and args.get("path"):
        render_file_preview(console, str(args["path"]), result)
    elif name == "web_search":
        render_web_search_results(console, result)
