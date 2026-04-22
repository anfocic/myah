"""Static Rich renderers and parsing helpers for tool outputs."""

from __future__ import annotations

import difflib
import os
import re

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax

# Map file extensions to Pygments language names that rich.syntax knows.
# Unknown extensions fall back to plain text via the None case.
_EXT_TO_LEXER = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".sh": "bash",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
}

# Matches "   123\tactual content" produced by tools.files.read_file so we
# can strip the prefix before letting Syntax add its own line numbers.
_CAT_N_LINE = re.compile(r"^\s*\d+\t(.*)$")

# Parses the "[n] Title" / "URL: ..." pairs emitted by tools.web_search.
# Keeping this display-side (rather than changing the tool output) means the
# model still reads a plain text summary while the user gets OSC 8 hyperlinks.
_WEB_RESULT_HEAD = re.compile(r"^\[(\d+)\] (.+)$")

# How many lines of a read_file preview to show inline. Enough to see the
# shape of a file without flooding the terminal; model still gets the rest.
PREVIEW_LINES = 15


def _lexer_for(path: str) -> str | None:
    _, ext = os.path.splitext(path)
    return _EXT_TO_LEXER.get(ext.lower())


def build_unified_diff(path: str, old: str, new: str, *, context: int = 2) -> str:
    """Return a unified diff for display and permission previews."""
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
            n=context,
        )
    )


def render_diff(console: Console, path: str, old: str, new: str) -> None:
    """Unified diff of old_string → new_string for edit_file."""
    diff = build_unified_diff(path, old, new, context=2)
    if not diff.strip():
        return
    syntax = Syntax(diff, "diff", theme="ansi_dark", word_wrap=True)
    console.print(Panel(syntax, border_style="dim", padding=(0, 1)))


def _strip_line_numbers(text: str) -> tuple[str, int]:
    """Return (code, start_line) for numbered read_file output."""
    lines = text.splitlines()
    stripped: list[str] = []
    start = 1
    first_seen = False
    for line in lines:
        m = _CAT_N_LINE.match(line)
        if m:
            if not first_seen:
                try:
                    start = int(line.split("\t", 1)[0].strip())
                except ValueError:
                    pass
                first_seen = True
            stripped.append(m.group(1))
        else:
            if line.startswith("..."):
                continue
            stripped.append(line)
    return "\n".join(stripped), start


def render_file_preview(console: Console, path: str, result: str) -> None:
    """Show the first PREVIEW_LINES of a read_file result with highlighting."""
    lexer = _lexer_for(path)
    if lexer is None:
        return
    code, start_line = _strip_line_numbers(result)
    lines = code.splitlines()
    if not lines:
        return
    truncated = lines[:PREVIEW_LINES]
    shown = "\n".join(truncated)
    syntax = Syntax(
        shown,
        lexer,
        theme="ansi_dark",
        line_numbers=True,
        start_line=start_line,
        word_wrap=False,
    )
    more = len(lines) - len(truncated)
    title = f"[dim]{path}[/dim]" + (
        f" [dim](showing {len(truncated)} of {len(lines)})[/dim]" if more > 0 else ""
    )
    console.print(Panel(syntax, title=title, border_style="dim", padding=(0, 1)))


def _parse_web_results(result: str) -> list[tuple[int, str, str]]:
    entries: list[tuple[int, str, str]] = []
    index: int | None = None
    title: str | None = None
    for line in result.splitlines():
        m = _WEB_RESULT_HEAD.match(line)
        if m:
            index = int(m.group(1))
            title = m.group(2).strip()
            continue
        if line.startswith("URL: ") and index is not None:
            url = line[5:].strip()
            if url and url != "No URL":
                entries.append((index, title or "Untitled", url))
            index = None
            title = None
    return entries


def render_web_search_results(console: Console, result: str) -> None:
    """Panel of clickable result titles (OSC 8) with the URL shown beneath."""
    entries = _parse_web_results(result)
    if not entries:
        return
    lines: list[str] = []
    for n, title, url in entries:
        lines.append(f"[dim]\\[{n}][/dim] [link={url}]{escape(title)}[/link]")
        lines.append(f"    [dim]{escape(url)}[/dim]")
    console.print(Panel("\n".join(lines), border_style="dim", padding=(0, 1)))
