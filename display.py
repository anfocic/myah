# display.py
"""Rich renderers for tool results.

Pure display layer — these functions only affect what the user sees in the
REPL. The model always receives the raw tool result string unchanged, so
clipping/highlighting here can't corrupt the loop.
"""
from __future__ import annotations

import difflib
import os
import re

from rich.console import Console
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


def _lexer_for(path: str) -> str | None:
    _, ext = os.path.splitext(path)
    return _EXT_TO_LEXER.get(ext.lower())


def render_diff(console: Console, path: str, old: str, new: str) -> None:
    """Unified diff of old_string → new_string for edit_file. Shown as a
    dimmed Panel so it's visually subordinate to the ↳ summary line."""
    diff = "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
            n=2,  # small context — old_string is usually already narrow
        )
    )
    if not diff.strip():
        return
    syntax = Syntax(diff, "diff", theme="ansi_dark", word_wrap=True)
    console.print(Panel(syntax, border_style="dim", padding=(0, 1)))


# Matches "   123\tactual content" produced by tools.files.read_file so we
# can strip the prefix before letting Syntax add its own line numbers.
_CAT_N_LINE = re.compile(r"^\s*\d+\t(.*)$")


def _strip_line_numbers(text: str) -> tuple[str, int]:
    """Return (code, start_line). start_line comes from the first numbered
    line, so a partial read (offset=50) still highlights with correct
    numbering."""
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
            # Trailing "(N more lines; use offset=...)" hint from read_file.
            # Skip it rather than feed it to the syntax highlighter.
            if line.startswith("..."):
                continue
            stripped.append(line)
    return "\n".join(stripped), start


# How many lines of a read_file preview to show inline. Enough to see the
# shape of a file without flooding the terminal; model still gets the rest.
PREVIEW_LINES = 15


def render_file_preview(console: Console, path: str, result: str) -> None:
    """Show the first PREVIEW_LINES of a read_file result with syntax
    highlighting. If the file is longer, show a "... (N more)" footer."""
    lexer = _lexer_for(path)
    if lexer is None:
        return  # not a language we highlight — skip to keep output terse
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
