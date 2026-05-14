"""Streaming response renderer for the REPL.

Two-phase: stream raw token deltas as they land (the "it's alive" signal —
each token appears the instant it arrives, no repaint dance), then on
`finish()` rewind those raw lines and re-render the full reply as Markdown so
headings, bold, bullets, and code blocks come out properly formatted.

The rewind trick relies on the buffer-backed console (`repl/screen.py`): its
`RepaintBuffer` exposes `mark()` / `rewind_to()`. Against a plain `Console`
(some tests, non-full-screen callers) there's nothing to rewind, so it falls
back to pure raw append — the old behavior.
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown


class StreamingMarkdown:
    """Stream raw deltas live, then render the finished reply as Markdown.

    Interface is a pair of methods that each receive the *full* accumulated
    content so callers don't have to track deltas themselves."""

    def __init__(self, console_: Console):
        self._console = console_
        self._started = False
        self._printed_len = 0
        # Rewind point in the buffer, just after the `⏺` marker. None when the
        # console isn't buffer-backed — then we keep the raw-append behavior.
        self._mark: int | None = None

    def _buffer(self):
        """The underlying RepaintBuffer if the console is buffer-backed, else
        None. `console.file` is the buffer for `BufferConsole`."""
        buf = getattr(self._console, "file", None)
        return buf if hasattr(buf, "rewind_to") else None

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._console.print("[dim]⏺[/dim]")
        buf = self._buffer()
        if buf is not None:
            # Mark *after* the ⏺ so the marker survives the end-of-turn rewind.
            self._mark = buf.mark()
        self._started = True

    def _append_raw(self, text: str) -> None:
        if not text:
            return
        self._console.out(text, end="", highlight=False)
        self._printed_len += len(text)

    def update(self, content: str) -> None:
        if not content:
            return
        self._ensure_started()
        self._append_raw(content[self._printed_len:])

    def finish(self, content: str) -> None:
        if not content:
            return
        self._ensure_started()
        buf = self._buffer()
        if buf is not None and self._mark is not None:
            # Swap the raw stream for the rendered Markdown.
            buf.rewind_to(self._mark)
            self._console.print(Markdown(content))
        else:
            # Plain console: no rewind possible — keep the raw text as-is.
            self._append_raw(content[self._printed_len:])
            if not content.endswith("\n"):
                self._console.print()
