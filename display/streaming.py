"""Streaming response renderer for the REPL."""

from __future__ import annotations

import shutil
import time

from rich.console import Console
from rich.markdown import Markdown

# Throttle for mid-stream Markdown repaints. ~12 Hz — fast enough to feel live,
# slow enough that Markdown re-parsing and the cursor-up/clear/reprint flicker
# stay imperceptible. `rich.live.Live` would do this diff-based and smoother,
# but it fights `patch_stdout` over the cursor (see CONCEPTS §24/§42) — so we
# drive the same dance manually with raw ANSI, which `patch_stdout(raw=True)`
# passes through cleanly.
_REPAINT_INTERVAL_MS = 80


class StreamingMarkdown:
    """Incrementally render streamed content with a hybrid strategy.

    Starts in markdown-repaint mode for short replies. Once the rendered
    response would exceed the viewport, it permanently switches to raw
    append mode: we clear the last repaint once, print the accumulated raw
    content once, and then append deltas without trying to redraw again.
    That avoids the "stalled until finish()" failure mode on long replies.
    """

    def __init__(self, console_: Console):
        self._console = console_
        self._last_lines = 0
        self._last_ms = 0.0
        self._started = False
        self._mode = "markdown"
        self._printed_len = 0

    def _render_and_count(self, content: str) -> tuple[str, int]:
        with self._console.capture() as cap:
            self._console.print(Markdown(content, code_theme="monokai"))
        output = cap.get()
        return output, output.count("\n")

    def _ensure_started(self) -> None:
        if not self._started:
            self._console.print()
            self._started = True

    def _clear_previous(self) -> None:
        if self._last_lines > 0:
            self._console.file.write(f"\r\033[{self._last_lines}A\033[J")

    def _append_raw(self, text: str) -> None:
        if not text:
            return
        self._console.out(text, end="", highlight=False)
        self._printed_len += len(text)

    def _switch_to_append_mode(self, content: str) -> None:
        self._clear_previous()
        self._console.file.flush()
        self._mode = "append"
        self._printed_len = 0
        self._append_raw(content)
        self._last_lines = 0

    def update(self, content: str) -> None:
        if not content:
            return
        now_ms = time.monotonic() * 1000
        if now_ms - self._last_ms < _REPAINT_INTERVAL_MS:
            return

        self._ensure_started()

        if self._mode == "append":
            self._append_raw(content[self._printed_len :])
            self._last_ms = now_ms
            return

        output, n_lines = self._render_and_count(content)
        term_height = shutil.get_terminal_size().lines or 24
        if n_lines >= term_height - 2:
            self._switch_to_append_mode(content)
            self._last_ms = now_ms
            return

        self._clear_previous()
        self._console.file.write(output)
        self._console.file.flush()
        self._last_lines = n_lines
        self._last_ms = now_ms

    def finish(self, content: str) -> None:
        """Definitive render at stream end — bypasses the throttle."""
        if not content:
            return
        self._ensure_started()
        if self._mode == "append":
            self._append_raw(content[self._printed_len :])
            if not content.endswith("\n"):
                self._console.print()
            self._last_lines = 0
            return
        self._clear_previous()
        self._console.file.flush()
        self._console.print(Markdown(content, code_theme="monokai"))
        self._last_lines = 0
