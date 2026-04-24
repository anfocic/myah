"""Streaming response renderer for the REPL.

Raw append, no repaints. The previous hybrid repaint-every-80ms mode
danced around `patch_stdout` + `rich.live.Live` cursor-fight, added
flicker, and needed a switch-to-append fallback for long replies. It
traded complexity for a "feels alive because it redraws" illusion. The
real aliveness comes from each token landing as it arrives — which raw
append gives us for free and without fighting the terminal.

Trade-off: we lose end-of-stream Markdown rendering (headings, code
blocks, tables). The payoff is ~80 lines of cursor-dance gone and no
more flicker on long replies. If Markdown rendering comes back, do it
with `rich.live.Live` on the alt-screen, not by patching over it.
"""

from __future__ import annotations

from rich.console import Console


class StreamingMarkdown:
    """Write raw deltas as they arrive. Interface is a pair of methods
    that each receive the full accumulated content so callers don't have
    to track deltas. Name kept for import stability even though we no
    longer render Markdown here."""

    def __init__(self, console_: Console):
        self._console = console_
        self._started = False
        self._printed_len = 0

    def _ensure_started(self) -> None:
        if not self._started:
            self._console.print("[dim]⏺[/dim]")
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
        self._append_raw(content[self._printed_len:])
        if not content.endswith("\n"):
            self._console.print()
