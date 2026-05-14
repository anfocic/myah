"""Screen plumbing for the full-screen REPL — the pieces between `rich` and
`prompt_toolkit` that have no UI-framework dependency and are unit-testable on
their own.

- `RepaintBuffer` — a file-like sink `rich.Console` writes into. Keeps an
  append-only list of rendered ANSI lines so the scrolling main pane can slice
  a viewport in O(viewport) rather than re-parsing the whole history per frame.
- `BufferConsole` — a `rich.Console` pointed at a `RepaintBuffer`, with
  `status()` neutered (a static buffer can't animate a spinner).
- `_NullStatus` — the no-op spinner stand-in.

`repl/app.py` owns the `prompt_toolkit.Application`; this module owns nothing
that imports it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from rich.console import Console


class _NullStatus:
    """No-op stand-in for `rich`'s `console.status()`.

    A buffer-backed console can't animate a spinner, so liveness comes from
    streamed tokens landing + the rail's STREAM state instead. `agent/loop.py`
    drives the status object with explicit `.start()` / `.stop()` calls *and*
    `main.py` uses it as a context manager — both paths are covered here."""

    def start(self) -> None:  # noqa: D102
        pass

    def stop(self) -> None:  # noqa: D102
        pass

    def update(self, *args, **kwargs) -> None:  # noqa: D102
        pass

    def __enter__(self) -> _NullStatus:
        return self

    def __exit__(self, *exc) -> bool:
        return False


class RepaintBuffer:
    """An append-only, thread-safe sink for `rich.Console` output.

    `rich` calls `write()` with chunks of already-rendered ANSI text; we split
    on newlines into `lines` (complete) plus a `_partial` tail (the in-progress
    last line, e.g. mid-stream tokens). The UI thread reads a viewport slice via
    `view()`; writes come from the turn worker and tool-worker threads. A single
    lock guards both. Every mutation calls `on_change` so the app can repaint."""

    def __init__(self, on_change: Callable[[], None] | None = None) -> None:
        self.lines: list[str] = []
        self._partial: str = ""
        self._lock = threading.RLock()
        # Default no-op so RepaintBuffer is usable without an Application
        # (boot phase, tests). `repl/app.py` swaps in `app.invalidate`.
        self.on_change: Callable[[], None] = on_change or (lambda: None)

    # -- file-like interface (what rich.Console writes to) --------------------

    def write(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            combined = self._partial + text
            parts = combined.split("\n")
            self.lines.extend(parts[:-1])
            self._partial = parts[-1]
        self.on_change()

    def flush(self) -> None:
        # rich flushes after each print/out; nothing to do — a flush must not
        # promote the in-progress `_partial` to a committed line.
        pass

    # -- read side (what the main pane renders from) --------------------------

    def line_count(self) -> int:
        with self._lock:
            return len(self.lines) + (1 if self._partial else 0)

    def view(self, top: int, count: int) -> list[str]:
        """Return the `count` logical lines starting at `top`, clamped. The
        in-progress `_partial` counts as the final logical line."""
        with self._lock:
            alllines = self.lines + ([self._partial] if self._partial else [])
        top = max(0, top)
        return alllines[top:top + count]

    def clear(self) -> None:
        """Wipe the scrollback — backs the `/clear` slash command."""
        with self._lock:
            self.lines = []
            self._partial = ""
        self.on_change()


class ScrollState:
    """Tracks the main pane's vertical scroll position.

    `follow_tail` is the load-bearing bit: while True, new output keeps the
    viewport pinned to the bottom (the common case — you want to see the latest
    tokens). PageUp / wheel-up disengages it (you're reading back); scrolling
    or paging back down to the bottom re-engages it. All methods are pure
    arithmetic on `(total_lines, height)` — no prompt_toolkit dependency, so
    the behavior is unit-testable in isolation."""

    def __init__(self) -> None:
        self.scroll_top: int = 0
        self.follow_tail: bool = True

    @staticmethod
    def _max_top(total_lines: int, height: int) -> int:
        return max(0, total_lines - height)

    def on_content(self, total_lines: int, height: int) -> None:
        """Reconcile after the buffer changed: snap to bottom if tailing,
        otherwise just keep the existing position in bounds."""
        max_top = self._max_top(total_lines, height)
        if self.follow_tail:
            self.scroll_top = max_top
        else:
            self.scroll_top = max(0, min(self.scroll_top, max_top))

    def page_up(self, height: int) -> None:
        self.follow_tail = False
        self.scroll_top = max(0, self.scroll_top - height)

    def page_down(self, total_lines: int, height: int) -> None:
        max_top = self._max_top(total_lines, height)
        self.scroll_top = min(max_top, self.scroll_top + height)
        self.follow_tail = self.scroll_top >= max_top

    def scroll(self, delta: int, total_lines: int, height: int) -> None:
        """Relative scroll for the mouse wheel — `delta < 0` up, `> 0` down."""
        max_top = self._max_top(total_lines, height)
        self.scroll_top = max(0, min(self.scroll_top + delta, max_top))
        self.follow_tail = self.scroll_top >= max_top

    def to_bottom(self, total_lines: int, height: int) -> None:
        """Jump to the latest output and resume tailing (the End key)."""
        self.scroll_top = self._max_top(total_lines, height)
        self.follow_tail = True


class BufferConsole(Console):
    """A `rich.Console` whose output flows into a `RepaintBuffer` instead of a
    terminal. `force_terminal=True` keeps ANSI escapes flowing (the buffer is
    not a TTY); `width` is explicit for the same reason — set it to the main
    pane's width. `status()` is neutered to a `_NullStatus`."""

    def __init__(self, buffer: RepaintBuffer, *, width: int) -> None:
        super().__init__(file=buffer, force_terminal=True, width=width)

    def status(self, *args, **kwargs) -> _NullStatus:  # type: ignore[override]
        return _NullStatus()
