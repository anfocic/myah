"""Thread bridge for the permission gate under the full-screen REPL.

`permissions.check_permission` runs on the turn-worker thread and must collect
a y/n/a decision — but it can't open a nested `pt_prompt` inside a live
`prompt_toolkit.Application`. The bridge inverts the flow: the worker posts a
`PermissionRequest` and blocks on its Event; the UI thread (a key binding)
reads the pending request, captures the keypress, and resolves it.

`repl/app.py` installs `PermissionBridge.ask` as the `permissions._ask_permission`
hook and drives `resolve()` from a key binding. This module has no
prompt_toolkit dependency, so the handshake is unit-testable with plain threads.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class PermissionRequest:
    """A single in-flight permission decision. `result` defaults to deny so an
    abort/exit that resolves it without an explicit choice fails safe."""

    prompt: str
    event: threading.Event = field(default_factory=threading.Event)
    result: str = "n"


class PermissionBridge:
    """Single-slot handoff between the turn worker and the UI thread.

    Only one request is ever pending: `check_permission` is called serially
    per tool on the worker, so each `ask()` blocks until resolved before the
    next is posted."""

    def __init__(self, on_change: Callable[[], None] | None = None) -> None:
        self._pending: PermissionRequest | None = None
        self._lock = threading.Lock()
        # Repaint hook — the app passes `app.invalidate` so the rail flips to
        # HOLD and the input line shows the y/n/a hint the moment one is posted.
        self.on_change: Callable[[], None] = on_change or (lambda: None)

    @property
    def pending(self) -> PermissionRequest | None:
        with self._lock:
            return self._pending

    def ask(self, prompt: str) -> str:
        """Worker side: post a request, block until the UI resolves it, return
        the choice ('y' / 'n' / 'a'). Installed as `permissions._ask_permission`."""
        req = PermissionRequest(prompt)
        with self._lock:
            self._pending = req
        self.on_change()
        req.event.wait()
        self.on_change()
        return req.result

    def resolve(self, choice: str) -> bool:
        """UI side: resolve the pending request (key binding, or 'n' on
        abort/exit). Returns True if there was one to resolve."""
        with self._lock:
            req = self._pending
            if req is None:
                return False
            req.result = choice
            self._pending = None
        req.event.set()
        return True
