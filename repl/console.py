"""Shared rich.Console proxy. Every module in `repl/` (plus the display and
slash-command layers) imports `console` from here, so styling stays coherent
and the underlying output sink can be swapped in exactly one place.

The proxy indirection is load-bearing: the full-screen REPL (`repl/app.py`)
swaps the inner Console for one whose `file=` is an in-memory repaint buffer
feeding the scrolling main pane. Because every call site does
`from repl.console import console` — a *name* binding — a plain module-global
rebind would never reach them. Delegating through a proxy object does.
"""
from rich.console import Console


class _ConsoleProxy:
    """Transparent stand-in for a rich.Console.

    Forwards every attribute access to a swappable inner Console, so the
    full-screen app can redirect all harness output into its buffer-backed
    console at startup without touching a single import site."""

    def __init__(self, inner: Console) -> None:
        # object.__setattr__ so we don't recurse through our own __setattr__.
        object.__setattr__(self, "_inner", inner)

    def _set_inner(self, inner: Console) -> None:
        """Swap the delegate. Called once by `repl/app.py` at startup."""
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name: str):
        # __getattr__ only fires when normal lookup misses, i.e. for everything
        # that isn't `_inner` / `_set_inner` — which is every Console method.
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(object.__getattribute__(self, "_inner"), name, value)


# Pre-app default: a plain terminal Console. Good enough for imports, tests,
# and the boot phase before `repl/app.py` swaps in the buffer-backed console.
console = _ConsoleProxy(Console(force_terminal=True))
