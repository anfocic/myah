"""Shared rich.Console singleton. Every module in `repl/` (plus display.py
and the slash-command layer) imports from here so styling stays coherent
and future shared state (themes, log files) has a single seat.

`force_terminal=True` is load-bearing: once the REPL is wrapped in
`prompt_toolkit.patch_stdout`, `sys.stdout` becomes a non-TTY proxy and
Rich will otherwise auto-strip colors. Forcing terminal mode keeps the
ANSI escapes flowing through patch_stdout, which re-emits them above the
pinned input line."""
from rich.console import Console

console = Console(force_terminal=True)
