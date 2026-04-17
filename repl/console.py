"""Shared rich.Console singleton. Every module in `repl/` (plus display.py
and the slash-command layer) imports from here so rich's cursor tracking
and live regions coordinate through a single instance. A per-module
Console() would usually work, but breaks down when one module's status
spinner needs to be stopped before another module prints."""
from rich.console import Console

console = Console()
