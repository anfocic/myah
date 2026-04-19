# permissions.py
"""User-in-the-loop guard for destructive tools.

Before a sensitive tool runs, the harness pauses and asks the user to approve
the specific call. This teaches the same trust model Claude Code uses — the
model proposes; the human allows.
"""
import json

from prompt_toolkit.shortcuts import prompt as pt_prompt

SENSITIVE_TOOLS = {"write_file", "edit_file", "bash", "git_checkout"}
_session_allowed: set[str] = set()


NEVER_TRUNCATE_KEYS = {"path", "command"}


def _render_args(args) -> str:
    """Pretty-print tool args, truncating long string values.

    Keys in NEVER_TRUNCATE_KEYS are shown in full — the user is authorizing
    a destructive action *on that specific target*, so truncating the path
    or command would let a long tail hide the real payload.
    """
    try:
        d = dict(args)
    except (TypeError, ValueError):
        return str(args)
    for k, v in list(d.items()):
        if k in NEVER_TRUNCATE_KEYS:
            continue
        if isinstance(v, str) and len(v) > 120:
            d[k] = v[:117] + "..."
    return json.dumps(d, indent=2, default=str)


def check_permission(console, name: str, args) -> bool:
    """Return True if the tool may run. Prompts the user for sensitive tools.

    The prompt uses `prompt_toolkit.shortcuts.prompt` directly — a one-shot
    session that slots cleanly under `patch_stdout` because the outer
    PromptSession has already returned control to the agent loop. No spinner
    to stop/restart now that rich.Status is retired (§TUI refactor)."""
    if name not in SENSITIVE_TOOLS or name in _session_allowed:
        return True

    console.print(
        f"\n[bold yellow]Permission requested[/bold yellow] [dim]for[/dim] "
        f"[bold]{name}[/bold]"
    )
    console.print(f"[dim]{_render_args(args)}[/dim]")
    try:
        choice = pt_prompt("Allow? [y]es / [n]o / [a]lways › ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        # Treat an aborted permission prompt as a denial — safer default.
        return False

    if choice.startswith("a"):
        _session_allowed.add(name)
        return True
    return choice.startswith("y")
