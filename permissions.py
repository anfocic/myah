# permissions.py
"""User-in-the-loop guard for destructive tools.

Before a sensitive tool runs, the harness pauses and asks the user to approve
the specific call. This teaches the same trust model Claude Code uses — the
model proposes; the human allows.
"""
import json

SENSITIVE_TOOLS = {"write_file", "edit_file"}
_session_allowed: set[str] = set()


def _render_args(args) -> str:
    """Pretty-print tool args, truncating long string values.

    `path` is never truncated — the user is deciding whether to authorize a
    destructive action on that specific file, so they need to see it in full.
    """
    try:
        d = dict(args)
    except (TypeError, ValueError):
        return str(args)
    for k, v in list(d.items()):
        if k == "path":
            continue
        if isinstance(v, str) and len(v) > 120:
            d[k] = v[:117] + "..."
    return json.dumps(d, indent=2, default=str)


def check_permission(console, status, name: str, args) -> bool:
    """Return True if the tool may run. Prompts the user for sensitive tools.

    `status` is a rich.Status; it must be stopped around the input() or the
    spinner corrupts the prompt. `console` is a rich.Console.
    """
    if name not in SENSITIVE_TOOLS or name in _session_allowed:
        return True

    if status:
        status.stop()

    console.print(
        f"\n[bold yellow]Permission requested[/bold yellow] [dim]for[/dim] "
        f"[bold]{name}[/bold]"
    )
    console.print(f"[dim]{_render_args(args)}[/dim]")
    choice = console.input(
        "[bold]Allow?[/bold] [dim]\\[y]es / \\[n]o / \\[a]lways ›[/dim] "
    ).strip().lower()

    if status:
        status.start()

    if choice.startswith("a"):
        _session_allowed.add(name)
        return True
    return choice.startswith("y")
