"""REPL UI helpers: prompt chrome, hint line, context-bar tag, tab
completion for slash commands. All cosmetic — the model sees none of
this; every line here is for the human at the keyboard."""
import readline
import subprocess

from repl.console import console
from repl.state import State


def _current_branch() -> str | None:
    """Best-effort current branch name. Returns None outside a repo or if
    git is missing. Called once per prompt; cheap enough to skip caching."""
    try:
        out = subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def build_prompt(state: State) -> str:
    """`You [branch · plan · debug] ›` — badges only rendered when the
    condition applies so the prompt stays clean in the common case."""
    parts = []
    branch = _current_branch()
    if branch:
        parts.append(branch)
    if state.get("plan_mode"):
        parts.append("[yellow]plan[/yellow]")
    if state.get("debug"):
        parts.append("[magenta]debug[/magenta]")
    badge = f" [dim]\\[{' · '.join(parts)}][/dim]" if parts else ""
    return f"[bold magenta]You[/bold magenta]{badge} [dim]›[/dim] "


def print_hint() -> None:
    console.print(
        "[dim]/help · /plan · /model · /compact · /rewind · "
        "ctrl+c to interrupt[/dim]"
    )


def install_slash_completer(commands: dict) -> None:
    """Bind Tab-completion of slash commands against the given registry.
    Takes `commands` as an arg instead of importing from repl.commands to
    avoid a circular dependency (commands.py already imports ctx_tag from
    this module).

    readline calls the completer repeatedly with increasing state indices
    until it returns None. We match only when the buffer starts with "/"
    so normal prose input stays untouched."""
    def completer(text: str, state: int):
        buf = readline.get_line_buffer()
        if not buf.startswith("/"):
            return None
        matches = [c for c in commands if c.startswith(buf)]
        return matches[state] if state < len(matches) else None

    readline.set_completer(completer)
    readline.set_completer_delims("")  # treat "/" as part of the word
    readline.parse_and_bind("tab: complete")


def ctx_tag(ctx_used: int, ctx_total: int) -> str:
    """Colored `[N%]` marker for the post-turn status line. Thresholds
    mirror trim_history's hysteresis bounds — green below 50% (cheap),
    yellow 50–80% (summarization incoming), red above 80% (auto-trim
    fires)."""
    pct = ctx_used / ctx_total
    if pct < 0.5:
        color = "green"
    elif pct < 0.8:
        color = "yellow"
    else:
        color = "red"
    return f"[dim]\\[[/dim][{color}]{pct:.0%}[/{color}][dim]][/dim]"
