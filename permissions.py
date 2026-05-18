# permissions.py
"""User-in-the-loop guard for destructive tools.

Before a sensitive tool runs, the harness pauses and asks the user to approve
the specific call. This teaches the same trust model Claude Code uses — the
model proposes; the human allows.
"""
import json
from pathlib import Path

from prompt_toolkit.shortcuts import prompt as pt_prompt
from rich import box
from rich.panel import Panel
from rich.syntax import Syntax

from display import build_unified_diff, phosphor

SENSITIVE_TOOLS = {
    "write_file", "edit_file", "bash", "git_checkout",
    "note_write", "note_append", "daily_note",
}
_session_allowed: set[str] = set()


NEVER_TRUNCATE_KEYS = {"path", "command"}
_PREVIEW_LINES = 12
_RISK_LABELS = {
    "bash": "shell command",
    "edit_file": "file edit",
    "write_file": "file write",
    "git_checkout": "git state change",
    "note_write": "vault note write",
    "note_append": "vault note append",
    "daily_note": "vault daily note write",
}


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


def _tool_id(meta: dict | None) -> str:
    return str((meta or {}).get("tool_id", ""))


def _render_halt_frame(console, name: str, risk: str, tool_id: str) -> None:
    """The design's HALT frame: a red double-ruled panel that stops the turn
    cold for a permission decision. The detailed diff/command preview still
    renders below it — the frame is the alarm, the preview is the evidence."""
    rows = [
        phosphor.bracket("HALT · PERMISSION REQUESTED", "red"),
        "",
        f"  [{phosphor.DIM}]tool[/]   [bold]{name}[/]",
        f"  [{phosphor.DIM}]risk[/]   [{phosphor.RED}]{risk}[/]",
    ]
    if tool_id:
        rows.append(f"  [{phosphor.DIM}]id[/]     [{phosphor.DIM}]{tool_id}[/]")
    rows += [
        "",
        f"  [{phosphor.DIM}][[/][bold]y[/][{phosphor.DIM}]] allow once    "
        f"[[/][bold]a[/][{phosphor.DIM}]] session    "
        f"[[/][bold]n[/][{phosphor.DIM}]] deny[/]",
    ]
    console.print()
    console.print(
        Panel(
            "\n".join(rows),
            border_style="red",
            box=box.DOUBLE,
            padding=(0, 1),
        )
    )


def _content_stats(content: str) -> tuple[int, int]:
    n_bytes = len(content.encode("utf-8"))
    n_lines = content.count("\n") + (1 if content else 0)
    return n_bytes, n_lines


def _content_preview(content: str, *, max_lines: int = _PREVIEW_LINES) -> str:
    lines = content.splitlines()
    if len(lines) <= max_lines:
        return content
    preview = "\n".join(lines[:max_lines])
    return preview + f"\n... ({len(lines) - max_lines} more lines)"


def _print_permission_preview(console, name: str, args) -> None:
    if name == "bash":
        command = str(args.get("command", ""))
        cwd = str(args.get("cwd", "."))
        timeout = int(args.get("timeout", 30))
        console.print(f"[dim]cwd[/dim] {cwd} [dim]· timeout[/dim] {timeout}s")
        console.print(
            Panel(
                Syntax(command, "bash", theme="ansi_dark", word_wrap=True),
                title="[dim]command[/dim]",
                border_style="dim",
                padding=(0, 1),
            )
        )
        return

    if name == "edit_file":
        path = str(args.get("path", ""))
        replace_all = bool(args.get("replace_all", False))
        old = str(args.get("old_string", ""))
        new = str(args.get("new_string", ""))
        mode = "replace all" if replace_all else "single replace"
        console.print(f"[dim]path[/dim] {path} [dim]· mode[/dim] {mode}")
        diff = build_unified_diff(path, old, new, context=1)
        if diff.strip():
            console.print(
                Panel(
                    Syntax(diff, "diff", theme="ansi_dark", word_wrap=True),
                    title="[dim]diff preview[/dim]",
                    border_style="dim",
                    padding=(0, 1),
                )
            )
        else:
            console.print("[dim]No textual diff to preview.[/dim]")
        return

    if name == "write_file":
        path = str(args.get("path", ""))
        content = str(args.get("content", ""))
        n_bytes, n_lines = _content_stats(content)
        line_label = "line" if n_lines == 1 else "lines"

        # If the target already exists, the model is overwriting — show a
        # diff so the user can see exactly what's changing, the same way
        # edit_file's preview does. Only fall back to a raw content preview
        # for true creation cases.
        existing: str | None = None
        try:
            p = Path(path)
            if p.is_file():
                existing = p.read_text()
        except OSError:
            existing = None

        mode = "overwrite" if existing is not None else "create"
        console.print(
            f"[dim]path[/dim] {path} [dim]· mode[/dim] {mode} "
            f"[dim]· size[/dim] {n_bytes} bytes "
            f"[dim]· {n_lines} {line_label}[/dim]"
        )

        if existing is not None:
            diff = build_unified_diff(path, existing, content, context=1)
            if diff.strip():
                console.print(
                    Panel(
                        Syntax(diff, "diff", theme="ansi_dark", word_wrap=True),
                        title="[dim]diff preview[/dim]",
                        border_style="dim",
                        padding=(0, 1),
                    )
                )
            else:
                console.print("[dim]No textual diff to preview (content unchanged).[/dim]")
            return

        preview = _content_preview(content)
        if preview:
            lexer = "bash" if Path(path).suffix == ".sh" else "text"
            console.print(
                Panel(
                    Syntax(preview, lexer, theme="ansi_dark", word_wrap=True),
                    title="[dim]content preview[/dim]",
                    border_style="dim",
                    padding=(0, 1),
                )
            )
        return

    if name == "git_checkout":
        console.print(f"[dim]branch[/dim] {args.get('branch', '')}")
        return

    console.print(f"[dim]{_render_args(args)}[/dim]")


def _allow_key(name: str, args) -> str:
    """Session-scoped approval key for one exact tool call.

    "Always allow" should approve this specific invocation shape, not every
    future call to the whole tool family (e.g. every `bash` command)."""
    try:
        encoded = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        encoded = repr(args)
    return f"{name}:{encoded}"


def _default_ask_permission(prompt_text: str) -> str:
    """Default permission asker — a one-shot `prompt_toolkit` prompt. Works
    for the inline REPL, subagents, and tests. The full-screen REPL replaces
    this via `set_permission_asker` with an Event-based bridge to its UI
    thread, since a nested `pt_prompt` can't run inside a live Application."""
    return pt_prompt(prompt_text)


# Swappable so the run-context (inline / full-screen / test) decides how the
# y/n/a decision is collected, while `check_permission` stays UI-agnostic.
_ask_permission = _default_ask_permission


def set_permission_asker(asker) -> None:
    """Install a custom permission asker: `asker(prompt_text) -> str`, where
    the return's leading char is significant ('y' / 'n' / 'a')."""
    global _ask_permission
    _ask_permission = asker


def check_permission(console, name: str, args, *, meta: dict | None = None) -> bool:
    """Return True if the tool may run. Prompts the user for sensitive tools.

    The y/n/a decision is collected through the swappable `_ask_permission`
    hook — a one-shot `pt_prompt` by default, or the full-screen app's
    UI-thread bridge when that's installed. The HALT frame and preview always
    render through the passed-in `console`."""
    key = _allow_key(name, args)
    if name not in SENSITIVE_TOOLS or key in _session_allowed:
        return True

    tool_id = _tool_id(meta)
    risk = _RISK_LABELS.get(name, "sensitive action")
    _render_halt_frame(console, name, risk, tool_id)
    _print_permission_preview(console, name, args)
    try:
        choice = _ask_permission("Allow? [y]es / [n]o / [a]lways › ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        # Treat an aborted permission prompt as a denial — safer default.
        return False

    if choice.startswith("a"):
        _session_allowed.add(key)
        return True
    return choice.startswith("y")
