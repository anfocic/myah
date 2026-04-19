"""REPL UI helpers: prompt chrome, context tag, slash-command completion.
All cosmetic — the model sees none of this; every line here is for the
human at the keyboard.

Input engine: prompt_toolkit. The main REPL owns a single `PromptSession`
(built via `build_session`) that carries persistent history + a custom
completer. Uses `FileHistory` at `~/.mia_input_history` — not compatible
with readline's prior format, which is why the filename changed when the
engine was swapped."""
import os
import subprocess
import time
from collections.abc import Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory

from repl.state import State

INPUT_HISTORY_FILE = os.path.expanduser("~/.mia_input_history")


_BRANCH_TTL_SECONDS = 2.0
_branch_cached_at: float = 0.0
_branch_cached_value: str | None = None


def _current_branch() -> str | None:
    """Best-effort current branch name. Returns None outside a repo or if
    git is missing. Cached for a short TTL so a `git checkout` in another
    terminal is reflected within a couple of seconds without paying a git
    fork on every prompt."""
    global _branch_cached_at, _branch_cached_value
    now = time.monotonic()
    if now - _branch_cached_at < _BRANCH_TTL_SECONDS:
        return _branch_cached_value
    try:
        out = subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1,
        ).strip()
        _branch_cached_value = out or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        _branch_cached_value = None
    _branch_cached_at = now
    return _branch_cached_value


_BRANCH_MAX = 18


def _short_branch(name: str) -> str:
    return name if len(name) <= _BRANCH_MAX else name[: _BRANCH_MAX - 1] + "…"


def build_prompt(state: State) -> FormattedText:
    """`You [branch · plan · debug] ›` as prompt_toolkit FormattedText.
    Badges only render when the condition applies so the prompt stays
    clean in the common case. Long branches get truncated so the prompt
    doesn't eat the visible line."""
    out: list[tuple[str, str]] = [("bold ansimagenta", "You")]
    parts: list[tuple[str, str]] = []
    branch = _current_branch()
    if branch:
        parts.append(("ansibrightblack", _short_branch(branch)))
    if state.get("plan_mode"):
        parts.append(("ansiyellow", "plan"))
    if state.get("debug"):
        parts.append(("ansimagenta", "debug"))
    if parts:
        out.append(("ansibrightblack", " ["))
        for i, (style, text) in enumerate(parts):
            if i > 0:
                out.append(("ansibrightblack", " · "))
            out.append((style, text))
        out.append(("ansibrightblack", "]"))
    out.append(("ansibrightblack", " › "))
    return FormattedText(out)


class SlashCompleter(Completer):
    """Completes slash commands at the start of the input. Mirrors the
    old readline-based completer: only fires when the buffer starts with
    "/", yields every command whose key prefix-matches the buffer.

    Taking `commands` as a constructor arg (rather than importing from
    repl.commands) keeps the same anti-cycle discipline the old code had
    — commands.py already imports `ctx_tag` from this module."""

    def __init__(self, commands: dict) -> None:
        self._commands = sorted(commands.keys())

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        buf = document.text_before_cursor
        if not buf.startswith("/"):
            return
        for cmd in self._commands:
            if cmd.startswith(buf):
                yield Completion(cmd, start_position=-len(buf))


def build_session(commands: dict) -> PromptSession:
    """Construct the REPL's single PromptSession. Held for the lifetime
    of the process so input history persists across turns without hitting
    disk on every prompt."""
    return PromptSession(
        history=FileHistory(INPUT_HISTORY_FILE),
        completer=SlashCompleter(commands),
        complete_while_typing=False,
    )


def ctx_tag(ctx_used: int, ctx_total: int) -> str:
    """Colored `ctx N%` marker for the post-turn status line. Green below
    70% (comfortable), yellow 70–85% (trim incoming), red above 85%
    (auto-trim fires). Thresholds intentionally above trim_history's 0.8
    bound so a fresh session with a large system prompt doesn't flash
    yellow on turn 1."""
    pct = ctx_used / ctx_total
    if pct < 0.70:
        color = "green"
    elif pct < 0.85:
        color = "yellow"
    else:
        color = "red"
    return f"[dim]ctx[/dim] [{color}]{pct:.0%}[/{color}]"
