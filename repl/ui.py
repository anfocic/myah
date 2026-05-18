"""REPL UI helpers: the prompt floor, transmission header, session rail,
context tag, and slash-command completion. All cosmetic — the model sees none
of this; every line here is for the human at the keyboard.

The full-screen REPL (`repl/app.py`) owns the `prompt_toolkit.Application`;
this module just builds the formatted-text fragments and rich-markup strings
that populate its rail, input prefix, and scrolling pane."""

import os
import subprocess
import time
from collections.abc import Iterable
from datetime import datetime

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText

from config import get_context_size
from display import phosphor
from providers import get_active_provider
from repl.state import State

_BRANCH_TTL_SECONDS = 2.0
_branch_cached_at: float = 0.0
_branch_cached_value: str | None = None


def _current_branch() -> str | None:
    """Best-effort current branch name. Returns None outside a repo or if
    git is missing. Cached for a short TTL so a `git checkout` in another
    terminal is reflected within a couple of seconds without paying a git
    fork on every repaint."""
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


def _tilde_cwd(cwd: str) -> str:
    """`~`-relative cwd for the prompt floor, trimmed to the last two path
    components so a deep tree doesn't crowd out the input line."""
    home = os.path.expanduser("~")
    path = "~" + cwd[len(home):] if cwd.startswith(home) else cwd
    parts = path.split(os.sep)
    if len(parts) > 3:
        return os.sep.join([parts[0], "…", *parts[-2:]])
    return path


def _ctx_pct(ctx_used: int, ctx_total: int) -> float:
    return (ctx_used / ctx_total) if ctx_total > 0 else 0.0


def _ctx_color(ctx_used: int, ctx_total: int) -> str:
    pct = _ctx_pct(ctx_used, ctx_total)
    if pct < 0.70:
        return "green"
    if pct < 0.85:
        return "yellow"
    return "red"


def _history_turns(history: list[dict]) -> int:
    return sum(1 for msg in history if msg.get("role") == "user")


def _mode_labels(state: State) -> list[str]:
    labels: list[str] = []
    if state.get("plan_mode"):
        labels.append("plan")
    if state.get("debug"):
        labels.append("debug")
    return labels


def build_prompt(state: State) -> FormattedText:
    """The Phosphor prompt floor — `myah@local:~/cwd$`. The accent hue carries
    `myah@local` and the `$`; the cwd rides in magenta (the design's branch/
    identity slot). Used as the input window's line prefix in `repl/app.py`.

    When a Ctrl+V image is staged for the next turn, the prompt floor leads
    with `[img NNk]` in cyan so the user knows what's about to be sent."""
    accent = phosphor.accent_pt()
    cwd = _tilde_cwd(state.get("cwd", os.getcwd()))
    fragments: list[tuple[str, str]] = []
    pending_size = state.get("_pending_image_size")
    if pending_size:
        kb = max(1, pending_size // 1024)
        fragments.append(("ansicyan bold", f"[img {kb}k] "))
    fragments.extend([
        (f"{accent} bold", "myah@local"),
        ("ansibrightblack", ":"),
        ("ansimagenta", cwd),
        (f"{accent} bold", "$ "),
    ])
    return FormattedText(fragments)


def compose_user_message(text: str, pending_image: tuple[str, str] | None):
    """Build the run_agent user_input from the prompt-buffer text and an
    optional pending image. Returns either the plain string (no image)
    or our internal list-of-blocks shape (text + image) so the provider
    translation layer can rewrite it per backend.

    `pending_image` is `(base64_data, media_type)` as produced by
    `tools.clipboard.get_clipboard_image()`. Empty text is preserved —
    the user can paste an image and hit Enter without typing."""
    if pending_image is None:
        return text
    b64, media = pending_image
    return [
        {"type": "text", "text": text},
        {"type": "image", "source": {
            "type": "base64", "media_type": media, "data": b64,
        }},
    ]


class SlashCompleter(Completer):
    """Completes slash commands at the start of the input. Only fires when the
    buffer starts with "/", yields every command whose key prefix-matches.

    Taking `commands` as a constructor arg (rather than importing from
    repl.commands) keeps the anti-cycle discipline — commands.py already
    imports `ctx_tag` from this module."""

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


def ctx_tag(ctx_used: int, ctx_total: int) -> str:
    """Colored `ctx N%` marker for status lines. Green below 70% (comfortable),
    yellow 70–85% (trim incoming), red above 85% (auto-trim fires). Thresholds
    intentionally above trim_history's 0.8 bound so a fresh session with a
    large system prompt doesn't flash yellow on turn 1."""
    pct = _ctx_pct(ctx_used, ctx_total)
    color = _ctx_color(ctx_used, ctx_total)
    return f"[dim]ctx[/dim] [{color}]{pct:.0%}[/{color}]"


def build_transmission_header(state: State) -> str:
    """The Phosphor turn preface — `░▒▓ TRANSMISSION NNN ▓▒░ · time · ctx%`
    closed by an accent rule. Each run reads as one transmission in the log."""
    turn_no = _history_turns(state["history"]) + 1
    ctx_used = state["ctx_used"]
    ctx_size = get_context_size()
    pct = _ctx_pct(ctx_used, ctx_size)
    now = datetime.now().strftime("%H:%M:%S")
    parts = [
        phosphor.bracket(f"TRANSMISSION {turn_no:03d}"),
        f"[{phosphor.DIM}]· {now} · ctx {pct:.0%}[/]",
    ]
    for mode in _mode_labels(state):
        color = "yellow" if mode == "plan" else "magenta"
        parts.append(f"[{color}]{mode}[/{color}]")
    parts.append(phosphor.rule(30))
    return " ".join(parts)


def render_session_rail(state: State, sess_state: str = "READY") -> str:
    """The Phosphor left-rail session console, resolved from current state.

    `sess_state` (READY / STREAM / HOLD / BOOT) drives the state pill's hue.
    The full-screen REPL re-renders the rail every frame with the live state;
    the `/session` command uses the READY default."""
    provider = get_active_provider()
    return phosphor.session_rail(
        sess_state=sess_state,
        branch=_current_branch(),
        turns=len(state["history"]) // 2,
        ctx_used=state.get("ctx_used", 0),
        ctx_total=get_context_size(),
        provider_label=f"{provider.name}:{provider.model}",
    )


def build_turn_footer(ctx_used: int, ctx_total: int, elapsed_s: float, stats: dict) -> str:
    """Post-turn stats grouped into one footer line."""
    parts = [
        f"[dim]{ctx_used:,}/{ctx_total:,}[/dim] {ctx_tag(ctx_used, ctx_total)}",
        f"[dim]{elapsed_s:.1f}s[/dim]",
    ]
    ttft_ms = stats.get("ttft_ms")
    if ttft_ms is not None:
        parts.append(f"[dim]ttft {ttft_ms}ms[/dim]")
    tok_per_s = stats.get("tok_per_s")
    if tok_per_s:
        parts.append(f"[dim]{tok_per_s:.0f} tok/s[/dim]")
    return " [dim]·[/dim] ".join(parts)
