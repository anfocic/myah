"""REPL UI helpers: prompt chrome, context tag, slash-command completion.
All cosmetic — the model sees none of this; every line here is for the
human at the keyboard.

Input engine: prompt_toolkit. The main REPL owns a single `PromptSession`
(built via `build_session`) that carries persistent history + a custom
completer. Uses `FileHistory` at `~/.mia_input_history` — not compatible
with readline's prior format, which is why the filename changed when the
engine was swapped."""

import colorsys
import os
import subprocess
import time
from collections.abc import Iterable
from datetime import datetime

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory

from config import INPUT_HISTORY_FILE, get_context_size
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


def _tilde_cwd(cwd: str) -> str:
    """`~`-relative cwd for the prompt floor, trimmed to the last two path
    components so a deep tree doesn't crowd out the input line."""
    home = os.path.expanduser("~")
    path = "~" + cwd[len(home):] if cwd.startswith(home) else cwd
    parts = path.split(os.sep)
    if len(parts) > 3:
        return os.sep.join([parts[0], "…", *parts[-2:]])
    return path


_MODEL_MAX = 40


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


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


def _rprompt_model(provider) -> str:
    """Compact provider:model label for the prompt-time chrome."""
    model = _clip(provider.model, _MODEL_MAX)
    return f"{provider.name}:{model}"


def _ctx_gradient_style(ctx_used: int, ctx_total: int) -> str:
    """Smooth green → yellow → red gradient driven by ctx fill.

    HSL hue slides from 120° (green) at 0% through 60° (yellow) at 50%
    to 0° (red) at 100%. Moderate lightness and saturation keep the
    color readable rather than neon. Cheap enough to compute on every
    prompt paint; cache isn't worth the complexity."""
    pct = max(0.0, min(1.0, _ctx_pct(ctx_used, ctx_total)))
    hue_deg = 120 * (1 - pct)
    # colorsys.hls_to_rgb argument order is (h, L, S), all in [0, 1].
    r, g, b = colorsys.hls_to_rgb(hue_deg / 360.0, 0.55, 0.55)
    return f"fg:#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x} bold"


def build_prompt(state: State) -> FormattedText:
    """The Phosphor prompt floor — `myah@local:~/cwd$`. The accent hue carries
    `myah@local` and the `$`; the cwd rides in magenta (the design's branch/
    identity slot). Status signals stay on the right via `rprompt`."""
    accent = phosphor.accent_pt()
    cwd = _tilde_cwd(state.get("cwd", os.getcwd()))
    return FormattedText(
        [
            (f"{accent} bold", "myah@local"),
            ("ansibrightblack", ":"),
            ("ansimagenta", cwd),
            (f"{accent} bold", "$ "),
        ]
    )


_DIM = "fg:ansibrightblack"


def build_rprompt(state: State) -> FormattedText:
    """Right-aligned prompt chrome: `branch · ctx% · provider:model [MODE]`.

    cwd moved to the left prompt floor, so the rprompt now carries the
    at-a-glance signals only. Percent is the one number with color — a
    smooth green→yellow→red gradient driven by ctx fill. A `[MODE]` pill
    trails when plan or debug mode is on. prompt_toolkit hides all of this
    automatically if the input grows wide enough to collide with it."""
    provider = get_active_provider()
    ctx_used = state.get("ctx_used", 0)
    ctx_size = get_context_size()
    pct = _ctx_pct(ctx_used, ctx_size)

    segments: list[tuple[str, str]] = []
    branch = _current_branch()
    if branch:
        segments.append((_DIM, _short_branch(branch)))
        segments.append((_DIM, " · "))
    segments.append((_ctx_gradient_style(ctx_used, ctx_size), f"{pct:.0%}"))
    segments.append((_DIM, " · "))
    segments.append((_DIM, _rprompt_model(provider)))
    if state.get("plan_mode"):
        segments.append(("ansiyellow bold", " [PLAN]"))
    if state.get("debug"):
        segments.append(("ansimagenta bold", " [DEBUG]"))
    return FormattedText(segments)


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


def build_session(commands: dict, state: State) -> PromptSession:
    """Construct the REPL's single PromptSession. Held for the lifetime
    of the process so input history persists across turns without hitting
    disk on every prompt.

    Status chrome rides on `rprompt` (same line as `You ›`), not on a
    `bottom_toolbar`. Inline chrome keeps the REPL to one visual row and
    side-steps prompt_toolkit's default reverse-video toolbar styling."""
    return PromptSession(
        history=FileHistory(INPUT_HISTORY_FILE),
        completer=SlashCompleter(commands),
        complete_while_typing=False,
        rprompt=lambda: build_rprompt(state),
    )


def ctx_tag(ctx_used: int, ctx_total: int) -> str:
    """Colored `ctx N%` marker for the post-turn status line. Green below
    70% (comfortable), yellow 70–85% (trim incoming), red above 85%
    (auto-trim fires). Thresholds intentionally above trim_history's 0.8
    bound so a fresh session with a large system prompt doesn't flash
    yellow on turn 1."""
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


def render_session_rail(state: State) -> str:
    """The Phosphor left-rail session console, resolved from current state.
    Shared by the boot screen and the `/session` command — a scrollback REPL
    can't pin it as a side panel, so it renders inline on demand instead."""
    provider = get_active_provider()
    return phosphor.session_rail(
        sess_state="READY",
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
