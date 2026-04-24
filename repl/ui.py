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

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory

from config import NUM_CTX
from providers import get_active_provider
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


def _toolbar_model(provider) -> str:
    """Short model name for the toolbar. For org/slug forms like
    `google/gemma-4-e4b` we keep only the slug after the slash — the org
    is redundant context on a status line. Non-slashed names (Anthropic
    `claude-sonnet-4-6`, Ollama tags like `qwen2.5:7b-instruct`) pass
    through unchanged. Provider prefix is dropped — the user already
    knows what they picked."""
    return _clip(provider.model.split("/")[-1], _MODEL_MAX)


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
    """A short prompt keeps the input line focused; status lives in the
    bottom toolbar instead of competing with what the user is typing."""
    return FormattedText(
        [
            ("bold white", "You"),
            ("ansibrightblack", " › "),
        ]
    )


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

    Status chrome lives on `rprompt` (same line as `You >`), not in a
    `bottom_toolbar` at the terminal bottom. Keeps the chrome inline
    with the input — one line of UI, not two — and side-steps
    prompt_toolkit's default reverse-video toolbar styling."""
    return PromptSession(
        history=FileHistory(INPUT_HISTORY_FILE),
        completer=SlashCompleter(commands),
        complete_while_typing=False,
        rprompt=lambda: build_rprompt(state),
    )


_DIM = "fg:ansibrightblack"


def build_rprompt(state: State) -> FormattedText:
    """Right-aligned prompt chrome, rendered inline with `You >`.

    Format: `<model> · <pct%> · <branch>`. Pct is the one number with
    a color (green→red gradient on ctx fill) — exact token counts live
    in `/stats` and `/context`. prompt_toolkit hides this automatically
    if the input grows wide enough to collide with it."""
    provider = get_active_provider()
    u = state["ctx_used"]
    pct_text = f"{_ctx_pct(u, NUM_CTX):.0%}"
    pct_style = _ctx_gradient_style(u, NUM_CTX)
    segments: list[tuple[str, str]] = [
        (_DIM, _toolbar_model(provider)),
        (_DIM, " · "),
        (pct_style, pct_text),
    ]
    branch = _current_branch()
    if branch:
        segments.append((_DIM, " · "))
        segments.append((_DIM, _short_branch(branch)))
    return FormattedText(segments)


def ctx_tag(ctx_used: int, ctx_total: int) -> str:
    """`ctx N%` marker. Silent below 70% — a green-0% splash on every turn
    is attention without information. Yellow 70–85% (trim incoming), red
    above 85% (auto-trim fires). Thresholds intentionally above
    trim_history's 0.8 bound so a fresh session with a large system
    prompt doesn't flash yellow on turn 1."""
    pct = _ctx_pct(ctx_used, ctx_total)
    if pct < 0.70:
        return f"[dim]ctx {pct:.0%}[/dim]"
    color = _ctx_color(ctx_used, ctx_total)
    return f"[dim]ctx[/dim] [{color}]{pct:.0%}[/{color}]"


def build_turn_header(state: State) -> str:
    """Compact turn preface so each run reads as a single unit in the log."""
    provider = get_active_provider()
    turn_no = _history_turns(state["history"]) + 1
    parts = [
        f"[bold]Turn {turn_no}[/bold]",
        f"[dim]{provider.name}:{_clip(provider.model, _MODEL_MAX)}[/dim]",
        f"[dim]{state['ctx_used']:,}/{NUM_CTX:,}[/dim] {ctx_tag(state['ctx_used'], NUM_CTX)}",
    ]
    branch = _current_branch()
    if branch:
        parts.insert(1, f"[dim]{_short_branch(branch)}[/dim]")
    for mode in _mode_labels(state):
        color = "yellow" if mode == "plan" else "magenta"
        parts.append(f"[{color}]{mode}[/{color}]")
    return " [dim]·[/dim] ".join(parts)


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
