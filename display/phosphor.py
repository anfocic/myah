"""Phosphor — the TUI's shared visual vocabulary.

A Claude Design handoff ("Myah TUI · Phosphor") established a Matrix-influenced
look for the REPL: an ASCII masthead, ``░▒▓ SECTION ▓▒░`` brackets, a left-rail
session console, glyph-forward tool lines. This module is the single home for
that vocabulary so every surface — tool events, slash commands, the permission
gate, prompt chrome — speaks it consistently.

Color discipline: everything maps to rich's *named* ANSI colors, never RGB.
The design called for "strict 16-color ANSI, max compat"; named colors honor
whatever palette the user's terminal is set to, so dark / light / high-contrast
all work for free without a theme switcher. The one knob is the accent hue
(``config.PHOSPHOR_ACCENT``) — the design's single live tweak.
"""

from __future__ import annotations

import config

# ── semantic palette ────────────────────────────────────────────────────────
# The design's "tokens · spec" screen fixes a semantic role per ANSI slot.
# These names are what the rest of the harness should reference rather than
# raw color names, so the mapping stays in one place.
RED = "red"  # destructive · halt · deny · error
GREEN = "green"  # ok · ready · accept
YELLOW = "yellow"  # warn · edit · streaming · plan mode
MAGENTA = "magenta"  # branch · spawn_subagent · agent identity
CYAN = "cyan"  # read · grep · glob · web_search
WHITE = "white"  # default fg · neutral text
DIM = "bright_black"  # secondaries · rules · labels
BRIGHT = "bright_white"  # emphasis · active state · headings

# Accent hue → rich color. The design maps the accent onto ANSI slots c2/c3/c6.
_ACCENT_COLORS = {"green": GREEN, "amber": YELLOW, "cyan": CYAN}
# Same mapping in prompt_toolkit's color vocabulary, for the prompt floor.
_ACCENT_PT = {"green": "ansigreen", "amber": "ansiyellow", "cyan": "ansicyan"}

# Tool category → bullet hue. Mirrors the design's TOOL_HUE map (and the old
# display.tools._TOOL_COLORS it replaces). Categories follow the "tokens · spec"
# screen: cyan = read/inspect, yellow = write, red = shell/destructive,
# magenta = agent, green = harness/meta.
TOOL_HUE = {
    # read / inspect
    "read_file": CYAN,
    "glob": CYAN,
    "grep": CYAN,
    "web_search": CYAN,
    "vault_search": CYAN,
    "git_status": CYAN,
    "git_log": CYAN,
    "git_diff": CYAN,
    "git_branch_list": CYAN,
    "note_read": CYAN,
    "note_search": CYAN,
    "note_list": CYAN,
    # write
    "edit_file": YELLOW,
    "write_file": YELLOW,
    "note_write": YELLOW,
    "note_append": YELLOW,
    "daily_note": YELLOW,
    # shell / destructive
    "bash": RED,
    "git_checkout": RED,
    # agent
    "spawn_subagent": MAGENTA,
    # harness / meta
    "get_current_time": GREEN,
    "harness_info": GREEN,
    "cd": GREEN,
    "pwd": GREEN,
}


def accent() -> str:
    """Rich color name for the active accent hue. Read at call time so a
    config reload (or a test monkeypatch) is reflected without re-import."""
    return _ACCENT_COLORS.get(config.PHOSPHOR_ACCENT, GREEN)


def accent_pt() -> str:
    """Active accent hue as a prompt_toolkit color name (for the prompt floor)."""
    return _ACCENT_PT.get(config.PHOSPHOR_ACCENT, "ansigreen")


def tool_hue(name: str) -> str:
    """Bullet hue for a tool, by category. Unknown tools render dim."""
    return TOOL_HUE.get(name, DIM)


# ── glyph helpers ───────────────────────────────────────────────────────────


def bracket(label: str, color: str | None = None) -> str:
    """The ``░▒▓ LABEL ▓▒░`` section bracket. Accent-colored by default;
    pass ``color`` to override (e.g. red for the HALT frame)."""
    c = color or accent()
    return f"[{c}]░▒▓[/] [{c} bold]{label}[/] [{c}]▓▒░[/]"


def rule(width: int = 60, char: str = "─") -> str:
    """A horizontal rule in the accent hue, dimmed — the trailing line that
    closes a transmission header."""
    return f"[{accent()} dim]{char * width}[/]"


def meter(used: int, total: int, width: int = 20) -> str:
    """A ``████░░░░`` fill bar, colored green / yellow / red by fill ratio.
    Same thresholds as the rest of the harness's ctx signals (70 / 85 %)."""
    ratio = (used / total) if total > 0 else 0.0
    ratio = max(0.0, min(1.0, ratio))
    filled = round(ratio * width)
    color = GREEN if ratio < 0.70 else YELLOW if ratio < 0.85 else RED
    return f"[{color}]{'█' * filled}[/][{DIM}]{'░' * (width - filled)}[/]"


# ── masthead ────────────────────────────────────────────────────────────────

# Block-letter "myah" banner, lifted from the design's PhMast component.
_BANNER = r""" ███╗   ███╗  ██╗   ██╗   █████╗   ██╗   ██╗
 ████╗ ████║  ╚██╗ ██╔╝  ██╔══██╗  ██║   ██║
 ██╔████╔██║   ╚████╔╝   ███████║  ███████║
 ██║╚██╔╝██║    ╚██╔╝    ██╔══██║  ██╔══██║
 ██║ ╚═╝ ██║     ██║     ██║  ██║  ██║  ██║"""


def masthead(mode: str = "full", *, subtitle: str = "personal harness") -> str:
    """The harness banner. ``full`` is the boot-screen ASCII art plus an
    identity line; ``compact`` is a single-line ``▮ myah`` strip; ``none``
    renders nothing. Side-by-side art + metadata (as in the web mock) doesn't
    survive terminal reflow, so the terminal translation stacks them."""
    if mode == "none":
        return ""
    a = accent()
    if mode == "compact":
        return f"[{a} bold]▮ myah[/] [{DIM}]· {subtitle}[/]"
    banner = "\n".join(f"[{a}]{line}[/]" for line in _BANNER.splitlines())
    identity = f"[{a} bold]myah[/] [{DIM}]· {subtitle}[/]"
    return f"{banner}\n{identity}"


# ── left rail ───────────────────────────────────────────────────────────────

_STATE_HUES = {"READY": None, "BOOT": None, "STREAM": YELLOW, "HOLD": RED}


def session_rail(
    *,
    sess_state: str = "READY",
    branch: str | None = None,
    turns: int = 0,
    ctx_used: int = 0,
    ctx_total: int = 0,
    provider_label: str = "",
) -> str:
    """The left-rail session console as a stacked block.

    The web mock pins this as a fixed side panel; a scrollback REPL can't
    (see ``display/streaming.py`` on why the harness stays off the alt
    screen), so it renders on the boot screen and on demand via ``/session``.
    """
    a = accent()
    state_hue = _STATE_HUES.get(sess_state, a) or a
    pct = (ctx_used / ctx_total * 100) if ctx_total > 0 else 0.0

    lines = [
        bracket("SESSION"),
        f"  [{DIM}]state[/]   [{state_hue} bold]{sess_state}[/]",
        f"  [{DIM}]branch[/]  [{MAGENTA}]⎇ {branch or '(detached)'}[/]",
        f"  [{DIM}]turns[/]   [{WHITE}]{turns:03d}[/]",
        f"  [{DIM}]model[/]   [{WHITE}]{provider_label or '(none)'}[/]",
        "",
        bracket("CTX"),
        f"  {meter(ctx_used, ctx_total)}",
        f"  [{DIM}]{pct:.0f}%  {ctx_used:,} / {ctx_total:,}[/]",
        "",
        bracket("TOOLS"),
        f"  [{CYAN}]●[/] [{DIM}]read[/]   [{WHITE}]read_file glob grep web_search[/]",
        f"  [{YELLOW}]●[/] [{DIM}]write[/]  [{WHITE}]edit_file write_file[/]",
        f"  [{RED}]●[/] [{DIM}]shell[/]  [{WHITE}]bash git_checkout[/]",
        f"  [{MAGENTA}]●[/] [{DIM}]agent[/]  [{WHITE}]spawn_subagent[/]",
        "",
        f"  [{DIM}]esc[/]  [{DIM}]abort turn[/]   "
        f"[{DIM}]^c[/]  [{DIM}]clear input[/]   "
        f"[{DIM}]^d[/]  [{DIM}]exit[/]",
    ]
    return "\n".join(lines)
