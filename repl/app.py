"""The full-screen REPL — a `prompt_toolkit.Application` with a pinned left
rail, a scrolling transmission-log main pane, and a pinned input line.

This replaces the old inline-prompt + `patch_stdout` scrollback REPL so the
Phosphor session rail can be a *real* persistent sidebar. The keystone trick:
`repl.console.console` is swapped for a `BufferConsole` whose output flows into
a `RepaintBuffer`, so every existing `console.print` site (`agent/loop.py`,
`repl/commands.py`, `display/*`, `permissions.py`) keeps working untouched —
their output just lands in the scrolling pane instead of stdout.

Threading: `run_agent` is synchronous, so each turn runs on a worker thread
while the event loop owns the main thread. The worker writes through the
buffer-backed console (every write triggers `app.invalidate()`); the permission
gate hands off to the UI thread via `PermissionBridge`; Ctrl+C injects a
`KeyboardInterrupt` into the worker, reusing `run_agent`'s own abort handling.
"""

from __future__ import annotations

import atexit
import copy
import ctypes
import shutil
import threading
import time

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from rich.console import Console as _RichConsole

import permissions
from agent import apply_summary, run_agent, trim_history
from config import INPUT_HISTORY_FILE, get_context_size
from display import on_tool_end, on_tool_start, phosphor
from permissions import check_permission
from providers import get_active_provider
from repl.commands import SLASH_COMMANDS, handle_slash
from repl.console import console
from repl.permission_bridge import PermissionBridge
from repl.persistence import has_saved_session, load_session, save_session
from repl.screen import BufferConsole, RepaintBuffer, ScrollState
from repl.state import State
from repl.tool_registry import TOOL_SCHEMAS, make_execute_tool
from repl.ui import (
    SlashCompleter,
    build_prompt,
    build_transmission_header,
    build_turn_footer,
    render_session_rail,
)

RAIL_WIDTH = 32
# Slash commands that shell out (spawn $EDITOR, run subprocesses) — these must
# run via `run_in_terminal` so they don't corrupt the full-screen layout.
_TERMINAL_SLASH = {"/config", "/eval"}


def _render_markup(markup: str, width: int) -> str:
    """Render a rich-markup string to an ANSI string at a fixed width — used to
    paint the rail (which `render_session_rail` returns as markup) into a
    prompt_toolkit window."""
    c = _RichConsole(force_terminal=True, width=width)
    with c.capture() as cap:
        c.print(markup, end="")
    return cap.get()


def _ok_line(label: str, detail: str) -> str:
    """One Phosphor boot line — `[ok]  label  detail`. The literal `[ok]` is
    escaped so rich's markup parser doesn't read it as a (bogus) style tag."""
    return (
        f"[{phosphor.GREEN}]\\[ok][/]   "
        f"[{phosphor.DIM}]{label:<8}[/] [{phosphor.WHITE}]{detail}[/]"
    )


class ReplApp:
    """Owns the full-screen Application: layout, key bindings, the turn-worker
    thread, and the permission bridge."""

    def __init__(self, state: State, *, resume: bool) -> None:
        self.state = state
        self.sess_state = "READY"
        self.turn_thread: threading.Thread | None = None
        self._main_height = max(4, shutil.get_terminal_size().lines - 4)
        self._resume = resume
        self._booted = False

        # Buffer-backed console — pointed at the scrolling main pane. Its width
        # is only set for real once the main Window has rendered (see
        # `_main_text`): terminal-size probes can hand back the 80-col fallback,
        # so the actual pane width is the only source to trust. This is a
        # placeholder until then.
        self.buffer = RepaintBuffer()
        self.console = BufferConsole(self.buffer, width=120)
        console._set_inner(self.console)  # type: ignore[attr-defined]  # proxy-only method (see repl/console.py)

        self.scroll = ScrollState()
        self.bridge = PermissionBridge()
        permissions.set_permission_asker(self.bridge.ask)

        # Same gate closure main.py used — captured by make_execute_tool so the
        # spawn_subagent branch forwards the same permission gate downward.
        def perm_check(name, args, meta=None):
            return check_permission(console, name, args, meta=meta)

        self.perm_check = perm_check
        self.execute_tool = make_execute_tool(state, permission_check=perm_check)

        self.input_buffer = Buffer(
            completer=SlashCompleter(SLASH_COMMANDS),
            history=FileHistory(INPUT_HISTORY_FILE),
            complete_while_typing=False,
            multiline=False,
            accept_handler=self._on_accept,
        )

        self.app = self._build_application()

        # Resume is opt-in on both ends (same rationale as the old main.py):
        # without --resume we neither load nor save. The boot screen is printed
        # later — on the first render, once the real pane width is known.
        if resume:
            load_session(state)
            atexit.register(save_session, state)

    # -- layout --------------------------------------------------------------

    def _build_application(self) -> Application:
        rail = Window(
            FormattedTextControl(self._rail_text),
            width=Dimension.exact(RAIL_WIDTH),
        )
        self._main_window = Window(
            FormattedTextControl(self._main_text),
            wrap_lines=False,
        )
        self._input_window = Window(
            BufferControl(self.input_buffer),
            height=1,
            get_line_prefix=self._prompt_prefix,
        )
        root = VSplit([
            rail,
            Window(width=1, char="│", style="fg:ansibrightblack"),
            HSplit([
                self._main_window,
                Window(height=1, char="─", style="fg:ansibrightblack"),
                self._input_window,
            ]),
        ])
        return Application(
            layout=Layout(root, focused_element=self._input_window),
            key_bindings=self._build_key_bindings(),
            full_screen=True,
            mouse_support=False,
            # Coalesce the streaming repaint storm — a fast local model can emit
            # hundreds of tokens/sec, each triggering an invalidate.
            min_redraw_interval=0.05,
            after_render=self._after_render,
        )

    def _after_render(self, _app) -> None:
        """Runs after every render — the first time `render_info` is populated.
        Sizes the buffer console to the real pane width (terminal-size probes
        can hand back the 80-col fallback; the rendered Window can't) and prints
        the boot screen once that width is correct."""
        ri = self._main_window.render_info
        if ri is None:
            return
        if ri.window_width and ri.window_width != self.console._width:
            # Picks up resizes too; already-rendered history keeps its old
            # width (documented v1 limitation).
            self.console._width = ri.window_width
        if not self._booted:
            self._booted = True
            self._print_boot_screen(self._resume)

    def _rail_text(self) -> ANSI:
        markup = render_session_rail(self.state, self.sess_state)
        return ANSI(_render_markup(markup, RAIL_WIDTH))

    def _main_text(self) -> ANSI:
        ri = self._main_window.render_info
        if ri is not None:
            self._main_height = max(1, ri.window_height)
        # Reconcile scroll position against the current buffer size: stay
        # pinned to the tail unless the user has paged up.
        self.scroll.on_content(self.buffer.line_count(), self._main_height)
        lines = self.buffer.view(self.scroll.scroll_top, self._main_height)
        return ANSI("\n".join(lines))

    def _prompt_prefix(self, line_number: int, wrap_count: int):
        if self.bridge.pending is not None:
            return FormattedText([
                ("fg:ansired bold", "permission "),
                ("fg:ansibrightblack", "[y/n/a] › "),
            ])
        return build_prompt(self.state)

    # -- key bindings --------------------------------------------------------

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        perm_pending = Condition(lambda: self.bridge.pending is not None)

        @kb.add("c-d")
        def _(event):
            # EOF only when the input line is empty — matches readline's feel.
            if not self.input_buffer.text:
                self._request_exit()

        @kb.add("c-c")
        def _(event):
            if self.bridge.pending is not None:
                self.bridge.resolve("n")
                self.app.invalidate()
            elif self._turn_running():
                self._abort_turn()
            else:
                self.input_buffer.reset()

        @kb.add("escape", eager=True)
        def _(event):
            if self._turn_running():
                self._abort_turn()

        @kb.add("pageup")
        def _(event):
            self.scroll.page_up(self._main_height)
            self.app.invalidate()

        @kb.add("pagedown")
        def _(event):
            self.scroll.page_down(self.buffer.line_count(), self._main_height)
            self.app.invalidate()

        # y/n/a resolve a pending permission. `eager` so they beat the input
        # buffer's character insertion; the filter keeps them inert otherwise.
        for choice in ("y", "n", "a"):
            @kb.add(choice, filter=perm_pending, eager=True)
            def _(event, choice=choice):
                self.bridge.resolve(choice)
                self.app.invalidate()

        return kb

    # -- input handling ------------------------------------------------------

    def _on_accept(self, buff: Buffer) -> bool:
        """Buffer accept handler. Returns False so the input line is cleared
        after every submission."""
        text = buff.text.strip()
        if not text:
            return False
        if self._turn_running():
            self.console.print(
                f"[{phosphor.DIM}]↳ turn in progress — wait for it to finish[/]"
            )
            self.app.invalidate()
            return False
        if text.lower() == "exit":
            self._request_exit()
            return False
        if text.startswith("/"):
            self._run_slash(text)
            return False
        self._start_turn(text)
        return False

    def _run_slash(self, text: str) -> None:
        cmd = text.split()[0]
        if cmd in _TERMINAL_SLASH:
            # Suspend the full-screen layout so $EDITOR / subprocesses get a
            # clean terminal, then redraw.
            run_in_terminal(lambda: self._dispatch_slash(text))
        else:
            self._dispatch_slash(text)

    def _dispatch_slash(self, text: str) -> None:
        handle_slash(text, self.state)
        if text.split()[0] == "/clear":
            self.buffer.clear()  # cmd_clear wipes history; also wipe scrollback
        # /retry stashes the prior input here for immediate resubmission.
        retry = self.state.pop("_retry_input", None)
        self.console.print()
        if retry is not None:
            self._start_turn(retry)
        else:
            self.app.invalidate()

    # -- turn worker ---------------------------------------------------------

    def _turn_running(self) -> bool:
        return self.turn_thread is not None and self.turn_thread.is_alive()

    def _start_turn(self, user_input: str) -> None:
        # Daemon thread: if a turn is wedged in a network read at exit, the
        # injected KeyboardInterrupt can't land until the socket returns, and a
        # non-daemon thread would then pin the whole process open. A daemon is
        # reaped at interpreter shutdown — and a wedged provider read has
        # nothing half-written to protect (run_agent already tolerates orphaned
        # tool workers for the same reason).
        self.turn_thread = threading.Thread(
            target=self._turn_worker, args=(user_input,), daemon=True
        )
        self.turn_thread.start()

    def _turn_worker(self, user_input: str) -> None:
        """Port of the old `main.py` per-turn block, run on a worker thread."""
        self.sess_state = "STREAM"
        self.app.invalidate()
        start = time.time()
        self.console.print(build_transmission_header(self.state))
        # Snapshot pre-turn history so /rewind can restore it (deep copy: the
        # entries are dicts and the next turn rebinds `history`).
        self.state["snapshots"].append(copy.deepcopy(self.state["history"]))
        dropped: list = []
        try:
            response, self.state["history"], self.state["ctx_used"], stats = run_agent(
                user_input, TOOL_SCHEMAS, self.execute_tool, self.state["history"],
                console=console,
                permission_check=self.perm_check,
                plan_mode=self.state["plan_mode"],
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
                debug=self.state["debug"],
                cwd=self.state["cwd"],
                todos=self.state.get("todos", []),
                vars_dict=self.state.get("vars", {}),
            )
            ctx_before_trim = self.state["ctx_used"]
            self.state["history"], dropped = trim_history(
                self.state["history"], self.state["ctx_used"], get_context_size(),
                tools=TOOL_SCHEMAS, model_name=get_active_provider().model,
            )
            if dropped:
                # console.status() is a no-op on the buffer-backed console.
                with console.status("[yellow]Summarizing dropped turns...[/yellow]"):
                    self.state["history"] = apply_summary(self.state["history"], dropped)
        except KeyboardInterrupt:
            self.console.print(
                f"  [{phosphor.YELLOW}]⤷ aborted — history unchanged[/] "
                f"[{phosphor.DIM}]· snapshot not pushed · provider call "
                f"cancelled[/]"
            )
            self._end_turn()
            return
        except BaseException as e:  # noqa: BLE001 — a worker crash must not kill the UI
            self.console.print(f"[{phosphor.RED}]turn error: {type(e).__name__}: {e}[/]")
            self._end_turn()
            return

        elapsed = time.time() - start
        self.state["last_turn"] = {
            "ctx_used": self.state["ctx_used"],
            "elapsed_s": elapsed,
            **(stats or {}),
        }
        self.state["turn_history"].append(self.state["last_turn"])
        if dropped:
            ctx_size = get_context_size()
            threshold = int(0.8 * ctx_size)
            self.console.print(
                f"[dim yellow]↳ trim_history fired: ctx was {ctx_before_trim} "
                f"(> threshold {threshold} = 80% of ctx_size={ctx_size}); dropped "
                f"{len(dropped) // 2} turn(s), summarized into context; will "
                f"re-settle after next provider call[/dim yellow]"
            )
        self.console.print(
            build_turn_footer(
                self.state["ctx_used"], get_context_size(), elapsed, stats or {}
            )
        )
        self.console.print()
        self._end_turn()

    def _end_turn(self) -> None:
        self.sess_state = "READY"
        self.app.invalidate()

    def _abort_turn(self) -> None:
        """Inject KeyboardInterrupt into the turn worker — `run_agent` already
        catches and re-raises it, so its abort path is reused verbatim. Not
        instant on a socket-blocked provider read (fires when control returns
        to Python), matching the old REPL's behavior."""
        t = self.turn_thread
        if t is not None and t.is_alive() and t.ident is not None:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(t.ident), ctypes.py_object(KeyboardInterrupt)
            )
        self.app.invalidate()

    # -- lifecycle -----------------------------------------------------------

    def _request_exit(self) -> None:
        # Unblock anything waiting on the permission bridge (default-deny), then
        # give a live turn a bounded chance to unwind before we drop the screen.
        self.bridge.resolve("n")
        t = self.turn_thread
        if t is not None and t.is_alive():
            self._abort_turn()
            t.join(timeout=2.0)
        self.app.exit()

    def _print_boot_screen(self, resume: bool) -> None:
        """Phosphor boot screen — masthead, an INITIALIZE checklist of real
        startup state, a READY bracket. The session rail is the persistent
        sidebar now, so it's no longer reprinted inline here."""
        provider = get_active_provider()
        self.console.print(phosphor.masthead("full"))
        self.console.print()
        self.console.print(phosphor.bracket("INITIALIZE"))
        self.console.print(_ok_line("config", "merged defaults · files · env"))
        self.console.print(_ok_line("provider", f"{provider.name} · {provider.model}"))
        self.console.print(
            _ok_line("model", f"{provider.model} · {get_context_size():,} ctx")
        )
        self.console.print(
            _ok_line(
                "tools",
                f"{len(TOOL_SCHEMAS)} registered · read · write · shell · agent",
            )
        )
        if resume and self.state["history"]:
            turns = len(self.state["history"]) // 2
            session_state = f"resumed · {turns} turn(s) restored · /clear to reset"
        elif resume:
            session_state = "resume requested · no prior session"
        elif has_saved_session():
            session_state = "fresh · prior session on disk · --resume to load"
        else:
            session_state = "fresh"
        self.console.print(_ok_line("session", session_state))
        self.console.print(_ok_line("history", INPUT_HISTORY_FILE))
        self.console.print()
        self.console.print(phosphor.bracket("READY"))
        self.console.print(
            f"[{phosphor.WHITE}]Type [{phosphor.accent()} bold]/help[/] for "
            f"commands · [{phosphor.accent()} bold]/model[/] to swap · "
            f"[{phosphor.DIM}]exit[/] to quit.[/]"
        )
        self.console.print(
            f"[{phosphor.DIM}]tip · plan mode rejects tool calls; the model "
            f"describes instead — toggle with [/][{phosphor.accent()}]/plan[/]"
        )
        self.console.print()

    def run(self) -> None:
        # Wire repaint hooks now that the Application exists — the boot screen
        # was printed with the buffer's default no-op on_change.
        self.buffer.on_change = self.app.invalidate
        self.bridge.on_change = self.app.invalidate
        self.app.run()
