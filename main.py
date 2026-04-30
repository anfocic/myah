# main.py
"""Myah's REPL entry point.

Thin by design: everything except the per-turn loop itself lives in sibling
modules (see `repl/` and `display.py`). This file's job is to wire them
together — parse CLI flags, install completer, run the turn loop, report
post-turn stats — and nothing else."""
import argparse
import atexit
import copy
import time

from prompt_toolkit.patch_stdout import patch_stdout

from agent import apply_summary, run_agent, trim_history
from config import NUM_CTX
from display import on_tool_end, on_tool_start
from permissions import check_permission
from providers import get_active_provider
from repl.commands import SLASH_COMMANDS, handle_slash
from repl.console import console
from repl.persistence import (
    has_saved_session,
    load_session,
    save_session,
)
from repl.state import State, new_state
from repl.tool_registry import make_execute_tool, tools
from repl.ui import build_prompt, build_session, build_turn_footer, build_turn_header


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="myah", description="Myah — a hand-rolled agent harness.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="load the prior session from ~/.mia_session.json (default: fresh start).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    state: State = new_state()
    session = build_session(SLASH_COMMANDS, state)
    console.print(
        "[bold]Myah ready.[/bold] "
        "Type [italic dim]/help[/italic dim] for commands, "
        "[italic dim]exit[/italic dim] to quit.\n"
    )
    # State is the single source of truth so slash commands always see the
    # latest values. trim_history rebinds `history`, so keeping a separate
    # local would silently drift once the first auto-trim fires.
    # Session persistence is opt-in on both ends: without --resume we
    # neither load nor save, so a forgotten flag can't silently overwrite
    # a session you wanted to keep. Weak local models also handle short
    # fresh contexts better than long restored ones, so "start clean" is
    # the right default for the target runtime.
    if args.resume:
        load_session(state)
        atexit.register(save_session, state)
        if state["history"]:
            turns = len(state["history"]) // 2
            console.print(
                f"[dim]↳ {turns} turn(s) restored · /clear to reset[/dim]\n"
            )
        else:
            console.print("[dim]↳ no prior session[/dim]\n")
    elif has_saved_session():
        console.print(
            "[dim]↳ prior session on disk · re-launch with --resume to "
            "load (this session won't be saved)[/dim]\n"
        )

    # Hoist permission_check out of the per-turn closure so make_execute_tool
    # can capture it — the spawn_subagent branch needs to forward the same
    # gate into the child run_agent. `_session_allowed` inside permissions.py
    # is the only mutable state, kept as a module global there, so a single
    # closure for the whole process is correct.
    def perm_check(name, args, meta=None):
        return check_permission(console, name, args, meta=meta)

    execute_tool = make_execute_tool(state, permission_check=perm_check)

    # patch_stdout redirects every stdout write (including Rich's) to
    # scroll *above* the prompt_toolkit input line. raw=True preserves
    # ANSI escape sequences Rich emits for colors. This is what gives us
    # the Claude-Code-style pinned prompt — input stays at the bottom
    # while responses and tool logs accumulate above.
    with patch_stdout(raw=True):
        while True:
            # /retry stashes the prior user input here so we skip reading stdin
            # and resubmit it directly. Cleared once consumed.
            user_input = state.pop("_retry_input", None)
            if user_input is None:
                try:
                    user_input = session.prompt(build_prompt(state))
                except KeyboardInterrupt:
                    # prompt_toolkit default: Ctrl+C raises KeyboardInterrupt
                    # rather than clearing. Treat it as "abort this input" and
                    # loop back, matching readline's feel.
                    continue
                except EOFError:
                    console.print("\n[dim]Exiting.[/dim]")
                    break
                if not user_input.strip():
                    continue
                if user_input.strip().lower() == "exit":
                    break
                if handle_slash(user_input, state):
                    console.print()
                    continue

            start = time.time()
            console.print(build_turn_header(state))
            # Snapshot pre-turn history so /rewind can restore it. Deep copy
            # because history entries are dicts; a shallow copy could alias and
            # mutate the snapshot when the next turn rebinds. The deque drops
            # the oldest entry automatically at maxlen.
            state["snapshots"].append(copy.deepcopy(state["history"]))
            dropped: list = []
            try:
                response, state["history"], state["ctx_used"], stats = run_agent(
                    user_input, tools, execute_tool, state["history"],
                    console=console,
                    permission_check=perm_check,
                    plan_mode=state["plan_mode"],
                    on_tool_start=on_tool_start,
                    on_tool_end=on_tool_end,
                    debug=state["debug"],
                )
                ctx_before_trim = state["ctx_used"]
                # tools + model_name keep the trim loop's count aligned
                # with the gate-check value run_agent computed (which is
                # tool-schema-aware). Without them, trim stops at
                # history <= target while the real prompt is still over
                # by the schema budget.
                state["history"], dropped = trim_history(
                    state["history"], state["ctx_used"], NUM_CTX,
                    tools=tools, model_name=get_active_provider().model,
                )
                if dropped:
                    with console.status(
                        "[yellow]Summarizing dropped turns...[/yellow]",
                        spinner="dots",
                    ):
                        state["history"] = apply_summary(state["history"], dropped)
            except KeyboardInterrupt:
                console.print("\n[dim yellow]↳ aborted — history unchanged[/dim yellow]\n")
                continue

            elapsed = time.time() - start
            # Stash per-turn metrics on state so `/stats` can surface them
            # on demand instead of printing a footer line every turn.
            state["last_turn"] = {
                "ctx_used": state["ctx_used"],
                "elapsed_s": elapsed,
                **(stats or {}),
            }
            if dropped:
                threshold = int(0.8 * NUM_CTX)
                console.print(
                    f"[dim yellow]↳ trim_history fired: ctx was "
                    f"{ctx_before_trim} (> threshold {threshold} = 80% of "
                    f"NUM_CTX={NUM_CTX}); dropped {len(dropped) // 2} "
                    f"turn(s), summarized into context; will re-settle "
                    f"after next provider call[/dim yellow]"
                )
            console.print(build_turn_footer(state["ctx_used"], NUM_CTX, elapsed, stats or {}))
            console.print()


if __name__ == "__main__":
    main()
