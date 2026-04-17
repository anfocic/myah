# main.py
"""Mia's REPL entry point.

Thin by design: everything except the per-turn loop itself lives in sibling
modules (see `repl/` and `display.py`). This file's job is to wire them
together — load session, install completer, run the turn loop, report
post-turn stats — and nothing else."""
import atexit
import copy
import time

from agent import apply_summary, run_agent, status_line, trim_history
from config import NUM_CTX
from display import on_tool_end, on_tool_start
from permissions import check_permission
from repl.commands import SLASH_COMMANDS, handle_slash
from repl.console import console
from repl.persistence import load_input_history, load_session, save_session
from repl.state import State, new_state
from repl.tool_registry import make_execute_tool, tools
from repl.ui import build_prompt, ctx_tag, install_slash_completer, print_hint


def main() -> None:
    load_input_history()
    install_slash_completer(SLASH_COMMANDS)
    console.print(
        "[bold]Agent ready.[/bold] "
        "Type [italic dim]/help[/italic dim] for commands, "
        "[italic dim]exit[/italic dim] to quit.\n"
    )
    # State is the single source of truth so slash commands always see the
    # latest values. trim_history rebinds `history`, so keeping a separate
    # local would silently drift once the first auto-trim fires.
    state: State = new_state()
    load_session(state)
    atexit.register(save_session, state)
    if state["history"]:
        turns = len(state["history"]) // 2
        console.print(
            f"[dim]↳ resumed session: {turns} turn(s) restored "
            f"(use /clear to start fresh)[/dim]\n"
        )
    execute_tool = make_execute_tool(state)

    while True:
        # /retry stashes the prior user input here so we skip reading stdin
        # and resubmit it directly. Cleared once consumed.
        user_input = state.pop("_retry_input", None)
        if user_input is None:
            print_hint()
            try:
                user_input = console.input(build_prompt(state))
            except (KeyboardInterrupt, EOFError):
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
        # Snapshot pre-turn history so /rewind can restore it. Deep copy
        # because history entries are dicts; a shallow copy could alias and
        # mutate the snapshot when the next turn rebinds. The deque drops
        # the oldest entry automatically at maxlen.
        state["snapshots"].append(copy.deepcopy(state["history"]))
        dropped: list = []
        try:
            with console.status(
                "[yellow]Thinking...[/yellow]", spinner="dots"
            ) as status:
                def perm_check(name, args):
                    return check_permission(console, status, name, args)

                response, state["history"], state["ctx_used"], stats = run_agent(
                    user_input, tools, execute_tool, state["history"],
                    status=status, console=console,
                    permission_check=perm_check,
                    plan_mode=state["plan_mode"],
                    on_tool_start=on_tool_start,
                    on_tool_end=on_tool_end,
                    debug=state["debug"],
                )
                state["history"], dropped = trim_history(
                    state["history"], state["ctx_used"], NUM_CTX
                )
                if dropped:
                    status.update(
                        status_line(
                            "Summarizing dropped turns...",
                            state["ctx_used"],
                            time.time() - start,
                        )
                    )
                    status.start()  # agent may have stopped it while streaming
                    state["history"] = apply_summary(state["history"], dropped)
        except KeyboardInterrupt:
            console.print("\n[dim yellow]↳ aborted — history unchanged[/dim yellow]\n")
            continue

        tag = ctx_tag(state["ctx_used"], NUM_CTX)
        elapsed = time.time() - start
        rate = stats.get("tok_per_s")
        rate_s = f" · [dim]{rate:.0f} tok/s[/dim]" if rate else ""
        console.print(f"[dim]{tag} · {elapsed:.1f}s[/dim]{rate_s}")
        if dropped:
            console.print(
                f"[dim yellow]↳ trimmed {len(dropped) // 2} old turn(s), "
                "summarized into context[/dim yellow]"
            )
        console.rule(style="dim")


if __name__ == "__main__":
    main()
