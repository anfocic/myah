"""Slash commands + dispatcher. The control-plane side of the control-plane /
data-plane split (CONCEPTS §22): slash input is handled by the REPL without
the model in the loop. Every handler takes `(state, arg="")` — commands
that don't use `arg` ignore it, so `handle_slash` can dispatch uniformly
without inspecting function signatures."""
from agent import apply_summary, compact_history
from providers import (
    ProviderError,
    build_provider,
    get_active_provider,
    list_ollama_models,
    set_active_provider,
)
from repl.console import console
from repl.persistence import wipe_session
from repl.state import State
from repl.tool_registry import TOOL_NAMES
from repl.ui import ctx_tag
from tools.harness import harness_snapshot


def cmd_help(state: State, arg: str = "") -> None:
    lines = ["[bold]Commands:[/bold]"]
    for name, (_, desc) in SLASH_COMMANDS.items():
        lines.append(f"  [cyan]{name}[/cyan] — {desc}")
    lines.append("  [cyan]exit[/cyan] — quit")
    console.print("\n".join(lines))


def cmd_clear(state: State, arg: str = "") -> None:
    state["history"].clear()
    state["snapshots"].clear()  # else /rewind resurrects wiped history
    state["ctx_used"] = 0
    wipe_session()
    console.print("[dim]↳ history cleared (session file wiped too)[/dim]")


def cmd_context(state: State, arg: str = "") -> None:
    s = harness_snapshot(state, TOOL_NAMES)
    tag = ctx_tag(s["ctx_used"], s["num_ctx"])
    plan = "[yellow]ON[/yellow]" if state.get("plan_mode") else "[dim]off[/dim]"
    console.print(
        f"[bold]model:[/bold] {s['model']} [dim]({s['provider']})[/dim]\n"
        f"[bold]num_ctx:[/bold] {s['num_ctx']}\n"
        f"[bold]ctx used:[/bold] {s['ctx_used']} {tag}\n"
        f"[bold]history turns:[/bold] {s['history_turns']}\n"
        f"[bold]plan mode:[/bold] {plan}\n"
        f"[bold]tools:[/bold] {', '.join(s['tools'])}"
    )


def cmd_plan(state: State, arg: str = "") -> None:
    state["plan_mode"] = not state.get("plan_mode", False)
    status = "[yellow]ON[/yellow]" if state["plan_mode"] else "[dim]off[/dim]"
    console.print(
        f"[dim]↳ plan mode[/dim] {status} "
        f"[dim]— in plan mode, tool calls are rejected; the model describes instead[/dim]"
    )


def cmd_debug(state: State, arg: str = "") -> None:
    state["debug"] = not state.get("debug", False)
    status = "[magenta]ON[/magenta]" if state["debug"] else "[dim]off[/dim]"
    console.print(
        f"[dim]↳ debug[/dim] {status} "
        f"[dim]— when on, the full messages array is printed before each provider call[/dim]"
    )


def cmd_retry(state: State, arg: str = "") -> None:
    """Pop the last user/assistant pair from history and re-run the user
    input. Useful when the model flubbed and we want a fresh sample without
    retyping. No-op (with a note) if there's no prior turn."""
    history = state["history"]
    if len(history) < 2 or history[-2].get("role") != "user":
        console.print("[dim]↳ no previous turn to retry[/dim]")
        return
    last_user = history[-2].get("content", "")
    del history[-2:]
    state["_retry_input"] = last_user
    console.print(f"[dim]↳ retrying: {last_user[:80]}[/dim]")


def cmd_compact(state: State, arg: str = "") -> None:
    """Proactive compact. Keeps the last 2 user/assistant pairs, summarizes
    the rest into a system note at the start of history. Different from the
    auto-trim hysteresis (§6) that waits for 80% ctx — /compact lets the
    user reset before an expensive operation instead of reacting to it."""
    new_history, dropped = compact_history(state["history"])
    if not dropped:
        console.print("[dim]↳ nothing to compact (history ≤ 2 turns)[/dim]")
        return
    n_turns = len(dropped) // 2
    console.print(f"[dim]↳ compacting {n_turns} turn(s)...[/dim]")
    summarized = apply_summary(new_history, dropped)
    state["history"] = summarized
    if summarized is new_history:
        # apply_summary returns its input verbatim when summarization fails
        console.print(
            f"[dim]↳ dropped {n_turns} turn(s) "
            "(summary failed — kept last 2 turns only)[/dim]"
        )
    else:
        console.print(f"[dim]↳ compacted {n_turns} turn(s) into summary[/dim]")
    state["ctx_used"] = 0  # real count settles after the next provider call


def cmd_rewind(state: State, arg: str = "") -> None:
    """Rewind N turns by popping the snapshot stack (§34). Snapshots are
    pushed before each run_agent call, so `/rewind 1` restores the state
    from just before the previous turn. N defaults to 1."""
    try:
        n = int(arg) if arg.strip() else 1
    except ValueError:
        console.print(f"[dim]↳ /rewind expects a number, got: {arg!r}[/dim]")
        return
    if n < 1:
        console.print("[dim]↳ /rewind N must be ≥ 1[/dim]")
        return
    snapshots = state["snapshots"]
    if not snapshots:
        console.print("[dim]↳ nothing to rewind[/dim]")
        return
    n = min(n, len(snapshots))
    state["history"] = snapshots[-n]
    for _ in range(n):
        snapshots.pop()
    state["ctx_used"] = 0
    turns = len(state["history"]) // 2
    console.print(
        f"[dim]↳ rewound {n} turn(s) ({turns} turn(s) remain)[/dim]"
    )


def cmd_model(state: State, arg: str = "") -> None:
    """List or swap the active model at runtime. No arg → list locally
    available ollama tags + the current model. With arg → swap.

    Arg shapes:
        /model qwen2.5:14b             — same provider, different model
        /model ollama:qwen2.5:14b      — force ollama backend
        /model openai-compat:gpt-4o-mini — cross-provider swap
    """
    current = get_active_provider()
    if not arg.strip():
        console.print(
            f"[bold]current:[/bold] {current.model} [dim]({current.name})[/dim]"
        )
        tags = list_ollama_models()
        if tags:
            console.print("[bold]ollama (local):[/bold]")
            for t in tags:
                marker = "[cyan]*[/cyan] " if t == current.model else "  "
                console.print(f"  {marker}{t}")
        else:
            console.print(
                "[dim]↳ no ollama daemon reachable (or no models pulled)[/dim]"
            )
        console.print(
            "[dim]↳ swap with /model <name> or /model <provider>:<name>[/dim]"
        )
        return

    # Parse "<provider>:<model>" vs plain "<model>" (keep current provider).
    if ":" in arg and arg.split(":", 1)[0] in {"ollama", "openai-compat"}:
        provider_name, model = arg.split(":", 1)
    else:
        provider_name, model = current.name, arg

    # For ollama, validate against the local tag list when possible. Saves
    # the user a failed-turn round-trip to discover a typo.
    if provider_name == "ollama":
        tags = list_ollama_models()
        if tags and model not in tags:
            console.print(
                f"[dim]↳ {model!r} not in local ollama tags "
                f"({', '.join(tags[:6])}{'...' if len(tags) > 6 else ''}). "
                "Pull it first (`ollama pull <name>`) or use an exact tag.[/dim]"
            )
            return

    try:
        new_provider = build_provider(provider_name, model)
    except (ValueError, ProviderError) as e:
        console.print(f"[dim red]↳ swap failed: {e}[/dim red]")
        return

    set_active_provider(new_provider)
    console.print(
        f"[dim]↳ switched to {model} "
        f"\\[{provider_name}] (takes effect next turn)[/dim]"
    )


SLASH_COMMANDS: dict = {
    "/help": (cmd_help, "show this list"),
    "/clear": (cmd_clear, "reset conversation history + wipe saved session"),
    "/context": (cmd_context, "show context window usage + harness info"),
    "/plan": (cmd_plan, "toggle plan mode (describe, don't execute)"),
    "/debug": (cmd_debug, "toggle debug (print messages array each turn)"),
    "/retry": (cmd_retry, "re-run the last turn (pops + resubmits)"),
    "/compact": (cmd_compact, "summarize older turns, keep the last 2"),
    "/rewind": (cmd_rewind, "undo N turns (default 1) via in-memory snapshots"),
    "/model": (cmd_model, "list or swap the active model (e.g. /model qwen2.5:14b)"),
}


def handle_slash(user_input: str, state: State) -> bool:
    """If user_input is a slash command, run it and return True. Else False.
    Every handler accepts `(state, arg="")`; commands that don't take an arg
    simply ignore it."""
    parts = user_input.strip().split(maxsplit=1)
    if not parts or not parts[0].startswith("/"):
        return False
    cmd = parts[0]
    arg = parts[1] if len(parts) > 1 else ""
    entry = SLASH_COMMANDS.get(cmd)
    if entry is None:
        console.print(f"[dim]↳ unknown command: {cmd} (try /help)[/dim]")
        return True
    handler, _ = entry
    handler(state, arg)
    return True
