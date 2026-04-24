"""Slash commands + dispatcher. The control-plane side of the control-plane /
data-plane split (CONCEPTS §22): slash input is handled by the REPL without
the model in the loop. Every handler takes `(state, arg="")` — commands
that don't use `arg` ignore it, so `handle_slash` can dispatch uniformly
without inspecting function signatures."""
from rich.panel import Panel

from agent import apply_summary, compact_history
from agent.system_prompt import build_system_prompt
from agent.tokens import estimate_tokens
from config import NUM_CTX
from providers import (
    SUPPORTED_PROVIDERS,
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
from repl.tool_registry import tools as REGISTERED_TOOLS
from repl.ui import ctx_tag


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


def _next_turn_messages(state: State) -> list[dict]:
    """Reconstruct the exact messages array run_agent would send at the
    start of a new turn: system prompt + durable history. No synthetic
    user turn — /context reports what's already committed, not what the
    next user input would add."""
    sys_prompt = build_system_prompt(plan_mode=state.get("plan_mode", False))
    return [{"role": "system", "content": sys_prompt}] + list(state["history"])


def _count_or_estimate(provider, messages: list[dict], tools: list[dict] | None) -> tuple[int, str]:
    """Try the provider's exact tokenizer; on any failure fall back to the
    char/4 estimator so /context and /profile never hard-fail.

    Returns (count, source_label). source_label is "exact" on success or
    a short "estimate (...)" string explaining why we fell back."""
    try:
        return provider.count_tokens(messages, tools), "exact"
    except ProviderError as e:
        return estimate_tokens(messages), f"estimate ({e})"
    except Exception as e:  # noqa: BLE001 — last-ditch; never break the REPL
        return estimate_tokens(messages), f"estimate ({type(e).__name__})"


def cmd_context(state: State, arg: str = "") -> None:
    provider = get_active_provider()
    messages = _next_turn_messages(state)
    count, source = _count_or_estimate(provider, messages, REGISTERED_TOOLS)
    state["ctx_used"] = count  # keep the per-turn tag in sync with the reading
    tag = ctx_tag(count, NUM_CTX)
    plan = "[yellow]ON[/yellow]" if state.get("plan_mode") else "[dim]off[/dim]"
    console.print(
        f"[bold]model:[/bold] {provider.model} [dim]({provider.name})[/dim]\n"
        f"[bold]num_ctx:[/bold] {NUM_CTX:,}\n"
        f"[bold]ctx (next turn):[/bold] {count:,} {tag} [dim]· {source}[/dim]\n"
        f"[bold]history turns:[/bold] {len(state['history']) // 2}\n"
        f"[bold]plan mode:[/bold] {plan}\n"
        f"[bold]tools:[/bold] {', '.join(TOOL_NAMES)}"
    )


def cmd_stats(state: State, arg: str = "") -> None:
    """Show the last turn's metrics. Captured by main.py into state["last_turn"]
    now that the per-turn footer was dropped from the REPL output."""
    last = state.get("last_turn")
    if not last:
        console.print("[dim]↳ no completed turn yet[/dim]")
        return
    tag = ctx_tag(last["ctx_used"], NUM_CTX)
    parts = [
        f"[bold]ctx:[/bold] {last['ctx_used']:,}/{NUM_CTX:,} {tag}",
        f"[bold]wall:[/bold] {last['elapsed_s']:.1f}s",
    ]
    if last.get("ttft_ms") is not None:
        parts.append(f"[bold]ttft:[/bold] {last['ttft_ms']}ms")
    if last.get("tok_per_s"):
        parts.append(f"[bold]rate:[/bold] {last['tok_per_s']:.0f} tok/s")
    if last.get("completion_tokens"):
        parts.append(f"[bold]gen:[/bold] {last['completion_tokens']} tok")
    console.print(" · ".join(parts))


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
            f"[dim]↳ dropped {n_turns} turn(s) (summary failed — kept last 2 turns only)[/dim]"
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
    console.print(f"[dim]↳ rewound {n} turn(s) ({turns} turn(s) remain)[/dim]")


_PROFILE_BAR_WIDTH = 28
_PROFILE_LABEL_LEN = 10


def _profile_bar(tokens: int, num_ctx: int) -> str:
    """Render a stacked-block bar showing `tokens` filled out of `num_ctx`.
    Clamps at the bar width, so over-budget states show a full bar (not a
    wrapped one) — the numeric readout next to it still shows the overflow."""
    if num_ctx <= 0:
        return "░" * _PROFILE_BAR_WIDTH
    ratio = max(0.0, min(1.0, tokens / num_ctx))
    filled = round(_PROFILE_BAR_WIDTH * ratio)
    return "█" * filled + "░" * (_PROFILE_BAR_WIDTH - filled)


def _profile_row(label: str, tokens: int, num_ctx: int) -> str:
    # No square brackets around the bar — rich parses `[text]` as markup
    # (e.g. `[bold]…[/bold]`) and silently eats anything that isn't a known
    # style. `[system ████]` → gone. Whitespace separation reads fine and
    # sidesteps the parser.
    pct = (tokens / num_ctx * 100) if num_ctx > 0 else 0.0
    bar = _profile_bar(tokens, num_ctx)
    return f"  {label:<{_PROFILE_LABEL_LEN}} {bar}  {tokens:>5,}  {pct:>4.1f}%"


def _blank_content(messages: list[dict], role: str) -> list[dict]:
    """Return a copy of messages with content="" on every message with the
    given role. Used for marginal-diff counting: replacing content keeps the
    role sequence valid (Anthropic rejects sequences with missing roles)
    while zeroing out that role's content-token contribution."""
    return [({**m, "content": ""} if m.get("role") == role else m) for m in messages]


def cmd_profile(state: State, arg: str = "") -> None:
    """Per-role breakdown of what the next turn would send, using the
    active provider's real tokenizer.

    Marginal-diff method — each row equals the delta between "full prompt"
    and "full prompt with that role's content blanked out". So every row
    is provider-exact for that category's content tokens. Rows sum to
    total-minus-framing; the remainder is shown as a "framing" row so
    nothing is hidden. Tool messages are intra-turn scratch space (not
    in state['history']) and don't show here."""
    provider = get_active_provider()
    messages = _next_turn_messages(state)

    try:
        total_all = provider.count_tokens(messages, REGISTERED_TOOLS)
        total_notools = provider.count_tokens(messages, None)
        total_nosys = provider.count_tokens(_blank_content(messages, "system"), None)
        total_nouser = provider.count_tokens(_blank_content(messages, "user"), None)
        total_noasst = provider.count_tokens(_blank_content(messages, "assistant"), None)
        source = f"exact via {provider.name}"
        tools_row = max(0, total_all - total_notools)
        system_row = max(0, total_notools - total_nosys)
        user_row = max(0, total_notools - total_nouser)
        asst_row = max(0, total_notools - total_noasst)
        framing = max(0, total_notools - (system_row + user_row + asst_row))
        total = total_all
    except Exception as e:  # noqa: BLE001 — last-ditch fallback (covers ProviderError)
        # Fallback: char/4 per role. Rows only sum to total (no framing row).
        history = state["history"]
        user_msgs = [m for m in history if m.get("role") == "user"]
        asst_msgs = [m for m in history if m.get("role") == "assistant"]
        system_row = estimate_tokens([{"role": "system", "content": messages[0]["content"]}])
        user_row = estimate_tokens(user_msgs)
        asst_row = estimate_tokens(asst_msgs)
        tools_row = 0
        framing = 0
        total = system_row + user_row + asst_row
        source = f"estimate ({type(e).__name__}: {e})"

    total_pct = (total / NUM_CTX * 100) if NUM_CTX > 0 else 0.0

    lines = [
        _profile_row("system", system_row, NUM_CTX),
        _profile_row("user", user_row, NUM_CTX),
        _profile_row("assistant", asst_row, NUM_CTX),
        _profile_row("tools", tools_row, NUM_CTX),
        _profile_row("framing", framing, NUM_CTX),
        "",
        f"  [bold]total:[/bold] {total:,} / {NUM_CTX:,}  ({total_pct:.1f}%)  [dim]· {source}[/dim]",
        "  [dim]tool messages are intra-turn and not shown[/dim]",
    ]

    title = (
        f"[bold]Context profile[/bold] [dim]· {provider.model} "
        f"({provider.name}) · NUM_CTX={NUM_CTX:,}[/dim]"
    )
    console.print(Panel("\n".join(lines), title=title, border_style="dim", padding=(1, 2)))


def cmd_eval(state: State, arg: str = "") -> None:
    """Run the eval suite against the active provider/model from inside the
    REPL. Mirrors `scripts/run_evals.py` but scoped to the current session.

    Arg shapes:
        /eval                         — run every task
        /eval list                    — list task ids, don't run
        /eval find_string             — run one task
        /eval find_string edit_rename — run a subset (space-separated)

    The active provider is used. Swap with /model first if you want to
    compare models. `run_suite` saves/restores the active provider itself,
    so the REPL session isn't disturbed if a task pins a different one."""
    # Imported lazily so the eval deps (rich.table, task modules, fixtures)
    # only load when the user actually runs /eval.
    from evals.runner import list_tasks, run_suite

    parts = arg.strip().split()
    if parts and parts[0] == "list":
        for tid in list_tasks():
            console.print(f"  {tid}")
        return

    task_ids = parts or None
    provider = get_active_provider()
    scope = f"{len(task_ids)} task(s)" if task_ids else "full suite"
    console.print(f"[dim]↳ running {scope} on {provider.model} ({provider.name})...[/dim]")
    run_suite(task_ids=task_ids, console=console)


def cmd_model(state: State, arg: str = "") -> None:
    """List or swap the active model at runtime. No arg → list locally
    available ollama tags + the current model. With arg → swap.

    Arg shapes:
        /model qwen2.5:14b               — same provider, different model
        /model ollama:qwen2.5:14b        — force ollama backend
        /model openai-compat:gpt-4o-mini — local OpenAI-compat server
        /model openai:gpt-4.1-mini       — first-party OpenAI
        /model anthropic:claude-sonnet-4-6 — first-party Anthropic
        /model deepseek:deepseek-chat    — DeepSeek
    """
    current = get_active_provider()
    if not arg.strip():
        console.print(f"[bold]current:[/bold] {current.model} [dim]({current.name})[/dim]")
        tags = list_ollama_models()
        if tags:
            console.print("[bold]ollama (local):[/bold]")
            for t in tags:
                marker = "[cyan]*[/cyan] " if t == current.model else "  "
                console.print(f"  {marker}{t}")
        else:
            console.print("[dim]↳ no ollama daemon reachable (or no models pulled)[/dim]")
        console.print("[dim]↳ swap with /model <name> or /model <provider>:<name>[/dim]")
        return

    # Parse "<provider>:<model>" vs plain "<model>" (keep current provider).
    # Only treat the first colon as a separator if the prefix is a known
    # provider name — otherwise `qwen2.5:14b` (which contains a colon in
    # its ollama tag) would be misread as provider="qwen2.5".
    if ":" in arg and arg.split(":", 1)[0] in SUPPORTED_PROVIDERS:
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
    console.print(f"[dim]↳ switched to {model} \\[{provider_name}] (takes effect next turn)[/dim]")


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
    "/profile": (cmd_profile, "per-role token breakdown of the next turn's prompt"),
    "/stats": (cmd_stats, "show the last turn's ctx/wall/ttft/rate"),
    "/eval": (cmd_eval, "run the eval suite (`/eval list` to list, `/eval <id>...` for a subset)"),
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
