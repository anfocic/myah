"""Slash commands + dispatcher. The control-plane side of the control-plane /
data-plane split (CONCEPTS §22): slash input is handled by the REPL without
the model in the loop. Every handler takes `(state, arg="")` — commands
that don't use `arg` ignore it, so `handle_slash` can dispatch uniformly
without inspecting function signatures."""
import json
import os

from rich.panel import Panel

from agent import apply_summary, compact_history
from agent.system_prompt import build_system_prompt
from agent.tokens import estimate_tokens
from config import get_context_size
from display import phosphor, render_todos
from providers import (
    SUPPORTED_PROVIDERS,
    ProviderError,
    build_provider,
    get_active_provider,
    list_ollama_models,
    set_active_provider,
)
from repl.config_loader import config_paths, get_config, get_provenance, reload_config
from repl.console import console
from repl.persistence import wipe_session
from repl.state import State
from repl.tool_registry import TOOL_NAMES
from repl.tool_registry import TOOL_SCHEMAS as REGISTERED_TOOLS
from repl.ui import ctx_tag, render_session_rail
from tools.notes import note_write


def _config_value_repr(val) -> str:
    """Compact repr for config values in /config display."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, str):
        return val
    return str(val)


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, object]:
    """Flatten a nested dict into dot-path keys for display."""
    out: dict[str, object] = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, path))
        else:
            out[path] = v
    return out


def cmd_config(state: State, arg: str = "") -> None:
    """Show, reload, or edit configuration.

    Arg shapes:
        /config          — show merged config with provenance
        /config reload   — re-read config files
        /config path     — print config file paths
        /config edit     — open user config in $EDITOR
    """
    parts = arg.strip().split()
    sub = parts[0] if parts else ""

    if sub == "reload":
        reload_config()
        console.print(
            "[dim]↳ config reloaded into cache; restart for changes to "
            "module-level constants (NUM_CTX, MODEL_NAME, etc.) to take "
            "effect in the running harness[/dim]"
        )
        return

    if sub == "path":
        for label, path in config_paths().items():
            exists = "[green]exists[/green]" if path.exists() else "[dim]missing[/dim]"
            console.print(f"  [bold]{label}:[/bold] {path} {exists}")
        return

    if sub == "edit":
        import shlex
        import subprocess

        user_path = config_paths()["user"]
        user_path.parent.mkdir(parents=True, exist_ok=True)
        if not user_path.exists():
            user_path.write_text("{\n}\n")
        editor = os.environ.get("EDITOR", "vi")
        console.print(f"[dim]↳ opening {user_path} in {editor}...[/dim]")
        # shlex.split preserves common multi-word EDITOR values like
        # "code --wait" while keeping us off the shell (no injection).
        subprocess.call([*shlex.split(editor), str(user_path)])
        return

    # Default: show merged config with provenance
    cfg = get_config()
    prov = get_provenance()
    flat = _flatten_dict(cfg)
    lines = ["[bold]Configuration[/bold] [dim](source in brackets)[/dim]"]
    for key in sorted(flat):
        val = flat[key]
        src = prov.get(key, "default")
        lines.append(
            f"  [dim]{key:<40}[/dim] {_config_value_repr(val):<12} [dim][{src}][/dim]"
        )
    console.print("\n".join(lines))


# Slash commands grouped for /help display. Dispatch still goes through the
# flat SLASH_COMMANDS dict; this only controls how the list is presented.
_HELP_GROUPS: list[tuple[str, list[str]]] = [
    ("info", ["/help", "/about", "/version", "/context", "/profile", "/stats", "/session", "/todos", "/vars"]),
    ("control", ["/cd", "/config", "/clear", "/save-session", "/export"]),
    ("modes", ["/plan", "/debug"]),
    ("undo", ["/retry", "/compact", "/rewind"]),
    ("provider", ["/model", "/eval"]),
]


def cmd_help(state: State, arg: str = "") -> None:
    a = phosphor.accent()
    lines = [
        phosphor.bracket("COMMANDS"),
        f"[{phosphor.DIM}]· dispatched by repl/commands.py — the model never "
        f"sees them[/]",
    ]
    for label, names in _HELP_GROUPS:
        lines.append(f"\n[{phosphor.DIM}]── {label} ──[/]")
        for name in names:
            entry = SLASH_COMMANDS.get(name)
            if entry is None:
                continue
            desc = entry[1]
            lines.append(f"  [{a}]{name:<11}[/] [{phosphor.WHITE}]{desc}[/]")
    lines.append(f"\n[{phosphor.DIM}]── exit ──[/]")
    lines.append(
        f"  [{phosphor.RED}]{'exit':<11}[/] "
        f"[{phosphor.WHITE}]quit the REPL · ^D also works[/]"
    )
    console.print("\n".join(lines))


def cmd_clear(state: State, arg: str = "") -> None:
    state["history"].clear()
    state["snapshots"].clear()  # else /rewind resurrects wiped history
    state["ctx_used"] = 0
    state["todos"] = []
    state["vars"] = {}
    wipe_session()
    console.print("[dim]↳ history cleared (session file wiped too)[/dim]")


def _next_turn_messages(state: State) -> list[dict]:
    """Reconstruct the exact messages array run_agent would send at the
    start of a new turn: system prompt + durable history. No synthetic
    user turn — /context reports what's already committed, not what the
    next user input would add."""
    sys_prompt = build_system_prompt(
        plan_mode=state.get("plan_mode", False),
        cwd=state.get("cwd"),
        todos=state.get("todos", []),
        vars_dict=state.get("vars", {}),
    )
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
    tag = ctx_tag(count, get_context_size())
    plan = "[yellow]ON[/yellow]" if state.get("plan_mode") else "[dim]off[/dim]"
    console.print(
        f"{phosphor.bracket('CONTEXT PROFILE')} "
        f"[{phosphor.DIM}]· {provider.model} · num_ctx {get_context_size():,} "
        f"· {source}[/]\n"
        f"[bold]ctx (next turn):[/bold] {count:,} {tag}\n"
        f"[bold]history turns:[/bold] {len(state['history']) // 2}\n"
        f"[bold]plan mode:[/bold] {plan}\n"
        f"[bold]tools:[/bold] {', '.join(TOOL_NAMES)}"
    )


_STAT_BAR_WIDTH = 24
_SPARK = "▁▂▃▄▅▆▇█"


def _stat_bar(value: float, cap: float, max_: float) -> str:
    """A horizontal mini-bar. `cap` is the "good below this" line; values
    past it shade accent → yellow → red (the design's /stats treatment)."""
    ratio = min(1.0, value / max_) if max_ > 0 else 0.0
    filled = round(ratio * _STAT_BAR_WIDTH)
    if value <= cap:
        color = phosphor.accent()
    elif value <= cap * 1.5:
        color = phosphor.YELLOW
    else:
        color = phosphor.RED
    return (
        f"[{color}]{'█' * filled}[/]"
        f"[{phosphor.DIM}]{'░' * (_STAT_BAR_WIDTH - filled)}[/]"
    )


def _sparkline(values: list[float]) -> str:
    """Map a series onto the ▁▂▃▄▅▆▇█ block ramp, scaled to its own range."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(_SPARK[min(7, int((v - lo) / span * 7))] for v in values)


def cmd_stats(state: State, arg: str = "") -> None:
    """The Phosphor /stats dashboard: mini-bars for the last turn's metrics
    over a sparkline trend of the recent window. Captured by main.py into
    state["last_turn"] / state["turn_history"]."""
    last = state.get("last_turn")
    if not last:
        console.print(f"[{phosphor.DIM}]↳ no completed turn yet[/]")
        return
    ctx_size = get_context_size()
    a = phosphor.accent()

    console.print(phosphor.bracket("LAST TURN"))
    rows: list[tuple[str, str, str]] = [
        ("ctx", _stat_bar(last["ctx_used"], 0.8 * ctx_size, ctx_size),
         f"{last['ctx_used']:,} / {ctx_size:,}"),
        ("wall", _stat_bar(last["elapsed_s"], 5, 10), f"{last['elapsed_s']:.1f}s"),
    ]
    if last.get("ttft_ms") is not None:
        rows.append(("ttft", _stat_bar(last["ttft_ms"], 500, 2000),
                     f"{last['ttft_ms']}ms"))
    if last.get("tok_per_s"):
        rows.append(("rate", _stat_bar(last["tok_per_s"], 30, 80),
                     f"{last['tok_per_s']:.0f} tok/s"))
    if last.get("completion_tokens"):
        rows.append(("gen", _stat_bar(last["completion_tokens"], 2000, 4096),
                     f"{last['completion_tokens']} tok"))
    if last.get("cost_usd") is not None:
        from providers.pricing import format_cost_usd
        # Bar caps are in cents, scaled so $0.01 fills the "good" zone and
        # $0.10 is the wall — keeps a normal turn readable without making
        # an Opus-trajectory row look identical to an ollama one.
        rows.append((
            "cost",
            _stat_bar(last["cost_usd"], cap=0.01, max_=0.10),
            format_cost_usd(last["cost_usd"]),
        ))
    for label, bar, value in rows:
        console.print(
            f"  [{phosphor.DIM}]{label:<6}[/] {bar}  [{phosphor.WHITE}]{value}[/]"
        )

    turns = list(state.get("turn_history", []))
    if len(turns) >= 2:
        console.print(f"\n{phosphor.bracket(f'TREND · LAST {len(turns)} TURNS')}")
        ctx_pcts = [t["ctx_used"] / ctx_size * 100 for t in turns]
        walls = [t["elapsed_s"] for t in turns]
        console.print(
            f"  [{phosphor.DIM}]ctx %[/]  [{a}]{_sparkline(ctx_pcts)}[/]  "
            f"[{phosphor.DIM}]{ctx_pcts[0]:.0f} → {ctx_pcts[-1]:.0f} %[/]"
        )
        console.print(
            f"  [{phosphor.DIM}]wall[/]   [{a}]{_sparkline(walls)}[/]  "
            f"[{phosphor.DIM}]{walls[0]:.1f} → {walls[-1]:.1f} s[/]"
        )
        rates = [t["tok_per_s"] for t in turns if t.get("tok_per_s")]
        if len(rates) >= 2:
            console.print(
                f"  [{phosphor.DIM}]tok/s[/]  [{a}]{_sparkline(rates)}[/]  "
                f"[{phosphor.DIM}]{rates[0]:.0f} → {rates[-1]:.0f} tok/s[/]"
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
    """Render a stacked-block bar showing `tokens` filled out of `num_ctx`,
    the filled run in the accent hue. Clamps at the bar width, so over-budget
    states show a full bar (not a wrapped one) — the numeric readout next to
    it still shows the overflow."""
    if num_ctx <= 0:
        return f"[{phosphor.DIM}]{'░' * _PROFILE_BAR_WIDTH}[/]"
    ratio = max(0.0, min(1.0, tokens / num_ctx))
    filled = round(_PROFILE_BAR_WIDTH * ratio)
    return (
        f"[{phosphor.accent()}]{'█' * filled}[/]"
        f"[{phosphor.DIM}]{'░' * (_PROFILE_BAR_WIDTH - filled)}[/]"
    )


def _profile_row(label: str, tokens: int, num_ctx: int) -> str:
    # The bar carries proper rich markup tags (not bare `[text]`, which the
    # parser would eat); the surrounding columns stay plain.
    pct = (tokens / num_ctx * 100) if num_ctx > 0 else 0.0
    bar = _profile_bar(tokens, num_ctx)
    return f"  {label:<{_PROFILE_LABEL_LEN}} {bar}  {tokens:>5,}  {pct:>4.1f}%"


def _blank_content(messages: list[dict], role: str) -> list[dict]:
    """Return a copy of messages with content="" on every message with the
    given role. Used for marginal-diff counting: replacing content keeps the
    role sequence valid (Anthropic rejects sequences with missing roles)
    while zeroing out that role's content-token contribution."""
    return [({**m, "content": ""} if m.get("role") == role else m) for m in messages]


def _profile_cost_breakdown(
    provider,
    *,
    system_tokens: int,
    user_tokens: int,
    assistant_tokens: int,
    tools_tokens: int,
) -> dict[str, float] | None:
    """Per-role USD attribution for the next turn's prompt composition.

    Uses input price for system/user/tool tokens (they enter the model as
    prompt) and output price for assistant tokens (those were generated and
    billed as completion when produced). Returns None when the active
    (provider, model) isn't in the price table — callers skip the section
    rather than print misleading zeros."""
    from providers.pricing import lookup_price

    price = lookup_price(provider.name, provider.model)
    if price is None:
        return None
    sys_usd = system_tokens * price.input_per_mtok / 1_000_000
    user_usd = user_tokens * price.input_per_mtok / 1_000_000
    asst_usd = assistant_tokens * price.output_per_mtok / 1_000_000
    tools_usd = tools_tokens * price.input_per_mtok / 1_000_000
    return {
        "system": sys_usd,
        "user": user_usd,
        "assistant": asst_usd,
        "tools": tools_usd,
        "total": sys_usd + user_usd + asst_usd + tools_usd,
    }


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

    ctx_size = get_context_size()
    total_pct = (total / ctx_size * 100) if ctx_size > 0 else 0.0

    lines = [
        _profile_row("system", system_row, ctx_size),
        _profile_row("user", user_row, ctx_size),
        _profile_row("assistant", asst_row, ctx_size),
        _profile_row("tools", tools_row, ctx_size),
        _profile_row("framing", framing, ctx_size),
        "",
        f"  [bold]total:[/bold] {total:,} / {ctx_size:,}  ({total_pct:.1f}%)  [dim]· {source}[/dim]",
        "  [dim]tool messages are intra-turn and not shown[/dim]",
    ]

    title = (
        f"{phosphor.bracket('CONTEXT PROFILE')} [{phosphor.DIM}]· "
        f"{provider.model} ({provider.name}) · num_ctx={ctx_size:,}[/]"
    )
    console.print(
        Panel("\n".join(lines), title=title, border_style=phosphor.DIM, padding=(1, 2))
    )

    cost = _profile_cost_breakdown(
        provider,
        system_tokens=system_row,
        user_tokens=user_row,
        assistant_tokens=asst_row,
        tools_tokens=tools_row,
    )
    if cost is not None:
        from providers.pricing import format_cost_usd

        total_usd = cost["total"]
        cost_lines = []
        for role in ("system", "user", "assistant", "tools"):
            usd = cost[role]
            share = (usd / total_usd * 100) if total_usd > 0 else 0.0
            cost_lines.append(
                f"  {role:<{_PROFILE_LABEL_LEN}} {format_cost_usd(usd):>10}  {share:>4.1f}%"
            )
        cost_lines.append("")
        cost_lines.append(
            f"  [bold]total:[/bold] {format_cost_usd(total_usd)}  "
            f"[dim]· next-turn prompt at "
            f"{provider.model} prices · assistant priced as output[/dim]"
        )
        cost_title = (
            f"{phosphor.bracket('COST BREAKDOWN')} [{phosphor.DIM}]· "
            f"{provider.name}:{provider.model}[/]"
        )
        console.print(
            Panel("\n".join(cost_lines), title=cost_title, border_style=phosphor.DIM, padding=(1, 2))
        )


def cmd_eval(state: State, arg: str = "") -> None:
    """Run the eval suite against the active provider/model — or across
    multiple models at once in matrix mode.

    Arg shapes:
        /eval                              — run every task on current model
        /eval list                         — list task ids, don't run
        /eval find_string                  — run one task on current model
        /eval find_string edit_rename      — run a subset
        /eval -m openai-compat:gpt-4.1-mini -m ollama:qwen2.5:7b-instruct
                                           — matrix: full suite on each model
        /eval -m ... -m ... task1 task2    — matrix with a task subset

    Matrix mode (`-m provider:model` repeated) swaps providers internally
    and prints a combined comparison table. Models need to be pre-loaded
    on backends that require explicit loads (LM Studio's `lms load`);
    the runner handles the swap + `ensure_exclusive` eviction but cannot
    load a model the server doesn't already have.

    For single-model runs the active provider is used; `run_suite` saves/
    restores it so the REPL isn't disturbed if a task pins a different one."""
    # Imported lazily so the eval deps (rich.table, task modules, fixtures)
    # only load when the user actually runs /eval. Reloading keeps the REPL
    # dev loop honest: eval runner fixes take effect without restarting Myah.
    import importlib

    from evals import runner as eval_runner

    if not os.environ.get("PYTEST_CURRENT_TEST"):
        eval_runner = importlib.reload(eval_runner)

    parts = arg.strip().split()
    if parts and parts[0] == "list":
        for tid in eval_runner.list_tasks():
            console.print(f"  {tid}")
        return

    models: list[tuple[str, str]] = []
    task_ids: list[str] = []
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok == "-m":
            if i + 1 >= len(parts):
                console.print("[red]/eval: -m needs a provider:model argument[/red]")
                return
            spec = parts[i + 1]
            provider_name, sep, model_name = spec.partition(":")
            if not sep or not model_name:
                console.print(
                    f"[red]/eval: -m expects provider:model, got {spec!r}[/red]"
                )
                return
            models.append((provider_name, model_name))
            i += 2
        else:
            task_ids.append(tok)
            i += 1

    tids = task_ids or None
    if models:
        scope = f"{len(task_ids)} task(s)" if task_ids else "full suite"
        console.print(
            f"[dim]↳ matrix run: {scope} across {len(models)} model(s): "
            f"{', '.join(f'{p}:{m}' for p, m in models)}[/dim]"
        )
        eval_runner.run_matrix(models=models, task_ids=tids, console=console)
        return

    provider = get_active_provider()
    scope = f"{len(task_ids)} task(s)" if task_ids else "full suite"
    console.print(f"[dim]↳ running {scope} on {provider.model} ({provider.name})...[/dim]")
    eval_runner.run_suite(task_ids=tids, console=console)


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
        a = phosphor.accent()
        console.print(phosphor.bracket("MODEL REGISTRY"))
        console.print(
            f"  [{phosphor.DIM}]active[/]  [{a} bold]★ {current.model}[/] "
            f"[{phosphor.DIM}]· {current.name}[/]"
        )
        tags = list_ollama_models()
        if tags:
            console.print(f"\n  [{a} bold]ollama[/] [{phosphor.DIM}]· local daemon[/]")
            for t in tags:
                if t == current.model:
                    console.print(f"  [{a}]●[/] [{phosphor.BRIGHT}]{t}[/]")
                else:
                    console.print(f"  [{phosphor.DIM}]◌[/] [{phosphor.WHITE}]{t}[/]")
        else:
            console.print(
                f"[{phosphor.DIM}]↳ no ollama daemon reachable (or no models "
                f"pulled)[/]"
            )
        console.print(
            f"\n  [{phosphor.DIM}]⤷ /model <name> · swap within active provider[/]\n"
            f"  [{phosphor.DIM}]⤷ /model <provider>:<name> · switch backend[/]"
        )
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


def cmd_cd(state: State, arg: str = "") -> None:
    """Change the harness working directory. No arg → print current directory."""
    import os

    path = arg.strip()
    if not path:
        console.print(state["cwd"])
        return

    current = state["cwd"]
    resolved = os.path.realpath(os.path.join(current, path))

    if not os.path.exists(resolved):
        console.print(f"[red]cd: {path}: No such file or directory[/red]")
        return
    if not os.path.isdir(resolved):
        console.print(f"[red]cd: {path}: Not a directory[/red]")
        return

    state["cwd"] = resolved
    console.print(state["cwd"])


def cmd_session(state: State, arg: str = "") -> None:
    """Render the Phosphor left-rail session console on demand.

    The web mock pins this as a fixed side panel; a scrollback REPL can't
    (see display/streaming.py on staying off the alt screen), so the rail
    prints here and on the boot screen instead."""
    console.print(render_session_rail(state))


def cmd_todos(state: State, arg: str = "") -> None:
    """Show the current todo list. Working memory the model maintains
    via the `todo_write` tool — see tools/todo.py. `/todos clear` wipes
    the list (useful when starting a new task that shouldn't inherit
    stale entries)."""
    if arg.strip() == "clear":
        state["todos"] = []
        console.print("[dim]↳ todo list cleared[/dim]")
        return
    render_todos(console, state.get("todos", []))


def cmd_vars(state: State, arg: str = "") -> None:
    """Show or wipe conversation variables. The named key-value store the
    model populates via set_var / get_var / list_vars / unset_var — see
    tools/vars.py.

    Arg shapes:
        /vars            — render current vars
        /vars clear      — wipe all vars
        /vars unset NAME — drop a single var
    """
    from tools.vars import format_vars

    parts = arg.strip().split(maxsplit=1)
    sub = parts[0] if parts else ""

    if sub == "clear":
        state["vars"] = {}
        console.print("[dim]↳ vars cleared[/dim]")
        return
    if sub == "unset":
        name = parts[1].strip() if len(parts) > 1 else ""
        vars_dict = state.get("vars", {}) or {}
        if name and name in vars_dict:
            del vars_dict[name]
            console.print(f"[dim]↳ unset {name}[/dim]")
        else:
            console.print(f"[dim]↳ unset: {name!r} not set[/dim]")
        return
    console.print(format_vars(state.get("vars", {}) or {}))


def cmd_export(state: State, arg: str = "") -> None:
    """Export the current conversation as a plain markdown transcript to
    an arbitrary path. Distinct from /save-session, which archives into
    the personal vault with Obsidian frontmatter.

    Arg shapes:
        /export                   — timestamped default in cwd
        /export <path>            — exact path (relative or absolute)
        /export <dir>/            — timestamped default inside that dir
    """
    from repl.export import export_conversation

    provider = get_active_provider()
    result = export_conversation(
        history=state.get("history", []),
        model=provider.model,
        provider=provider.name,
        path=arg.strip() or None,
    )
    if result.startswith(("Error", "Refused")):
        console.print(f"[dim red]↳ {result}[/dim red]")
    else:
        console.print(f"[dim]↳ exported to {result}[/dim]")


def _mia_version() -> str:
    """Best-effort lookup of the installed package version. Falls back to
    'dev' when the package isn't installed (e.g. running from a source
    checkout without `pip install -e .`)."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover — Python < 3.8 ditched a while back
        return "unknown"
    try:
        return version("myah")
    except PackageNotFoundError:
        return "dev"


def cmd_version(state: State, arg: str = "") -> None:
    """Print the mia package version."""
    console.print(f"mia {_mia_version()}")


def cmd_about(state: State, arg: str = "") -> None:
    """Print a compact manifest of the running harness: package version,
    active provider/model, context window, num tools, plan mode."""
    provider = get_active_provider()
    a = phosphor.accent()
    lines = [
        phosphor.bracket("MIA"),
        f"  [{phosphor.DIM}]version  [/] [{a}]{_mia_version()}[/]",
        f"  [{phosphor.DIM}]model    [/] [{phosphor.WHITE}]{provider.model}[/] "
        f"[{phosphor.DIM}]· {provider.name}[/]",
        f"  [{phosphor.DIM}]num_ctx  [/] [{phosphor.WHITE}]{get_context_size():,}[/]",
        f"  [{phosphor.DIM}]tools    [/] [{phosphor.WHITE}]{len(TOOL_NAMES)} registered[/]",
        f"  [{phosphor.DIM}]plan mode[/] "
        + ("[yellow]ON[/yellow]" if state.get("plan_mode") else "[dim]off[/dim]"),
        f"  [{phosphor.DIM}]cwd      [/] [{phosphor.WHITE}]{state.get('cwd', '')}[/]",
    ]
    console.print("\n".join(lines))


def cmd_save_session(state: State, arg: str = "") -> None:
    """Save the current conversation to the vault as a markdown archive.

    Arg shapes:
        /save-session            — auto-title from first user message
        /save-session <title>    — use the given title
    """
    import re
    from datetime import date, datetime

    history = state.get("history", [])
    if not history:
        console.print("[dim]↳ no conversation history to save[/dim]")
        return

    provider = get_active_provider()
    model = provider.model
    provider_name = provider.name
    ctx_size = provider.context_size

    # Build title
    title = arg.strip()
    if not title:
        first_user = ""
        for msg in history:
            if msg.get("role") == "user":
                first_user = msg.get("content", "")
                break
        title = first_user.strip().split("\n")[0][:60]
        if not title:
            title = "untitled-session"

    # Slugify for filename
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", title.lower()).strip("-")[:40]
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    filename = f"sessions/{timestamp}-{slug}.md"

    # Build frontmatter
    turn_count = sum(1 for msg in history if msg.get("role") == "assistant")
    total_user_msgs = sum(1 for msg in history if msg.get("role") == "user")
    last_turn = state.get("last_turn", {})
    ctx_used = last_turn.get("ctx_used", state.get("ctx_used", 0))

    lines = [
        "---",
        f"date: {date.today().isoformat()}",
        f"datetime: {datetime.now().isoformat()}",
        f"model: {model}",
        f"provider: {provider_name}",
        f"context_size: {ctx_size}",
        f"turns: {turn_count}",
        f"user_messages: {total_user_msgs}",
        f"ctx_used_last_turn: {ctx_used}",
        "tags: session",
        "---",
        "",
        f"# Session: {title}",
        "",
        "## Summary",
        "",
        "_Add your own summary here, or ask the model to extract entities and decisions._",
        "",
        "## Transcript",
        "",
    ]

    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"**user:** {content}")
            lines.append("")
        elif role == "assistant":
            lines.append(f"**assistant:** {content}")
            lines.append("")
        elif role == "system":
            lines.append("*[system note omitted]*")
            lines.append("")

    lines.extend([
        "## Extracted Data",
        "",
        "- **Entities:**",
        "- **Decisions:**",
        "- **TODOs:**",
        "",
        "## Raw Telemetry",
        "",
        "```json",
        json.dumps(last_turn, indent=2, default=str),
        "```",
    ])

    content = "\n".join(lines)
    result = note_write(filename, content)
    if result.startswith("Error") or result.startswith("Refused"):
        console.print(f"[dim red]↳ {result}[/dim red]")
    else:
        note_path = result.replace("Note written: ", "")
        console.print(f"[dim]↳ saved session to [[{note_path}]][/dim]")


SLASH_COMMANDS: dict = {
    "/help": (cmd_help, "show this list"),
    "/version": (cmd_version, "print the mia package version"),
    "/about": (cmd_about, "show a compact manifest of the running harness"),
    "/cd": (cmd_cd, "change the harness working directory (or print it with no argument)"),
    "/session": (cmd_session, "show the session console (state/ctx/tools rail)"),
    "/config": (cmd_config, "show/reload/edit configuration (reload | path | edit)"),
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
    "/save-session": (cmd_save_session, "archive current conversation to the vault (`/save-session <title>`)"),
    "/export": (cmd_export, "export transcript as portable markdown (`/export [<path>]`)"),
    "/todos": (cmd_todos, "show the model's working todo list (`/todos clear` to wipe)"),
    "/vars": (cmd_vars, "show conversation vars (`/vars clear` to wipe, `/vars unset NAME`)"),
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
