# main.py
import atexit
import copy
import json
import os
import readline  # noqa: F401 — import enables arrow-key line editing + history in input()
import subprocess
import time
from collections import deque
from typing import NotRequired, TypedDict

from rich.console import Console

from agent import (
    apply_summary,
    compact_history,
    run_agent,
    status_line,
    trim_history,
)
from display import render_diff, render_file_preview
from config import NUM_CTX
from permissions import check_permission
from providers import (
    ProviderError,
    build_provider,
    get_active_provider,
    list_ollama_models,
    set_active_provider,
)
from tools.bash import bash as run_bash
from tools.files import edit_file, read_file, write_file
from tools.git import git_checkout
from tools.harness import harness_info, harness_snapshot
from tools.search import glob, grep
from tools.utils import get_current_time

console = Console()

# Cap on the in-memory snapshot stack used by /rewind. Each snapshot is a
# deep copy of history taken before a run_agent call; 20 is generous enough
# to cover an interactive session without unbounded growth.
REWIND_MAX_SNAPSHOTS = 20

HISTORY_FILE = os.path.expanduser("~/.mia_history")
SESSION_FILE = os.path.expanduser("~/.mia_session.json")


# Runtime-wise this is still a plain dict — TypedDict is a type-checker-only
# construct, not a class. It documents the allowed keys/types so editors
# catch typos like `state["plaan_mode"]` without changing any access shape.
# _retry_input is transient: cmd_retry sets it, the REPL loop pops it.
class State(TypedDict):
    history: list
    ctx_used: int
    plan_mode: bool
    debug: bool
    snapshots: deque
    _retry_input: NotRequired[str]


def _load_input_history() -> None:
    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)
    atexit.register(_save_input_history)


def _save_input_history() -> None:
    try:
        readline.write_history_file(HISTORY_FILE)
    except OSError:
        pass


def _load_session(state: State) -> None:
    try:
        with open(SESSION_FILE) as f:
            loaded = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return
    if not isinstance(loaded, list):
        return
    # Validate each entry: without this filter a hand-edited or corrupted
    # session file slides through isinstance(list) and crashes later when
    # run_agent iterates and calls .get("role") on a non-dict.
    valid = [
        e for e in loaded
        if isinstance(e, dict)
        and isinstance(e.get("role"), str)
        and isinstance(e.get("content"), str)
    ]
    if len(valid) != len(loaded):
        console.print(
            f"[dim yellow]↳ session file had {len(loaded) - len(valid)} "
            "malformed entr(ies); dropped them[/dim yellow]"
        )
    state["history"] = valid


def _save_session(state: State) -> None:
    # Guarded against partial writes by going through a temp file.
    try:
        tmp = SESSION_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state["history"], f)
        os.replace(tmp, SESSION_FILE)
    except OSError:
        pass


def _wipe_session() -> None:
    try:
        os.remove(SESSION_FILE)
    except FileNotFoundError:
        pass

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file. Returns the contents with line numbers prepended (cat -n style) so you can reference specific lines. By default returns the first 1000 lines; use offset and limit to paginate longer files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path to read"},
                    "offset": {
                        "type": "integer",
                        "description": "1-indexed line to start from. Defaults to 1.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of lines to return. Defaults to 1000.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes content to a file at the given path. Overwrites the whole file — prefer edit_file for surgical changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to write to",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Surgically replace a string in a file. old_string must uniquely identify the target unless replace_all is true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path to edit"},
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to find. Must be unique in the file unless replace_all is true.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence instead of requiring uniqueness",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files by name or glob pattern, recursively. Use this to resolve a bare filename (e.g. 'search.py') to its full path before reading or editing. Accepts 'search.py', '*.py', or '**/*.md'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Filename or glob pattern to match",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search from. Defaults to current working directory.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Regex search across files under a path. Returns matching file paths by default, or path:line:text when output_mode is 'content'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Python regex to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search. Defaults to the current working directory.",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional glob filter like '*.py' or '**/*.md'",
                    },
                    "output_mode": {
                        "type": "string",
                        "description": "'files_with_matches' (default) or 'content'",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command. Use for git, running tests, builds, package management, or any shell-only operation. Returns stdout, stderr, and exit code. Requires user permission for each call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory. Defaults to the REPL's current directory.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds. Defaults to 30.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "harness_info",
            "description": "Introspect the harness you are running in: current model, context window size (num_ctx), context used in the previous turn, number of conversation turns in history, working directory, git branch, today's date, and the list of tools available to you. Call this when the user asks about the harness, model, or context usage, or when you need to decide whether to summarize / shorten your reply because ctx is getting tight.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_checkout",
            "description": "Switch to a git branch. Equivalent to `git checkout <branch>`. ALWAYS use this whenever the user asks to switch, check out, or move to a branch — never simulate the action in text and never fabricate output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "branch": {
                        "type": "string",
                        "description": "Branch name to switch to (e.g. 'main', 'feat/foo').",
                    },
                },
                "required": ["branch"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in tools]


def make_execute_tool(state: State):
    """Factory so the harness_info tool can close over the live REPL state
    (ctx_used, history). Passing state through run_agent would leak REPL
    internals into its signature; a closure keeps agent.py state-ignorant."""

    def execute_tool(name, args):
        # Args come from the model, which occasionally omits required keys.
        # Return the error as a tool result so the model can recover instead of
        # crashing the REPL.
        try:
            if name == "get_current_time":
                return get_current_time()
            elif name == "read_file":
                return read_file(
                    args["path"],
                    int(args.get("offset", 1)),
                    int(args["limit"]) if "limit" in args else None,
                )
            elif name == "write_file":
                return write_file(args["path"], args["content"])
            elif name == "edit_file":
                return edit_file(
                    args["path"],
                    args["old_string"],
                    args["new_string"],
                    bool(args.get("replace_all", False)),
                )
            elif name == "glob":
                return glob(args["pattern"], args.get("path", "."))
            elif name == "grep":
                return grep(
                    args["pattern"],
                    args.get("path", "."),
                    args.get("glob"),
                    args.get("output_mode", "files_with_matches"),
                )
            elif name == "bash":
                return run_bash(
                    args["command"],
                    args.get("cwd", "."),
                    int(args.get("timeout", 30)),
                )
            elif name == "harness_info":
                return harness_info(state, TOOL_NAMES)
            elif name == "git_checkout":
                return git_checkout(args["branch"])
            return "Tool not found"
        except KeyError as e:
            return f"Missing required argument: {e}"

    return execute_tool


# ── Tool-call display ────────────────────────────────────────────────────────
# Fires between the assistant's streaming reply and the next "Thinking..."
# spinner. Keeps tool activity visible so the user can follow the loop rather
# than watching silent pauses. Display-only — the model sees the raw result.

_SALIENT_ARG_KEYS = ("path", "command", "pattern", "query")


def _args_preview(args: dict) -> str:
    """One-line arg summary. Prefers the salient key per tool (path/command/
    pattern); falls back to the first value so new tools render something."""
    if not args:
        return ""
    for k in _SALIENT_ARG_KEYS:
        if k in args:
            v = str(args[k])
            return v if len(v) <= 70 else v[:67] + "..."
    first = next(iter(args.values()), "")
    s = str(first)
    return s if len(s) <= 70 else s[:67] + "..."


def on_tool_start(name: str, args: dict) -> None:
    preview = _args_preview(args)
    if preview:
        console.print(
            f"[cyan]⏺[/cyan] [bold]{name}[/bold][dim]({preview})[/dim]"
        )
    else:
        console.print(f"[cyan]⏺[/cyan] [bold]{name}[/bold]")


def on_tool_end(name: str, args: dict, result: str, ok: bool) -> None:
    if not ok:
        if result.startswith("User denied"):
            console.print("  [dim]↳[/dim] [red]denied[/red]")
        elif result.startswith("Plan mode:"):
            console.print("  [dim]↳[/dim] [yellow]blocked (plan mode)[/yellow]")
        elif result.startswith("Tool raised:"):
            first = result.split("\n", 1)[0]
            console.print(f"  [dim]↳[/dim] [red]{first}[/red]")
        else:
            first = result.splitlines()[0] if result else "(empty)"
            console.print(f"  [dim]↳ {first}[/dim]")
        return
    lines = result.splitlines()
    n = len(lines)
    if n <= 1:
        line = result.strip() or "(empty)"
        if len(line) > 80:
            line = line[:77] + "..."
        console.print(f"  [dim]↳ {line}[/dim]")
    else:
        first = next((l for l in lines if l.strip()), "")
        if len(first) > 80:
            first = first[:77] + "..."
        console.print(f"  [dim]↳ {n} lines · {first}[/dim]")

    # Rich per-tool renderers. Only fire when the tool ran successfully and
    # the args we need are present — a malformed edit call shouldn't break
    # display.
    if name == "edit_file" and args.get("old_string") is not None:
        render_diff(
            console,
            str(args.get("path", "")),
            str(args.get("old_string", "")),
            str(args.get("new_string", "")),
        )
    elif name == "read_file" and args.get("path"):
        render_file_preview(console, str(args["path"]), result)


# ── Prompt chrome ────────────────────────────────────────────────────────────
# Branch + mode badge in the prompt prefix, hint line above it, rule between
# turns. All cosmetic — the model sees none of this.


def _current_branch() -> str | None:
    """Best-effort current branch name. Returns None outside a repo or if
    git is missing. Called once per prompt; cheap enough to skip caching."""
    try:
        out = subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _build_prompt(state: State) -> str:
    """`You [branch · plan · debug] ›` — badges only rendered when the
    condition applies so the prompt stays clean in the common case."""
    parts = []
    branch = _current_branch()
    if branch:
        parts.append(branch)
    if state.get("plan_mode"):
        parts.append("[yellow]plan[/yellow]")
    if state.get("debug"):
        parts.append("[magenta]debug[/magenta]")
    badge = f" [dim]\\[{' · '.join(parts)}][/dim]" if parts else ""
    return f"[bold magenta]You[/bold magenta]{badge} [dim]›[/dim] "


def _print_hint() -> None:
    console.print(
        "[dim]/help · /plan · /model · /compact · /rewind · "
        "ctrl+c to interrupt[/dim]"
    )


# ── Tab completion for slash commands ───────────────────────────────────────
# readline calls the completer repeatedly with increasing state indices until
# it returns None. We match only when the buffer starts with "/" so normal
# prose input stays untouched.


def _install_slash_completer() -> None:
    def completer(text: str, state: int):
        buf = readline.get_line_buffer()
        if not buf.startswith("/"):
            return None
        matches = [c for c in SLASH_COMMANDS if c.startswith(buf)]
        return matches[state] if state < len(matches) else None

    readline.set_completer(completer)
    readline.set_completer_delims("")  # treat "/" as part of the word
    readline.parse_and_bind("tab: complete")


def ctx_tag(ctx_used: int, ctx_total: int) -> str:
    pct = ctx_used / ctx_total
    if pct < 0.5:
        color = "green"
    elif pct < 0.8:
        color = "yellow"
    else:
        color = "red"
    return f"[dim]\\[[/dim][{color}]{pct:.0%}[/{color}][dim]][/dim]"


# ── Slash commands ────────────────────────────────────────────────────────────
# Handled by the REPL before the turn reaches the model. They mutate shared
# REPL state via the `state` dict (history list, last ctx_used). Keeping them
# out of the tool layer teaches the control-plane / data-plane split: the
# model never sees these — they're harness UX.

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
    _wipe_session()
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


if __name__ == "__main__":
    _load_input_history()
    _install_slash_completer()
    console.print(
        "[bold]Agent ready.[/bold] "
        "Type [italic dim]/help[/italic dim] for commands, "
        "[italic dim]exit[/italic dim] to quit.\n"
    )
    # state is the single source of truth so slash commands always see the
    # latest values. trim_history rebinds the list, so we can't keep a
    # separate local `history` without drift.
    state: State = {
        "history": [],
        "ctx_used": 0,
        "plan_mode": False,
        "debug": False,
        # In-memory only (not persisted): bounded stack of pre-turn history
        # copies used by /rewind. See §34 — persisting would bloat the session
        # file and let rewind survive across restarts, which conflicts with
        # the session-resume model ("what's on disk" = "what I saw last").
        "snapshots": deque(maxlen=REWIND_MAX_SNAPSHOTS),
    }
    _load_session(state)
    atexit.register(_save_session, state)
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
            _print_hint()
            try:
                user_input = console.input(_build_prompt(state))
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
        # the oldest entry automatically when it hits maxlen.
        state["snapshots"].append(copy.deepcopy(state["history"]))
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
                f"[dim yellow]↳ trimmed {len(dropped) // 2} old turn(s), summarized into context[/dim yellow]"
            )
        console.rule(style="dim")
