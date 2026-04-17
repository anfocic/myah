# main.py
import atexit
import os
import readline  # noqa: F401 — import enables arrow-key line editing + history in input()
import time

from rich.console import Console

from agent import status_line, run_agent, summarize_dropped, trim_history
from config import NUM_CTX
from permissions import check_permission
from tools.bash import bash as run_bash
from tools.files import edit_file, read_file, write_file
from tools.harness import harness_info, harness_snapshot
from tools.search import glob, grep
from tools.utils import get_current_time

console = Console()

HISTORY_FILE = os.path.expanduser("~/.mia_history")


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
]

TOOL_NAMES = [t["function"]["name"] for t in tools]


def make_execute_tool(state: dict):
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
            return "Tool not found"
        except KeyError as e:
            return f"Missing required argument: {e}"

    return execute_tool


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

def cmd_help(state):
    lines = ["[bold]Commands:[/bold]"]
    for name, (_, desc) in SLASH_COMMANDS.items():
        lines.append(f"  [cyan]{name}[/cyan] — {desc}")
    lines.append("  [cyan]exit[/cyan] — quit")
    console.print("\n".join(lines))


def cmd_clear(state):
    state["history"].clear()
    state["ctx_used"] = 0
    console.print("[dim]↳ history cleared[/dim]")


def cmd_context(state):
    s = harness_snapshot(state, TOOL_NAMES)
    tag = ctx_tag(s["ctx_used"], s["num_ctx"])
    console.print(
        f"[bold]model:[/bold] {s['model']} [dim]({s['provider']})[/dim]\n"
        f"[bold]num_ctx:[/bold] {s['num_ctx']}\n"
        f"[bold]ctx used:[/bold] {s['ctx_used']} {tag}\n"
        f"[bold]history turns:[/bold] {s['history_turns']}\n"
        f"[bold]tools:[/bold] {', '.join(s['tools'])}"
    )


SLASH_COMMANDS: dict = {
    "/help": (cmd_help, "show this list"),
    "/clear": (cmd_clear, "reset conversation history"),
    "/context": (cmd_context, "show context window usage + harness info"),
}


def handle_slash(user_input: str, state: dict) -> bool:
    """If user_input is a slash command, run it and return True. Else False."""
    cmd = user_input.strip().split(maxsplit=1)[0]
    if not cmd.startswith("/"):
        return False
    entry = SLASH_COMMANDS.get(cmd)
    if entry is None:
        console.print(f"[dim]↳ unknown command: {cmd} (try /help)[/dim]")
        return True
    handler, _ = entry
    handler(state)
    return True


if __name__ == "__main__":
    _load_input_history()
    console.print(
        "[bold]Agent ready.[/bold] "
        "Type [italic dim]/help[/italic dim] for commands, "
        "[italic dim]exit[/italic dim] to quit.\n"
    )
    # state is the single source of truth so slash commands always see the
    # latest values. trim_history rebinds the list, so we can't keep a
    # separate local `history` without drift.
    state: dict = {"history": [], "ctx_used": 0}
    execute_tool = make_execute_tool(state)
    while True:
        try:
            user_input = console.input("[bold magenta]You[/bold magenta] [dim]›[/dim] ")
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
        try:
            with console.status(
                "[yellow]Thinking...[/yellow]", spinner="dots"
            ) as status:
                def perm_check(name, args):
                    return check_permission(console, status, name, args)

                response, state["history"], state["ctx_used"] = run_agent(
                    user_input, tools, execute_tool, state["history"],
                    status=status, console=console,
                    permission_check=perm_check,
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
                    summary = summarize_dropped(dropped)
                    if summary:
                        state["history"].insert(
                            0,
                            {
                                "role": "system",
                                "content": f"Summary of earlier conversation: {summary}",
                            },
                        )
        except KeyboardInterrupt:
            console.print("\n[dim yellow]↳ aborted — history unchanged[/dim yellow]\n")
            continue

        tag = ctx_tag(state["ctx_used"], NUM_CTX)
        elapsed = time.time() - start
        console.print(f"[dim]{tag} · {elapsed:.1f}s[/dim]\n")
        if dropped:
            console.print(
                f"[dim yellow]↳ trimmed {len(dropped) // 2} old turn(s), summarized into context[/dim yellow]\n"
            )
