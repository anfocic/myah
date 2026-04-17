# main.py
import time

from rich.console import Console

from agent import status_line, run_agent, summarize_dropped, trim_history
from config import NUM_CTX
from permissions import check_permission
from tools.files import edit_file, read_file, write_file
from tools.search import grep
from tools.utils import get_current_time

console = Console()

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
]


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
        elif name == "grep":
            return grep(
                args["pattern"],
                args.get("path", "."),
                args.get("glob"),
                args.get("output_mode", "files_with_matches"),
            )
        return "Tool not found"
    except KeyError as e:
        return f"Missing required argument: {e}"


def ctx_tag(ctx_used: int, ctx_total: int) -> str:
    pct = ctx_used / ctx_total
    if pct < 0.5:
        color = "green"
    elif pct < 0.8:
        color = "yellow"
    else:
        color = "red"
    return f"[dim]\\[[/dim][{color}]{pct:.0%}[/{color}][dim]][/dim]"


if __name__ == "__main__":
    console.print(
        "[bold]Agent ready.[/bold] Type [italic dim]exit[/italic dim] to quit.\n"
    )
    history = []
    while True:
        user_input = console.input("[bold magenta]You[/bold magenta] [dim]›[/dim] ")
        if user_input.strip().lower() == "exit":
            break
        start = time.time()
        with console.status(
            "[yellow]Thinking...[/yellow]", spinner="dots"
        ) as status:
            def perm_check(name, args):
                return check_permission(console, status, name, args)

            response, history, ctx_used = run_agent(
                user_input, tools, execute_tool, history,
                status=status, console=console,
                permission_check=perm_check,
            )
            history, dropped = trim_history(history, ctx_used, NUM_CTX)
            if dropped:
                status.update(
                    status_line(
                        "Summarizing dropped turns...",
                        ctx_used,
                        time.time() - start,
                    )
                )
                status.start()  # agent may have stopped it while streaming
                summary = summarize_dropped(dropped)
                if summary:
                    history.insert(
                        0,
                        {
                            "role": "system",
                            "content": f"Summary of earlier conversation: {summary}",
                        },
                    )

        tag = ctx_tag(ctx_used, NUM_CTX)
        elapsed = time.time() - start
        console.print(f"[dim]{tag} · {elapsed:.1f}s[/dim]\n")
        if dropped:
            console.print(
                f"[dim yellow]↳ trimmed {len(dropped) // 2} old turn(s), summarized into context[/dim yellow]\n"
            )
