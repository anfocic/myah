# main.py
import time

from rich.console import Console

from agent import status_line, run_agent, summarize_dropped, trim_history
from config import NUM_CTX
from tools.files import read_file, write_file
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
            "description": "Reads the contents of a file at the given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path to read"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes content to a file at the given path",
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
]


def execute_tool(name, args):
    if name == "get_current_time":
        return get_current_time()
    elif name == "read_file":
        return read_file(args["path"])
    elif name == "write_file":
        return write_file(args["path"], args["content"])
    return "Tool not found"


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
            response, history, ctx_used = run_agent(
                user_input, tools, execute_tool, history, status=status
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
        console.print(f"\n[bold cyan]Mia[/bold cyan] {tag} [dim]›[/dim] {response}\n")
        if dropped:
            console.print(
                f"[dim yellow]↳ trimmed {len(dropped) // 2} old turn(s), summarized into context[/dim yellow]\n"
            )
