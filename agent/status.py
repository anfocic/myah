"""Spinner status-line composition + per-turn JSONL logging.

These are the two forms of "what just happened" emitted every turn: a
single-line status string for the live rich spinner, and a single JSON
line appended to logs/agent.jsonl for post-hoc study (token counts,
latency, the actual tool calls the model emitted)."""
import json
import time
from pathlib import Path

from providers import Usage, get_active_provider

LOG_FILE = Path("logs/agent.jsonl")


def _fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def status_line(
    verb: str,
    tokens: int,
    elapsed: float,
    progress: tuple[int, int] | None = None,
) -> str:
    """Compose the spinner status line: verb · progress · tokens · elapsed."""
    parts = [f"[yellow]{verb}[/yellow]"]
    if progress:
        parts.append(f"[dim]({progress[0]}/{progress[1]})[/dim]")
    parts.append(f"[dim]· ↑ {_fmt_tokens(tokens)} tokens[/dim]")
    parts.append(f"[dim]· {elapsed:.0f}s[/dim]")
    return " ".join(parts)


def log_response(
    usage: Usage | None,
    messages: list,
    *,
    content: str,
    tool_calls: list,
    ttft_ms: int | None = None,
) -> None:
    """Append a single-line JSON record per turn for post-hoc study. Works
    for every provider because it takes normalized `Usage` + `ToolCall`s."""
    LOG_FILE.parent.mkdir(exist_ok=True)
    provider = get_active_provider()
    entry = {
        "ts": time.time(),
        "provider": provider.name,
        "model": provider.model,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "ttft_ms": ttft_ms,
        "content": content,
        "tool_calls": [
            {"name": tc.name, "arguments": tc.arguments} for tc in tool_calls
        ],
        "messages_in_prompt": len(messages),
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
