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


def status_line(
    verb: str,
    progress: tuple[int, int] | None = None,
) -> str:
    """Compose the spinner status line: `verb (N/M)`.

    Earlier versions added an `↑ N tokens · Xs` tail, but the token count
    was stale (it was last turn's `ctx_used`, not current prompt size) and
    therefore misleading. Keeping only the verb + optional progress gives
    real feedback ("something is happening, and here's what") without
    pretending to report numbers we can't cheaply measure mid-turn."""
    if progress:
        return f"[yellow]{verb}[/yellow] [dim]({progress[0]}/{progress[1]})[/dim]"
    return f"[yellow]{verb}[/yellow]"


def log_response(
    usage: Usage | None,
    messages: list,
    *,
    content: str,
    tool_calls: list,
    ttft_ms: int | None = None,
    reasoning: str = "",
) -> None:
    """Append a single-line JSON record per turn for post-hoc study. Works
    for every provider because it takes normalized `Usage` + `ToolCall`s.

    `reasoning` is the chain-of-thought that thinking-capable models
    (qwen3 via LM Studio, DeepSeek R1, ...) surface separately from the
    visible assistant reply. It's not part of message history, but logging
    it here lets post-hoc analysis see what the model was doing when a
    turn looks mysteriously short in `content`.
    """
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
        "reasoning": reasoning,
        "tool_calls": [
            {"name": tc.name, "arguments": tc.arguments} for tc in tool_calls
        ],
        "messages_in_prompt": len(messages),
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
