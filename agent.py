# agent.py
import json
import time
from pathlib import Path

import ollama

from config import MODEL_NAME, NUM_CTX

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


def log_response(response, messages: list) -> None:
    """Append a single-line JSON record per ollama.chat call for post-hoc study."""
    LOG_FILE.parent.mkdir(exist_ok=True)
    entry = {
        "ts": time.time(),
        "prompt_eval_count": getattr(response, "prompt_eval_count", None),
        "eval_count": getattr(response, "eval_count", None),
        "eval_duration_ms": (getattr(response, "eval_duration", 0) or 0) // 1_000_000,
        "total_duration_ms": (getattr(response, "total_duration", 0) or 0) // 1_000_000,
        "content": response.message.content,
        "tool_calls": [
            {"name": tc.function.name, "arguments": tc.function.arguments}
            for tc in (response.message.tool_calls or [])
        ],
        "messages_in_prompt": len(messages),
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~1 token per 4 characters of message content."""
    total = sum(len(m.get("content") or "") for m in messages)
    return total // 4


def trim_history(
    history: list, ctx_used: int, num_ctx: int,
    high: float = 0.8, target: float = 0.5,
) -> tuple[list, list]:
    """If ctx is over `high`, drop oldest user/assistant pairs until history
    fits under `target`. Returns (new_history, dropped_messages)."""
    if ctx_used <= high * num_ctx:
        return history, []

    target_tokens = int(target * num_ctx)
    dropped: list = []
    while len(history) >= 2 and estimate_tokens(history) > target_tokens:
        dropped.extend(history[:2])
        history = history[2:]
    return history, dropped


def summarize_dropped(dropped: list) -> str:
    """Compress dropped turns into a terse note via the same model."""
    if not dropped:
        return ""
    transcript = "\n".join(
        f"{m['role']}: {m.get('content', '')}" for m in dropped
    )
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Summarize the following conversation turns in 2-3 terse sentences. Capture user intent, key facts, and any tool results. No filler.",
            },
            {"role": "user", "content": transcript},
        ],
        options={"num_ctx": NUM_CTX},
    )
    return (response.message.content or "").strip()


def run_agent(
    user_input: str,
    tools: list,
    execute_tool,
    history: list = [],
    status=None,
):
    messages = (
        [
            {
                "role": "system",
                "content": f"""You are Mia, a personal assistant.
            You are running on the {MODEL_NAME} model served locally via Ollama.
            You were NOT built by OpenAI or Anthropic. If asked who you are or what model
            you are, answer truthfully based on the line above. Do not claim any other origin.

            Rules:
            - Always use tools when the task requires it
            - After using a tool, always respond with a short confirmation message
            - Never return an empty response
            - For tasks needing multiple steps, do them one at a time""",
            }
        ]
        + history
        + [{"role": "user", "content": user_input}]
    )

    start = time.time()
    ctx_used = estimate_tokens(messages)

    while True:
        if status:
            status.update(status_line("Thinking...", ctx_used, time.time() - start))

        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            options={"num_ctx": NUM_CTX},
        )
        log_response(response, messages)

        # Ollama returns the real prompt token count; fall back to estimate
        ctx_used = getattr(response, "prompt_eval_count", None) or estimate_tokens(messages)

        message = response.message

        if message.tool_calls:
            messages.append({"role": "assistant", "content": message.content or ""})
            total = len(message.tool_calls)
            for i, tool_call in enumerate(message.tool_calls, start=1):
                if status:
                    status.update(
                        status_line(
                            f"Running {tool_call.function.name}",
                            ctx_used,
                            time.time() - start,
                            progress=(i, total),
                        )
                    )
                result = execute_tool(
                    tool_call.function.name, tool_call.function.arguments
                )
                messages.append({"role": "tool", "content": str(result)})
        else:
            if not message.content:
                message.content = "Done."
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": message.content})
            return message.content, history, ctx_used
