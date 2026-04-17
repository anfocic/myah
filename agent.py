# agent.py
import json
import time
from pathlib import Path

import ollama

from config import MODEL_NAME, NUM_CTX, STREAM_DELAY_MS, TOOL_RESULT_MAX_BYTES

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
    response,
    messages: list,
    *,
    content_override: str | None = None,
    tool_calls_override: list | None = None,
    ttft_ms: int | None = None,
) -> None:
    """Append a single-line JSON record per ollama.chat call for post-hoc study.

    Streaming mode builds content + tool_calls from chunks, so pass them in via
    the `*_override` args. Non-streaming reads them from `response.message`.
    """
    LOG_FILE.parent.mkdir(exist_ok=True)
    tool_calls = (
        tool_calls_override
        if tool_calls_override is not None
        else (response.message.tool_calls or [])
    )
    entry = {
        "ts": time.time(),
        "prompt_eval_count": getattr(response, "prompt_eval_count", None),
        "eval_count": getattr(response, "eval_count", None),
        "eval_duration_ms": (getattr(response, "eval_duration", 0) or 0) // 1_000_000,
        "total_duration_ms": (getattr(response, "total_duration", 0) or 0) // 1_000_000,
        "ttft_ms": ttft_ms,
        "content": content_override if content_override is not None else response.message.content,
        "tool_calls": [
            {"name": tc.function.name, "arguments": tc.function.arguments}
            for tc in tool_calls
        ],
        "messages_in_prompt": len(messages),
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~1 token per 4 characters of message content."""
    total = sum(len(m.get("content") or "") for m in messages)
    return total // 4


def truncate_tool_result(result: str, max_bytes: int = TOOL_RESULT_MAX_BYTES) -> str:
    """Cap a tool result's size so one giant read_file can't blow the ctx
    window. Keeps head + tail (the useful bits are usually at the edges)."""
    if len(result) <= max_bytes:
        return result
    dropped = len(result) - max_bytes
    head = max_bytes // 2
    tail = max_bytes - head
    return (
        result[:head]
        + f"\n\n...[truncated {dropped} chars]...\n\n"
        + result[-tail:]
    )


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
    console=None,
    permission_check=None,
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
            - For tasks needing multiple steps, do them one at a time
            - If the user gives a bare filename like 'search.py', call `glob` first to resolve it to a full path, then read/edit that path""",
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
            status.start()  # idempotent; re-arms spinner after a streaming phase

        content_parts: list[str] = []
        tool_calls: list = []
        final_chunk = None
        first_content_seen = False
        ttft_ms: int | None = None

        try:
            for chunk in ollama.chat(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                options={"num_ctx": NUM_CTX},
                stream=True,
            ):
                final_chunk = chunk
                msg = chunk.message

                if msg.content:
                    if not first_content_seen:
                        first_content_seen = True
                        ttft_ms = int((time.time() - start) * 1000)
                        if status:
                            status.stop()
                        if console:
                            console.print(
                                "\n[bold cyan]Mia[/bold cyan] [dim]›[/dim] ",
                                end="",
                                highlight=False,
                            )
                    content_parts.append(msg.content)
                    if console:
                        console.print(msg.content, end="", highlight=False)
                        if STREAM_DELAY_MS:
                            time.sleep(STREAM_DELAY_MS / 1000)

                if msg.tool_calls:
                    tool_calls.extend(msg.tool_calls)
        except KeyboardInterrupt:
            # Let main.py decide how to render the abort; just clean up transient
            # UI state so the REPL isn't left with a spinner running or a
            # half-streamed line. history is untouched because we only append
            # on successful completion below.
            if status:
                status.stop()
            if first_content_seen and console:
                console.print()
            raise

        if first_content_seen and console:
            console.print()  # close out the streamed line with a newline

        content = "".join(content_parts)
        log_response(
            final_chunk,
            messages,
            content_override=content,
            tool_calls_override=tool_calls,
            ttft_ms=ttft_ms,
        )

        ctx_used = (
            getattr(final_chunk, "prompt_eval_count", None) or estimate_tokens(messages)
        )

        if tool_calls:
            messages.append({"role": "assistant", "content": content})
            total = len(tool_calls)
            for i, tool_call in enumerate(tool_calls, start=1):
                name = tool_call.function.name
                args = tool_call.function.arguments
                if permission_check and not permission_check(name, args):
                    messages.append(
                        {"role": "tool", "content": "User denied this tool call."}
                    )
                    continue
                if status:
                    status.update(
                        status_line(
                            f"Running {name}",
                            ctx_used,
                            time.time() - start,
                            progress=(i, total),
                        )
                    )
                result = execute_tool(name, args)
                messages.append(
                    {"role": "tool", "content": truncate_tool_result(str(result))}
                )
        else:
            if not content:
                content = "Done."
                if console:
                    console.print(
                        f"\n[bold cyan]Mia[/bold cyan] [dim]›[/dim] {content}"
                    )
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": content})
            return content, history, ctx_used
