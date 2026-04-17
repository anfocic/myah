# agent.py
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ollama
from rich.live import Live
from rich.markdown import Markdown

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


def _build_system_prompt(plan_mode: bool = False) -> str:
    """Base persona + (if the cwd has a CLAUDE.md) project context appended."""
    base = f"""You are Mia, a personal assistant.
You are running on the {MODEL_NAME} model served locally via Ollama.
You were NOT built by OpenAI or Anthropic. If asked who you are or what model
you are, answer truthfully based on the line above. Do not claim any other origin.

Rules:
- Always use tools when the task requires it
- After using a tool, always respond with a short confirmation message
- Never return an empty response
- For tasks needing multiple steps, do them one at a time
- If the user gives a bare filename like 'search.py', call `glob` first to resolve it to a full path, then read/edit that path"""

    parts = [base]

    # Re-read every turn so edits to CLAUDE.md take effect without restarting.
    # File is typically small; re-read cost is negligible vs. an LLM call.
    claude_md = Path("CLAUDE.md")
    if claude_md.is_file():
        try:
            parts.append(
                "Project context (CLAUDE.md — the user's instructions for this repo):\n"
                + claude_md.read_text()
            )
        except OSError:
            pass

    if plan_mode:
        parts.append(
            "PLAN MODE is ON. Describe what you WOULD do, step-by-step, and wait "
            "for the user to confirm. Do NOT call any tools yet — tool calls will "
            "be rejected until the user turns plan mode off with /plan."
        )

    return "\n\n".join(parts)


def run_agent(
    user_input: str,
    tools: list,
    execute_tool,
    history: list = [],
    status=None,
    console=None,
    permission_check=None,
    plan_mode: bool = False,
):
    messages = (
        [{"role": "system", "content": _build_system_prompt(plan_mode)}]
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
        live: Live | None = None

        try:
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
                                    "\n[bold cyan]Mia[/bold cyan] [dim]›[/dim]"
                                )
                                live = Live(
                                    Markdown(""),
                                    console=console,
                                    refresh_per_second=12,
                                    vertical_overflow="visible",
                                )
                                live.start()
                        content_parts.append(msg.content)
                        if live is not None:
                            live.update(Markdown("".join(content_parts)))
                            if STREAM_DELAY_MS:
                                time.sleep(STREAM_DELAY_MS / 1000)

                    if msg.tool_calls:
                        tool_calls.extend(msg.tool_calls)
            except KeyboardInterrupt:
                # Let main.py decide how to render the abort. history is
                # untouched because we only append on successful completion.
                if status:
                    status.stop()
                raise
        finally:
            if live is not None:
                live.stop()

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
            results = _run_tools_parallel(
                tool_calls,
                execute_tool,
                permission_check,
                plan_mode=plan_mode,
                status=status,
                ctx_used=ctx_used,
                start=start,
            )
            for content_str in results:
                messages.append(
                    {"role": "tool", "content": truncate_tool_result(content_str)}
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


def _run_tools_parallel(
    tool_calls: list,
    execute_tool,
    permission_check,
    *,
    plan_mode: bool,
    status,
    ctx_used: int,
    start: float,
) -> list[str]:
    """Serial permission gate (user prompts can't be parallel) → parallel
    execution for approved calls → results returned in original tool_calls
    order so the model sees them aligned with its request.

    In plan_mode, nothing executes; every call returns a canned string.
    Permission prompts are skipped too (no work is happening)."""
    n = len(tool_calls)
    results: list[str | None] = [None] * n

    if plan_mode:
        return [
            "Plan mode is on — tool call not executed. Describe the plan instead."
            for _ in tool_calls
        ]

    approved: list[tuple[int, str, dict]] = []
    for i, tc in enumerate(tool_calls):
        name = tc.function.name
        args = tc.function.arguments
        if permission_check and not permission_check(name, args):
            results[i] = "User denied this tool call."
        else:
            approved.append((i, name, args))

    if approved:
        with ThreadPoolExecutor(max_workers=len(approved)) as ex:
            future_to_idx = {
                ex.submit(execute_tool, name, args): i
                for (i, name, args) in approved
            }
            done = 0
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                try:
                    results[i] = str(fut.result())
                except Exception as e:
                    # Tool errors are data, not exceptions — let the model recover.
                    results[i] = f"Tool raised: {type(e).__name__}: {e}"
                done += 1
                if status:
                    status.update(
                        status_line(
                            "Running tools",
                            ctx_used,
                            time.time() - start,
                            progress=(done, len(approved)),
                        )
                    )

    return [r if r is not None else "" for r in results]
