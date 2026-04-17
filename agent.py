# agent.py
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from rich.live import Live
from rich.markdown import Markdown

from config import (
    MODEL_NAME,
    MODEL_PROVIDER,
    NUM_CTX,
    STREAM_DELAY_MS,
    TOOL_RESULT_MAX_BYTES,
)
from providers import ProviderError, Usage, get_provider

_provider = get_provider()

# Tools the plan-mode gate lets through unchanged. Everything else gets
# short-circuited so the model can investigate (glob/grep/read) while
# planning, but can't mutate state until plan mode is turned off.
READ_ONLY_TOOLS = frozenset(
    {"glob", "grep", "read_file", "get_current_time", "harness_info"}
)

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
    entry = {
        "ts": time.time(),
        "provider": _provider.name,
        "model": _provider.model,
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
    """Compress dropped turns into a terse note via the active provider."""
    if not dropped:
        return ""
    transcript = "\n".join(
        f"{m['role']}: {m.get('content', '')}" for m in dropped
    )
    try:
        content, _ = _provider.chat(
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the following conversation turns in 2-3 terse sentences. Capture user intent, key facts, and any tool results. No filler.",
                },
                {"role": "user", "content": transcript},
            ],
            num_ctx=NUM_CTX,
        )
    except ProviderError:
        return ""
    return content


def _git(*args: str) -> str | None:
    """Run a git command, return stripped stdout, or None on any failure
    (not a repo, git missing, detached state, etc.)."""
    try:
        out = subprocess.check_output(
            ["git", *args],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _env_block() -> str:
    """Compact environment snapshot prepended to every system prompt so the
    model has cwd / platform / git state on turn 1 without burning a tool
    call. Kept small — ~80-120 tokens depending on git state."""
    lines = [
        f"cwd: {os.getcwd()}",
        f"platform: {platform.system().lower()} ({platform.machine()})",
        f"date: {date.today().isoformat()}",
    ]
    branch = _git("branch", "--show-current")
    if branch:
        main_ref = _git("rev-parse", "--abbrev-ref", "origin/HEAD")
        main = main_ref.split("/", 1)[-1] if main_ref else "main"
        porcelain = _git("status", "--porcelain")
        dirty = len(porcelain.splitlines()) if porcelain else 0
        lines.append(f"git: branch={branch} main={main} dirty={dirty}")
    else:
        lines.append("git: (not a repository)")
    return "<env>\n" + "\n".join(lines) + "\n</env>"


_SERVED_VIA = {
    "ollama": "served locally via Ollama",
    "openai-compat": "served via an OpenAI-compatible HTTP API",
}


def _build_system_prompt(plan_mode: bool = False) -> str:
    """Base persona + env block + (if the cwd has a CLAUDE.md) project
    context + (if plan mode) planning rules."""
    served = _SERVED_VIA.get(MODEL_PROVIDER, f"served via {MODEL_PROVIDER}")
    base = f"""You are Mia, a personal assistant.
You are running on the {MODEL_NAME} model {served}.
Answer truthfully about what model and provider you are based on the line above.

Rules:
- Always use tools when the task requires it
- After using a tool, always respond with a short confirmation message
- Never return an empty response
- For tasks needing multiple steps, do them one at a time
- If the user gives a bare filename like 'search.py', call `glob` first to resolve it to a full path, then read/edit that path"""

    parts = [base, _env_block()]

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
            "PLAN MODE is ON.\n\n"
            "BEFORE proposing anything, you MUST investigate the codebase. For any "
            "plan that touches existing code, call `glob` and/or `grep` to find "
            "what exists, then `read_file` on the relevant files. Your plan must "
            "reference specific files and line numbers you have actually read — "
            "generic advice (\"add logging\", \"improve errors\", \"use rich\") "
            "is not acceptable. If the user's request mentions an existing feature "
            "or file, read the current implementation first.\n\n"
            "After investigating, describe the proposed changes step-by-step and "
            "wait for the user to confirm. Mutating tools (write_file, edit_file, "
            "bash) are rejected automatically until /plan is toggled off. Read-only "
            f"tools ({', '.join(sorted(READ_ONLY_TOOLS))}) still work."
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
    on_tool_start=None,
    on_tool_end=None,
    debug: bool = False,
):
    messages = (
        [{"role": "system", "content": _build_system_prompt(plan_mode)}]
        + history
        + [{"role": "user", "content": user_input}]
    )

    start = time.time()
    ctx_used = estimate_tokens(messages)

    while True:
        if debug and console:
            _debug_dump_messages(console, messages)
        if status:
            status.update(status_line("Thinking...", ctx_used, time.time() - start))
            status.start()  # idempotent; re-arms spinner after a streaming phase

        content_parts: list[str] = []
        tool_calls: list = []
        final_usage: Usage | None = None
        first_content_seen = False
        ttft_ms: int | None = None
        live: Live | None = None

        try:
            try:
                for chunk in _provider.stream_chat(messages, tools, NUM_CTX):
                    if chunk.content_delta:
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
                        content_parts.append(chunk.content_delta)
                        if live is not None:
                            live.update(Markdown("".join(content_parts)))
                            if STREAM_DELAY_MS:
                                time.sleep(STREAM_DELAY_MS / 1000)

                    if chunk.tool_calls:
                        tool_calls.extend(chunk.tool_calls)

                    if chunk.done:
                        final_usage = chunk.usage
            except KeyboardInterrupt:
                # Let main.py decide how to render the abort. history is
                # untouched because we only append on successful completion.
                if status:
                    status.stop()
                raise
            except ProviderError as e:
                if status:
                    status.stop()
                if console:
                    console.print(
                        f"\n[red]Provider error ({_provider.name}):[/red] {e}"
                    )
                # Same discipline as Ctrl+C: history untouched, return to REPL.
                return "", history, ctx_used, {
                    "ttft_ms": None, "completion_tokens": None, "tok_per_s": None,
                }
        finally:
            if live is not None:
                live.stop()

        content = "".join(content_parts)
        log_response(
            final_usage,
            messages,
            content=content,
            tool_calls=tool_calls,
            ttft_ms=ttft_ms,
        )

        ctx_used = (
            (final_usage.prompt_tokens if final_usage else None)
            or estimate_tokens(messages)
        )

        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": content,
                # Carry the tool calls on the assistant message. Ollama ignores
                # unknown fields; the openai-compat adapter uses them to build
                # the OpenAI-shaped tool_calls array with synthesized IDs.
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in tool_calls
                ],
            })
            results = _run_tools_parallel(
                tool_calls,
                execute_tool,
                permission_check,
                plan_mode=plan_mode,
                status=status,
                ctx_used=ctx_used,
                start=start,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
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
            # Streaming rate: completion tokens divided by seconds spent
            # generating (elapsed minus ttft). Only meaningful when both
            # are known; otherwise None so main.py can skip rendering.
            stream_s = None
            if ttft_ms is not None:
                stream_s = (time.time() - start) - (ttft_ms / 1000)
            completion_tokens = (
                final_usage.completion_tokens if final_usage else None
            )
            tok_per_s = None
            if completion_tokens and stream_s and stream_s > 0:
                tok_per_s = completion_tokens / stream_s
            stats = {
                "ttft_ms": ttft_ms,
                "completion_tokens": completion_tokens,
                "tok_per_s": tok_per_s,
            }
            return content, history, ctx_used, stats


def _debug_dump_messages(console, messages: list) -> None:
    """Print the full messages array being sent to the provider. Cheap to
    render — rich will wrap long content. Toggled by /debug in main.py."""
    console.print(
        f"[dim]── debug: {len(messages)} messages → provider ──[/dim]"
    )
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        content = m.get("content", "") or ""
        preview = content if len(content) <= 400 else content[:397] + "..."
        console.print(f"[dim]#{i} [{role}][/dim] {preview}")
        if m.get("tool_calls"):
            console.print(f"[dim]   tool_calls: {m['tool_calls']}[/dim]")
    console.print("[dim]── end debug ──[/dim]")


def _run_tools_parallel(
    tool_calls: list,
    execute_tool,
    permission_check,
    *,
    plan_mode: bool,
    status,
    ctx_used: int,
    start: float,
    on_tool_start=None,
    on_tool_end=None,
) -> list[str]:
    """Serial permission gate (user prompts can't be parallel) → parallel
    execution for approved calls → results returned in original tool_calls
    order so the model sees them aligned with its request.

    In plan_mode, read-only tools (grep/glob/read_file/etc.) still run so
    the model can ground its plan in the actual code. Mutating tools are
    short-circuited with a canned string; no permission prompt fires for
    them since no work would happen.

    on_tool_start/on_tool_end are optional display callbacks. start fires
    in request order; end fires in completion order (parallel). end receives
    args so the display layer can correlate (e.g. render a diff from the
    same old_string/new_string it saw in start). ok=False means denied,
    plan-blocked, or raised."""

    def _notify_start(name, args):
        if on_tool_start:
            try:
                on_tool_start(name, args)
            except Exception:
                pass  # display callback must never break the loop

    def _notify_end(name, args, result, ok):
        if on_tool_end:
            try:
                on_tool_end(name, args, result, ok)
            except Exception:
                pass

    n = len(tool_calls)
    results: list[str | None] = [None] * n

    approved: list[tuple[int, str, dict]] = []
    for i, tc in enumerate(tool_calls):
        name = tc.name
        args = tc.arguments
        _notify_start(name, args)
        if plan_mode and name not in READ_ONLY_TOOLS:
            msg = (
                f"Plan mode: {name} is a mutating tool and was not executed. "
                "Use read-only tools (glob, grep, read_file, harness_info) to "
                "investigate, then describe the proposed changes."
            )
            results[i] = msg
            _notify_end(name, args, msg, False)
            continue
        if permission_check and not permission_check(name, args):
            msg = "User denied this tool call."
            results[i] = msg
            _notify_end(name, args, msg, False)
        else:
            approved.append((i, name, args))

    if approved:
        with ThreadPoolExecutor(max_workers=len(approved)) as ex:
            future_to_idx = {
                ex.submit(execute_tool, name, args): (i, name, args)
                for (i, name, args) in approved
            }
            done = 0
            for fut in as_completed(future_to_idx):
                i, name, args = future_to_idx[fut]
                try:
                    out = str(fut.result())
                    results[i] = out
                    _notify_end(name, args, out, True)
                except Exception as e:
                    # Tool errors are data, not exceptions — let the model recover.
                    msg = f"Tool raised: {type(e).__name__}: {e}"
                    results[i] = msg
                    _notify_end(name, args, msg, False)
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
