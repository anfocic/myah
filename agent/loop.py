"""The agentic loop itself — `run_agent` plus its two helpers
(`_run_tools_parallel`, `_debug_dump_messages`).

Shape per CONCEPTS §1:

  build messages → provider stream_chat →
     if tool_calls: execute → append results → loop back
     else: return final content

Everything the loop needs about context/tokens/status/prompt is imported
from siblings in this package. The loop doesn't know about rich styling,
the REPL's State dict, or where messages came from — those all belong
upstairs in main.py / repl/."""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent import READ_ONLY_TOOLS
from agent.context import MICROCOMPACT_CTX_THRESHOLD, microcompact
from agent.status import log_response, status_line
from agent.system_prompt import build_system_prompt
from agent.tokens import estimate_tokens, truncate_tool_result
from config import NUM_CTX, STREAM_DELAY_MS
from providers import ProviderError, Usage, get_active_provider
from security import annotate_if_injected


def run_agent(
    user_input: str,
    tools: list,
    execute_tool,
    history: list | None = None,
    console=None,
    permission_check=None,
    plan_mode: bool = False,
    on_tool_start=None,
    on_tool_end=None,
    debug: bool = False,
    subagent: bool = False,
):
    # Sentinel + per-call init avoids the mutable-default-arg footgun: a
    # literal [] default is shared across every call that omits history, so
    # one call's turn leaks into the next caller's turn-1.
    if history is None:
        history = []
    messages = (
        [{"role": "system", "content": build_system_prompt(plan_mode, subagent=subagent)}]
        + history
        + [{"role": "user", "content": user_input}]
    )

    start = time.time()
    ctx_used = estimate_tokens(messages)

    while True:
        # Intra-turn context relief. Fires when a tool-heavy turn has racked
        # up enough results to push ctx past the threshold. Does nothing on
        # iteration 1 (no tool messages yet) or when the count is small.
        if ctx_used > MICROCOMPACT_CTX_THRESHOLD * NUM_CTX:
            n_elided = microcompact(messages)
            if n_elided:
                if console:
                    console.print(
                        f"[dim yellow]↳ microcompact: elided "
                        f"{n_elided} old tool result(s)[/dim yellow]"
                    )
                ctx_used = estimate_tokens(messages)
        if debug and console:
            _debug_dump_messages(console, messages)
        if console:
            # Under patch_stdout, an animated region fights the pinned
            # prompt; an append-only log line avoids the cursor war.
            console.print(status_line("Thinking..."))

        content_parts: list[str] = []
        tool_calls: list = []
        final_usage: Usage | None = None
        first_content_seen = False
        ttft_ms: int | None = None

        provider = get_active_provider()
        try:
            for chunk in provider.stream_chat(messages, tools, NUM_CTX):
                if chunk.content_delta:
                    if not first_content_seen:
                        first_content_seen = True
                        ttft_ms = int((time.time() - start) * 1000)
                        if console:
                            # Blank separator before the response begins.
                            console.print()
                    content_parts.append(chunk.content_delta)
                    # Stream tokens as plain text so they scroll above the
                    # pinned prompt under patch_stdout. Markdown formatting
                    # is applied once at end of stream (see below) because
                    # in-place re-rendering via rich.live.Live is incompatible
                    # with prompt_toolkit's patched stdout — they both drive
                    # the terminal's cursor and fight over it.
                    if console:
                        console.out(chunk.content_delta, end="", highlight=False)
                        if STREAM_DELAY_MS:
                            time.sleep(STREAM_DELAY_MS / 1000)

                if chunk.tool_calls:
                    tool_calls.extend(chunk.tool_calls)

                if chunk.done:
                    final_usage = chunk.usage
        except KeyboardInterrupt:
            # Let main.py decide how to render the abort. history is
            # untouched because we only append on successful completion.
            raise
        except ProviderError as e:
            if console:
                console.print(
                    f"\n[red]Provider error ({provider.name}):[/red] {e}"
                )
            # Same discipline as Ctrl+C: history untouched, return to REPL.
            return "", history, ctx_used, {
                "ttft_ms": None, "completion_tokens": None, "tok_per_s": None,
            }
        finally:
            if first_content_seen and console:
                # Newline terminates the plain-text streaming chunk line so
                # the post-turn status line doesn't land at the end of the
                # last token.
                console.print()

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
                console=console,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
            )
            for content_str in results:
                # Two-pass processing on every tool result:
                # 1. annotate_if_injected scans for prompt-injection markers
                #    and prepends a warning if any are found. The original
                #    content is preserved — the model needs to see the
                #    actual bytes to answer the user's question; the
                #    warning just contextualizes it as DATA not instructions.
                # 2. truncate_tool_result caps the size so a 5MB read_file
                #    can't blow the context window (§20).
                safe = annotate_if_injected(content_str)
                messages.append(
                    {"role": "tool", "content": truncate_tool_result(safe)}
                )
        else:
            if not content:
                content = "Done."
                if console:
                    console.print(f"\n{content}")
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
    console=None,
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

    # Fail-closed default: if no caller-supplied check, only read-only tools
    # run. Protects subagents/tests that plug into run_agent without wiring
    # up the REPL's permission prompt — otherwise bash/edit/write would
    # execute silently with no human in the loop.
    if permission_check is None:
        permission_check = lambda name, args: name in READ_ONLY_TOOLS

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
        if not permission_check(name, args):
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
                if console:
                    console.print(
                        status_line("Running tools", progress=(done, len(approved)))
                    )

    return [r if r is not None else "" for r in results]
