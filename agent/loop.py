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

import inspect
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from agent import READ_ONLY_TOOLS
from agent.context import MICROCOMPACT_CTX_THRESHOLD, microcompact
from agent.status import log_response
from agent.system_prompt import build_system_prompt
from agent.tokens import count_tokens, truncate_tool_result
from config import MAX_AGENT_ITERATIONS, NUM_CTX, SPIN_WINDOW, get_context_size
from display import StreamingMarkdown
from providers import Provider, ProviderError, Usage, get_active_provider
from providers.base import ToolCall
from security import annotate_if_injected


@dataclass
class TurnResult:
    """What one provider turn produced once its stream is fully consumed.

    Separated from `run_agent`'s orchestration so the streaming mechanics
    (four chunk channels, UI rendering, exit paths) can be unit-tested
    without message assembly or tool dispatch. See
    `vault/wiki/code/agent-loop.md` §Phase 3 for the mental model."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None
    ttft_ms: int | None = None
    # Set when `stream_chat` raised `ProviderError`. KeyboardInterrupt is
    # NOT caught here — it re-raises so the REPL's outer handler sees it.
    provider_error: ProviderError | None = None


def _stream_provider_turn(
    provider: Provider,
    messages: list[dict],
    tools: list,
    num_ctx: int,
    *,
    console,
    start_time: float,
) -> TurnResult:
    """Consume one `provider.stream_chat` iterator to completion.

    Owns: spinner lifecycle, markdown renderer init/finish, four chunk
    channels (content / reasoning / tool_calls / done), dim-italic
    reasoning block open/close, cleanup on KeyboardInterrupt (re-raises)
    and ProviderError (returns with `provider_error` set).

    Does NOT decide whether to loop, append to messages/history, invoke
    tools, or compute stats — that stays the caller's job so `run_agent`
    still reads top to bottom as a sequence of phases."""
    thinking = console.status("[yellow]Thinking...[/yellow]", spinner="dots") if console else None
    if thinking:
        thinking.start()

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_rendering = False
    tool_calls: list[ToolCall] = []
    final_usage: Usage | None = None
    first_content_seen = False
    ttft_ms: int | None = None
    renderer = StreamingMarkdown(console) if console else None

    try:
        for chunk in provider.stream_chat(messages, tools, num_ctx):
            if chunk.reasoning_delta:
                # Swap the spinner for an in-line dim stream the first
                # time reasoning shows up. Every subsequent chunk just
                # prints incrementally so the user sees the model
                # thinking in real time without the markdown renderer
                # (which is reserved for the final assistant reply).
                if thinking:
                    thinking.stop()
                    thinking = None
                if not reasoning_rendering and console:
                    console.print("↳ thinking: ", end="", style="dim italic")
                    reasoning_rendering = True
                reasoning_parts.append(chunk.reasoning_delta)
                if console:
                    # markup=False so a `[bracket]` in the reasoning
                    # trace isn't parsed as a Rich tag; style carries
                    # the formatting so the delta itself stays literal.
                    console.print(
                        chunk.reasoning_delta,
                        end="",
                        style="dim italic",
                        soft_wrap=True,
                        markup=False,
                    )

            if chunk.content_delta:
                if thinking:
                    thinking.stop()
                    thinking = None
                if reasoning_rendering and console:
                    # Reasoning is done once real content starts. Close
                    # the dim block with a newline so the markdown
                    # renderer below starts cleanly.
                    console.print()
                    reasoning_rendering = False
                if not first_content_seen:
                    first_content_seen = True
                    ttft_ms = int((time.time() - start_time) * 1000)
                content_parts.append(chunk.content_delta)
                if renderer:
                    renderer.update("".join(content_parts))

            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)

            if chunk.done:
                final_usage = chunk.usage

        # Reasoning-only turn (no content, no tool_calls): close the dim
        # block so the prompt doesn't land on the same line.
        if reasoning_rendering and console:
            console.print()
            reasoning_rendering = False
    except KeyboardInterrupt:
        if thinking:
            thinking.stop()
        if reasoning_rendering and console:
            console.print()
        if renderer:
            renderer.finish("".join(content_parts))
        elif first_content_seen and console:
            console.print()
        raise
    except ProviderError as e:
        if thinking:
            thinking.stop()
        if reasoning_rendering and console:
            console.print()
        if renderer:
            renderer.finish("".join(content_parts))
        elif first_content_seen and console:
            console.print()
        return TurnResult(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=tool_calls,
            usage=final_usage,
            ttft_ms=ttft_ms,
            provider_error=e,
        )

    # Normal completion — tool-only turns never saw a content delta to
    # trigger spinner stop, so clear it now before tool output starts.
    if thinking:
        thinking.stop()

    content = "".join(content_parts)
    if renderer:
        renderer.finish(content)

    return TurnResult(
        content=content,
        reasoning="".join(reasoning_parts),
        tool_calls=tool_calls,
        usage=final_usage,
        ttft_ms=ttft_ms,
    )


def _halt_run(
    *,
    reason: str,
    detail: str,
    user_input: str,
    history: list,
    ctx_used: int,
    console,
):
    """Build the shared (content, history, ctx_used, stats) tuple returned
    when a loop guard fires. Mirrors the non-error final-turn return shape
    so eval runners, subagents, and the REPL don't need branch-specific
    handling — they just read `stats["halt_reason"]` when they care."""
    content = f"[halted: {reason}] {detail}"
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": content})
    if console:
        console.print(f"\n[yellow]↳ loop halted ({reason}): {detail}[/yellow]")
    stats = {
        "ttft_ms": None,
        "completion_tokens": None,
        "tok_per_s": None,
        "halt_reason": reason,
    }
    return content, history, ctx_used, stats


def _call_with_optional_meta(func, *args, meta: dict | None = None):
    """Call a callback that may or may not accept tool metadata.

    Older tests and helper closures still use `(name, args)` callables, while
    the TUI callbacks now take an extra `meta` payload with tool id/timing.
    This adapter keeps both shapes working without forcing unrelated code to
    change."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args)

    params = sig.parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return func(*args, meta=meta)
    if "meta" in sig.parameters:
        return func(*args, meta=meta)

    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= len(args) + 1:
        return func(*args, meta)
    return func(*args)


def _timed_tool_call(execute_tool, name: str, args: dict) -> tuple[bool, object, float]:
    """Run one tool and always return `(ok, payload, duration_s)`."""
    started_at = time.monotonic()
    try:
        return True, execute_tool(name, args), time.monotonic() - started_at
    except Exception as e:
        return False, e, time.monotonic() - started_at


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
    cwd: str | None = None,
):
    # Sentinel + per-call init avoids the mutable-default-arg footgun: a
    # literal [] default is shared across every call that omits history, so
    # one call's turn leaks into the next caller's turn-1.
    if history is None:
        history = []
    messages = (
        [{"role": "system", "content": build_system_prompt(plan_mode, subagent=subagent, cwd=cwd)}]
        + history
        + [{"role": "user", "content": user_input}]
    )

    start = time.time()
    # Eagerly fetch the provider so token counting uses the live model name
    # (encoding cache is keyed by model; /model swaps must not drift).
    provider = get_active_provider()
    ctx_used = count_tokens(messages, tools=tools, model_name=provider.model)
    # Accumulated across every iteration of the while-loop so callers (evals,
    # /debug, subagent reporters) can see the reasoning from every turn in
    # a multi-tool trajectory, not only the final answer's reasoning.
    reasoning_total: list[str] = []

    # Loop guards (CONCEPTS §loop-control). Both are per-call — a fresh
    # user turn resets the counters so a long session doesn't gradually
    # starve the cap or accumulate phantom spin history.
    iter_count = 0
    recent_calls: deque[tuple[str, str]] = deque(maxlen=SPIN_WINDOW)

    while True:
        iter_count += 1
        if iter_count > MAX_AGENT_ITERATIONS:
            return _halt_run(
                reason="iter_cap",
                detail=f"hit {MAX_AGENT_ITERATIONS} iterations without a final answer",
                user_input=user_input,
                history=history,
                ctx_used=ctx_used,
                console=console,
            )

        # Intra-turn context relief. Fires when a tool-heavy turn has racked
        # up enough results to push ctx past the threshold. Does nothing on
        # iteration 1 (no tool messages yet) or when the count is small.
        ctx_size = get_context_size()
        if ctx_used > MICROCOMPACT_CTX_THRESHOLD * ctx_size:
            ctx_before = ctx_used
            n_elided = microcompact(messages)
            if n_elided:
                ctx_used = count_tokens(messages, tools=tools, model_name=provider.model)
                if console:
                    threshold = int(MICROCOMPACT_CTX_THRESHOLD * ctx_size)
                    console.print(
                        f"[dim yellow]↳ microcompact fired: ctx {ctx_before} → "
                        f"{ctx_used} (threshold {threshold} = "
                        f"{int(MICROCOMPACT_CTX_THRESHOLD * 100)}% of "
                        f"ctx_size={ctx_size}); elided {n_elided} old tool "
                        f"result(s)[/dim yellow]"
                    )
        if debug and console:
            _debug_dump_messages(console, messages)

        turn = _stream_provider_turn(
            provider,
            messages,
            tools,
            NUM_CTX,
            console=console,
            start_time=start,
        )

        if turn.reasoning:
            reasoning_total.append(turn.reasoning)

        if turn.provider_error is not None:
            if console:
                console.print(
                    f"\n[red]Provider error ({provider.name}):[/red] {turn.provider_error}"
                )
            return (
                "",
                history,
                ctx_used,
                {
                    "ttft_ms": None,
                    "completion_tokens": None,
                    "tok_per_s": None,
                    "provider_error": f"{provider.name}: {turn.provider_error}",
                    "reasoning": "\n\n".join(reasoning_total),
                },
            )

        log_response(
            turn.usage,
            messages,
            content=turn.content,
            tool_calls=turn.tool_calls,
            ttft_ms=turn.ttft_ms,
            reasoning=turn.reasoning,
        )

        ctx_used = (turn.usage.prompt_tokens if turn.usage else None) or count_tokens(
            messages, tools=tools, model_name=provider.model
        )

        if turn.tool_calls:
            # Spinning guard: if the model keeps asking for the same
            # (name, args) tuple across consecutive rounds, halt before
            # executing again. Deque is size-capped, so we only ever look
            # at the most recent SPIN_WINDOW tool calls — a legitimate
            # re-read of the same file later in a trajectory doesn't
            # trigger. Canonicalize args via json with sort_keys=True so
            # semantically equal dicts compare equal.
            for tc in turn.tool_calls:
                args_canonical = json.dumps(tc.arguments, sort_keys=True, default=str)
                recent_calls.append((tc.name, args_canonical))
            if len(recent_calls) == SPIN_WINDOW and len(set(recent_calls)) == 1:
                name, args_canonical = recent_calls[0]
                return _halt_run(
                    reason="spinning",
                    detail=(f"{name}({args_canonical}) repeated {SPIN_WINDOW} consecutive times"),
                    user_input=user_input,
                    history=history,
                    ctx_used=ctx_used,
                    console=console,
                )

            assistant_msg: dict = {
                "role": "assistant",
                "content": turn.content,
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments} for tc in turn.tool_calls
                ],
                "reasoning_content": turn.reasoning,
            }
            messages.append(assistant_msg)
            results = _run_tools_parallel(
                turn.tool_calls,
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
                messages.append({"role": "tool", "content": truncate_tool_result(safe)})
        else:
            content = turn.content
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
            if turn.ttft_ms is not None:
                stream_s = (time.time() - start) - (turn.ttft_ms / 1000)
            completion_tokens = turn.usage.completion_tokens if turn.usage else None
            tok_per_s = None
            if completion_tokens and stream_s and stream_s > 0:
                tok_per_s = completion_tokens / stream_s
            stats = {
                "ttft_ms": turn.ttft_ms,
                "completion_tokens": completion_tokens,
                "tok_per_s": tok_per_s,
                # Full chain-of-thought across every turn in this
                # trajectory, joined by blank lines. Empty string for
                # non-reasoning models. Read by eval runner + /debug.
                "reasoning": "\n\n".join(reasoning_total),
            }
            return content, history, ctx_used, stats


def _debug_dump_messages(console, messages: list) -> None:
    """Print the full messages array being sent to the provider. Cheap to
    render — rich will wrap long content. Toggled by /debug in main.py."""
    console.print(f"[dim]── debug: {len(messages)} messages → provider ──[/dim]")
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
    in request order; end fires in completion order (parallel). Both receive
    a small `meta` payload with a stable tool id (`T01`, `T02`, ...) and, on
    end, the execution duration in seconds. ok=False means denied,
    plan-blocked, or raised."""

    def _notify_start(name, args, meta):
        if on_tool_start:
            try:
                _call_with_optional_meta(on_tool_start, name, args, meta=meta)
            except Exception:
                pass  # display callback must never break the loop

    def _notify_end(name, args, result, ok, meta):
        if on_tool_end:
            try:
                _call_with_optional_meta(on_tool_end, name, args, result, ok, meta=meta)
            except Exception:
                pass

    n = len(tool_calls)
    results: list[str | None] = [None] * n

    # Fail-closed default: if no caller-supplied check, only read-only tools
    # run. Protects subagents/tests that plug into run_agent without wiring
    # up the REPL's permission prompt — otherwise bash/edit/write would
    # execute silently with no human in the loop.
    if permission_check is None:
        permission_check = lambda name, args, meta=None: name in READ_ONLY_TOOLS

    approved: list[tuple[int, str, dict, dict]] = []
    total = len(tool_calls)
    for i, tc in enumerate(tool_calls):
        name = tc.name
        args = tc.arguments
        meta = {
            "tool_id": f"T{i + 1:02d}",
            "index": i + 1,
            "total": total,
        }
        _notify_start(name, args, meta)
        if plan_mode and name not in READ_ONLY_TOOLS:
            readonly = ", ".join(sorted(READ_ONLY_TOOLS))
            msg = (
                f"Plan mode: {name} is a mutating tool and was not executed. "
                f"Use read-only tools ({readonly}) to "
                "investigate, then describe the proposed changes."
            )
            results[i] = msg
            _notify_end(name, args, msg, False, meta)
            continue
        if not _call_with_optional_meta(permission_check, name, args, meta=meta):
            msg = "User denied this tool call."
            results[i] = msg
            _notify_end(name, args, msg, False, meta)
        else:
            approved.append((i, name, args, meta))

    if approved:
        # Once a batch contains any mutating tool, run the whole batch
        # serially in request order. This avoids stale reads and write races
        # across same-turn edits against the same workspace state.
        run_serially = any(name not in READ_ONLY_TOOLS for _, name, _, _ in approved)

        # No separate "Running tools (n/m)" status line in either branch —
        # on_tool_start / on_tool_end already render each tool's own block
        # (`● name args` / `  └ summary`), which carries the same progress
        # signal plus the result content.
        if run_serially:
            for i, name, args, meta in approved:
                ok, payload, duration_s = _timed_tool_call(execute_tool, name, args)
                end_meta = meta | {"duration_s": duration_s}
                if ok:
                    out = str(payload)
                    results[i] = out
                    _notify_end(name, args, out, True, end_meta)
                else:
                    e = payload
                    msg = f"Tool raised: {type(e).__name__}: {e}"
                    results[i] = msg
                    _notify_end(name, args, msg, False, end_meta)
        else:
            # NOTE: cancel_futures=True only cancels futures that haven't
            # started yet. A bash tool already mid-`subprocess.run` keeps
            # running on its worker thread until it returns — Python doesn't
            # interrupt threads. The Ctrl+C path makes the REPL responsive
            # again; in-flight workers just become orphans until they finish.
            ex = ThreadPoolExecutor(max_workers=len(approved))
            try:
                future_to_idx = {
                    ex.submit(_timed_tool_call, execute_tool, name, args): (i, name, args, meta)
                    for (i, name, args, meta) in approved
                }
                for fut in as_completed(future_to_idx):
                    i, name, args, meta = future_to_idx[fut]
                    ok, payload, duration_s = fut.result()
                    end_meta = meta | {"duration_s": duration_s}
                    if ok:
                        out = str(payload)
                        results[i] = out
                        _notify_end(name, args, out, True, end_meta)
                    else:
                        e = payload
                        # Tool errors are data, not exceptions — let the model recover.
                        msg = f"Tool raised: {type(e).__name__}: {e}"
                        results[i] = msg
                        _notify_end(name, args, msg, False, end_meta)
            except KeyboardInterrupt:
                ex.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                # Idempotent: if KeyboardInterrupt already shut us down,
                # this is a no-op. Otherwise it cleans up cleanly even when
                # an unrelated exception escaped the loop.
                ex.shutdown(wait=False)

    return [r if r is not None else "" for r in results]
