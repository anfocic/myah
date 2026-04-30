"""Subagent tool — spawns a nested `run_agent` for a delegated task.

Isolated history (fresh `[]`), same provider, same `execute_tool` (so
tool dispatch and the permission gate behave identically), and the
`spawn_subagent` schema stripped from the child's tool list so the model
in the subagent doesn't see it as a callable option. A lock-guarded
depth counter is belt-and-suspenders: even if the child's model
hallucinates a call to `spawn_subagent`, the counter short-circuits it
before a second nested `run_agent` can start.

See CONCEPTS §43 for the design rationale and tradeoffs."""
import threading
from typing import Any

# Flat hierarchy on purpose: at most one subagent in flight at a time,
# period. A 7B model with a 4k context can't usefully coordinate deeper
# trees — each level duplicates the system prompt + env block, and the
# parent's turn is blocked the whole time the subtree runs. The lock is
# for the edge case where the parent emits two `spawn_subagent` calls
# in a single turn — `_run_tools_parallel` would race otherwise.
_MAX_DEPTH = 1
_depth = 0
_depth_lock = threading.Lock()


def spawn_subagent(
    task: str,
    tools: list[dict[str, Any]],
    execute_tool,
    permission_check,
    console=None,
    plan_mode: bool = False,
    cwd: str | None = None,
) -> str:
    """Run a nested `run_agent` focused on `task`. Returns the child's
    final content wrapped in a `<subagent_result>` marker so the parent
    sees it distinctly from its own content — same convention as the
    prompt-injection annotation layer uses (§41).

    `plan_mode` is threaded through so a subagent spawned from a
    plan-mode parent inherits the read-only gate. The model in the
    subagent sees the plan-mode system prompt and mutating tools get
    short-circuited by `_run_tools_parallel` identically to the parent."""
    # Lazy import — importing agent.loop at module-load time would create
    # a cycle the moment tools/subagent is pulled into repl/tool_registry,
    # which the agent layer imports transitively via execute_tool.
    from agent.loop import run_agent

    global _depth
    # Atomic check-and-increment so two parallel tool-call dispatches
    # from the same parent turn agree on "one subagent at a time."
    with _depth_lock:
        if _depth >= _MAX_DEPTH:
            return (
                "<subagent_error>"
                "Max subagent depth reached — nested spawns are not allowed "
                "and the harness already has one subagent in flight. "
                "This call was refused without running."
                "</subagent_error>"
            )
        _depth += 1

    # Strip spawn_subagent from the child's schema list so the model in
    # the subagent doesn't see it as a callable option. The depth counter
    # above is the real safety net; this filter just avoids wasting ~80
    # tokens every turn describing a tool we'd refuse to run anyway.
    child_tools = [
        t for t in tools if t["function"]["name"] != "spawn_subagent"
    ]

    if console:
        preview = task if len(task) <= 80 else task[:77] + "…"
        console.print(
            f"[dim cyan]↳ subagent spawning: {preview}[/dim cyan]"
        )

    try:
        content, _history, _ctx, _stats = run_agent(
            user_input=task,
            tools=child_tools,
            execute_tool=execute_tool,
            history=[],
            console=console,
            permission_check=permission_check,
            plan_mode=plan_mode,
            subagent=True,
            cwd=cwd,
        )
    finally:
        with _depth_lock:
            _depth -= 1

    if console:
        console.print("[dim cyan]↳ subagent finished[/dim cyan]")

    return f"<subagent_result>\n{content}\n</subagent_result>"
