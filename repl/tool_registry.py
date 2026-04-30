"""Tool schema list + dispatcher factory.

The schema list is what the model sees (OpenAI function-calling format);
it's serialized into every turn's prompt. `make_execute_tool` closes over
REPL state so the `harness_info` tool can report live context — this is
the lexical-capture trick from CONCEPTS §23 that keeps `agent.py` unaware
of state shape.

Tool registration now lives in each tool module (see `tools/spec.py`).
Importing a submodule calls `register()`, populating a shared registry.
Only four tools remain special-cased here because they need REPL state or
closure access that generic adapters cannot provide:

- `pwd` / `cd`    — read/write state["cwd"]
- `harness_info`  — needs the live state dict + tool name list
- `spawn_subagent` — needs the full execute_tool closure + permission_check
"""

from typing import Any

# Import every tool submodule so their `register()` calls populate the
# shared registry. The side-effect is intentional and idempotent.
import tools.bash  # noqa: F401
import tools.files  # noqa: F401
import tools.git  # noqa: F401
import tools.search  # noqa: F401
import tools.utils  # noqa: F401
import tools.vault  # noqa: F401
import tools.web_search  # noqa: F401
from repl.console import console
from repl.state import State
from tools.cd import cd, pwd
from tools.harness import harness_info
from tools.spec import get_registry
from tools.subagent import spawn_subagent

# Special-case schemas for tools that need state/closure access and therefore
# cannot be registered via the generic adapter pattern in their home modules.
_SPECIAL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "pwd",
            "description": "Print the harness current working directory. Returns the absolute path as a string.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cd",
            "description": "Change the harness working directory. Resolves <path> relative to the current harness cwd. Use .. for parent. Refuses to escape the cwd guard. Returns the new directory path, or an error message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to change to. Supports relative paths (.., ../sibling) and absolute paths.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "harness_info",
            "description": "Introspect the harness you are running in: current model, context window size (num_ctx), context used in the previous turn, number of conversation turns in history, working directory, git branch, today's date, and the list of tools available to you. Call this when the user asks about the harness, model, or context usage, or when you need to decide whether to summarize / shorten your reply because ctx is getting tight.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_subagent",
            "description": (
                "Delegate a self-contained investigative subtask to a subagent "
                "running with a fresh context window. The subagent has the same "
                "tools you do (except spawn_subagent itself). You'll receive its "
                "final answer as a single tool result. Good for: 'find every caller "
                "of X and summarize', 'read these three files and report their "
                "relationship', 'grep the repo for TODOs in auth code'. Bad for: "
                "open-ended conversation, questions that need clarification, or "
                "tasks that would trivially fit in one or two of your own tool calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "The task for the subagent, written as a self-contained "
                            "instruction. Include any file paths, patterns, or "
                            "constraints the subagent needs — it starts with empty "
                            "history and cannot see your prior conversation."
                        ),
                    },
                },
                "required": ["task"],
            },
        },
    },
]

_registry = get_registry()

# Build the full schema list seen by the model: generic registry first,
# then the special cases. The order only matters for readability.
# Named TOOL_SCHEMAS rather than `tools` to avoid shadowing the `tools`
# package imported above for side-effect registration — the collision
# made mypy resolve every downstream `from repl.tool_registry import
# tools` to the package module instead of this list.
TOOL_SCHEMAS: list[dict[str, Any]] = [t.schema for t in _registry.values()] + _SPECIAL_SCHEMAS

TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]


def make_execute_tool(state: State, permission_check=None):
    """Factory so the harness_info tool can close over the live REPL state
    (ctx_used, history). Passing state through run_agent would leak REPL
    internals into its signature; a closure keeps agent.py state-ignorant.

    `permission_check` is captured here too so the `spawn_subagent` branch
    can thread it into the child `run_agent`. The subagent thereby
    inherits the same permission gate — the user still approves destructive
    tool calls the subagent attempts, just like for the parent."""

    def execute_tool(name, args):
        # Args come from the model, which occasionally omits required keys.
        # Return the error as a tool result so the model can recover instead of
        # crashing the REPL.
        # Path-resolution rule: every model-supplied path arg below goes through
        # resolve_against(cwd, ...) before reaching the tool, so the model's
        # `cd` movement is honored uniformly. Tools themselves stay
        # cwd-state-ignorant; this dispatcher is the single bridge.
        # state.get() so test fixtures that hand-construct a State (instead of
        # using new_state()) don't crash.
        import os as _os

        cwd = state.get("cwd") or _os.getcwd()
        try:
            if name == "pwd":
                return pwd(lambda: cwd)
            elif name == "cd":
                return cd(
                    lambda: cwd,
                    lambda new_cwd: state.__setitem__("cwd", new_cwd),
                    args["path"],
                )
            elif name == "harness_info":
                return harness_info(state, TOOL_NAMES)
            elif name == "spawn_subagent":
                return spawn_subagent(
                    task=args["task"],
                    tools=tools,
                    execute_tool=execute_tool,
                    permission_check=permission_check,
                    console=console,
                    plan_mode=bool(state.get("plan_mode", False)),
                    cwd=cwd,
                )

            tool = _registry.get(name)
            if tool is None:
                return "Tool not found"
            return tool.adapter(args, cwd)
        except KeyError as e:
            return f"Missing required argument: {e}"

    return execute_tool
