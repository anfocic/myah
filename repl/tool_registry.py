"""Tool schema list + dispatcher factory.

The schema list is what the model sees (OpenAI function-calling format);
it's serialized into every turn's prompt. `make_execute_tool` closes over
REPL state so the `harness_info` tool can report live context — this is
the lexical-capture trick from CONCEPTS §23 that keeps `agent.py` unaware
of state shape.

Adding a new tool is three edits in this file: import the function, add a
schema entry, add an `execute_tool` branch. CONCEPTS §36 argues for
narrow named tools over generic shell (for small models), which is why
`git_checkout` exists even though `bash` could do the same job."""
from typing import Any

from repl.console import console
from repl.state import State
from tools.bash import bash as run_bash
from tools.files import edit_file, read_file, write_file
from tools.git import git_checkout
from tools.harness import harness_info
from tools.search import glob, grep
from tools.subagent import spawn_subagent
from tools.utils import get_current_time

# Tool schemas are nested dicts (OpenAI function-calling format). The
# explicit `list[dict[str, Any]]` annotation keeps mypy from inferring a
# `Collection[str]` value type from the first entry, which blocks the
# nested indexing in the TOOL_NAMES comprehension below.
tools: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file. Returns the contents with line numbers prepended (cat -n style) so you can reference specific lines. By default returns the first 1000 lines; use offset and limit to paginate longer files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path to read"},
                    "offset": {
                        "type": "integer",
                        "description": "1-indexed line to start from. Defaults to 1.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of lines to return. Defaults to 1000.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes content to a file at the given path. Overwrites the whole file — prefer edit_file for surgical changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to write to",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Surgically replace a string in a file. old_string must uniquely identify the target unless replace_all is true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path to edit"},
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to find. Must be unique in the file unless replace_all is true.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence instead of requiring uniqueness",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files by name or glob pattern, recursively. Use this to resolve a bare filename (e.g. 'search.py') to its full path before reading or editing. Accepts 'search.py', '*.py', or '**/*.md'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Filename or glob pattern to match",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search from. Defaults to current working directory.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Regex search across files under a path. Returns matching file paths by default, or path:line:text when output_mode is 'content'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Python regex to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search. Defaults to the current working directory.",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional glob filter like '*.py' or '**/*.md'",
                    },
                    "output_mode": {
                        "type": "string",
                        "description": "'files_with_matches' (default) or 'content'",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command. Use for git, running tests, builds, package management, or any shell-only operation. Returns stdout, stderr, and exit code. Requires user permission for each call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory. Defaults to the REPL's current directory.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds. Defaults to 30.",
                    },
                },
                "required": ["command"],
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
            "name": "git_checkout",
            "description": "Switch to a git branch. Equivalent to `git checkout <branch>`. ALWAYS use this whenever the user asks to switch, check out, or move to a branch — never simulate the action in text and never fabricate output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "branch": {
                        "type": "string",
                        "description": "Branch name to switch to (e.g. 'main', 'feat/foo').",
                    },
                },
                "required": ["branch"],
            },
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

TOOL_NAMES = [t["function"]["name"] for t in tools]


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
        try:
            if name == "get_current_time":
                return get_current_time()
            elif name == "read_file":
                return read_file(
                    args["path"],
                    int(args.get("offset", 1)),
                    int(args["limit"]) if "limit" in args else None,
                )
            elif name == "write_file":
                return write_file(args["path"], args["content"])
            elif name == "edit_file":
                return edit_file(
                    args["path"],
                    args["old_string"],
                    args["new_string"],
                    bool(args.get("replace_all", False)),
                )
            elif name == "glob":
                return glob(args["pattern"], args.get("path", "."))
            elif name == "grep":
                return grep(
                    args["pattern"],
                    args.get("path", "."),
                    args.get("glob"),
                    args.get("output_mode", "files_with_matches"),
                )
            elif name == "bash":
                return run_bash(
                    args["command"],
                    args.get("cwd", "."),
                    int(args.get("timeout", 30)),
                )
            elif name == "harness_info":
                return harness_info(state, TOOL_NAMES)
            elif name == "git_checkout":
                return git_checkout(args["branch"])
            elif name == "spawn_subagent":
                return spawn_subagent(
                    task=args["task"],
                    tools=tools,
                    execute_tool=execute_tool,
                    permission_check=permission_check,
                    console=console,
                    plan_mode=bool(state.get("plan_mode", False)),
                )
            return "Tool not found"
        except KeyError as e:
            return f"Missing required argument: {e}"

    return execute_tool
