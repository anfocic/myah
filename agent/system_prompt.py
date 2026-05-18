"""System prompt assembly. Base persona + env block + (if present) the
cwd's CLAUDE.md + (if plan mode) plan-mode rules. Re-read every turn so a
live edit to CLAUDE.md or a /plan toggle takes effect on the next call —
the cost is a ~5KB file read vs. a whole LLM round-trip."""
import os
import platform
import subprocess
from datetime import date
from pathlib import Path

# Imported for the plan-mode system prompt's "read-only tools" list.
# Defined in agent/__init__.py and also used by agent/loop.py's tool gate.
from agent import READ_ONLY_TOOLS
from providers import get_active_provider


def _git(*args: str, cwd: str | None = None) -> str | None:
    """Run a git command, return stripped stdout, or None on any failure
    (not a repo, git missing, detached state, etc.).

    `cwd` selects the directory git runs in so the env block can describe
    the repo the model is currently sitting inside, not the process startup
    repo, when the two diverge after `cd`."""
    try:
        out = subprocess.check_output(
            ["git", *args],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            cwd=cwd,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _env_block(cwd: str | None = None) -> str:
    """Compact environment snapshot prepended to every system prompt so the
    model has cwd / platform / git state on turn 1 without burning a tool
    call. Kept small — ~80-120 tokens depending on git state.

    `cwd` is the harness-tracked working directory (state["cwd"]) so the
    block reflects the model's `cd` movement. None falls back to the
    process cwd, which is correct on turn 1 before any cd has happened."""
    effective_cwd = cwd or os.getcwd()
    lines = [
        f"cwd: {effective_cwd}",
        f"platform: {platform.system().lower()} ({platform.machine()})",
        f"date: {date.today().isoformat()}",
    ]
    branch = _git("branch", "--show-current", cwd=effective_cwd)
    if branch:
        main_ref = _git("rev-parse", "--abbrev-ref", "origin/HEAD", cwd=effective_cwd)
        main = main_ref.split("/", 1)[-1] if main_ref else "main"
        porcelain = _git("status", "--porcelain", cwd=effective_cwd)
        dirty = len(porcelain.splitlines()) if porcelain else 0
        lines.append(f"git: branch={branch} main={main} dirty={dirty}")
    else:
        lines.append("git: (not a repository)")
    return "<env>\n" + "\n".join(lines) + "\n</env>"


_SERVED_VIA = {
    "ollama": "served locally via Ollama",
    "openai-compat": "served via an OpenAI-compatible HTTP API",
    "openai": "served by OpenAI",
    "anthropic": "served by Anthropic",
    "deepseek": "served by DeepSeek",
}


def _todos_block(todos: list | None) -> str | None:
    """Render the live todo list as a `<todos>` block for the system prompt.

    Returns None when the list is empty so the prompt stays compact. The list
    is read once per turn; mid-turn mutations show up via the `todo_write`
    tool result, not via this block."""
    if not todos:
        return None
    lines: list[str] = []
    for t in todos:
        label = t.activeForm if t.status == "in_progress" else t.content
        lines.append(f"[{t.status}] {label}")
    return "<todos>\n" + "\n".join(lines) + "\n</todos>"


def _vars_block(vars_dict: dict[str, str] | None) -> str | None:
    """Render conversation variables as a `<vars>` block.

    Truncates each value to PROMPT_VALUE_PREVIEW_CHARS so a single fat
    var can't dominate the prompt. Full values are still reachable via
    `get_var`. Returns None when empty so the prompt stays compact."""
    if not vars_dict:
        return None
    from tools.vars import PROMPT_VALUE_PREVIEW_CHARS

    lines: list[str] = []
    for name in sorted(vars_dict):
        value = vars_dict[name]
        if len(value) > PROMPT_VALUE_PREVIEW_CHARS:
            preview = value[: PROMPT_VALUE_PREVIEW_CHARS - 3] + "..."
            lines.append(f"{name} = {preview}  (truncated; get_var for full)")
        else:
            lines.append(f"{name} = {value}")
    return "<vars>\n" + "\n".join(lines) + "\n</vars>"


def build_system_prompt_parts(
    plan_mode: bool = False, subagent: bool = False, cwd: str | None = None,
    todos: list | None = None,
    vars_dict: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return the system prompt as its named parts so callers like /profile
    can show a per-source token breakdown. `build_system_prompt` is a thin
    wrapper that joins these with `\\n\\n`.

    Keys are stable: `persona`, `env`, `claude_md` (may be missing),
    `plan_mode` (present only when plan_mode=True). Dict iteration order
    matches prompt assembly order (Python 3.7+ insertion order)."""
    provider = get_active_provider()
    served = _SERVED_VIA.get(provider.name, f"served via {provider.name}")
    if subagent:
        persona = f"""You are a Myah subagent — a focused helper spawned by the main agent to complete a single task.
You are running on the {provider.model} model {served}.

The single user message you received IS your task. Investigate using tools if the task requires it, then return a concise final answer. The main agent will receive your final message verbatim as a tool result.

Subagent rules:
- Answer the task directly. Do NOT ask clarifying questions — you cannot; the parent is blocked waiting for your reply.
- Be concise. One paragraph or a short list is usually right. No pleasantries, no "Would you like me to...".
- Do not attempt to spawn further subagents; nested spawning is disabled.
- If a tool call fails, do not retry the same call — try a different tool or different arguments.
- Everything else applies: never fabricate tool output, never claim state changed without calling a tool, prefer surgical tools (edit_file, glob, grep) over shell (bash)."""
    else:
        persona = f"""You are Mia, the user's personal agent. You help manage their Obsidian vault, draft writing, research the web, and work with their files and code.
You are running on the {provider.model} model {served}.
Answer truthfully about what model and provider you are based on the line above.

Rules:
- CRITICAL: You cannot perform actions by describing them. The only way to change state (checkout a branch, edit a file, run a command, read a file) is to call the relevant tool. Writing "Switching to main now" or "Running the tests..." without calling `bash` is a LIE — the state did not change.
- CRITICAL: Never fabricate tool output. If you have not just called a tool, you do not know what it would have printed. Do not invent "HEAD is now at abc123...", "tests passed", file contents, or any other imagined result.
- Always use tools when the task requires it. Anything involving git, the filesystem, shell commands, or reading/editing files requires a tool call.
- If the user asks about current events, recent facts, or the live web, call `web_search` instead of guessing from model memory.
- After using a tool, respond with a short confirmation message grounded in the actual tool output.
- Never return an empty response
- For tasks needing multiple steps, do them one at a time
- If the user gives a bare filename like 'search.py', call `glob` first to resolve it to a full path, then read/edit that path
- If a tool call fails or returns an error, do NOT retry the exact same call. Read the error, then try a different approach: a different tool, different arguments, a narrower scope, or ask the user. Repeating the identical call rarely succeeds and burns the context window.
- For multi-step work (3+ steps), call `todo_write` at the start to plan, then update the list as you go — exactly one item in `in_progress` at a time, mark items `completed` the moment you finish them (not in batches at the end). The current list is shown in the `<todos>` block in this prompt; do not duplicate items already there.
- For intermediate facts you want to recall later without re-deriving (the user's email, a path you computed, a session id, the result of a slow lookup), stash them with `set_var(name, value)` and read them back with `get_var(name)`. Currently-set vars are shown in the `<vars>` block of this prompt — check there before re-computing. Use `unset_var` when something is no longer relevant.
- For a self-contained investigative subtask (e.g. "find every place X is called and summarize"), consider calling `spawn_subagent` — the subagent runs with a fresh context window, so its tool chatter doesn't eat yours.
- The user keeps a personal Obsidian vault. Use `note_search` and `note_read` to recall what they have written before answering from memory, and `note_write` / `note_append` / `daily_note` to capture notes, drafts, and logs. Prefer `[[wikilinks]]` between related notes.
- A separate project knowledge vault exists at `vault/` (sibling to `CLAUDE.md`). When working on this codebase, call `vault_search` to check for existing templates, documented decisions, or prior examples before implementing new features."""

    parts: dict[str, str] = {
        "persona": persona,
        "env": _env_block(cwd),
    }

    # Re-read every turn so edits to CLAUDE.md take effect without restarting.
    # Anchored to the harness cwd (state["cwd"]) so a `cd` into a sibling
    # project surfaces THAT project's CLAUDE.md, not the startup one.
    claude_md = Path(cwd) / "CLAUDE.md" if cwd else Path("CLAUDE.md")
    if claude_md.is_file():
        try:
            parts["claude_md"] = (
                "Project context (CLAUDE.md — the user's instructions for this repo):\n"
                + claude_md.read_text()
            )
        except OSError:
            pass

    todos_part = _todos_block(todos) if not subagent else None
    if todos_part:
        parts["todos"] = todos_part

    vars_part = _vars_block(vars_dict) if not subagent else None
    if vars_part:
        parts["vars"] = vars_part

    if plan_mode:
        parts["plan_mode"] = (
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

    return parts


def build_system_prompt(
    plan_mode: bool = False, subagent: bool = False, cwd: str | None = None,
    todos: list | None = None,
    vars_dict: dict[str, str] | None = None,
) -> str:
    """Base persona + env block + (if the cwd has a CLAUDE.md) project
    context + (if plan mode) planning rules. Reads model + provider from
    the live adapter so /model swaps take effect on the next turn.

    `cwd` is the harness-tracked working directory so the env block, the
    git status line, and the CLAUDE.md lookup all reflect the model's
    current location after `cd`. None falls back to the process cwd.

    `todos` is the live working-memory list (state["todos"]). Injected as
    a `<todos>` block when non-empty so the model sees it every turn.
    Subagents never see the parent's todos.

    `vars_dict` is the live conversation-variable map (state["vars"]).
    Injected as a `<vars>` block when non-empty so the model sees the
    keys (and previews) without re-issuing get_var on every turn.

    `subagent=True` swaps the persona for a subagent-focused one: no
    pleasantries, no clarifying questions, return a concise final answer
    because the parent agent is waiting on a tool result. See §43."""
    return "\n\n".join(
        build_system_prompt_parts(plan_mode, subagent, cwd, todos, vars_dict).values()
    )
