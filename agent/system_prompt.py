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
    "openai": "served by OpenAI",
    "anthropic": "served by Anthropic",
    "deepseek": "served by DeepSeek",
}


def build_system_prompt(plan_mode: bool = False, subagent: bool = False) -> str:
    """Base persona + env block + (if the cwd has a CLAUDE.md) project
    context + (if plan mode) planning rules. Reads model + provider from
    the live adapter so /model swaps take effect on the next turn.

    `subagent=True` swaps the persona for a subagent-focused one: no
    pleasantries, no clarifying questions, return a concise final answer
    because the parent agent is waiting on a tool result. See §43."""
    provider = get_active_provider()
    served = _SERVED_VIA.get(provider.name, f"served via {provider.name}")
    if subagent:
        base = f"""You are a Mia subagent — a focused helper spawned by the main agent to complete a single task.
You are running on the {provider.model} model {served}.

The single user message you received IS your task. Investigate using tools if the task requires it, then return a concise final answer. The main agent will receive your final message verbatim as a tool result.

Subagent rules:
- Answer the task directly. Do NOT ask clarifying questions — you cannot; the parent is blocked waiting for your reply.
- Be concise. One paragraph or a short list is usually right. No pleasantries, no "Would you like me to...".
- Do not attempt to spawn further subagents; nested spawning is disabled.
- Everything else applies: never fabricate tool output, never claim state changed without calling a tool, prefer surgical tools (edit_file, glob, grep) over shell (bash)."""
    else:
        base = f"""You are Mia, a personal assistant.
You are running on the {provider.model} model {served}.
Answer truthfully about what model and provider you are based on the line above.

Rules:
- CRITICAL: You cannot perform actions by describing them. The only way to change state (checkout a branch, edit a file, run a command, read a file) is to call the relevant tool. Writing "Switching to main now" or "Running the tests..." without calling `bash` is a LIE — the state did not change.
- CRITICAL: Never fabricate tool output. If you have not just called a tool, you do not know what it would have printed. Do not invent "HEAD is now at abc123...", "tests passed", file contents, or any other imagined result.
- Always use tools when the task requires it. Anything involving git, the filesystem, shell commands, or reading/editing files requires a tool call.
- After using a tool, respond with a short confirmation message grounded in the actual tool output.
- Never return an empty response
- For tasks needing multiple steps, do them one at a time
- If the user gives a bare filename like 'search.py', call `glob` first to resolve it to a full path, then read/edit that path
- For a self-contained investigative subtask (e.g. "find every place X is called and summarize"), consider calling `spawn_subagent` — the subagent runs with a fresh context window, so its tool chatter doesn't eat yours."""

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
