import os
import subprocess
from datetime import date


def _git_branch() -> str:
    try:
        out = subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "(detached)"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "(not a git repo)"


def harness_info(
    state: dict,
    *,
    model: str,
    provider: str,
    num_ctx: int,
    tool_names: list[str],
) -> str:
    """Snapshot of the harness the model is running in. Called by the model
    when it needs to introspect ctx budget, environment, or its own tool
    inventory mid-turn. `state` is the live REPL state dict (ctx_used reflects
    the previous turn's settled value — there is no 'current turn' value
    because the model is asking from inside it)."""
    ctx_used = state.get("ctx_used", 0)
    history_turns = len(state.get("history", [])) // 2
    pct = (ctx_used / num_ctx) if num_ctx else 0.0
    return (
        f"model: {model} ({provider})\n"
        f"num_ctx: {num_ctx}\n"
        f"ctx_used: {ctx_used} ({pct:.1%}) — snapshot from previous turn\n"
        f"history_turns: {history_turns}\n"
        f"cwd: {os.getcwd()}\n"
        f"git_branch: {_git_branch()}\n"
        f"date: {date.today().isoformat()}\n"
        f"tools: {', '.join(tool_names)}"
    )
