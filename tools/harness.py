import os
import subprocess
from datetime import date

from config import MODEL_NAME, MODEL_PROVIDER, NUM_CTX


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


def harness_snapshot(state: dict, tool_names: list[str]) -> dict:
    """Single source of truth for fields both /context (rich) and the
    harness_info tool (plaintext) render."""
    ctx_used = state["ctx_used"]
    return {
        "model": MODEL_NAME,
        "provider": MODEL_PROVIDER,
        "num_ctx": NUM_CTX,
        "ctx_used": ctx_used,
        "ctx_pct": (ctx_used / NUM_CTX) if NUM_CTX else 0.0,
        "history_turns": len(state["history"]) // 2,
        "plan_mode": bool(state.get("plan_mode", False)),
        "cwd": os.getcwd(),
        "git_branch": _git_branch(),
        "date": date.today().isoformat(),
        "tools": tool_names,
    }


def harness_info(state: dict, tool_names: list[str]) -> str:
    # ctx_used is the previous turn's settled value — there is no "current
    # turn" count because the model is calling this from inside one.
    s = harness_snapshot(state, tool_names)
    return (
        f"model: {s['model']} ({s['provider']})\n"
        f"num_ctx: {s['num_ctx']}\n"
        f"ctx_used: {s['ctx_used']} ({s['ctx_pct']:.1%}) — snapshot from previous turn\n"
        f"history_turns: {s['history_turns']}\n"
        f"plan_mode: {'ON' if s['plan_mode'] else 'off'}\n"
        f"cwd: {s['cwd']}\n"
        f"git_branch: {s['git_branch']}\n"
        f"date: {s['date']}\n"
        f"tools: {', '.join(s['tools'])}"
    )
