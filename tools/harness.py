import os
import subprocess
from collections.abc import Mapping
from datetime import date
from typing import Any

from config import NUM_CTX, get_context_size
from providers import get_active_provider


def _git_branch(cwd: str | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=cwd,
        ).strip()
        return out or "(detached)"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "(not a git repo)"


# Read-only shape — accepts both a plain dict and main.py's State TypedDict.
# Declaring the full State shape here would pull a circular import from main.
def harness_snapshot(state: Mapping[str, Any], tool_names: list[str]) -> dict:
    """Single source of truth for fields both /context (rich) and the
    harness_info tool (plaintext) render. Reads model + provider from the
    live adapter so /model swaps show up immediately."""
    ctx_used = state["ctx_used"]
    provider = get_active_provider()
    cwd = state.get("cwd") or os.getcwd()
    ctx_size = get_context_size()
    return {
        "model": provider.model,
        "provider": provider.name,
        "num_ctx": ctx_size,
        "ctx_used": ctx_used,
        "ctx_pct": (ctx_used / ctx_size) if ctx_size else 0.0,
        "history_turns": len(state["history"]) // 2,
        "plan_mode": bool(state.get("plan_mode", False)),
        "cwd": cwd,
        "git_branch": _git_branch(cwd),
        "date": date.today().isoformat(),
        "tools": tool_names,
    }


def harness_info(state: Mapping[str, Any], tool_names: list[str]) -> str:
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
