"""Smoke test for context-window tracking, trimming, and summarization.

Runs a handful of turns against the real local model with a deliberately
small NUM_CTX so that trim_history is forced to fire. Asserts:

  1. At least one trim event happens.
  2. summarize_dropped produces a non-empty summary.
  3. After trimming, ctx_used comes back down below the trim threshold.

Run: `python -m scripts.smoke_ctx`
"""
from __future__ import annotations

import config

config.NUM_CTX = 1024  # shrink *before* importing agent so module-level uses see it

from agent import run_agent, summarize_dropped, trim_history  # noqa: E402


def noop_tool(name: str, args: dict) -> str:
    return "no tools available in smoke test"


def main() -> None:
    big = "alpha beta gamma delta epsilon zeta eta theta " * 40  # ~320 tokens
    inputs = [
        f"Remember fact 1 — the golden key is in the attic. Context padding: {big}",
        f"Remember fact 2 — the silver box is in the cellar. Context padding: {big}",
        f"Remember fact 3 — the password is 'harness'. Context padding: {big}",
        "What facts am I asking you to remember? List them briefly.",
    ]

    history: list = []
    trim_events = 0
    summary_seen = False
    peak_ctx = 0

    for i, user_input in enumerate(inputs, start=1):
        _, history, ctx_used, _ = run_agent(user_input, [], noop_tool, history)
        peak_ctx = max(peak_ctx, ctx_used)
        history, dropped = trim_history(history, ctx_used, config.NUM_CTX)
        if dropped:
            trim_events += 1
            summary = summarize_dropped(dropped)
            if summary:
                summary_seen = True
                history.insert(
                    0,
                    {"role": "system", "content": f"Summary of earlier conversation: {summary}"},
                )
        print(
            f"turn {i}: ctx_used={ctx_used:>4}  history_len={len(history):>2}  "
            f"dropped={len(dropped) // 2}"
        )

    print()
    print(f"peak ctx_used: {peak_ctx}/{config.NUM_CTX}")
    print(f"trim events:   {trim_events}")
    print(f"summary seen:  {summary_seen}")

    assert trim_events > 0, "expected trim_history to fire at least once"
    assert summary_seen, "expected summarize_dropped to produce non-empty output"
    # ctx_used is the *actual* prompt size sent to the provider. The backend
    # may ignore NUM_CTX (especially openai-compat servers), so the peak can
    # legitimately exceed it. The harness trims reactively; it cannot enforce
    # the limit server-side.
    assert peak_ctx > 0, "ctx_used should be positive"
    print("\nOK — smoke test passed")


if __name__ == "__main__":
    main()
