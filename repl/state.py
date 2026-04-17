"""REPL state shape. Runtime-wise this is still a plain dict —
TypedDict is a type-checker-only construct, so editors catch typos like
`state["plaan_mode"]` without changing any access shape. `_retry_input`
is transient: cmd_retry sets it, the REPL loop pops it."""
from collections import deque
from typing import NotRequired, TypedDict

# Cap on the in-memory snapshot stack used by /rewind. Each snapshot is a
# deep copy of history taken before a run_agent call; 20 is generous enough
# to cover an interactive session without unbounded growth.
REWIND_MAX_SNAPSHOTS = 20


class State(TypedDict):
    history: list
    ctx_used: int
    plan_mode: bool
    debug: bool
    snapshots: deque
    _retry_input: NotRequired[str]


def new_state() -> State:
    """Freshly-initialized State with the right defaults. Centralized here
    so tests and main.py don't drift from the canonical shape."""
    return {
        "history": [],
        "ctx_used": 0,
        "plan_mode": False,
        "debug": False,
        "snapshots": deque(maxlen=REWIND_MAX_SNAPSHOTS),
    }
