"""REPL state shape. Runtime-wise this is still a plain dict —
TypedDict is a type-checker-only construct, so editors catch typos like
`state["plaan_mode"]` without changing any access shape. `_retry_input`
is transient: cmd_retry sets it, the REPL loop pops it."""
import os
from collections import deque
from typing import TYPE_CHECKING, NotRequired, TypedDict

if TYPE_CHECKING:
    from tools.todo import Todo

# Cap on the in-memory snapshot stack used by /rewind. Each snapshot is a
# deep copy of history taken before a run_agent call; 20 is generous enough
# to cover an interactive session without unbounded growth.
REWIND_MAX_SNAPSHOTS = 20

# How many recent per-turn metric dicts /stats keeps for its sparkline trend.
# Matches the design's "last 8 turns" trend block.
TURN_HISTORY_MAX = 8


class State(TypedDict):
    history: list
    ctx_used: int
    plan_mode: bool
    debug: bool
    snapshots: deque
    # Harness's tracked working directory. Kept separate from os.getcwd()
    # so the security guard (is_within_cwd) is always anchored to the
    # original process cwd, while the model can navigate freely within it.
    cwd: str
    _retry_input: NotRequired[str]
    # Per-turn metrics captured at end of each run_agent call. `/stats`
    # renders these on demand so the REPL doesn't have to print a footer
    # line on every turn. Absent before the first turn completes.
    last_turn: NotRequired[dict]
    # Rolling window of the last TURN_HISTORY_MAX per-turn metric dicts,
    # feeding /stats' sparkline trend. main.py appends after each turn.
    turn_history: deque
    # Model-maintained working memory. todo_write replaces this list in
    # place; system_prompt re-reads it every turn so the model always
    # sees the live state. See tools/todo.py.
    todos: list["Todo"]
    # Conversation variables — a named string store the model populates
    # via set_var / get_var / list_vars / unset_var. Session-scoped (not
    # currently persisted across `--resume`). See tools/vars.py.
    vars: dict[str, str]


def new_state() -> State:
    """Freshly-initialized State with the right defaults. Centralized here
    so tests and main.py don't drift from the canonical shape."""
    return {
        "history": [],
        "ctx_used": 0,
        "plan_mode": False,
        "debug": False,
        "snapshots": deque(maxlen=REWIND_MAX_SNAPSHOTS),
        "turn_history": deque(maxlen=TURN_HISTORY_MAX),
        "cwd": os.getcwd(),
        "todos": [],
        "vars": {},
    }
