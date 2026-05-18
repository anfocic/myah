"""Session persistence for the conversation transcript. Input history
(arrow-key recall) now lives in `repl/ui.py` under prompt_toolkit's
`FileHistory`, so this module only concerns itself with one file:

- `~/.mia_session.json` holds the conversation transcript (user/assistant
  pairs only, never tool messages per §3) plus the live todo list.
  Atomic write via temp + rename.

The file format accepts two shapes:
- legacy: a bare list of message dicts (history only).
- current: an object `{"history": [...], "todos": [...]}`.

Writes always use the object shape; reads honor both so an older session
file still loads.

All operations fail silently (return / pass) on I/O errors — losing a
session file is annoying, not fatal, and the REPL must still boot."""
import json
import os

from config import SESSION_FILE
from repl.console import console
from repl.state import State
from tools.todo import deserialize_todos, serialize_todos


def _valid_history_entries(raw: list) -> tuple[list, int]:
    """Filter raw entries to those with the right shape. Returns (valid,
    dropped_count). Used by load_session to surface a one-line warning
    when a hand-edited or corrupted file silently loses data."""
    valid = [
        e for e in raw
        if isinstance(e, dict)
        and isinstance(e.get("role"), str)
        and isinstance(e.get("content"), str)
    ]
    return valid, len(raw) - len(valid)


def load_session(state: State) -> None:
    try:
        with open(SESSION_FILE) as f:
            loaded = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return

    # Legacy: bare list of messages.
    if isinstance(loaded, list):
        valid, dropped = _valid_history_entries(loaded)
        if dropped:
            console.print(
                f"[dim yellow]↳ session file had {dropped} "
                "malformed entr(ies); dropped them[/dim yellow]"
            )
        state["history"] = valid
        return

    # Current: object with history + todos. Unknown dict shapes (no
    # "history" or "todos" key) are treated as foreign — don't clobber
    # in-memory state.
    if isinstance(loaded, dict) and ("history" in loaded or "todos" in loaded):
        raw_history = loaded.get("history", [])
        if isinstance(raw_history, list):
            valid, dropped = _valid_history_entries(raw_history)
            if dropped:
                console.print(
                    f"[dim yellow]↳ session file had {dropped} "
                    "malformed entr(ies); dropped them[/dim yellow]"
                )
            state["history"] = valid
        if "todos" in loaded:
            state["todos"] = deserialize_todos(loaded["todos"])


def save_session(state: State) -> None:
    """Atomic via temp + rename so a crash mid-write can't produce a file
    that fails to parse on next startup. Prunes completed todos on save
    so resume starts clean but in-flight work survives."""
    todos = state.get("todos", []) or []
    open_todos = [t for t in todos if t.status != "completed"]
    payload = {
        "history": state["history"],
        "todos": serialize_todos(open_todos),
    }
    try:
        tmp = SESSION_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, SESSION_FILE)
    except OSError:
        pass


def wipe_session() -> None:
    try:
        os.remove(SESSION_FILE)
    except FileNotFoundError:
        pass


def has_saved_session() -> bool:
    """True if a session file exists with at least one turn. Used to hint
    the user (when they launched without --resume) that prior work is on
    disk but won't be loaded unless they opt in."""
    try:
        with open(SESSION_FILE) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if isinstance(data, list):
        return len(data) > 0
    if isinstance(data, dict):
        h = data.get("history")
        return isinstance(h, list) and len(h) > 0
    return False
