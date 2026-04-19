"""Session persistence for the conversation transcript. Input history
(arrow-key recall) now lives in `repl/ui.py` under prompt_toolkit's
`FileHistory`, so this module only concerns itself with one file:

- `~/.mia_session.json` holds the conversation transcript (user/assistant
  pairs only, never tool messages per §3). Atomic write via temp + rename.

All operations fail silently (return / pass) on I/O errors — losing a
session file is annoying, not fatal, and the REPL must still boot."""
import json
import os

from repl.console import console
from repl.state import State

SESSION_FILE = os.path.expanduser("~/.mia_session.json")


def load_session(state: State) -> None:
    """Validate per-entry shape: without this filter, a hand-edited or
    corrupted session file slides through isinstance(list) and crashes
    later when run_agent iterates and calls .get("role") on a non-dict."""
    try:
        with open(SESSION_FILE) as f:
            loaded = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return
    if not isinstance(loaded, list):
        return
    valid = [
        e for e in loaded
        if isinstance(e, dict)
        and isinstance(e.get("role"), str)
        and isinstance(e.get("content"), str)
    ]
    if len(valid) != len(loaded):
        console.print(
            f"[dim yellow]↳ session file had {len(loaded) - len(valid)} "
            "malformed entr(ies); dropped them[/dim yellow]"
        )
    state["history"] = valid


def save_session(state: State) -> None:
    """Atomic via temp + rename so a crash mid-write can't produce a file
    that fails to parse on next startup."""
    try:
        tmp = SESSION_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state["history"], f)
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
    return isinstance(data, list) and len(data) > 0
