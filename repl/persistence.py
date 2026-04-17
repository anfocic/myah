"""Session + input-history persistence. Two user-dir files:

- `~/.mia_session.json` holds the conversation transcript (user/assistant
  pairs only, never tool messages per §3). Atomic write via temp + rename.
- `~/.mia_history` holds the readline input history so arrow-key recall
  survives restarts.

All operations fail silently (return / pass) on I/O errors — losing a
session file is annoying, not fatal, and the REPL must still boot."""
import atexit
import json
import os
import readline

from repl.console import console
from repl.state import State

HISTORY_FILE = os.path.expanduser("~/.mia_history")
SESSION_FILE = os.path.expanduser("~/.mia_session.json")


def load_input_history() -> None:
    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)
    atexit.register(save_input_history)


def save_input_history() -> None:
    try:
        readline.write_history_file(HISTORY_FILE)
    except OSError:
        pass


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
