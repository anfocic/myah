"""/rewind behavior: snapshot stack semantics, clamping, /clear interaction."""
from collections import deque

from main import REWIND_MAX_SNAPSHOTS, cmd_clear, cmd_rewind


def _history(n_turns: int) -> list[dict]:
    out = []
    for i in range(n_turns):
        out.append({"role": "user", "content": f"u{i}"})
        out.append({"role": "assistant", "content": f"a{i}"})
    return out


def test_rewind_one_turn_restores_prior_state(state):
    state["snapshots"].append(_history(1))          # snapshot before turn 2
    state["snapshots"].append(_history(2))          # snapshot before turn 3
    state["history"] = _history(3)

    cmd_rewind(state)

    assert state["history"] == _history(2)
    assert len(state["snapshots"]) == 1


def test_rewind_clamps_to_stack_depth(state):
    state["snapshots"].append(_history(1))
    state["history"] = _history(2)

    cmd_rewind(state, "99")  # ask for more than we have

    assert state["history"] == _history(1)
    assert len(state["snapshots"]) == 0


def test_rewind_on_empty_stack_is_noop(state):
    state["history"] = _history(1)
    original = list(state["history"])

    cmd_rewind(state)

    assert state["history"] == original


def test_rewind_rejects_non_integer_arg(state):
    state["snapshots"].append([])
    cmd_rewind(state, "abc")
    # stack is unchanged — rejection, not a partial apply
    assert len(state["snapshots"]) == 1


def test_clear_wipes_snapshots_too(state, tmp_path, monkeypatch):
    # /clear must drop the snapshot stack or /rewind would resurrect the
    # history the user just asked to nuke.
    monkeypatch.setattr("main.SESSION_FILE", str(tmp_path / "session.json"))
    state["history"] = _history(2)
    state["snapshots"].append(_history(1))

    cmd_clear(state)

    assert state["history"] == []
    assert len(state["snapshots"]) == 0


def test_rewind_stack_capped_at_maxlen():
    snapshots: deque = deque(maxlen=REWIND_MAX_SNAPSHOTS)
    for _ in range(REWIND_MAX_SNAPSHOTS + 5):
        snapshots.append([])
    assert len(snapshots) == REWIND_MAX_SNAPSHOTS
