"""Tests for the todo_write tool: validation, whole-list-replace,
at-most-one in_progress invariant, formatting, persistence round-trip."""
import json

import pytest

from repl import persistence
from repl.tool_registry import make_execute_tool
from tools.todo import (
    Todo,
    deserialize_todos,
    format_todos,
    parse_todos,
    serialize_todos,
    todo_write,
)

# ── parse + validate ─────────────────────────────────────────────────────────


def test_parse_empty_list_is_valid():
    assert parse_todos([]) == []


def test_parse_valid_list():
    raw = [
        {"content": "Fix bug", "activeForm": "Fixing bug", "status": "pending"},
        {"content": "Write tests", "activeForm": "Writing tests", "status": "in_progress"},
    ]
    parsed = parse_todos(raw)
    assert len(parsed) == 2
    assert parsed[0].content == "Fix bug"
    assert parsed[1].status == "in_progress"


def test_parse_rejects_non_list():
    with pytest.raises(ValueError, match="must be a list"):
        parse_todos("not a list")


def test_parse_rejects_missing_keys():
    with pytest.raises(ValueError, match="missing required key"):
        parse_todos([{"content": "x", "status": "pending"}])  # no activeForm


def test_parse_rejects_blank_content():
    with pytest.raises(ValueError, match="content must be a non-empty string"):
        parse_todos([{"content": "", "activeForm": "x", "status": "pending"}])


def test_parse_rejects_invalid_status():
    with pytest.raises(ValueError, match="status must be one of"):
        parse_todos([{"content": "x", "activeForm": "y", "status": "blocked"}])


def test_parse_rejects_two_in_progress():
    raw = [
        {"content": "a", "activeForm": "doing a", "status": "in_progress"},
        {"content": "b", "activeForm": "doing b", "status": "in_progress"},
    ]
    with pytest.raises(ValueError, match="at most one"):
        parse_todos(raw)


def test_parse_allows_zero_in_progress():
    raw = [{"content": "a", "activeForm": "doing a", "status": "pending"}]
    assert len(parse_todos(raw)) == 1


# ── format ───────────────────────────────────────────────────────────────────


def test_format_empty():
    assert format_todos([]) == "(no todos)"


def test_format_uses_active_form_only_when_in_progress():
    todos = [
        Todo("Fix bug", "Fixing bug", "pending"),
        Todo("Write tests", "Writing tests", "in_progress"),
        Todo("Ship it", "Shipping it", "completed"),
    ]
    out = format_todos(todos)
    assert "[ ] Fix bug" in out
    assert "[~] Writing tests" in out
    assert "[x] Ship it" in out


# ── todo_write (tool dispatcher entry) ───────────────────────────────────────


def test_todo_write_replaces_state():
    state = {"todos": [Todo("old", "doing old", "pending")]}
    raw = [{"content": "new", "activeForm": "doing new", "status": "pending"}]
    result = todo_write(state, raw)
    assert len(state["todos"]) == 1
    assert state["todos"][0].content == "new"
    assert "new" in result


def test_todo_write_empty_clears():
    state = {"todos": [Todo("a", "b", "pending")]}
    result = todo_write(state, [])
    assert state["todos"] == []
    assert "cleared" in result


def test_todo_write_returns_error_string_on_invalid():
    state = {"todos": []}
    result = todo_write(state, "not a list")
    assert result.startswith("todo_write rejected")
    assert state["todos"] == []  # untouched


def test_todo_write_returns_error_on_two_in_progress():
    state = {"todos": []}
    raw = [
        {"content": "a", "activeForm": "doing a", "status": "in_progress"},
        {"content": "b", "activeForm": "doing b", "status": "in_progress"},
    ]
    result = todo_write(state, raw)
    assert result.startswith("todo_write rejected")


# ── dispatcher integration ───────────────────────────────────────────────────


def test_dispatcher_invokes_todo_write(state):
    execute = make_execute_tool(state)
    raw = [{"content": "task", "activeForm": "doing task", "status": "pending"}]
    result = execute("todo_write", {"todos": raw})
    assert "task" in result
    assert len(state["todos"]) == 1


def test_dispatcher_missing_todos_arg(state):
    execute = make_execute_tool(state)
    # args.get("todos") returns None → tool rejects it as data
    result = execute("todo_write", {})
    assert result.startswith("todo_write rejected")


# ── serialize round-trip ─────────────────────────────────────────────────────


def test_serialize_deserialize_round_trip():
    todos = [
        Todo("a", "doing a", "pending"),
        Todo("b", "doing b", "in_progress"),
        Todo("c", "doing c", "completed"),
    ]
    restored = deserialize_todos(serialize_todos(todos))
    assert len(restored) == 3
    assert restored[1].activeForm == "doing b"


def test_deserialize_garbage_returns_empty():
    assert deserialize_todos("garbage") == []
    assert deserialize_todos([{"bad": "shape"}]) == []


# ── persistence ──────────────────────────────────────────────────────────────


def test_save_load_round_trip_preserves_todos(tmp_path, monkeypatch, state):
    path = str(tmp_path / "session.json")
    monkeypatch.setattr(persistence, "SESSION_FILE", path)
    state["history"] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    state["todos"] = [
        Todo("a", "doing a", "in_progress"),
        Todo("b", "doing b", "pending"),
    ]
    persistence.save_session(state)

    state["history"] = []
    state["todos"] = []
    persistence.load_session(state)
    assert len(state["history"]) == 2
    assert len(state["todos"]) == 2
    assert state["todos"][0].activeForm == "doing a"


def test_save_prunes_completed_todos(tmp_path, monkeypatch, state):
    path = str(tmp_path / "session.json")
    monkeypatch.setattr(persistence, "SESSION_FILE", path)
    state["todos"] = [
        Todo("done", "doing done", "completed"),
        Todo("active", "doing active", "in_progress"),
    ]
    persistence.save_session(state)
    with open(path) as f:
        on_disk = json.load(f)
    assert len(on_disk["todos"]) == 1
    assert on_disk["todos"][0]["content"] == "active"


def test_load_legacy_bare_list_format(tmp_path, monkeypatch, state):
    """A session file written by the old code (bare list) should still load."""
    path = str(tmp_path / "session.json")
    monkeypatch.setattr(persistence, "SESSION_FILE", path)
    legacy = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    with open(path, "w") as f:
        json.dump(legacy, f)
    persistence.load_session(state)
    assert len(state["history"]) == 2
    assert state["todos"] == []
