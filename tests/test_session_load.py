"""Session-file load validation. The bug the validation exists to prevent:
a corrupt file whose top-level is a list slides through isinstance(list)
and crashes later when run_agent iterates and calls .get("role")."""
import json

from repl import persistence


def _write_session(tmp_path, content) -> str:
    path = tmp_path / "session.json"
    with open(path, "w") as f:
        json.dump(content, f)
    return str(path)


def test_valid_session_loads_intact(tmp_path, monkeypatch, state):
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    path = _write_session(tmp_path, history)
    monkeypatch.setattr(persistence, "SESSION_FILE", path)

    persistence.load_session(state)
    assert state["history"] == history


def test_malformed_entries_are_dropped(tmp_path, monkeypatch, state):
    corrupt = [
        {"role": "user", "content": "ok"},
        {"role": "user"},                         # missing content
        "not a dict",                             # wrong type
        {"role": 42, "content": "x"},             # role wrong type
        {"role": "assistant", "content": "ok2"},
    ]
    path = _write_session(tmp_path, corrupt)
    monkeypatch.setattr(persistence, "SESSION_FILE", path)

    persistence.load_session(state)
    assert len(state["history"]) == 2
    assert state["history"][0]["content"] == "ok"
    assert state["history"][1]["content"] == "ok2"


def test_non_list_top_level_is_ignored(tmp_path, monkeypatch, state):
    path = _write_session(tmp_path, {"not": "a list"})
    monkeypatch.setattr(persistence, "SESSION_FILE", path)

    state["history"] = [{"role": "user", "content": "preserved"}]
    persistence.load_session(state)
    # untouched — load bailed
    assert state["history"][0]["content"] == "preserved"


def test_missing_file_is_silently_ignored(tmp_path, monkeypatch, state):
    monkeypatch.setattr(persistence, "SESSION_FILE", str(tmp_path / "does_not_exist.json"))

    state["history"] = [{"role": "user", "content": "preserved"}]
    persistence.load_session(state)
    assert state["history"][0]["content"] == "preserved"
