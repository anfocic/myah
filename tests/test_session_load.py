"""Session-file load validation. The bug the validation exists to prevent:
a corrupt file whose top-level is a list slides through isinstance(list)
and crashes later when run_agent iterates and calls .get("role")."""
import json


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
    import main
    monkeypatch.setattr(main, "SESSION_FILE", path)

    main._load_session(state)
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
    import main
    monkeypatch.setattr(main, "SESSION_FILE", path)

    main._load_session(state)
    assert len(state["history"]) == 2
    assert state["history"][0]["content"] == "ok"
    assert state["history"][1]["content"] == "ok2"


def test_non_list_top_level_is_ignored(tmp_path, monkeypatch, state):
    path = _write_session(tmp_path, {"not": "a list"})
    import main
    monkeypatch.setattr(main, "SESSION_FILE", path)

    state["history"] = [{"role": "user", "content": "preserved"}]
    main._load_session(state)
    # untouched — load bailed
    assert state["history"][0]["content"] == "preserved"


def test_missing_file_is_silently_ignored(tmp_path, monkeypatch, state):
    import main
    monkeypatch.setattr(main, "SESSION_FILE", str(tmp_path / "does_not_exist.json"))

    state["history"] = [{"role": "user", "content": "preserved"}]
    main._load_session(state)
    assert state["history"][0]["content"] == "preserved"
