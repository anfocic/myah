"""Fail-closed permission gate: a run_agent call without permission_check
must not silently execute destructive tools. Read-only tools still run."""
import threading
import time

import pytest
from rich.console import Console

import permissions
from agent import _run_tools_parallel
from providers.base import ToolCall


@pytest.fixture(autouse=True)
def clear_session_allowed():
    permissions._session_allowed.clear()
    yield
    permissions._session_allowed.clear()


def test_no_permission_check_denies_sensitive_tools():
    calls = [
        ToolCall(name="read_file", arguments={"path": "x"}),
        ToolCall(name="write_file", arguments={"path": "x", "content": "y"}),
        ToolCall(name="bash", arguments={"command": "ls"}),
    ]
    executed: list[str] = []

    def fake_execute(name, args):
        executed.append(name)
        return "ok"

    results = _run_tools_parallel(
        calls,
        fake_execute,
        permission_check=None,
        plan_mode=False,
    )

    # Only read_file ran
    assert executed == ["read_file"]
    assert results[0] == "ok"
    assert results[1] == "User denied this tool call."
    assert results[2] == "User denied this tool call."


def test_explicit_permission_check_overrides_default():
    calls = [ToolCall(name="write_file", arguments={"path": "x", "content": "y"})]
    executed: list[str] = []

    def fake_execute(name, args):
        executed.append(name)
        return "wrote"

    def allow_all(name, args):
        return True

    results = _run_tools_parallel(
        calls,
        fake_execute,
        permission_check=allow_all,
        plan_mode=False,
    )

    assert executed == ["write_file"]
    assert results[0] == "wrote"


def test_always_allow_is_scoped_to_exact_call(monkeypatch):
    responses = iter(["a", "n"])
    monkeypatch.setattr(permissions, "pt_prompt", lambda _prompt: next(responses))

    class DummyConsole:
        def print(self, *args, **kwargs):
            pass

    console = DummyConsole()

    assert permissions.check_permission(console, "bash", {"command": "pwd"}) is True
    # Same exact call is auto-approved.
    assert permissions.check_permission(console, "bash", {"command": "pwd"}) is True
    # Different command prompts again and can still be denied.
    assert permissions.check_permission(
        console,
        "bash",
        {"command": "rm -rf /tmp/demo"},
    ) is False


def test_mutating_tools_run_serially_to_avoid_races():
    calls = [
        ToolCall(name="write_file", arguments={"path": "x", "content": "1"}),
        ToolCall(
            name="edit_file",
            arguments={"path": "x", "old_string": "1", "new_string": "2"},
        ),
    ]
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_execute(name, args):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return name

    def allow_all(name, args):
        return True

    results = _run_tools_parallel(
        calls,
        fake_execute,
        permission_check=allow_all,
        plan_mode=False,
    )

    assert max_active == 1
    assert results == ["write_file", "edit_file"]


def test_tool_callbacks_receive_tool_id_and_duration():
    calls = [ToolCall(name="read_file", arguments={"path": "x.py"})]
    starts: list[tuple[str, dict]] = []
    ends: list[tuple[str, dict]] = []

    def fake_execute(name, args):
        time.sleep(0.01)
        return "contents"

    def allow_all(name, args):
        return True

    def on_start(name, args, meta=None):
        starts.append((name, meta or {}))

    def on_end(name, args, result, ok, meta=None):
        assert ok is True
        ends.append((result, meta or {}))

    results = _run_tools_parallel(
        calls,
        fake_execute,
        permission_check=allow_all,
        plan_mode=False,
        on_tool_start=on_start,
        on_tool_end=on_end,
    )

    assert results == ["contents"]
    assert starts[0][0] == "read_file"
    assert starts[0][1]["tool_id"] == "T01"
    assert ends[0][0] == "contents"
    assert ends[0][1]["tool_id"] == "T01"
    assert ends[0][1]["duration_s"] >= 0.0


def test_bash_permission_preview_includes_tool_id_and_command(monkeypatch):
    monkeypatch.setattr(permissions, "pt_prompt", lambda _prompt: "n")
    console = Console(record=True, force_terminal=False, width=100)

    allowed = permissions.check_permission(
        console,
        "bash",
        {"command": "pytest -q", "cwd": "tests", "timeout": 12},
        meta={"tool_id": "T07"},
    )

    exported = console.export_text()
    assert allowed is False
    assert "Permission requested" in exported
    assert "T07" in exported
    assert "shell command" in exported
    assert "pytest -q" in exported
    assert "tests" in exported
    assert "12s" in exported


def test_edit_file_permission_preview_renders_diff(monkeypatch):
    monkeypatch.setattr(permissions, "pt_prompt", lambda _prompt: "n")
    console = Console(record=True, force_terminal=False, width=100)

    allowed = permissions.check_permission(
        console,
        "edit_file",
        {
            "path": "x.py",
            "old_string": "old = 1\n",
            "new_string": "new = 2\n",
        },
        meta={"tool_id": "T02"},
    )

    exported = console.export_text()
    assert allowed is False
    assert "T02" in exported
    assert "x.py" in exported
    assert "diff preview" in exported
    assert "--- x.py" in exported
    assert "+new = 2" in exported
