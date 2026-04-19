"""Fail-closed permission gate: a run_agent call without permission_check
must not silently execute destructive tools. Read-only tools still run."""
from agent import _run_tools_parallel
from providers.base import ToolCall


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
