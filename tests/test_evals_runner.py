"""Tests for the eval runner and check dispatcher.

Grader unit tests use synthetic bundles (no provider, no tools). The
end-to-end test reuses the scripted FakeProvider from
`tests/test_integration.py` to drive one fake task through `run_suite`.
"""
from __future__ import annotations

from pathlib import Path

from evals import checks as checks_mod
from evals import runner
from providers.base import ToolCall
from tests.test_integration import FakeProvider, ScriptedTurn, install_provider  # noqa: F401

# ---------- check dispatcher unit tests ----------

def _bundle(**kw) -> dict:
    base = {
        "content": "",
        "trace": [],
        "stats": {},
        "ctx_used": 0,
        "cwd": Path("."),
        "fixture_dir": None,
    }
    base.update(kw)
    return base


def test_tool_trace_pass_and_fail():
    b = _bundle(trace=[{"name": "grep"}, {"name": "read_file"}])
    ok, _ = checks_mod.dispatch(
        {"type": "tool_trace", "must_call": ["grep"], "must_not_call": ["bash"]}, b
    )
    assert ok

    ok, why = checks_mod.dispatch(
        {"type": "tool_trace", "must_call": ["edit_file"]}, b
    )
    assert not ok and "expected calls not made" in why

    ok, why = checks_mod.dispatch(
        {"type": "tool_trace", "must_not_call": ["read_file"]}, b
    )
    assert not ok and "forbidden calls made" in why


def test_tool_trace_call_count_max():
    b = _bundle(trace=[{"name": "grep"}] * 5)
    ok, _ = checks_mod.dispatch({"type": "tool_trace", "call_count_max": 5}, b)
    assert ok
    ok, why = checks_mod.dispatch({"type": "tool_trace", "call_count_max": 4}, b)
    assert not ok and "too many tool calls" in why


def test_content_regex_and_negate():
    b = _bundle(content="the answer is config.py and tokens.py")
    assert checks_mod.dispatch({"type": "content_regex", "pattern": r"config\.py"}, b)[0]
    assert not checks_mod.dispatch({"type": "content_regex", "pattern": r"missing"}, b)[0]
    # negate: pass when pattern is NOT present
    assert checks_mod.dispatch({"type": "content_regex", "pattern": r"missing", "negate": True}, b)[0]
    assert not checks_mod.dispatch({"type": "content_regex", "pattern": r"config", "negate": True}, b)[0]


def test_content_substr():
    b = _bundle(content="Hello World")
    assert checks_mod.dispatch({"type": "content_substr", "value": "World"}, b)[0]
    assert checks_mod.dispatch({"type": "content_substr", "value": "world", "ignorecase": True}, b)[0]
    assert not checks_mod.dispatch({"type": "content_substr", "value": "world"}, b)[0]


def test_fs_file_contains(tmp_path: Path):
    f = tmp_path / "sample.py"
    f.write_text("def new_name():\n    pass\n")
    b = _bundle(cwd=tmp_path)
    assert checks_mod.dispatch(
        {"type": "fs_file_contains", "path": "sample.py", "pattern": r"^def new_name\("}, b
    )[0]
    assert checks_mod.dispatch(
        {"type": "fs_file_contains", "path": "sample.py", "pattern": r"^def old_name\(", "negate": True}, b
    )[0]
    assert not checks_mod.dispatch(
        {"type": "fs_file_contains", "path": "missing.py", "pattern": r"."}, b
    )[0]


def test_fs_file_equals_literal(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("abc")
    b = _bundle(cwd=tmp_path)
    assert checks_mod.dispatch(
        {"type": "fs_file_equals", "path": "x.txt", "expected": "abc"}, b
    )[0]
    assert not checks_mod.dispatch(
        {"type": "fs_file_equals", "path": "x.txt", "expected": "abd"}, b
    )[0]


def test_fs_grep_count_eq_ge_le(tmp_path: Path):
    f = tmp_path / "src.py"
    f.write_text("foo\nfoo bar\nbaz\nfoo\n")
    b = _bundle(cwd=tmp_path)
    # eq is the default op
    assert checks_mod.dispatch(
        {"type": "fs_grep_count", "path": "src.py", "pattern": r"foo", "expected": 3}, b
    )[0]
    ok, why = checks_mod.dispatch(
        {"type": "fs_grep_count", "path": "src.py", "pattern": r"foo", "expected": 2}, b
    )
    assert not ok and "found 3" in why
    # ge passes when count >= expected
    assert checks_mod.dispatch(
        {"type": "fs_grep_count", "path": "src.py", "pattern": r"foo",
         "expected": 2, "op": "ge"}, b
    )[0]
    # le passes when count <= expected
    assert checks_mod.dispatch(
        {"type": "fs_grep_count", "path": "src.py", "pattern": r"foo",
         "expected": 5, "op": "le"}, b
    )[0]
    # missing file is a fail, not a raise
    assert not checks_mod.dispatch(
        {"type": "fs_grep_count", "path": "absent.py", "pattern": r"x", "expected": 0}, b
    )[0]


def test_bash_exit_zero_pass_and_fail(tmp_path: Path):
    b = _bundle(cwd=tmp_path)
    # Exit 0
    assert checks_mod.dispatch(
        {"type": "bash_exit_zero", "cmd": "true"}, b
    )[0]
    # Exit nonzero — why should surface exit code and command
    ok, why = checks_mod.dispatch(
        {"type": "bash_exit_zero", "cmd": "false"}, b
    )
    assert not ok and "exit 1" in why


def test_bash_exit_zero_respects_cwd(tmp_path: Path):
    # Writes a sentinel into the task's cwd, then a second check reads it.
    # Proves the `cmd` runs with cwd=bundle["cwd"].
    b = _bundle(cwd=tmp_path)
    assert checks_mod.dispatch(
        {"type": "bash_exit_zero", "cmd": "echo hi > marker.txt"}, b
    )[0]
    assert (tmp_path / "marker.txt").exists()


def test_bash_exit_zero_cwd_rel(tmp_path: Path):
    sub = tmp_path / "sub"
    sub.mkdir()
    b = _bundle(cwd=tmp_path)
    assert checks_mod.dispatch(
        {"type": "bash_exit_zero", "cmd": "pwd | grep -q sub", "cwd_rel": "sub"}, b
    )[0]


def test_bash_exit_zero_timeout(tmp_path: Path):
    b = _bundle(cwd=tmp_path)
    ok, why = checks_mod.dispatch(
        {"type": "bash_exit_zero", "cmd": "sleep 2", "timeout_s": 1}, b
    )
    assert not ok and "timed out" in why


def test_python_callable_and_dict_form():
    b = _bundle(content="42")
    ok, _ = checks_mod.dispatch(lambda bundle: bundle["content"] == "42", b)
    assert ok
    ok, _ = checks_mod.dispatch({"type": "python", "fn": lambda bundle: (False, "nope")}, b)
    assert not ok


def test_unknown_check_type_returns_failure():
    ok, why = checks_mod.dispatch({"type": "does_not_exist"}, _bundle())
    assert not ok and "unknown check type" in why


# ---------- runner end-to-end against FakeProvider ----------

def test_run_suite_e2e(monkeypatch, tmp_path, install_provider):  # noqa: F811
    """Drive one fake task through run_suite. Scripts a fake model that
    emits a grep tool call then a final answer mentioning the target file,
    verifies the task passes, and checks JSONL output."""
    install_provider([
        ScriptedTurn(
            content_chunks=[],
            tool_calls=[ToolCall(name="grep", arguments={"pattern": "TOOL_RESULT_MAX_BYTES"})],
        ),
        ScriptedTurn(content_chunks=["Found in config.py"]),
    ])

    fake_task = {
        "id": "fake_find",
        "prompt": "find it",
        "setup": {"fs": None},
        "provider": None,
        "plan_mode": False,
        "permission": "allow_all",
        "limits": {"max_tool_calls": 4, "wall_timeout_s": 30},
        "checks": [
            {"type": "tool_trace", "must_call": ["grep"]},
            {"type": "content_regex", "pattern": r"config\.py"},
        ],
    }

    monkeypatch.setattr(runner, "discover_tasks", lambda: [fake_task])
    monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path)

    results = runner.run_suite()
    assert len(results) == 1
    r = results[0]
    assert r.passed, f"expected pass, got failures: {r.check_results}, err={r.error}"
    assert len(r.trace) == 1
    assert r.trace[0]["name"] == "grep"

    # JSONL file was written
    written = list(tmp_path.glob("*.jsonl"))
    assert len(written) == 1
    assert written[0].read_text().count("\n") == 1


def test_run_suite_fails_when_forbidden_tool_is_called(monkeypatch, tmp_path, install_provider):  # noqa: F811
    install_provider([
        ScriptedTurn(
            content_chunks=[],
            tool_calls=[ToolCall(name="bash", arguments={"command": "grep foo"})],
        ),
        ScriptedTurn(content_chunks=["Done."]),
    ])

    fake_task = {
        "id": "fake_refuse",
        "prompt": "find it",
        "permission": "allow_all",
        "checks": [
            {"type": "tool_trace", "must_not_call": ["bash"]},
        ],
    }
    monkeypatch.setattr(runner, "discover_tasks", lambda: [fake_task])
    monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path)

    results = runner.run_suite()
    assert len(results) == 1
    assert not results[0].passed
    assert not results[0].check_results[0]["pass"]
    assert "bash" in results[0].check_results[0]["why"]


def test_list_tasks_returns_phase1_ids():
    """Phase 1 tasks must all be discoverable, legacy ids must be gone."""
    ids = runner.list_tasks()
    assert "find_symbol_all" in ids
    assert "fix_failing_test" in ids
    assert "tdd_new_fn" in ids
    assert "commit_msg_from_diff" in ids
    assert "find_string" not in ids
    assert "edit_rename" not in ids
