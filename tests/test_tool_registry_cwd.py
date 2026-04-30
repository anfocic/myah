"""Integration tests: the dispatcher resolves paths against state["cwd"].

The unit tests in test_tools_cd.py prove resolve_against does the right
math; these tests prove the dispatcher actually feeds it through. They use
real tmp_path filesystems and the real dispatcher so a regression in the
wiring (forgetting resolve_against on a new tool, dropping cwd= on a git
call) gets caught.

Security boundary: process cwd is the repo root, so files in tmp_path
would be refused by is_within_cwd. We set MYAH_ALLOW_OUTSIDE_CWD=1 to
exercise the resolution logic without fighting the path guard — these
tests are about cwd plumbing, not the security layer.
"""
from __future__ import annotations

import os

import pytest

from repl.state import new_state
from repl.tool_registry import make_execute_tool


@pytest.fixture
def env_allow_outside(monkeypatch):
    monkeypatch.setenv("MYAH_ALLOW_OUTSIDE_CWD", "1")


@pytest.fixture
def state_in(tmp_path, env_allow_outside):
    """Build a fresh State whose cwd is rooted at `tmp_path`."""
    s = new_state()
    s["cwd"] = str(tmp_path)
    return s


def test_read_file_relative_path_resolves_against_state_cwd(state_in, tmp_path):
    """Relative path passed to read_file goes through resolve_against and
    lands in state["cwd"], not the process cwd. This is the bug the
    cd-wiring fixes."""
    target = tmp_path / "hello.txt"
    target.write_text("hi")

    execute_tool = make_execute_tool(state_in)
    result = execute_tool("read_file", {"path": "hello.txt"})

    assert "hi" in result


def test_write_file_relative_path_resolves_against_state_cwd(state_in, tmp_path):
    execute_tool = make_execute_tool(state_in)
    result = execute_tool("write_file", {"path": "out.txt", "content": "wrote"})

    assert "successfully" in result
    assert (tmp_path / "out.txt").read_text() == "wrote"


def test_edit_file_relative_path_resolves_against_state_cwd(state_in, tmp_path):
    target = tmp_path / "edit_me.txt"
    target.write_text("before")

    execute_tool = make_execute_tool(state_in)
    result = execute_tool("edit_file", {
        "path": "edit_me.txt",
        "old_string": "before",
        "new_string": "after",
    })

    assert "successfully" in result.lower() or "edited" in result.lower() or "replaced" in result.lower()
    assert target.read_text() == "after"


def test_glob_relative_path_resolves_against_state_cwd(state_in, tmp_path):
    (tmp_path / "alpha.py").write_text("")
    (tmp_path / "beta.py").write_text("")

    execute_tool = make_execute_tool(state_in)
    result = execute_tool("glob", {"pattern": "*.py"})

    assert "alpha.py" in result
    assert "beta.py" in result


def test_grep_relative_path_resolves_against_state_cwd(state_in, tmp_path):
    (tmp_path / "needle.txt").write_text("findme please\n")
    (tmp_path / "haystack.txt").write_text("nothing here\n")

    execute_tool = make_execute_tool(state_in)
    result = execute_tool("grep", {"pattern": "findme"})

    assert "needle.txt" in result


def test_absolute_path_overrides_state_cwd(state_in, tmp_path):
    """Absolute paths bypass state cwd — the model meant exactly that
    path. This documents the intended behavior."""
    sibling = tmp_path.parent / (tmp_path.name + "_sibling")
    sibling.mkdir(exist_ok=True)
    target = sibling / "x.txt"
    target.write_text("absolute")

    execute_tool = make_execute_tool(state_in)
    result = execute_tool("read_file", {"path": str(target)})

    assert "absolute" in result


def test_pwd_returns_state_cwd_not_process_cwd(state_in, tmp_path):
    execute_tool = make_execute_tool(state_in)
    result = execute_tool("pwd", {})

    assert result == str(tmp_path)
    # And it's distinct from the process cwd.
    assert os.path.realpath(result) != os.path.realpath(os.getcwd())


def test_cd_then_read_file_uses_new_cwd(state_in, tmp_path):
    """End-to-end: model cd's into a subdir, then reads a relative file.
    Before this fix, the read would silently look in the process cwd."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "inside.txt").write_text("found-it")

    execute_tool = make_execute_tool(state_in)

    cd_result = execute_tool("cd", {"path": "sub"})
    assert cd_result == str(sub)
    assert state_in["cwd"] == str(sub)

    read_result = execute_tool("read_file", {"path": "inside.txt"})
    assert "found-it" in read_result


def test_bash_uses_state_cwd_when_arg_omitted(state_in, tmp_path):
    """bash without an explicit cwd arg runs in state["cwd"]."""
    execute_tool = make_execute_tool(state_in)
    result = execute_tool("bash", {"command": "pwd"})

    # macOS prepends /private to /var/... realpaths, so compare with realpath.
    assert os.path.realpath(str(tmp_path)) in os.path.realpath(result.split("\n")[0]) or \
           str(tmp_path) in result
