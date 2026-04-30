"""Unit tests for tools/cd.py — pwd, cd, and resolve_against.

These test the pure logic of cd/pwd with a fake state getter/setter.
No filesystem manipulation needed — the state closures are the interface.
resolve_against is tested separately as the bridge that lets file/git/bash
tools honor the harness cwd.
"""
from __future__ import annotations

import os

import pytest

from tools.cd import cd, pwd, resolve_against


@pytest.fixture
def fake_state():
    """A dict that starts at the current OS cwd, mimicking a fresh REPL state."""
    return {"cwd": os.getcwd()}


@pytest.fixture
def getters(fake_state):
    """Return (getter, setter) lambdas for the fake state."""
    def getter():
        return fake_state["cwd"]

    def setter(new_cwd):
        fake_state["cwd"] = new_cwd

    return getter, setter


# ---------- pwd ----------

def test_pwd_returns_cwd(getters):
    getter, _ = getters
    assert pwd(getter) == os.getcwd()


def test_pwd_returns_absolute_path(getters):
    getter, _ = getters
    result = pwd(getter)
    assert os.path.isabs(result)


# ---------- cd ----------

def test_cd_to_subdir(getters, fake_state, tmp_path):
    getter, setter = getters
    sub = tmp_path / "sub"
    sub.mkdir()
    result = cd(getter, setter, str(sub))
    assert result == str(sub)
    assert fake_state["cwd"] == str(sub)


def test_cd_dot_means_noop(getters, fake_state, tmp_path):
    getter, setter = getters
    fake_state["cwd"] = str(tmp_path)
    result = cd(getter, setter, ".")
    # Returns the path but doesn't call setter (no-op).
    assert result == str(tmp_path)


def test_cd_dotdot_parent(getters, fake_state, tmp_path):
    getter, setter = getters
    sub = tmp_path / "child"
    sub.mkdir()
    fake_state["cwd"] = str(sub)
    result = cd(getter, setter, "..")
    assert result == str(tmp_path)
    assert fake_state["cwd"] == str(tmp_path)


def test_cd_nonexistent_rejected(getters, fake_state, tmp_path):
    getter, setter = getters
    fake_state["cwd"] = str(tmp_path)
    result = cd(getter, setter, "does_not_exist")
    assert "No such file or directory" in result
    # State unchanged.
    assert fake_state["cwd"] == str(tmp_path)


def test_cd_can_navigate_anywhere_in_harness_cwd(getters, fake_state, tmp_path):
    """cd itself has no boundary -- it tracks wherever the harness cwd is.

    The security boundary is enforced at file-tool call time (is_within_cwd
    checks the real os.getcwd()), not at cd time. This test documents
    that cd can move the harness cwd freely within the process tree.
    """
    getter, setter = getters
    child = tmp_path / "child"
    child.mkdir()
    fake_state["cwd"] = str(tmp_path)
    result = cd(getter, setter, "child")
    assert result == str(child)
    assert fake_state["cwd"] == str(child)


def test_cd_to_file_rejected(getters, fake_state, tmp_path):
    getter, setter = getters
    fake_state["cwd"] = str(tmp_path)
    file_path = tmp_path / "afile.txt"
    file_path.write_text("content")
    result = cd(getter, setter, "afile.txt")
    assert "Not a directory" in result
    # State unchanged.
    assert fake_state["cwd"] == str(tmp_path)


def test_cd_empty_path_rejected(getters, fake_state, tmp_path):
    getter, setter = getters
    fake_state["cwd"] = str(tmp_path)
    result = cd(getter, setter, "")
    assert "must not be empty" in result
    assert fake_state["cwd"] == str(tmp_path)


def test_cd_result_is_the_new_absolute_path(getters, fake_state, tmp_path):
    getter, setter = getters
    sub = tmp_path / "nested" / "deep"
    sub.mkdir(parents=True)
    fake_state["cwd"] = str(tmp_path)
    result = cd(getter, setter, "nested/deep")
    assert os.path.isabs(result)
    assert result == str(sub)


def test_cd_updates_state(getters, fake_state, tmp_path):
    getter, setter = getters
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    fake_state["cwd"] = str(a)
    cd(getter, setter, "../b")
    assert fake_state["cwd"] == str(b)


# ---------- resolve_against ----------

def test_resolve_against_relative_joined_to_cwd(tmp_path):
    """The whole point: relative path resolves against the supplied cwd,
    not against the process cwd. This is what makes the model's `cd`
    movement matter for downstream tool calls."""
    result = resolve_against(str(tmp_path), "foo.py")
    assert result == os.path.join(str(tmp_path), "foo.py")


def test_resolve_against_absolute_passes_through(tmp_path):
    """Absolute paths are returned unchanged (after expanduser) — the
    model meant exactly that path, not relative to cwd."""
    abs_path = str(tmp_path / "abs.py")
    result = resolve_against("/some/other/cwd", abs_path)
    assert result == abs_path


def test_resolve_against_dotdot_collapsed(tmp_path):
    """normpath collapses .. so the result is clean for downstream
    is_within_cwd / open() calls."""
    sub = tmp_path / "child"
    result = resolve_against(str(sub), "../sibling/foo.py")
    assert result == str(tmp_path / "sibling" / "foo.py")


def test_resolve_against_expands_tilde():
    """~ expansion produces an absolute path under the user home, which
    is then returned unchanged because it became absolute."""
    result = resolve_against("/whatever", "~/foo.py")
    assert result.startswith(os.path.expanduser("~"))
    assert result.endswith("foo.py")


def test_resolve_against_dot_returns_cwd(tmp_path):
    """`.` resolves to cwd itself — the form glob/grep use as default."""
    result = resolve_against(str(tmp_path), ".")
    assert os.path.normpath(result) == os.path.normpath(str(tmp_path))
