"""Unit tests for the git tool suite: git_status, git_diff, git_log, git_branch_list.

Uses the same subprocess-mocking pattern as test_git_checkout.py so tests
stay fast and repo-independent.
"""
import subprocess
from types import SimpleNamespace

import pytest

from tools import git as git_mod
from tools.git import (
    git_branch_list,
    git_diff,
    git_log,
    git_status,
)


# ---------------------------------------------------------------------------
# Shared mock infrastructure
# ---------------------------------------------------------------------------


class GitMocker:
    """Controls the outcome of _git() subprocess calls made by each tool.

    Set `result` to a dict with keys: returncode, stdout, stderr.
    `exc` can be set to an exception to raise instead of returning a result.
    `calls` records every argv list passed to subprocess.run.
    """

    def __init__(self, monkeypatch):
        self.calls = []
        self.result = {"returncode": 0, "stdout": "", "stderr": ""}
        self.exc: Exception | None = None

        def fake_run(argv, **kwargs):
            self.calls.append(argv)
            if self.exc:
                raise self.exc
            return SimpleNamespace(**self.result)

        monkeypatch.setattr(git_mod.subprocess, "run", fake_run)


# ---------------------------------------------------------------------------
# git_status
# ---------------------------------------------------------------------------

def test_git_status_porcelain_default(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {"returncode": 0, "stdout": "M  foo.py\n?? untracked.txt\n", "stderr": ""}
    out = git_status()
    assert "M  foo.py" in out
    assert "?? untracked.txt" in out
    assert m.calls[-1] == ["git", "status", "--porcelain"]


def test_git_status_non_porcelain(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {"returncode": 0, "stdout": "On branch main\n", "stderr": ""}
    out = git_status(porcelain=False)
    assert "On branch main" in out
    assert m.calls[-1] == ["git", "status"]


def test_git_status_clean_tree_returns_paren_message(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {"returncode": 0, "stdout": "", "stderr": ""}
    out = git_status()
    assert out == "(clean working tree)"


def test_git_status_failure_returns_error(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {"returncode": 128, "stdout": "", "stderr": "fatal: not a git repository"}
    out = git_status()
    assert "git status failed" in out
    assert "not a git repository" in out


def test_git_status_timeout_raises(monkeypatch):
    m = GitMocker(monkeypatch)
    m.exc = subprocess.TimeoutExpired(cmd="git status", timeout=10)
    with pytest.raises(TimeoutError) as exc_info:
        git_status()
    assert "timed out" in str(exc_info.value)


def test_git_status_git_not_found_raises(monkeypatch):
    m = GitMocker(monkeypatch)
    m.exc = FileNotFoundError("git")
    with pytest.raises(RuntimeError) as exc_info:
        git_status()
    assert "git is not installed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# git_diff
# ---------------------------------------------------------------------------

def test_git_diff_worktree_default(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {"returncode": 0, "stdout": "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n", "stderr": ""}
    out = git_diff()
    assert m.calls[-1] == ["git", "diff", "--no-color", "--no-ext-diff"]
    assert "--- a/foo.py" in out


def test_git_diff_index(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {"returncode": 0, "stdout": "--- a/foo.py\n+++ b/foo.py\n", "stderr": ""}
    out = git_diff(target="index")
    assert "--cached" in m.calls[-1]


def test_git_diff_commit_requires_ref(monkeypatch):
    out = git_diff(target="commit")
    assert "requires a ref argument" in out
    # No subprocess calls should have been made
    # (the guard is before the _git call)


def test_git_diff_commit_with_ref(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {"returncode": 0, "stdout": "commit abc123\ndiff --git a/foo.py b/foo.py\n", "stderr": ""}
    out = git_diff(target="commit", ref="abc123")
    assert "abc123" in m.calls[-1]
    assert "abc123" in out


def test_git_diff_no_diff_returns_paren_message(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {"returncode": 0, "stdout": "", "stderr": ""}
    out = git_diff()
    assert out == "(no differences)"


def test_git_diff_failure_returns_error(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {"returncode": 128, "stdout": "", "stderr": "bad revision 'nope'"}
    out = git_diff()
    assert "git diff failed" in out


def test_git_diff_timeout_raises(monkeypatch):
    m = GitMocker(monkeypatch)
    m.exc = subprocess.TimeoutExpired(cmd="git diff", timeout=10)
    with pytest.raises(TimeoutError) as exc_info:
        git_diff()
    assert "timed out" in str(exc_info.value)


# ---------------------------------------------------------------------------
# git_log
# ---------------------------------------------------------------------------

def test_git_log_default_limit_10(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {
        "returncode": 0,
        "stdout": "abc123 Fix bug\ndef456 Add feature\n",
        "stderr": "",
    }
    out = git_log()
    assert "-n10" in m.calls[-1]
    assert "Fix bug" in out


def test_git_log_respects_limit_arg(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {"returncode": 0, "stdout": "abc123\n", "stderr": ""}
    git_log(limit=5)
    assert "-n5" in m.calls[-1]


def test_git_log_caps_limit_at_100(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {"returncode": 0, "stdout": "", "stderr": ""}
    git_log(limit=500)
    assert "-n100" in m.calls[-1]


def test_git_log_with_stat(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {
        "returncode": 0,
        "stdout": "abc123 Fix bug\n 1 file changed, 5 insertions(+)\n",
        "stderr": "",
    }
    out = git_log(limit=3, with_stat=True)
    assert "--stat" in m.calls[-1]
    assert "file changed" in out


def test_git_log_no_commits_returns_paren_message(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {"returncode": 0, "stdout": "", "stderr": ""}
    out = git_log()
    assert out == "(no commits)"


def test_git_log_failure_returns_error(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {"returncode": 128, "stdout": "", "stderr": "fatal: bad revision 'nope'"}
    out = git_log()
    assert "git log failed" in out


def test_git_log_timeout_raises(monkeypatch):
    m = GitMocker(monkeypatch)
    m.exc = subprocess.TimeoutExpired(cmd="git log", timeout=10)
    with pytest.raises(TimeoutError) as exc_info:
        git_log()
    assert "timed out" in str(exc_info.value)


# ---------------------------------------------------------------------------
# git_branch_list
# ---------------------------------------------------------------------------

def test_git_branch_list_local_default(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {
        "returncode": 0,
        "stdout": "  main    abc123 Fix bug\n  feat/x def456 Add feature\n",
        "stderr": "",
    }
    out = git_branch_list()
    assert "main" in out
    assert "feat/x" in out
    assert "-r" not in m.calls[-1]  # not remote
    assert "-v" in m.calls[-1]  # verbose


def test_git_branch_list_remote(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {
        "returncode": 0,
        "stdout": "  origin/main abc123 Update README\n",
        "stderr": "",
    }
    out = git_branch_list(remote=True)
    assert "origin/main" in out
    assert "-r" in m.calls[-1]


def test_git_branch_list_no_local_branches(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {"returncode": 0, "stdout": "", "stderr": ""}
    out = git_branch_list()
    assert out == "(no local branches)"


def test_git_branch_list_no_remote_branches(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {"returncode": 0, "stdout": "", "stderr": ""}
    out = git_branch_list(remote=True)
    assert out == "(no remote-tracking branches)"


def test_git_branch_list_failure_returns_error(monkeypatch):
    m = GitMocker(monkeypatch)
    m.result = {"returncode": 128, "stdout": "", "stderr": "fatal: not a git repository"}
    out = git_branch_list()
    assert "git branch failed" in out


def test_git_branch_list_timeout_raises(monkeypatch):
    m = GitMocker(monkeypatch)
    m.exc = subprocess.TimeoutExpired(cmd="git branch", timeout=10)
    with pytest.raises(TimeoutError) as exc_info:
        git_branch_list()
    assert "timed out" in str(exc_info.value)
