"""git_checkout tool — guards + subprocess behavior.

Dash-prefix guard catches the most common injection shape (`-f`,
`--force`, `--orphan`). The subprocess paths are mocked so tests stay
fast and don't require a real git repo with the right set of branches."""
import subprocess
from types import SimpleNamespace

import pytest

from tools import git as git_mod
from tools.git import git_checkout

# ---------- guards ----------

def test_rejects_dash_prefix_branch():
    result = git_checkout("-f")
    assert result.startswith("Refusing")
    assert "-f" in result


def test_rejects_double_dash_prefix_branch():
    result = git_checkout("--force")
    assert result.startswith("Refusing")


def test_rejects_empty_branch():
    result = git_checkout("")
    assert result.startswith("Refusing")


# ---------- mocked subprocess paths ----------

@pytest.fixture
def mock_git(monkeypatch):
    """Control the outcome of the two subprocess calls git_checkout makes.

    `call_output(branch)`  → what `git branch --show-current` returns.
    `call_run(returncode, stdout, stderr)` → what `git checkout` produces.
    `raise_run(exc)` → stub `subprocess.run` to raise the exception.
    """
    state = {
        "current": "main",
        "run_returncode": 0,
        "run_stdout": "",
        "run_stderr": "",
        "run_raises": None,
    }

    def fake_check_output(argv, **kwargs):
        # git_checkout calls check_output once for `branch --show-current`.
        assert argv[:3] == ["git", "branch", "--show-current"]
        return state["current"] + "\n"

    def fake_run(argv, **kwargs):
        if state["run_raises"] is not None:
            raise state["run_raises"]
        return SimpleNamespace(
            returncode=state["run_returncode"],
            stdout=state["run_stdout"],
            stderr=state["run_stderr"],
        )

    monkeypatch.setattr(git_mod.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(git_mod.subprocess, "run", fake_run)
    return state


def test_success_reports_transition(mock_git):
    mock_git["current"] = "feat/x"
    mock_git["run_returncode"] = 0
    out = git_checkout("main")
    assert out == "Switched from feat/x to main"


def test_success_with_detached_head_shows_placeholder(mock_git):
    mock_git["current"] = ""  # detached HEAD → `--show-current` empty
    out = git_checkout("main")
    assert "(detached HEAD)" in out
    assert "to main" in out


def test_unknown_branch_surfaces_stderr(mock_git):
    mock_git["run_returncode"] = 1
    mock_git["run_stderr"] = "error: pathspec 'nope' did not match any file(s) known to git"
    out = git_checkout("nope")
    assert "git checkout failed" in out
    assert "nope" in out


def test_dirty_tree_error_surfaces_message(mock_git):
    mock_git["run_returncode"] = 1
    mock_git["run_stderr"] = (
        "error: Your local changes to the following files would be "
        "overwritten by checkout"
    )
    out = git_checkout("main")
    assert "git checkout failed" in out
    assert "local changes" in out


def test_falls_back_to_stdout_when_stderr_empty(mock_git):
    mock_git["run_returncode"] = 1
    mock_git["run_stderr"] = ""
    mock_git["run_stdout"] = "some stdout diagnostic"
    out = git_checkout("main")
    assert "some stdout diagnostic" in out


def test_timeout_on_checkout_returns_clear_message(mock_git):
    mock_git["run_raises"] = subprocess.TimeoutExpired(cmd="git checkout", timeout=10)
    out = git_checkout("main")
    assert "timed out" in out.lower()


def test_git_not_installed_returns_clear_message(mock_git):
    mock_git["run_raises"] = FileNotFoundError("git")
    out = git_checkout("main")
    assert "git is not installed" in out


def test_show_current_failure_falls_back_to_unknown(monkeypatch):
    """If `git branch --show-current` fails for any reason, the success
    message still gets assembled — just with 'unknown' as the prior
    branch instead of crashing."""
    def raising_check_output(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=128, cmd=args[0])

    def fake_run(argv, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(git_mod.subprocess, "check_output", raising_check_output)
    monkeypatch.setattr(git_mod.subprocess, "run", fake_run)

    out = git_checkout("main")
    assert "Switched from unknown to main" == out
