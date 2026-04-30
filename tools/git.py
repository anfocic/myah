# tools/git.py
"""Narrow git wrappers. Separate from `bash` so small models can pick them
by name — `git_checkout` is much more discoverable than "remember you could
shell out". Added because qwen2.5:7b kept narrating branch switches in text
instead of invoking `bash(git checkout ...)`.

All tools here are intentionally read-only (no `git reset`, `git rebase`,
`git commit --amend`, etc.). The harness exposes mutation through named,
user-approved channels only."""
import subprocess
from typing import Literal

_subprocess_timeout = 10  # seconds for all git subprocess calls


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _git(
    argv: list[str],
    cwd: str | None = None,
    timeout: int = _subprocess_timeout,
) -> subprocess.CompletedProcess:
    """Run a git subprocess and return the result. Raises on missing git.

    `cwd` selects the directory git runs in. None = inherit the process
    cwd; the harness passes state["cwd"] through so the model's `cd`
    movement is honored when these tools are invoked via the dispatcher."""
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"git {' '.join(argv[1:])} timed out after {timeout}s")
    except FileNotFoundError:
        raise RuntimeError("git is not installed")


# ---------------------------------------------------------------------------
# git_checkout
# ---------------------------------------------------------------------------


def git_checkout(branch: str, cwd: str | None = None) -> str:
    """Run `git checkout <branch>`. Reports the transition so the model can
    confirm the state change. Errors (unknown branch, dirty tree blocking
    checkout) come back as tool output for the model to recover from."""
    # argv-safe against shell injection (no shell=True), but git itself still
    # parses a leading "-" as a flag. Reject those before they become
    # accidental `git checkout -f` / `--orphan` invocations.
    if not branch or branch.startswith("-"):
        return f"Refusing to check out suspicious branch name: {branch!r}"

    try:
        current = subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            cwd=cwd,
        ).strip() or "(detached HEAD)"
    except (subprocess.SubprocessError, FileNotFoundError):
        current = "unknown"

    try:
        result = subprocess.run(
            ["git", "checkout", branch],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return "git checkout timed out"
    except FileNotFoundError:
        return "git is not installed"

    if result.returncode != 0:
        msg = (result.stderr or result.stdout).strip() or "unknown error"
        return f"git checkout failed: {msg}"
    return f"Switched from {current} to {branch}"


# ---------------------------------------------------------------------------
# git_status
# ---------------------------------------------------------------------------


def git_status(porcelain: bool = True, cwd: str | None = None) -> str:
    """Return the output of `git status`.

    By default uses porcelain mode (machine-parseable, no ANSI). Pass
    ``porcelain=False`` for the full human-readable format."""
    flag = "--porcelain" if porcelain else "--no-porcelain"
    result = _git(["git", "status", flag], cwd=cwd)
    if result.returncode != 0:
        return f"git status failed: {(result.stderr or result.stdout).strip()}"
    output = result.stdout
    if not output.strip():
        return "(clean working tree)"
    return output


# ---------------------------------------------------------------------------
# git_diff
# ---------------------------------------------------------------------------


def git_diff(
    target: Literal["worktree", "index", "commit"] = "worktree",
    ref: str = "",
    no_color: bool = True,
    cwd: str | None = None,
) -> str:
    """Return the output of `git diff` for the specified target.

    target — controls what is diffed:
      worktree  — unstaged changes vs the index (default)
      index     — staged changes vs HEAD
      commit    — changes introduced by a commit (ref is required)

    ref — for target='commit' this names the commit to inspect.
          For target='worktree' or 'index', ref is not applicable and
          is rejected to prevent a common semantic confusion (passing a ref
          to a worktree diff does NOT show "worktree vs that ref").

    ref is required when target is ``commit``; otherwise it is rejected."""
    if target == "commit":
        if not ref:
            return "git_diff with target='commit' requires a ref argument"
        argv = ["git", "diff", "--no-color", "--no-ext-diff", ref]
    elif target == "index":
        if ref:
            return "git_diff: ref is only valid when target='commit'; for 'worktree' or 'index' omit it"
        argv = ["git", "diff", "--cached", "--no-color", "--no-ext-diff"]
    else:
        if ref:
            return "git_diff: ref is only valid when target='commit'; for 'worktree' or 'index' omit it"
        argv = ["git", "diff", "--no-color", "--no-ext-diff"]

    result = _git(argv, cwd=cwd)
    if result.returncode != 0:
        return f"git diff failed: {(result.stderr or result.stdout).strip()}"
    output = result.stdout
    if not output.strip():
        return "(no differences)"
    return output


# ---------------------------------------------------------------------------
# git_log
# ---------------------------------------------------------------------------


def git_log(
    limit: int = 10,
    no_color: bool = True,
    with_stat: bool = False,
    cwd: str | None = None,
) -> str:
    """Return the output of `git log`.

    limit — number of commits to show (default 10, capped at 100).
    with_stat — if True, include file change statistics per commit."""
    limit = max(1, min(limit, 100))
    argv = ["git", "log", f"-n{limit}"]
    if with_stat:
        argv.append("--stat")
    if no_color:
        argv.extend(["--no-color", "--no-ext-diff"])
    result = _git(argv, cwd=cwd)
    if result.returncode != 0:
        return f"git log failed: {(result.stderr or result.stdout).strip()}"
    output = result.stdout
    if not output.strip():
        return "(no commits)"
    return output


# ---------------------------------------------------------------------------
# git_branch_list
# ---------------------------------------------------------------------------


def git_branch_list(
    remote: bool = False,
    verbose: bool = True,
    cwd: str | None = None,
) -> str:
    """Return the output of `git branch`.

    remote — if True, list remote-tracking branches instead of local ones.
    verbose — if True, show the last commit message and tracking info."""
    argv = ["git", "branch"]
    if remote:
        argv.append("-r")
    if verbose:
        argv.append("-v")
    result = _git(argv, cwd=cwd)
    if result.returncode != 0:
        return f"git branch failed: {(result.stderr or result.stdout).strip()}"
    output = result.stdout
    if not output.strip():
        if remote:
            return "(no remote-tracking branches)"
        return "(no local branches)"
    return output
