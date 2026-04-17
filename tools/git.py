# tools/git.py
"""Narrow git wrappers. Separate from `bash` so small models can pick them
by name — `git_checkout` is much more discoverable than "remember you could
shell out". Added because qwen2.5:7b kept narrating branch switches in text
instead of invoking `bash(git checkout ...)`."""
import subprocess


def git_checkout(branch: str) -> str:
    """Run `git checkout <branch>`. Reports the transition so the model can
    confirm the state change. Errors (unknown branch, dirty tree blocking
    checkout) come back as tool output for the model to recover from."""
    try:
        current = subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip() or "(detached HEAD)"
    except (subprocess.SubprocessError, FileNotFoundError):
        current = "unknown"

    try:
        result = subprocess.run(
            ["git", "checkout", branch],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return "git checkout timed out"
    except FileNotFoundError:
        return "git is not installed"

    if result.returncode != 0:
        msg = (result.stderr or result.stdout).strip() or "unknown error"
        return f"git checkout failed: {msg}"
    return f"Switched from {current} to {branch}"
