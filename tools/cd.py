"""Harness working directory tools: pwd and cd, plus path resolution helper.

Narrow by design: no shell involvement, no permission prompts. These let the
model navigate the workspace without burning a tool call on `bash("cd ...")`
or asking the user to navigate for it.

cd is listed in READ_ONLY_TOOLS (and allowed in plan mode) because the only
state it mutates is the harness's tracked cwd; no files, no external state.

Security boundary: file tools still call is_within_cwd() against the real
os.getcwd() (the process startup directory). cd cannot relax that boundary
-- it can only move within it. A model that cd's outside the boundary will
still see all file reads/writes refused.

resolve_against() is the bridge: relative paths the model passes are joined
to state["cwd"] before file tools see them, so 'foo.py' after `cd subdir`
resolves to <startup>/subdir/foo.py instead of <startup>/foo.py.
"""
import os


def pwd(state_cwd_getter) -> str:
    """Print the harness current working directory.

    Returns the absolute path as a plain string. The harness cwd starts at
    the process cwd at startup and is updated by calls to cd() -- it is
    distinct from os.getcwd() which always reads the live process-level cwd.
    """
    return state_cwd_getter()


def cd(state_cwd_getter, state_cwd_setter, path: str) -> str:
    """Change the harness working directory.

    Resolves <path> relative to the current harness cwd (not the process cwd).
    Supports .. and other relative forms. The harness cwd is purely informational
    -- file tools always check is_within_cwd() against the real os.getcwd()
    at call time, so cd cannot be used to escape the security boundary.

    Returns the new harness cwd on success, or an error message on failure
    (non-existent path or not a directory).
    """
    if not path:
        return "cd: path must not be empty"

    current = state_cwd_getter()
    resolved = os.path.realpath(os.path.join(current, path))

    if not os.path.exists(resolved):
        return f"cd: {path}: No such file or directory"

    if not os.path.isdir(resolved):
        return f"cd: {path}: Not a directory"

    # Silent no-op if already there (avoids spurious state change).
    if resolved == os.path.realpath(current):
        return resolved

    state_cwd_setter(resolved)
    return resolved


def resolve_against(cwd: str, path: str) -> str:
    """Resolve `path` against the harness `cwd`.

    Absolute paths (after expanduser) are returned unchanged. Relative paths
    are joined to `cwd` and normalized (.. collapsed, redundant slashes
    removed). The result is always absolute as long as `cwd` is absolute.

    This is the single bridge between the model's path arguments and the
    file/shell/git tools. By resolving here, every tool sees an absolute
    path that already reflects the model's `cd` movement, so the tools
    themselves stay free of cwd state.
    """
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.normpath(os.path.join(cwd, expanded))
