# tools/search.py
import os
import re
from pathlib import Path

from security import is_within_cwd, refuse_outside_cwd
from tools.spec import register

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "logs", "node_modules"}
MAX_FILE_BYTES = 1_000_000
MAX_RESULTS = 50


def _should_skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.startswith(".")


def _iter_files(root: Path, glob: str | None):
    pattern = glob or "*"
    for p in root.rglob(pattern):
        if not p.is_file():
            continue
        if any(_should_skip_dir(part) for part in p.relative_to(root).parts[:-1]):
            continue
        if not is_within_cwd(str(p)):
            continue
        yield p


def grep(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    output_mode: str = "files_with_matches",
):
    """Regex search over files. stdlib-only.

    output_mode: "files_with_matches" (list of paths) or "content" (path:line:text).
    Caps results at MAX_RESULTS, skips binaries, dotdirs, and files > 1MB.
    """
    if not is_within_cwd(path):
        return refuse_outside_cwd(path)

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"

    root = Path(os.path.expanduser(path)).resolve()
    if not root.exists():
        return f"Path not found: {path}"
    if root.is_file():
        files = [root]
    else:
        files = list(_iter_files(root, glob))

    hits: list[str] = []
    truncated = False

    for fp in files:
        try:
            if fp.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue

        try:
            text = fp.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        rel = fp.relative_to(root) if root.is_dir() else fp.name

        if output_mode == "files_with_matches":
            if regex.search(text):
                hits.append(str(rel))
                if len(hits) >= MAX_RESULTS:
                    truncated = True
                    break
        else:  # "content"
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{rel}:{lineno}:{line}")
                    if len(hits) >= MAX_RESULTS:
                        truncated = True
                        break
            if truncated:
                break

    if not hits:
        return "No matches."

    out = "\n".join(hits)
    if truncated:
        out += f"\n... (truncated at {MAX_RESULTS} results)"
    return out


def glob(pattern: str, path: str = "."):
    """Find files matching a glob pattern, recursively.

    Accepts bare filenames ('search.py'), simple globs ('*.py'), or recursive
    globs ('**/*.md'). Sorted output for determinism. Skips the same dirs grep
    skips (.git, venv, logs, etc.) so the model doesn't get drowned in noise.

    Primary use: the model sees a bare filename in the user's message and
    needs to resolve it to a full path before calling read_file or edit_file.
    """
    if not is_within_cwd(path):
        return refuse_outside_cwd(path)

    root = Path(os.path.expanduser(path)).resolve()
    if not root.exists():
        return f"Path not found: {path}"
    if root.is_file():
        return str(root.name)

    matches: list[str] = []
    truncated = False
    # Bare filenames with no glob char should match anywhere under root —
    # prepend '**/' so 'search.py' behaves like the user expects.
    effective = pattern
    if not any(c in pattern for c in "*?[") and "/" not in pattern:
        effective = f"**/{pattern}"

    for p in sorted(root.rglob(effective)):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(root).parts
        if any(_should_skip_dir(part) for part in rel_parts[:-1]):
            continue
        if not is_within_cwd(str(p)):
            continue
        matches.append("/".join(rel_parts))
        if len(matches) >= MAX_RESULTS:
            truncated = True
            break

    if not matches:
        return f"No files matching {pattern!r}"

    out = "\n".join(matches)
    if truncated:
        out += f"\n... (truncated at {MAX_RESULTS} results)"
    return out


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def _glob_adapter(args: dict, cwd: str):
    from tools.cd import resolve_against
    return glob(
        args["pattern"],
        resolve_against(cwd, args.get("path", ".")),
    )


register(
    name="glob",
    description="Find files by name or glob pattern, recursively. Use this to resolve a bare filename (e.g. 'search.py') to its full path before reading or editing. Accepts 'search.py', '*.py', or '**/*.md'.",
    adapter=_glob_adapter,
    properties={
        "pattern": {"type": "string", "description": "Filename or glob pattern to match"},
        "path": {
            "type": "string",
            "description": "Directory to search from. Defaults to current working directory.",
        },
    },
    required=["pattern"],
    read_only=True,
)


def _grep_adapter(args: dict, cwd: str):
    from tools.cd import resolve_against
    return grep(
        args["pattern"],
        resolve_against(cwd, args.get("path", ".")),
        args.get("glob"),
        args.get("output_mode", "files_with_matches"),
    )


register(
    name="grep",
    description="Regex search across files under a path. Returns matching file paths by default, or path:line:text when output_mode is 'content'.",
    adapter=_grep_adapter,
    properties={
        "pattern": {"type": "string", "description": "Python regex to search for"},
        "path": {
            "type": "string",
            "description": "Directory or file to search. Defaults to the current working directory.",
        },
        "glob": {
            "type": "string",
            "description": "Optional glob filter like '*.py' or '**/*.md'",
        },
        "output_mode": {
            "type": "string",
            "description": "'files_with_matches' (default) or 'content'",
        },
    },
    required=["pattern"],
    read_only=True,
)
