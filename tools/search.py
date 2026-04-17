# tools/search.py
import os
import re
from pathlib import Path

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
