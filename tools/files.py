# tools/files.py
import os
import tempfile

from security import is_within_cwd, refuse_outside_cwd
from tools.spec import register

DEFAULT_READ_LINES = 1000
MAX_LINE_CHARS = 500


def read_file(path: str, offset: int = 1, limit: int | None = None) -> str:
    """Read a file and return it with line numbers prepended (cat -n style).

    offset: 1-indexed line to start from. Default 1 (top of file).
    limit: max lines to return. Default DEFAULT_READ_LINES; use offset+limit
        to paginate through files that exceed it.

    Line numbers let the model answer "what's on line N?" without having to
    count newlines in its head (small models are bad at that — they fall
    back to grep and hallucinate). Individual lines over MAX_LINE_CHARS are
    truncated so one pathological log line can't blow the context window.
    """
    if not is_within_cwd(path):
        return refuse_outside_cwd(path)
    try:
        with open(os.path.expanduser(path)) as f:
            all_lines = f.readlines()
    except FileNotFoundError:
        return f"File not found: {path}"
    except UnicodeDecodeError:
        return f"Cannot read {path}: binary or non-UTF-8 file"
    except Exception as e:
        return f"Error reading file: {str(e)}"

    total = len(all_lines)
    if total == 0:
        return f"{path} is empty."

    if offset < 1:
        offset = 1
    if offset > total:
        return f"{path} has {total} lines; offset {offset} is past the end."

    take = limit if limit is not None else DEFAULT_READ_LINES
    end = min(offset - 1 + take, total)
    slice_ = all_lines[offset - 1 : end]

    rendered = []
    for i, line in enumerate(slice_, start=offset):
        line = line.rstrip("\n")
        if len(line) > MAX_LINE_CHARS:
            line = line[: MAX_LINE_CHARS - 3] + "..."
        rendered.append(f"{i:>6}\t{line}")

    out = "\n".join(rendered)
    remaining = total - end
    if remaining > 0:
        out += f"\n... ({remaining} more lines; use offset={end + 1} to continue)"
    return out


def write_file(path: str, content: str) -> str:
    if not is_within_cwd(path):
        return refuse_outside_cwd(path)
    try:
        with open(os.path.expanduser(path), "w") as f:
            f.write(content)
        return f"File written successfully: {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Surgical string replacement. Refuses ambiguous edits unless replace_all=True.

    Mirrors the Claude Code `Edit` tool: the model must produce an `old_string`
    that uniquely identifies the target, otherwise the edit is rejected.
    """
    if not is_within_cwd(path):
        return refuse_outside_cwd(path)
    try:
        expanded = os.path.expanduser(path)
        with open(expanded) as f:
            content = f.read()
    except FileNotFoundError:
        return f"File not found: {path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

    count = content.count(old_string)
    if count == 0:
        return f"old_string not found in {path}"
    if count > 1 and not replace_all:
        return (
            f"old_string appears {count} times in {path} — ambiguous. "
            f"Provide more surrounding context to make it unique, or pass replace_all=true."
        )

    new_content = (
        content.replace(old_string, new_string)
        if replace_all
        else content.replace(old_string, new_string, 1)
    )

    # Atomic write: if we're killed mid-write, the original file stays intact.
    # Use `tempfile.NamedTemporaryFile` in the same directory so `os.replace`
    # is guaranteed to be atomic (cross-fs renames are not), and so we get
    # a collision-free temp name — the older `expanded + ".tmp"` scheme
    # would silently clobber an unrelated `foo.py.tmp` left by a prior
    # interrupted edit (or a user's own backup convention).
    parent = os.path.dirname(expanded) or "."
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=parent,
            prefix=".edit_file.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = tmp.name
            tmp.write(new_content)
        os.replace(tmp_path, expanded)
        tmp_path = None  # successfully renamed, nothing to clean up
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return f"Error writing file: {str(e)}"

    replaced = count if replace_all else 1
    return f"Edited {path}: {replaced} replacement(s)"


# ---------------------------------------------------------------------------
# Adapters — bridge from model args (relative paths, strings) to tool fns.
# Each adapter receives the raw args dict and the harness cwd, then resolves
# paths and supplies defaults exactly as the old monolithic dispatcher did.
# ---------------------------------------------------------------------------


def _read_file_adapter(args: dict, cwd: str):
    from tools.cd import resolve_against
    return read_file(
        resolve_against(cwd, args["path"]),
        int(args.get("offset", 1)),
        int(args["limit"]) if "limit" in args else None,
    )


register(
    name="read_file",
    description="Read a file. Returns the contents with line numbers prepended (cat -n style) so you can reference specific lines. By default returns the first 1000 lines; use offset and limit to paginate longer files.",
    adapter=_read_file_adapter,
    properties={
        "path": {"type": "string", "description": "The file path to read"},
        "offset": {
            "type": "integer",
            "description": "1-indexed line to start from. Defaults to 1.",
        },
        "limit": {
            "type": "integer",
            "description": "Max number of lines to return. Defaults to 1000.",
        },
    },
    required=["path"],
    read_only=True,
)


def _write_file_adapter(args: dict, cwd: str):
    from tools.cd import resolve_against
    return write_file(resolve_against(cwd, args["path"]), args["content"])


register(
    name="write_file",
    description="Writes content to a file at the given path. Overwrites the whole file — prefer edit_file for surgical changes.",
    adapter=_write_file_adapter,
    properties={
        "path": {"type": "string", "description": "The file path to write to"},
        "content": {"type": "string", "description": "The content to write"},
    },
    required=["path", "content"],
    read_only=False,
)


def _edit_file_adapter(args: dict, cwd: str):
    from tools.cd import resolve_against
    return edit_file(
        resolve_against(cwd, args["path"]),
        args["old_string"],
        args["new_string"],
        bool(args.get("replace_all", False)),
    )


register(
    name="edit_file",
    description="Surgically replace a string in a file. old_string must uniquely identify the target unless replace_all is true.",
    adapter=_edit_file_adapter,
    properties={
        "path": {"type": "string", "description": "The file path to edit"},
        "old_string": {
            "type": "string",
            "description": "Exact text to find. Must be unique in the file unless replace_all is true.",
        },
        "new_string": {"type": "string", "description": "Replacement text"},
        "replace_all": {
            "type": "boolean",
            "description": "Replace every occurrence instead of requiring uniqueness",
        },
    },
    required=["path", "old_string", "new_string"],
    read_only=False,
)
