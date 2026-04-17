# tools/files.py
import os

DEFAULT_READ_LINES = 1000
MAX_LINE_CHARS = 500


def read_file(path: str, offset: int = 1, limit: int | None = None):
    """Read a file and return it with line numbers prepended (cat -n style).

    offset: 1-indexed line to start from. Default 1 (top of file).
    limit: max lines to return. Default DEFAULT_READ_LINES; use offset+limit
        to paginate through files that exceed it.

    Line numbers let the model answer "what's on line N?" without having to
    count newlines in its head (small models are bad at that — they fall
    back to grep and hallucinate). Individual lines over MAX_LINE_CHARS are
    truncated so one pathological log line can't blow the context window.
    """
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


def write_file(path: str, content: str):
    try:
        with open(os.path.expanduser(path), "w") as f:
            f.write(content)
        return f"File written successfully: {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False):
    """Surgical string replacement. Refuses ambiguous edits unless replace_all=True.

    Mirrors the Claude Code `Edit` tool: the model must produce an `old_string`
    that uniquely identifies the target, otherwise the edit is rejected.
    """
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
    tmp_path = expanded + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            f.write(new_content)
        os.replace(tmp_path, expanded)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return f"Error writing file: {str(e)}"

    replaced = count if replace_all else 1
    return f"Edited {path}: {replaced} replacement(s)"
