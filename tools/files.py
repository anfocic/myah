# tools/files.py
import os


def read_file(path: str):
    try:
        with open(os.path.expanduser(path), "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


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
        with open(expanded, "r") as f:
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
