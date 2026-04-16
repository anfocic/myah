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
