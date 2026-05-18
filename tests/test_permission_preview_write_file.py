"""Permission preview for `write_file`: show a unified diff against the
existing file when overwriting, and indicate create vs overwrite mode in
the metadata line."""
from io import StringIO
from pathlib import Path

from rich.console import Console

import permissions


def _capture_preview(name: str, args: dict) -> str:
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    permissions._print_permission_preview(console, name, args)
    return buf.getvalue()


def test_write_file_create_mode_shows_content_preview(tmp_path: Path):
    target = tmp_path / "new.txt"
    out = _capture_preview("write_file", {"path": str(target), "content": "hello"})
    assert "mode" in out and "create" in out
    assert "content preview" in out
    assert "hello" in out
    # No diff for new file.
    assert "diff preview" not in out


def test_write_file_overwrite_mode_shows_diff(tmp_path: Path):
    target = tmp_path / "doc.txt"
    target.write_text("line1\nline2\n")
    out = _capture_preview(
        "write_file", {"path": str(target), "content": "line1\nLINE-2\n"},
    )
    assert "mode" in out and "overwrite" in out
    assert "diff preview" in out
    # `LINE-2` is the added line in the diff.
    assert "LINE-2" in out
    # No raw content preview when we have a diff.
    assert "content preview" not in out


def test_write_file_overwrite_identical_content_says_unchanged(tmp_path: Path):
    target = tmp_path / "doc.txt"
    target.write_text("same\n")
    out = _capture_preview(
        "write_file", {"path": str(target), "content": "same\n"},
    )
    assert "overwrite" in out
    assert "unchanged" in out


def test_write_file_size_and_line_count_shown(tmp_path: Path):
    target = tmp_path / "new.txt"
    out = _capture_preview(
        "write_file", {"path": str(target), "content": "one\ntwo\nthree\n"},
    )
    assert "14 bytes" in out  # "one\ntwo\nthree\n" = 14 bytes
    assert "lines" in out
