"""Unit tests for tools/files.py — read_file, write_file, edit_file.

Each test chdir's into a pytest `tmp_path` so `is_within_cwd` accepts the
fixture-file paths. The chdir dance is the price of having a real
security guard; the alternative (injecting a cwd into every tool) would
bleed test-only plumbing into production tools."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.files import DEFAULT_READ_LINES, MAX_LINE_CHARS, edit_file, read_file, write_file


@pytest.fixture
def in_tmp(tmp_path: Path, monkeypatch):
    """Move cwd into tmp_path so security.is_within_cwd accepts the test files."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------- read_file ----------

def test_read_file_returns_numbered_lines(in_tmp: Path):
    (in_tmp / "a.py").write_text("x = 1\ny = 2\nz = 3\n")
    out = read_file("a.py")
    # cat -n style: right-aligned line number + tab + content.
    assert "     1\tx = 1" in out
    assert "     2\ty = 2" in out
    assert "     3\tz = 3" in out


def test_read_file_missing_returns_error_not_raise(in_tmp: Path):
    out = read_file("does_not_exist.py")
    assert out == "File not found: does_not_exist.py"


def test_read_file_empty_returns_note(in_tmp: Path):
    (in_tmp / "empty.py").write_text("")
    assert read_file("empty.py") == "empty.py is empty."


def test_read_file_offset_paginates(in_tmp: Path):
    (in_tmp / "long.py").write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")
    out = read_file("long.py", offset=5, limit=3)
    assert "     5\tline5" in out
    assert "     6\tline6" in out
    assert "     7\tline7" in out
    # Content outside the window is not rendered.
    assert "line4" not in out
    assert "line8" not in out


def test_read_file_offset_past_end_explains(in_tmp: Path):
    (in_tmp / "short.py").write_text("only\n")
    out = read_file("short.py", offset=99)
    assert "has 1 lines" in out
    assert "offset 99" in out


def test_read_file_long_line_truncated(in_tmp: Path):
    big = "a" * (MAX_LINE_CHARS + 50)
    (in_tmp / "wide.py").write_text(big + "\n")
    out = read_file("wide.py")
    # Truncation marker replaces the tail with "..." (3 chars).
    assert "..." in out
    # Length of the content portion (after the tab) is MAX_LINE_CHARS.
    content = out.split("\t", 1)[1]
    assert len(content) == MAX_LINE_CHARS


def test_read_file_remaining_suffix_points_to_next_offset(in_tmp: Path):
    (in_tmp / "paginate.py").write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")
    out = read_file("paginate.py", limit=3)
    # 10 lines total, limit 3 → 7 remaining, next offset = 4.
    assert "7 more lines" in out
    assert "offset=4" in out


def test_read_file_no_remaining_suffix_when_complete(in_tmp: Path):
    (in_tmp / "small.py").write_text("one\ntwo\n")
    out = read_file("small.py")
    assert "more lines" not in out


def test_read_file_default_limit_caps_to_default_read_lines(in_tmp: Path):
    # Write more than DEFAULT_READ_LINES so we can assert the truncation point.
    n = DEFAULT_READ_LINES + 50
    (in_tmp / "huge.py").write_text("\n".join(f"line{i}" for i in range(1, n + 1)) + "\n")
    out = read_file("huge.py")
    # Only DEFAULT_READ_LINES shown; the remaining count should flag the rest.
    assert f"line{DEFAULT_READ_LINES}" in out
    assert f"line{DEFAULT_READ_LINES + 1}" not in out
    assert "more lines" in out


def test_read_file_binary_returns_clear_message(in_tmp: Path):
    (in_tmp / "bin.dat").write_bytes(b"\xff\xfe\xfd\xfc")
    out = read_file("bin.dat")
    assert "Cannot read" in out
    assert "binary or non-UTF-8" in out


def test_read_file_refuses_outside_cwd(in_tmp: Path):
    out = read_file("/etc/passwd")
    assert "Refusing path outside cwd" in out


def test_read_file_negative_offset_treated_as_1(in_tmp: Path):
    (in_tmp / "neg.py").write_text("a\nb\nc\n")
    out = read_file("neg.py", offset=-5)
    assert "     1\ta" in out


# ---------- write_file ----------

def test_write_file_happy_path(in_tmp: Path):
    result = write_file("new.txt", "hello\n")
    assert "written successfully" in result
    assert (in_tmp / "new.txt").read_text() == "hello\n"


def test_write_file_overwrites_existing(in_tmp: Path):
    (in_tmp / "doc.txt").write_text("old content")
    write_file("doc.txt", "new content")
    assert (in_tmp / "doc.txt").read_text() == "new content"


def test_write_file_refuses_outside_cwd(in_tmp: Path):
    out = write_file("/tmp/outside.txt", "nope")
    assert "Refusing path outside cwd" in out


def test_write_file_error_on_permission_denied(in_tmp: Path):
    """A read-only parent dir should surface a useful error, not raise."""
    ro = in_tmp / "readonly"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        out = write_file("readonly/nope.txt", "x")
        assert "Error writing file" in out
    finally:
        # Restore perms so tmp_path cleanup works.
        ro.chmod(0o755)


# ---------- edit_file ----------

def test_edit_file_single_replacement(in_tmp: Path):
    (in_tmp / "a.py").write_text("x = 1\ny = 2\nz = 3\n")
    out = edit_file("a.py", "y = 2", "y = 42")
    assert "1 replacement(s)" in out
    assert (in_tmp / "a.py").read_text() == "x = 1\ny = 42\nz = 3\n"


def test_edit_file_missing_old_string_rejected(in_tmp: Path):
    (in_tmp / "a.py").write_text("x = 1\n")
    out = edit_file("a.py", "not_there", "x")
    assert "not found" in out
    # File unchanged.
    assert (in_tmp / "a.py").read_text() == "x = 1\n"


def test_edit_file_ambiguous_rejected_without_replace_all(in_tmp: Path):
    (in_tmp / "a.py").write_text("foo\nfoo\nbar\n")
    out = edit_file("a.py", "foo", "baz")
    assert "2 times" in out
    assert "replace_all" in out
    # File untouched.
    assert (in_tmp / "a.py").read_text() == "foo\nfoo\nbar\n"


def test_edit_file_replace_all_allows_ambiguous(in_tmp: Path):
    (in_tmp / "a.py").write_text("foo\nfoo\nbar\n")
    out = edit_file("a.py", "foo", "baz", replace_all=True)
    assert "2 replacement(s)" in out
    assert (in_tmp / "a.py").read_text() == "baz\nbaz\nbar\n"


def test_edit_file_missing_file_returns_error(in_tmp: Path):
    out = edit_file("does_not_exist.py", "x", "y")
    assert "File not found" in out


def test_edit_file_refuses_outside_cwd(in_tmp: Path):
    out = edit_file("/etc/hosts", "127.0.0.1", "x")
    assert "Refusing path outside cwd" in out


def test_edit_file_atomic_leaves_no_tmp_on_success(in_tmp: Path):
    (in_tmp / "a.py").write_text("hello\n")
    edit_file("a.py", "hello", "world")
    # Regardless of which tempfile scheme the implementation uses, no
    # `.tmp` or generated temp should linger in the directory after a
    # successful edit.
    leftover = [p.name for p in in_tmp.iterdir() if p.name != "a.py"]
    assert leftover == [], f"unexpected leftover files: {leftover}"


def test_edit_file_pre_existing_tmp_sibling_is_not_clobbered(in_tmp: Path):
    """Regression guard: if the user has an unrelated `foo.py.tmp` sitting
    in the directory (e.g. a backup, or debris from a previous aborted
    edit), editing `foo.py` must not silently overwrite it."""
    (in_tmp / "a.py").write_text("hello\n")
    sentinel = in_tmp / "a.py.tmp"
    sentinel.write_text("precious unrelated content")

    edit_file("a.py", "hello", "world")

    # a.py was updated correctly.
    assert (in_tmp / "a.py").read_text() == "world\n"
    # The sibling sentinel survived.
    assert sentinel.exists()
    assert sentinel.read_text() == "precious unrelated content"


def test_edit_file_original_preserved_on_write_error(in_tmp: Path, monkeypatch):
    """If the write half of the atomic replace fails, the original file
    must not be corrupted."""
    (in_tmp / "a.py").write_text("original\n")

    # Sabotage os.replace so the final rename fails after the temp write
    # has succeeded. Simulates a disk-full / permission error mid-edit.
    import tools.files as files_mod
    original_replace = os.replace

    def broken_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(files_mod.os, "replace", broken_replace)

    out = edit_file("a.py", "original", "edited")
    assert "Error" in out
    # The original file on disk is untouched.
    assert (in_tmp / "a.py").read_text() == "original\n"
    # And no orphaned tmp remains after the error path cleans up.
    _ = original_replace  # keep reference to silence unused-var warnings
    leftover = [p.name for p in in_tmp.iterdir() if p.name != "a.py"]
    assert leftover == [], f"unexpected leftover files: {leftover}"
