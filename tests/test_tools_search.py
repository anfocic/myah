"""Unit tests for tools/search.py — glob + grep.

Same `chdir into tmp_path` convention as test_tools_files.py — the
security.is_within_cwd guard resolves against live cwd, so tests have
to move there to set up realistic fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.search import MAX_RESULTS, glob, grep


@pytest.fixture
def in_tmp(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------- glob ----------

def test_glob_bare_filename_finds_file_anywhere(in_tmp: Path):
    (in_tmp / "pkg").mkdir()
    (in_tmp / "pkg" / "search.py").write_text("")
    out = glob("search.py")
    assert "pkg/search.py" in out


def test_glob_simple_pattern(in_tmp: Path):
    (in_tmp / "a.py").write_text("")
    (in_tmp / "b.py").write_text("")
    (in_tmp / "c.txt").write_text("")
    out = glob("*.py")
    assert "a.py" in out
    assert "b.py" in out
    assert "c.txt" not in out


def test_glob_recursive_pattern(in_tmp: Path):
    (in_tmp / "docs").mkdir()
    (in_tmp / "docs" / "intro.md").write_text("")
    (in_tmp / "README.md").write_text("")
    out = glob("**/*.md")
    assert "README.md" in out
    assert "docs/intro.md" in out


def test_glob_skips_dotdirs_and_venvs(in_tmp: Path):
    (in_tmp / ".git").mkdir()
    (in_tmp / ".git" / "HEAD").write_text("")
    (in_tmp / "venv").mkdir()
    (in_tmp / "venv" / "bin.py").write_text("")
    (in_tmp / "real.py").write_text("")
    out = glob("*.py")
    assert "real.py" in out
    assert "bin.py" not in out


def test_glob_no_match_returns_message(in_tmp: Path):
    out = glob("*.nonexistent")
    assert "No files matching" in out


def test_glob_refuses_outside_cwd(in_tmp: Path):
    assert "Refusing path outside cwd" in glob("*", "/etc")


def test_glob_missing_path_returns_error(in_tmp: Path):
    assert "Path not found" in glob("*", "does_not_exist")


def test_glob_sorted_deterministic(in_tmp: Path):
    (in_tmp / "zebra.py").write_text("")
    (in_tmp / "alpha.py").write_text("")
    (in_tmp / "mango.py").write_text("")
    out = glob("*.py")
    # Sorted output → alpha before mango before zebra.
    lines = out.splitlines()
    assert lines == sorted(lines)


def test_glob_truncates_at_max_results(in_tmp: Path):
    for i in range(MAX_RESULTS + 5):
        (in_tmp / f"f{i:03d}.py").write_text("")
    out = glob("*.py")
    assert "truncated at" in out
    # The listing portion has exactly MAX_RESULTS lines; the trailing
    # "truncated" marker is on its own line.
    match_lines = [
        ln for ln in out.splitlines() if ln.endswith(".py")
    ]
    assert len(match_lines) == MAX_RESULTS


# ---------- grep ----------

def test_grep_files_with_matches_default_mode(in_tmp: Path):
    (in_tmp / "a.py").write_text("TODO: fix this\n")
    (in_tmp / "b.py").write_text("nothing here\n")
    out = grep("TODO")
    assert "a.py" in out
    assert "b.py" not in out


def test_grep_content_mode_returns_path_line_text(in_tmp: Path):
    (in_tmp / "a.py").write_text("x = 1\nTODO: fix\ny = 2\n")
    out = grep("TODO", output_mode="content")
    # Format: `<path>:<lineno>:<line>`
    assert "a.py:2:TODO: fix" in out


def test_grep_no_matches_returns_message(in_tmp: Path):
    (in_tmp / "a.py").write_text("clean\n")
    assert grep("absent_pattern") == "No matches."


def test_grep_invalid_regex_returns_clear_error(in_tmp: Path):
    (in_tmp / "a.py").write_text("x\n")
    out = grep("[invalid(")
    assert "Invalid regex" in out


def test_grep_skips_binary_files_silently(in_tmp: Path):
    (in_tmp / "bin.dat").write_bytes(b"\xff\xfe text \xfd")
    (in_tmp / "a.py").write_text("text\n")
    out = grep("text")
    assert "a.py" in out
    assert "bin.dat" not in out


def test_grep_respects_glob_filter(in_tmp: Path):
    (in_tmp / "a.py").write_text("match\n")
    (in_tmp / "b.md").write_text("match\n")
    out = grep("match", glob="*.py")
    assert "a.py" in out
    assert "b.md" not in out


def test_grep_refuses_outside_cwd(in_tmp: Path):
    assert "Refusing path outside cwd" in grep(".", "/etc")


def test_grep_missing_path_returns_error(in_tmp: Path):
    assert "Path not found" in grep(".", "does_not_exist")


def test_grep_on_single_file_works(in_tmp: Path):
    f = in_tmp / "a.py"
    f.write_text("foo\nbar\n")
    # files_with_matches (default) returns just the filename when the
    # target path IS a file — the filename itself is the "hit".
    out = grep("foo", str(f))
    assert "a.py" in out
    # content mode returns path:line:text, so the matched text shows up.
    out_content = grep("foo", str(f), output_mode="content")
    assert ":1:foo" in out_content


def test_grep_skips_dotdirs_and_venvs(in_tmp: Path):
    (in_tmp / ".git").mkdir()
    (in_tmp / ".git" / "HEAD").write_text("TODO: from git\n")
    (in_tmp / "venv").mkdir()
    (in_tmp / "venv" / "bin.py").write_text("TODO: from venv\n")
    (in_tmp / "real.py").write_text("TODO: real\n")
    out = grep("TODO")
    assert "real.py" in out
    assert "venv" not in out
    assert ".git" not in out


def test_grep_truncates_at_max_results(in_tmp: Path):
    for i in range(MAX_RESULTS + 5):
        (in_tmp / f"f{i:03d}.py").write_text("needle\n")
    out = grep("needle")
    assert "truncated at" in out
    # Result lines are each one file path; count them to verify the cap.
    path_lines = [ln for ln in out.splitlines() if ln.endswith(".py")]
    assert len(path_lines) == MAX_RESULTS
