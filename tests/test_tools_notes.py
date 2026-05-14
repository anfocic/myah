"""Tests for tools/notes.py — the personal Obsidian vault tools.

Uses tmp_path as a fake vault so the tests exercise real filesystem I/O
without touching the user's actual vault.
"""
from __future__ import annotations

from datetime import date

import pytest

import tools.notes as notes_mod
from tools.notes import (
    daily_note,
    note_append,
    note_list,
    note_read,
    note_search,
    note_write,
)


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """Point the notes module at a tmp vault."""
    monkeypatch.setattr(notes_mod, "MIA_VAULT_PATH", str(tmp_path))
    monkeypatch.setattr(notes_mod, "MIA_DAILY_DIR", "daily")
    return tmp_path


# ── write / read ────────────────────────────────────────────────────────────

def test_write_then_read(vault):
    note_write("ideas", "# Ideas\n\nfirst one")
    out = note_read("ideas")
    assert "ideas.md" in out
    assert "first one" in out


def test_write_creates_parent_folders(vault):
    note_write("blog/drafts/post", "draft body")
    assert (vault / "blog" / "drafts" / "post.md").is_file()


def test_read_resolves_bare_name_across_folders(vault):
    note_write("notes/deep/buried", "needle")
    # bare name, no folder — falls back to vault-wide search
    out = note_read("buried")
    assert "needle" in out


def test_read_resolves_wikilink(vault):
    note_write("topics/python", "snake stuff")
    out = note_read("[[python]]")
    assert "snake stuff" in out


def test_read_missing_note(vault):
    assert "not found" in note_read("nonexistent").lower()


# ── path confinement ────────────────────────────────────────────────────────

def test_write_refuses_escape(vault):
    result = note_write("../outside", "evil")
    assert "Refused" in result
    assert not (vault.parent / "outside.md").exists()


def test_read_refuses_escape(vault):
    assert "Refused" in note_read("../../etc/passwd")


# ── append ──────────────────────────────────────────────────────────────────

def test_append_creates_when_missing(vault):
    note_append("log", "line one")
    assert "line one" in note_read("log")


def test_append_adds_blank_line_separator(vault):
    note_write("log", "existing")
    note_append("log", "added")
    body = (vault / "log.md").read_text()
    assert body == "existing\n\nadded\n"


# ── list ────────────────────────────────────────────────────────────────────

def test_list_returns_relative_paths(vault):
    note_write("a", "x")
    note_write("sub/b", "y")
    listed = note_list().splitlines()
    assert "a.md" in listed
    assert "sub/b.md" in listed


def test_list_skips_obsidian_internal_dirs(vault):
    note_write("real", "x")
    obsidian = vault / ".obsidian"
    obsidian.mkdir()
    (obsidian / "config.md").write_text("internal")
    listed = note_list().splitlines()
    assert listed == ["real.md"]


def test_list_folder_filter(vault):
    note_write("top", "x")
    note_write("blog/post", "y")
    listed = note_list("blog").splitlines()
    assert listed == ["blog/post.md"]


# ── search ──────────────────────────────────────────────────────────────────

def test_search_finds_content(vault):
    note_write("one", "the quick brown fox")
    note_write("two", "lazy dog")
    out = note_search("quick")
    assert "one.md" in out
    assert "two.md" not in out


def test_search_is_case_insensitive(vault):
    note_write("one", "Capitalized Word")
    assert "one.md" in note_search("capitalized")


def test_search_invalid_regex(vault):
    note_write("one", "x")
    assert "Invalid regex" in note_search("[unclosed")


def test_search_no_matches(vault):
    note_write("one", "nothing here")
    assert "No matches" in note_search("absent")


# ── daily note ──────────────────────────────────────────────────────────────

def test_daily_note_creates_and_appends(vault):
    daily_note("morning thought")
    today = date.today().isoformat()
    path = vault / "daily" / f"{today}.md"
    assert path.is_file()
    assert "morning thought" in path.read_text()


def test_daily_note_returns_content(vault):
    daily_note("entry one")
    out = daily_note()
    assert "entry one" in out
    assert date.today().isoformat() in out


def test_daily_note_empty_when_none(vault):
    out = daily_note()
    assert "empty" in out.lower()
