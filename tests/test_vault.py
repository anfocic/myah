"""Tests for tools/vault.py — vault_search and _iter_vault_files.

Uses tmp_path to build a fake vault on disk so the tests exercise real
filesystem I/O without depending on an actual vault/ directory.
"""
from __future__ import annotations

import pytest

import tools.vault as vault_mod
from tools.vault import _iter_vault_files, vault_search


@pytest.fixture
def fake_vault(tmp_path, monkeypatch):
    """Point VAULT_DIR at tmp_path so tests don't touch the real vault."""
    monkeypatch.setattr(vault_mod, "VAULT_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# _iter_vault_files
# ---------------------------------------------------------------------------

def test_iter_vault_files_yields_md_files(fake_vault):
    (fake_vault / "a.md").write_text("a")
    (fake_vault / "b.md").write_text("b")
    (fake_vault / "c.txt").write_text("c")

    files = list(_iter_vault_files(fake_vault))
    names = {f.name for f in files}
    assert names == {"a.md", "b.md"}


def test_iter_vault_files_skips_subdirs_in_skip_set(fake_vault):
    raw_dir = fake_vault / "raw"
    raw_dir.mkdir()
    (raw_dir / "sessions.md").write_text("session")

    files = list(_iter_vault_files(fake_vault))
    assert not files


def test_iter_vault_files_skips_dotgit(fake_vault):
    git_dir = fake_vault / ".git"
    git_dir.mkdir()
    (git_dir / "config.md").write_text("config")

    files = list(_iter_vault_files(fake_vault))
    assert not files


def test_iter_vault_files_skips_pycache(fake_vault):
    pycache = fake_vault / "__pycache__"
    pycache.mkdir()
    (pycache / "cache.md").write_text("cache")

    files = list(_iter_vault_files(fake_vault))
    assert not files


def test_iter_vault_files_does_not_skip_file_named_raw(fake_vault):
    (fake_vault / "raw.md").write_text("raw at root")

    files = list(_iter_vault_files(fake_vault))
    assert len(files) == 1
    assert files[0].name == "raw.md"


def test_iter_vault_files_deeply_nested(fake_vault):
    sub = fake_vault / "a" / "b" / "c"
    sub.mkdir(parents=True)
    (sub / "deep.md").write_text("deep")

    files = list(_iter_vault_files(fake_vault))
    assert len(files) == 1


def test_iter_vault_files_missing_root():
    from pathlib import Path

    files = list(_iter_vault_files(Path("/nonexistent/path")))
    assert files == []


# ---------------------------------------------------------------------------
# vault_search — basic matching
# ---------------------------------------------------------------------------

def test_vault_search_finds_simple_match(fake_vault):
    (fake_vault / "README.md").write_text("This is a test file with content.")

    result = vault_search("test file")
    assert "README.md" in result
    assert "test file" in result


def test_vault_search_case_insensitive(fake_vault):
    (fake_vault / "upper.md").write_text("UPPERCASE TEXT")

    result = vault_search("uppercase")
    assert "upper.md" in result


def test_vault_search_regex_supported(fake_vault):
    (fake_vault / "regex.md").write_text("abc123 def456")

    result = vault_search(r"\d+")
    assert "regex.md" in result
    assert "123" in result or "456" in result


def test_vault_search_invalid_regex_returns_error(fake_vault):
    (fake_vault / "x.md").write_text("hello")

    result = vault_search("[invalid")
    assert "Invalid regex" in result


def test_vault_search_no_matches(fake_vault):
    (fake_vault / "x.md").write_text("nothing relevant")

    result = vault_search("pineapple")
    assert result == "No matches in vault."


def test_vault_search_multiple_files(fake_vault):
    (fake_vault / "a.md").write_text("needle")
    (fake_vault / "b.md").write_text("needle again")
    (fake_vault / "c.md").write_text("no match")

    result = vault_search("needle")
    assert "a.md" in result
    assert "b.md" in result
    assert "c.md" not in result


def test_vault_search_returns_relative_paths(fake_vault):
    sub = fake_vault / "subdir"
    sub.mkdir()
    (sub / "inside.md").write_text("findme")

    result = vault_search("findme")
    assert "subdir/inside.md" in result


# ---------------------------------------------------------------------------
# vault_search — categories
# ---------------------------------------------------------------------------

def test_vault_search_category_code(fake_vault):
    code_dir = fake_vault / "wiki" / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "auth.md").write_text("login flow")

    result = vault_search("login", category="code")
    assert "wiki/code/auth.md" in result


def test_vault_search_category_meta(fake_vault):
    meta_dir = fake_vault / "wiki" / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "decisions.md").write_text("we chose sqlite")

    result = vault_search("sqlite", category="meta")
    assert "wiki/meta/decisions.md" in result


def test_vault_search_category_plans(fake_vault):
    plans_dir = fake_vault / "wiki" / "plans"
    plans_dir.mkdir(parents=True)
    (plans_dir / "v2.md").write_text("migrate to v2")

    result = vault_search("migrate", category="plans")
    assert "wiki/plans/v2.md" in result


def test_vault_search_category_templates(fake_vault):
    tmpl_dir = fake_vault / "templates"
    tmpl_dir.mkdir()
    (tmpl_dir / "skill.md").write_text("skill template")

    result = vault_search("skill", category="templates")
    assert "templates/skill.md" in result


def test_vault_search_unknown_category_uses_full_vault(fake_vault):
    (fake_vault / "root.md").write_text("found here")

    result = vault_search("found", category="nonexistent")
    assert "root.md" in result


def test_vault_search_category_only_searches_that_subtree(fake_vault):
    code_dir = fake_vault / "wiki" / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "match.md").write_text("needle")
    (fake_vault / "nomatch.md").write_text("needle too")

    result = vault_search("needle", category="code")
    assert "wiki/code/match.md" in result
    assert "nomatch.md" not in result


# ---------------------------------------------------------------------------
# vault_search — missing directories
# ---------------------------------------------------------------------------

def test_vault_search_missing_root_returns_error(monkeypatch):
    monkeypatch.setattr(vault_mod, "VAULT_DIR", vault_mod.Path("/nonexistent/vault/path"))
    result = vault_search("query")
    assert "Vault path not found" in result


def test_vault_search_missing_category_dir(fake_vault):
    result = vault_search("query", category="code")
    assert "Vault path not found" in result


# ---------------------------------------------------------------------------
# vault_search — edge cases
# ---------------------------------------------------------------------------

def test_vault_search_skips_large_files(fake_vault, monkeypatch):
    (fake_vault / "big.md").write_text("x" * 1000)
    (fake_vault / "small.md").write_text("needle")

    monkeypatch.setattr(vault_mod, "MAX_FILE_BYTES", 50)

    result = vault_search("needle")
    assert "small.md" in result
    assert "big.md" not in result


def test_vault_search_handles_stat_failure(fake_vault, monkeypatch):
    """OSError raised on stat() after iterating is caught, file skipped."""
    (fake_vault / "good.md").write_text("needle here")

    def fake_iter(root):
        yield from _iter_vault_files(root)

        class BrokenPath:
            def __init__(self):
                self.name = "broken.md"

            def stat(self):
                raise OSError("vanished")

            def read_text(self, **kw):
                return "needle"

            def relative_to(self, base):
                return fake_vault.__class__("broken.md")

        yield BrokenPath()

    monkeypatch.setattr(vault_mod, "_iter_vault_files", fake_iter)

    result = vault_search("needle")
    assert "good.md" in result
    assert "broken.md" not in result


def test_vault_search_handles_unreadable_file_text(fake_vault):
    """Files that raise UnicodeDecodeError on read_text are skipped."""
    (fake_vault / "binary.bin").write_bytes(b"\x80\x81\x82")

    result = vault_search("anything")
    assert result == "No matches in vault."


def test_vault_search_snippet_has_file_prefix(fake_vault):
    (fake_vault / "notes.md").write_text("before needle after")

    result = vault_search("needle")
    assert "notes.md" in result
    assert "needle" in result


def test_vault_search_caps_at_max_results(fake_vault, monkeypatch):
    monkeypatch.setattr(vault_mod, "MAX_RESULTS", 2)
    for i in range(5):
        (fake_vault / f"file{i}.md").write_text(f"match {i}")

    result = vault_search("match")
    hits = result.count(".md")
    assert hits <= 2


def test_vault_search_empty_query(fake_vault):
    (fake_vault / "a.md").write_text("content")

    result = vault_search("")
    assert "a.md" in result
    assert "content" in result
