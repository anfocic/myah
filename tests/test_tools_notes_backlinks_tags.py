"""Tests for note_backlinks + note_by_tag.

Backlinks: a `[[wikilink]]` scan over the vault that returns notes whose
body mentions the target, with case-insensitive matching and tolerance
for aliases (`[[foo|bar]]`) and heading fragments (`[[foo#H]]`).

Tags: matches both inline `#tag` in the body and `tags:` in YAML-ish
frontmatter (inline-list, flow-list, and block-list shapes)."""
from __future__ import annotations

import pytest

import tools.notes as notes_mod
from tools.notes import note_backlinks, note_by_tag, note_write


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setattr(notes_mod, "MIA_VAULT_PATH", str(tmp_path))
    monkeypatch.setattr(notes_mod, "MIA_DAILY_DIR", "daily")
    return tmp_path


# ── note_backlinks ───────────────────────────────────────────────────────────


def test_backlinks_finds_direct_wikilink(vault):
    note_write("target", "# Target")
    note_write("a", "see [[target]] for details")
    note_write("b", "no link here")
    out = note_backlinks("target")
    assert "a.md" in out and "b.md" not in out


def test_backlinks_match_is_case_insensitive(vault):
    note_write("Target", "# Target")
    note_write("a", "see [[TARGET]]")
    note_write("b", "see [[target]]")
    out = note_backlinks("Target")
    assert "a.md" in out and "b.md" in out


def test_backlinks_tolerates_alias(vault):
    note_write("target", "# Target")
    note_write("a", "see [[target|that thing]]")
    out = note_backlinks("target")
    assert "a.md" in out


def test_backlinks_tolerates_heading_fragment(vault):
    note_write("target", "# Target\n\n## Section")
    note_write("a", "see [[target#Section]]")
    out = note_backlinks("target")
    assert "a.md" in out


def test_backlinks_excludes_self_link(vault):
    note_write("target", "# Target\n\nlinks back to [[target]] for some reason")
    note_write("a", "see [[target]]")
    out = note_backlinks("target")
    assert "a.md" in out
    assert "target.md" not in out.replace("a.md", "")  # only the a.md mention


def test_backlinks_no_hits(vault):
    note_write("target", "# Target")
    note_write("a", "nothing here")
    assert note_backlinks("target") == "No backlinks."


def test_backlinks_strips_brackets_in_query(vault):
    note_write("target", "# Target")
    note_write("a", "see [[target]]")
    out = note_backlinks("[[target]]")
    assert "a.md" in out


def test_backlinks_strips_md_suffix_in_query(vault):
    note_write("target", "# Target")
    note_write("a", "see [[target]]")
    out = note_backlinks("target.md")
    assert "a.md" in out


def test_backlinks_refuses_empty(vault):
    out = note_backlinks("   ")
    assert out.startswith("Refused")


# ── note_by_tag ──────────────────────────────────────────────────────────────


def test_tag_inline_in_body(vault):
    note_write("a", "this note is about #python and stuff")
    note_write("b", "unrelated")
    out = note_by_tag("python")
    assert "a.md" in out and "b.md" not in out


def test_tag_match_is_case_insensitive(vault):
    note_write("a", "about #Python")
    out = note_by_tag("python")
    assert "a.md" in out


def test_tag_leading_hash_optional(vault):
    note_write("a", "about #python")
    assert "a.md" in note_by_tag("#python")
    assert "a.md" in note_by_tag("python")


def test_tag_in_frontmatter_inline_list(vault):
    note_write("a", "---\ntags: [python, ml]\n---\n\nbody")
    note_write("b", "---\ntags: [rust]\n---\n\nbody")
    out = note_by_tag("python")
    assert "a.md" in out and "b.md" not in out
    assert "frontmatter" in out


def test_tag_in_frontmatter_flow_csv(vault):
    note_write("a", "---\ntags: python, ml\n---\n\nbody")
    assert "a.md" in note_by_tag("ml")


def test_tag_in_frontmatter_block_list(vault):
    note_write("a", "---\ntags:\n  - python\n  - ml\n---\n\nbody")
    out = note_by_tag("ml")
    assert "a.md" in out and "frontmatter" in out


def test_tag_in_frontmatter_quoted_value(vault):
    note_write("a", "---\ntags: [\"python\"]\n---\n\nbody")
    assert "a.md" in note_by_tag("python")


def test_tag_no_hits(vault):
    note_write("a", "no tags here")
    assert note_by_tag("python") == "No notes with that tag."


def test_tag_inline_requires_word_boundary(vault):
    # `#pythonista` should NOT match `#python`.
    note_write("a", "about #pythonista")
    assert note_by_tag("python") == "No notes with that tag."


def test_tag_refuses_empty(vault):
    assert note_by_tag("   ").startswith("Refused")
    assert note_by_tag("#").startswith("Refused")


def test_tag_reports_body_vs_frontmatter_source(vault):
    note_write("body_only", "tagged inline #zone")
    note_write("fm_only", "---\ntags: [zone]\n---\n\nno inline")
    out = note_by_tag("zone")
    assert "body_only.md  (body)" in out
    assert "fm_only.md  (frontmatter)" in out
