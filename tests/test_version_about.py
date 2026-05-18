"""Tests for /version and /about — small introspection commands."""
import re
from io import StringIO

import pytest
from rich.console import Console

import repl.commands as cmds_mod
from repl.commands import cmd_about, cmd_version


@pytest.fixture
def buf_console(monkeypatch) -> StringIO:
    buf = StringIO()
    fake = Console(file=buf, force_terminal=False, width=120)
    monkeypatch.setattr(cmds_mod, "console", fake)
    return buf


# ── /version ─────────────────────────────────────────────────────────────────


def test_version_prints_mia_prefix(buf_console, state):
    cmd_version(state, "")
    out = buf_console.getvalue()
    assert out.startswith("mia ")
    # Either a real version string or the 'dev' fallback when not installed.
    body = out.strip().removeprefix("mia ").strip()
    assert body  # non-empty
    # PEP 440-ish: digits, letters, dots, hyphens, plus signs.
    assert re.fullmatch(r"[A-Za-z0-9._+-]+", body)


def test_version_helper_falls_back_to_dev_when_package_missing(monkeypatch):
    """If the package isn't installed, _mia_version() must return 'dev'
    rather than crash."""
    import importlib.metadata as md

    def boom(_):
        raise md.PackageNotFoundError("myah")

    monkeypatch.setattr(md, "version", boom)
    assert cmds_mod._mia_version() == "dev"


# ── /about ───────────────────────────────────────────────────────────────────


def test_about_includes_model_and_provider(buf_console, state):
    cmd_about(state, "")
    out = buf_console.getvalue()
    # Active provider in the tests is a real one; we just assert the
    # labels and that *some* model identifier is present.
    assert "MIA" in out
    assert "model" in out
    assert "num_ctx" in out
    assert "tools" in out
    assert "plan mode" in out


def test_about_reflects_plan_mode_toggle(buf_console, state):
    state["plan_mode"] = True
    cmd_about(state, "")
    out = buf_console.getvalue()
    assert "ON" in out


def test_about_reflects_cwd(buf_console, state, tmp_path):
    state["cwd"] = str(tmp_path)
    cmd_about(state, "")
    out = buf_console.getvalue()
    assert str(tmp_path) in out
