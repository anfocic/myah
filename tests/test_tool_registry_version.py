"""Tool registry versioning — Tool.version field, schema surfacing,
and harness_info exposure.

The version field lets a tool evolve its semantics without breaking
session files that recorded a call by name. These tests pin the public
contract: default is "1", custom versions flow through the schema,
and harness_info reports the version so the model can introspect.
"""
from __future__ import annotations

import pytest

from repl.state import new_state
from tools import spec
from tools.harness import harness_info, harness_snapshot


@pytest.fixture
def isolated_registry(monkeypatch):
    """Swap the module-level registry for an empty dict so each test
    starts clean and doesn't collide with the real harness tools (which
    register themselves at import time)."""
    monkeypatch.setattr(spec, "_registry", {})
    return spec._registry


def _noop_adapter(args, cwd):
    return "ok"


def test_register_default_version_is_one(isolated_registry):
    spec.register(
        name="t_default",
        description="A tool.",
        adapter=_noop_adapter,
    )
    tool = isolated_registry["t_default"]
    assert tool.version == "1"


def test_register_custom_version_stored_on_dataclass(isolated_registry):
    spec.register(
        name="t_v2",
        description="A tool.",
        adapter=_noop_adapter,
        version="2",
    )
    tool = isolated_registry["t_v2"]
    assert tool.version == "2"


def test_default_version_appears_in_schema_description(isolated_registry):
    spec.register(
        name="t_default",
        description="A tool.",
        adapter=_noop_adapter,
    )
    desc = isolated_registry["t_default"].schema["function"]["description"]
    assert desc.endswith("(v1)")


def test_custom_version_appears_in_schema_description(isolated_registry):
    spec.register(
        name="t_v2",
        description="Does the new thing.",
        adapter=_noop_adapter,
        version="2",
    )
    desc = isolated_registry["t_v2"].schema["function"]["description"]
    assert desc == "Does the new thing. (v2)"


def test_version_is_independent_of_read_only_flag(isolated_registry):
    """Versioning and read_only are orthogonal — bumping a version
    shouldn't accidentally flip permission gating."""
    spec.register(
        name="t_ro",
        description="Read-only tool.",
        adapter=_noop_adapter,
        version="3",
        read_only=True,
    )
    tool = isolated_registry["t_ro"]
    assert tool.version == "3"
    assert tool.read_only is True


def _fake_provider(monkeypatch):
    """Stub get_active_provider so harness_info doesn't reach for a
    live adapter during tests."""
    class _P:
        name = "fakeprov"
        model = "fakemodel"

    from tools import harness as harness_mod

    monkeypatch.setattr(harness_mod, "get_active_provider", lambda: _P())


def _state(ctx_used: int = 0):
    s = new_state()
    s["ctx_used"] = ctx_used
    return s


def test_harness_snapshot_includes_tool_versions(monkeypatch):
    _fake_provider(monkeypatch)
    snap = harness_snapshot(
        _state(),
        tool_names=["alpha", "beta"],
        tool_versions={"alpha": "1", "beta": "2"},
    )
    assert snap["tool_versions"] == {"alpha": "1", "beta": "2"}


def test_harness_snapshot_defaults_missing_versions_to_one(monkeypatch):
    _fake_provider(monkeypatch)
    snap = harness_snapshot(
        _state(),
        tool_names=["alpha", "beta"],
        tool_versions={"alpha": "3"},
    )
    assert snap["tool_versions"] == {"alpha": "3", "beta": "1"}


def test_harness_info_renders_versions_in_tools_line(monkeypatch):
    _fake_provider(monkeypatch)
    text = harness_info(
        _state(),
        tool_names=["alpha", "beta"],
        tool_versions={"alpha": "1", "beta": "2"},
    )
    assert "alpha (v1)" in text
    assert "beta (v2)" in text


def test_harness_info_back_compat_without_versions_arg(monkeypatch):
    """Callers that don't pass tool_versions still get a valid render —
    everything reads as v1."""
    _fake_provider(monkeypatch)
    text = harness_info(_state(), tool_names=["alpha"])
    assert "alpha (v1)" in text


def test_live_registry_exposes_versions_through_tool_versions_map():
    """Smoke test against the real wiring: TOOL_VERSIONS in
    repl/tool_registry.py covers every entry in TOOL_NAMES, including
    special-cased schemas, so harness_info never hits the default."""
    from repl.tool_registry import TOOL_NAMES, TOOL_VERSIONS

    for name in TOOL_NAMES:
        assert name in TOOL_VERSIONS, f"{name} missing from TOOL_VERSIONS"
        assert TOOL_VERSIONS[name], f"{name} has empty version"
