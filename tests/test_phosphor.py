"""Phosphor vocabulary module — accent resolution, brackets, masthead, rail."""

import config
from display import phosphor


def test_accent_maps_each_hue_to_a_named_ansi_color(monkeypatch):
    for hue, expected in [("green", "green"), ("amber", "yellow"), ("cyan", "cyan")]:
        monkeypatch.setattr(config, "PHOSPHOR_ACCENT", hue)
        assert phosphor.accent() == expected


def test_accent_falls_back_to_green_for_unknown_hue(monkeypatch):
    monkeypatch.setattr(config, "PHOSPHOR_ACCENT", "chartreuse")
    assert phosphor.accent() == "green"


def test_bracket_wraps_label_in_glyph_triple():
    out = phosphor.bracket("SESSION")
    assert "░▒▓" in out
    assert "▓▒░" in out
    assert "SESSION" in out


def test_rule_repeats_to_requested_width():
    out = phosphor.rule(width=12)
    assert "─" * 12 in out


def test_masthead_full_renders_banner_and_identity():
    out = phosphor.masthead("full", subtitle="personal harness")
    assert "█" in out  # block-letter banner art
    assert "myah" in out
    assert "personal harness" in out


def test_masthead_compact_is_single_line_without_banner_art():
    out = phosphor.masthead("compact", subtitle="personal harness")
    assert "▮ myah" in out
    assert "█" not in out


def test_masthead_none_is_empty():
    assert phosphor.masthead("none") == ""


def test_session_rail_has_all_three_sections():
    out = phosphor.session_rail(
        sess_state="READY",
        branch="main",
        turns=4,
        ctx_used=9011,
        ctx_total=32768,
        provider_label="ollama:qwen3",
    )
    assert "SESSION" in out
    assert "CTX" in out
    assert "TOOLS" in out
    assert "main" in out
    assert "READY" in out
    assert "█" in out or "░" in out  # ctx meter


def test_tool_hue_covers_known_tools_and_defaults_dim():
    assert phosphor.tool_hue("read_file") == "cyan"
    assert phosphor.tool_hue("bash") == "red"
    assert phosphor.tool_hue("spawn_subagent") == "magenta"
    assert phosphor.tool_hue("unknown_tool") == "bright_black"
