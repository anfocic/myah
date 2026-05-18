"""Tests for conversation-variable tools: validation, CRUD, dispatcher
integration, system-prompt rendering."""
from repl.tool_registry import make_execute_tool
from tools.vars import (
    MAX_VALUE_CHARS,
    PROMPT_VALUE_PREVIEW_CHARS,
    format_vars,
    get_var,
    list_vars,
    set_var,
    unset_var,
)

# ── set_var ──────────────────────────────────────────────────────────────────


def test_set_var_new(state):
    out = set_var(state, "foo", "bar")
    assert state["vars"] == {"foo": "bar"}
    assert "new" in out


def test_set_var_replace_distinguishes(state):
    set_var(state, "foo", "old")
    out = set_var(state, "foo", "new")
    assert state["vars"]["foo"] == "new"
    assert "replaced" in out


def test_set_var_unchanged_distinguishes(state):
    set_var(state, "foo", "bar")
    out = set_var(state, "foo", "bar")
    assert "unchanged" in out


def test_set_var_rejects_blank_name(state):
    out = set_var(state, "", "x")
    assert out.startswith("set_var rejected")
    assert state["vars"] == {}


def test_set_var_rejects_too_long_name(state):
    out = set_var(state, "a" * 65, "x")
    assert out.startswith("set_var rejected")


def test_set_var_rejects_bad_chars(state):
    out = set_var(state, "has space", "x")
    assert out.startswith("set_var rejected")


def test_set_var_rejects_non_string_name(state):
    out = set_var(state, 42, "x")
    assert out.startswith("set_var rejected")


def test_set_var_rejects_non_string_value(state):
    out = set_var(state, "foo", 42)
    assert out.startswith("set_var rejected")


def test_set_var_rejects_oversized_value(state):
    out = set_var(state, "foo", "x" * (MAX_VALUE_CHARS + 1))
    assert out.startswith("set_var rejected")
    assert state["vars"] == {}


# ── get_var ──────────────────────────────────────────────────────────────────


def test_get_var_hit(state):
    set_var(state, "foo", "bar")
    assert get_var(state, "foo") == "bar"


def test_get_var_miss_returns_sentinel(state):
    assert get_var(state, "ghost") == "(unset)"


def test_get_var_rejects_bad_name(state):
    out = get_var(state, "has space")
    assert out.startswith("get_var rejected")


# ── unset_var ────────────────────────────────────────────────────────────────


def test_unset_var_present(state):
    set_var(state, "foo", "bar")
    out = unset_var(state, "foo")
    assert state["vars"] == {}
    assert "unset" in out


def test_unset_var_absent_is_noop(state):
    out = unset_var(state, "ghost")
    assert "already unset" in out


# ── list_vars / format_vars ──────────────────────────────────────────────────


def test_list_vars_empty(state):
    assert list_vars(state) == "(no vars set)"


def test_list_vars_sorted(state):
    set_var(state, "b", "2")
    set_var(state, "a", "1")
    out = list_vars(state)
    assert out.index("a = 1") < out.index("b = 2")


def test_format_vars_truncates_long_value():
    out = format_vars({"k": "x" * 200})
    assert "..." in out
    assert len([line for line in out.splitlines() if "k = " in line]) == 1


# ── dispatcher integration ───────────────────────────────────────────────────


def test_dispatcher_set_var(state):
    execute = make_execute_tool(state)
    out = execute("set_var", {"name": "foo", "value": "bar"})
    assert "new" in out
    assert state["vars"] == {"foo": "bar"}


def test_dispatcher_get_var(state):
    execute = make_execute_tool(state)
    execute("set_var", {"name": "foo", "value": "bar"})
    assert execute("get_var", {"name": "foo"}) == "bar"


def test_dispatcher_unset_var(state):
    execute = make_execute_tool(state)
    execute("set_var", {"name": "foo", "value": "bar"})
    out = execute("unset_var", {"name": "foo"})
    assert "unset" in out
    assert state["vars"] == {}


def test_dispatcher_list_vars(state):
    execute = make_execute_tool(state)
    assert execute("list_vars", {}) == "(no vars set)"
    execute("set_var", {"name": "a", "value": "1"})
    assert "a = 1" in execute("list_vars", {})


def test_dispatcher_missing_args(state):
    execute = make_execute_tool(state)
    # args.get returns None → tool rejects as data
    assert execute("set_var", {}).startswith("set_var rejected")
    assert execute("get_var", {}).startswith("get_var rejected")


# ── system prompt block ──────────────────────────────────────────────────────


def test_vars_block_omitted_when_empty():
    from agent.system_prompt import _vars_block
    assert _vars_block({}) is None
    assert _vars_block(None) is None


def test_vars_block_renders_keys_alphabetically():
    from agent.system_prompt import _vars_block
    out = _vars_block({"b": "2", "a": "1"})
    assert out is not None
    assert "<vars>" in out and "</vars>" in out
    assert out.index("a = 1") < out.index("b = 2")


def test_vars_block_truncates_long_values():
    from agent.system_prompt import _vars_block
    out = _vars_block({"k": "x" * (PROMPT_VALUE_PREVIEW_CHARS + 50)})
    assert out is not None
    assert "truncated" in out
    assert "get_var for full" in out
