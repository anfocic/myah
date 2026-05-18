"""Conversation variables: a per-session named key-value store the model
can read and write through tool calls.

Solves "persistent cross-turn state without history pollution": the model
can stash an intermediate value ("I figured out the user's billing email
is X — remember that for later") without re-deriving it next turn and
without that long aside cluttering the assistant's user-visible reply.

Plan-mode-safe — the only effect is on harness-local working memory, no
external side effects.

Four tools share one state["vars"] dict reached through the dispatcher
closure (see repl/tool_registry.py), mirroring how todo_write hooks
into state["todos"]:

- set_var(name, value)   — create or replace
- get_var(name)          — read; '(unset)' on miss
- list_vars()            — render current map
- unset_var(name)        — delete; no-op on miss

The set is intentionally narrow: no scoping, no namespaces, no value
types beyond strings. JSON-encode richer payloads on the caller side.
"""
from __future__ import annotations

import re
from typing import Any

# Conservative name shape — alphanumeric + underscore + dash, ≤ 64 chars.
# Tight enough to keep var-name collisions obvious; loose enough that the
# model can pick natural slugs without ceremony.
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Cap on a single value — matches the rough order of TOOL_RESULT_MAX_BYTES
# so a `get_var` round-trip can't blow the tool-result truncation budget
# in surprising ways.
MAX_VALUE_CHARS = 10_000

# Truncation cap when rendering the live <vars> block in the system prompt.
# Full values still come through get_var; this just keeps the prompt block
# small on a turn where the model would otherwise pay for every value
# in tokens every turn.
PROMPT_VALUE_PREVIEW_CHARS = 120


def _check_name(name: Any) -> str | None:
    """Return None if `name` is a valid var name, else an error message
    the caller can return as the tool result string."""
    if not isinstance(name, str):
        return f"name must be a string, got {type(name).__name__}"
    if not _NAME_RE.match(name):
        return (
            f"name {name!r} is invalid — use 1-64 chars of [A-Za-z0-9_-]"
        )
    return None


def _check_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return f"value must be a string, got {type(value).__name__}"
    if len(value) > MAX_VALUE_CHARS:
        return f"value too long ({len(value)} > {MAX_VALUE_CHARS} chars)"
    return None


def format_vars(vars_dict: dict[str, str]) -> str:
    """Human-readable rendering used by /vars and list_vars."""
    if not vars_dict:
        return "(no vars set)"
    lines = []
    for name in sorted(vars_dict):
        value = vars_dict[name]
        preview = value if len(value) <= 80 else value[:77] + "..."
        lines.append(f"{name} = {preview}")
    return "\n".join(lines)


def set_var(state: dict, name: Any, value: Any) -> str:
    err = _check_name(name) or _check_value(value)
    if err:
        return f"set_var rejected: {err}"
    vars_dict: dict[str, str] = state.setdefault("vars", {})
    previous = vars_dict.get(name)
    vars_dict[name] = value
    if previous is None:
        return f"set {name} (new, {len(value)} chars)"
    if previous == value:
        return f"set {name} (unchanged, {len(value)} chars)"
    return f"set {name} (replaced, {len(value)} chars)"


def get_var(state: dict, name: Any) -> str:
    err = _check_name(name)
    if err:
        return f"get_var rejected: {err}"
    vars_dict: dict[str, str] = state.get("vars", {}) or {}
    if name not in vars_dict:
        return "(unset)"
    return vars_dict[name]


def unset_var(state: dict, name: Any) -> str:
    err = _check_name(name)
    if err:
        return f"unset_var rejected: {err}"
    vars_dict: dict[str, str] = state.get("vars", {}) or {}
    if name not in vars_dict:
        return f"{name}: already unset"
    del vars_dict[name]
    return f"unset {name}"


def list_vars(state: dict) -> str:
    return format_vars(state.get("vars", {}) or {})


SET_VAR_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "set_var",
        "description": (
            "Stash a named string value in conversation memory. Use for "
            "intermediate facts you want to recall next turn without "
            "re-deriving them or polluting your reply. Whole-value replace; "
            "no merge. Names are 1-64 chars of [A-Za-z0-9_-]. Values are "
            f"strings up to {MAX_VALUE_CHARS} chars (JSON-encode richer "
            "payloads yourself)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Variable name."},
                "value": {"type": "string", "description": "String value."},
            },
            "required": ["name", "value"],
        },
    },
}

GET_VAR_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_var",
        "description": (
            "Read a previously-stashed conversation variable. Returns the "
            "value, or the literal string '(unset)' if the name is unknown. "
            "Names are 1-64 chars of [A-Za-z0-9_-]."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Variable name."},
            },
            "required": ["name"],
        },
    },
}

LIST_VARS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_vars",
        "description": (
            "List every conversation variable currently set, one per line as "
            "`name = value` (long values are truncated in this view; use "
            "get_var for the full text). Returns '(no vars set)' when empty."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

UNSET_VAR_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "unset_var",
        "description": (
            "Forget a conversation variable. No-op when the name is unknown. "
            "Returns a one-line confirmation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Variable name."},
            },
            "required": ["name"],
        },
    },
}

VAR_SCHEMAS: list[dict[str, Any]] = [
    SET_VAR_SCHEMA,
    GET_VAR_SCHEMA,
    LIST_VARS_SCHEMA,
    UNSET_VAR_SCHEMA,
]
