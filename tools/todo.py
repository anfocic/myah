"""Working-memory todo list. The model maintains a structured task list
across turns and updates it as it works — frees attention from
re-deriving "what was I doing again?" on long multi-step tasks.

Single tool: `todo_write(todos)` replaces the entire list. Whole-list-
replace (not CRUD) matches the Claude Code reference and keeps the
model holding the truth: every call is the new full state.

Validation enforces at-most-one in-progress so the discipline of
"work on one thing at a time" is structural, not just persuasive.

State location: `state["todos"]` reached through the dispatcher closure
(see repl/tool_registry.py). Plan-mode-safe because the only effect is
on the harness's own working memory — no external side effects.
"""
from dataclasses import asdict, dataclass
from typing import Any, Literal

Status = Literal["pending", "in_progress", "completed"]
VALID_STATUSES: tuple[str, ...] = ("pending", "in_progress", "completed")


@dataclass
class Todo:
    content: str
    activeForm: str
    status: Status


def _parse_todo(raw: Any, idx: int) -> Todo:
    """Coerce one raw item from the tool args into a Todo, raising
    ValueError with a positional hint so the model can self-correct."""
    if not isinstance(raw, dict):
        raise ValueError(f"todo[{idx}]: expected object, got {type(raw).__name__}")
    missing = [k for k in ("content", "activeForm", "status") if k not in raw]
    if missing:
        raise ValueError(f"todo[{idx}]: missing required key(s): {', '.join(missing)}")
    content = raw["content"]
    active = raw["activeForm"]
    status = raw["status"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"todo[{idx}]: content must be a non-empty string")
    if not isinstance(active, str) or not active.strip():
        raise ValueError(f"todo[{idx}]: activeForm must be a non-empty string")
    if status not in VALID_STATUSES:
        raise ValueError(
            f"todo[{idx}]: status must be one of {VALID_STATUSES}, got {status!r}"
        )
    return Todo(content=content.strip(), activeForm=active.strip(), status=status)


def parse_todos(raw_list: Any) -> list[Todo]:
    """Parse + validate a whole list. Enforces at-most-one in-progress."""
    if not isinstance(raw_list, list):
        raise ValueError(f"todos must be a list, got {type(raw_list).__name__}")
    todos = [_parse_todo(item, i) for i, item in enumerate(raw_list)]
    in_progress = [t for t in todos if t.status == "in_progress"]
    if len(in_progress) > 1:
        raise ValueError(
            f"at most one todo may be 'in_progress' at a time "
            f"(found {len(in_progress)})"
        )
    return todos


_GLYPH: dict[str, str] = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "completed": "[x]",
}


def format_todos(todos: list[Todo]) -> str:
    """Plain-text checklist. The active form is shown for in-progress
    items ('Fixing the bug'); content for the rest ('Fix the bug')."""
    if not todos:
        return "(no todos)"
    lines = []
    for t in todos:
        label = t.activeForm if t.status == "in_progress" else t.content
        lines.append(f"{_GLYPH[t.status]} {label}")
    return "\n".join(lines)


def serialize_todos(todos: list[Todo]) -> list[dict[str, str]]:
    """Round-trip-safe dict form for persistence."""
    return [asdict(t) for t in todos]


def deserialize_todos(raw: Any) -> list[Todo]:
    """Lenient inverse of serialize_todos. Returns [] on any structural
    problem so a stale or corrupted session file can still load."""
    try:
        return parse_todos(raw)
    except (ValueError, TypeError):
        return []


def todo_write(state: dict, raw_todos: Any) -> str:
    """Replace state["todos"] with the validated parse of raw_todos.

    Returns the rendered new list so the model sees confirmation
    inline. Validation errors return as data (the model can self-
    correct) rather than raising."""
    try:
        todos = parse_todos(raw_todos)
    except ValueError as e:
        return f"todo_write rejected: {e}"
    state["todos"] = todos
    if not todos:
        return "todo list cleared"
    return "todo list updated:\n" + format_todos(todos)


TODO_WRITE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "todo_write",
        "description": (
            "Create or replace your working todo list. Pass the WHOLE list "
            "every call; the harness replaces it in place. Use this for "
            "multi-step tasks (3+ steps) to track progress and avoid losing "
            "context across turns. Rules: exactly one item may be "
            "'in_progress' at a time; mark items 'completed' the instant "
            "you finish them, not in a batch at the end; do not pre-emptively "
            "mark something completed. Each todo needs `content` (imperative, "
            "e.g. 'Fix the bug'), `activeForm` (present continuous, e.g. "
            "'Fixing the bug'), and `status` (pending | in_progress | "
            "completed). Pass an empty list to clear."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "The complete new todo list. Replaces prior list.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Imperative task description.",
                            },
                            "activeForm": {
                                "type": "string",
                                "description": "Present-continuous form, shown while in_progress.",
                            },
                            "status": {
                                "type": "string",
                                "enum": list(VALID_STATUSES),
                                "description": "One of: pending, in_progress, completed.",
                            },
                        },
                        "required": ["content", "activeForm", "status"],
                    },
                },
            },
            "required": ["todos"],
        },
    },
}
