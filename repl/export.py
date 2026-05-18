"""Conversation export to a portable markdown file.

Distinct from `/save-session`, which archives into the personal vault with
Obsidian frontmatter + extracted-data scaffolding. `/export` writes a
plain markdown transcript to an arbitrary path (cwd by default) so the
user can paste it into a doc, email it, attach it to a bug report, or
drop it in a non-vault repo.

Header is intentionally light: a single H1 + a one-line metadata caption.
No JSON telemetry, no extraction prompts."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


def _default_filename() -> str:
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    return f"mia-transcript-{stamp}.md"


def _format_transcript(history: list[dict], model: str, provider: str) -> str:
    """Render the durable conversation as markdown. Tool messages don't
    appear in `history` (intra-turn scratch space per agent/loop.py), so
    we only see user + assistant turns plus the synthetic-summary system
    notes that `apply_summary` injects after a compact."""
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Mia conversation export",
        "",
        f"_Exported {when} · {model} ({provider})_",
        "",
        "---",
        "",
    ]
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lines.append("### user")
            lines.append("")
            lines.append(content)
        elif role == "assistant":
            lines.append("### assistant")
            lines.append("")
            lines.append(content)
        elif role == "system":
            lines.append("### system note")
            lines.append("")
            lines.append(content)
        else:
            continue
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_conversation(
    history: list[dict], model: str, provider: str, path: str | None = None,
) -> str:
    """Write the transcript to disk and return the absolute path.

    Returns an error string starting with "Error" or "Refused" on failure
    so the slash command can route output uniformly (same convention as
    note_write). `path` accepts an absolute or relative location; a bare
    filename lands in the current working directory. None → a timestamped
    default in cwd."""
    if not history:
        return "Refused: no conversation history to export"
    target = Path(path).expanduser() if path else Path(_default_filename())
    # Resolve against cwd so a relative path lands where the user expects.
    target = target.resolve()
    if target.is_dir():
        target = target / _default_filename()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_format_transcript(history, model, provider))
    except OSError as e:
        return f"Error: {e}"
    return str(target)
