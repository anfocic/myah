"""Skills loader — discovers user-defined markdown skill files and
renders them into a `<skills>` block for the system prompt.

A skill is a markdown file under `MIA_SKILLS_PATH` (default
`~/.mia/skills/`). Optional YAML frontmatter sets `name` and
`description`; when absent, name falls back to the file's stem and
description falls back to the first non-blank body line. The block is
rebuilt every turn so a freshly added file lands on the next call —
same low-cost pattern as `<todos>` / `<vars>` / CLAUDE.md.

Subagents do NOT receive the block: skills are per-user playbook and
nested agents stay focused on the delegated subtask.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Best-effort YAML-frontmatter split. Returns (meta_dict, remaining_body).
    Recognizes only `key: value` lines — anything fancier degrades to empty
    meta and the original text. We deliberately avoid a YAML dependency for
    a four-key file format."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    raw = text[4:end]
    body = text[end + 4:].lstrip("\n")
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        meta[key] = value
    return meta, body


def _first_nonblank_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def load_skills(skills_dir: str | Path) -> list[Skill]:
    """Read every *.md file under `skills_dir` and return them sorted by
    name. A missing directory yields []. Per-file parse failures are
    swallowed so one broken skill cannot disable the loader."""
    d = Path(skills_dir).expanduser()
    if not d.is_dir():
        return []

    out: list[Skill] = []
    for path in sorted(d.glob("*.md")):
        try:
            text = path.read_text()
        except OSError:
            continue
        try:
            meta, body = _parse_frontmatter(text)
        except Exception:
            meta, body = {}, text
        name = meta.get("name") or path.stem
        description = meta.get("description") or _first_nonblank_line(body)
        out.append(Skill(name=name, description=description, body=body.strip()))

    out.sort(key=lambda s: s.name)
    return out


def skills_block(skills: list[Skill]) -> str | None:
    """Render the loaded skills as a `<skills>` block. Each skill gets a
    header line (name + description) followed by its body. Returns None
    when the list is empty so the system prompt omits the block entirely."""
    if not skills:
        return None
    sections: list[str] = []
    for s in skills:
        header = f"## {s.name} — {s.description}" if s.description else f"## {s.name}"
        sections.append(f"{header}\n\n{s.body}".rstrip())
    return "<skills>\n" + "\n\n".join(sections) + "\n</skills>"
