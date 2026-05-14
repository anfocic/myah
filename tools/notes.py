"""Personal Obsidian vault tools — Mia reads and writes the user's vault
through the `note_*` tools, confined to `config.MIA_VAULT_PATH`.

Distinct from `vault_search` (tools/vault.py), which queries this repo's own
dev knowledge base. These tools operate on the user's personal notes.

Security model: every path is resolved inside the vault root. A reference
that escapes the vault (via `..` or an absolute path elsewhere) is refused.
This mirrors the cwd guard in tools/files.py but is anchored to the vault,
since the vault legitimately lives outside the harness cwd.
"""

import re
from datetime import date
from pathlib import Path

from config import MIA_DAILY_DIR, MIA_VAULT_PATH
from tools.spec import register

SKIP_DIRS = {".git", ".obsidian", ".trash", "__pycache__"}
MAX_FILE_BYTES = 1_000_000
MAX_RESULTS = 20


def _vault_root() -> Path:
    """Vault root, created on first use so a fresh vault needs no setup step."""
    root = Path(MIA_VAULT_PATH)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve(ref: str) -> Path | None:
    """Resolve a note reference to an absolute path inside the vault.

    Accepts bare names ('ideas'), wikilinks ('[[ideas]]'), and subpaths
    ('notes/ideas.md'). A missing '.md' suffix is added. Returns None if the
    resolved path escapes the vault root.
    """
    root = _vault_root().resolve()
    name = ref.strip().strip("[]").strip()
    if not name.endswith(".md"):
        name += ".md"
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _find_by_name(ref: str) -> Path | None:
    """Locate a note anywhere in the vault by basename. Supports wikilink
    resolution, where the link names a note but not its folder."""
    root = _vault_root()
    stem = ref.strip().strip("[]").strip().removesuffix(".md")
    for p in sorted(root.rglob(f"{stem}.md")):
        if not any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            return p
    return None


def _iter_notes(root: Path):
    for p in root.rglob("*.md"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        yield p


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def note_read(path: str) -> str:
    """Read a note. Falls back to a vault-wide basename search so a bare name
    or `[[wikilink]]` still resolves when the folder is unknown."""
    target = _resolve(path)
    if target is None:
        return f"Refused: '{path}' escapes the vault."
    if not target.is_file():
        found = _find_by_name(path)
        if found is None:
            return f"Note not found: {path}"
        target = found
    try:
        text = target.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        return f"Cannot read {path}: {e}"
    rel = target.relative_to(_vault_root())
    return f"{rel}\n\n{text}"


def note_write(path: str, content: str) -> str:
    """Create or overwrite a note. Parent folders are created as needed."""
    target = _resolve(path)
    if target is None:
        return f"Refused: '{path}' escapes the vault."
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as e:
        return f"Error writing {path}: {e}"
    return f"Note written: {target.relative_to(_vault_root())}"


def note_append(path: str, content: str) -> str:
    """Append a block to a note, creating it if missing. A blank line is
    inserted between existing content and the new block."""
    target = _resolve(path)
    if target is None:
        return f"Refused: '{path}' escapes the vault."
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        prefix = ""
        if target.is_file():
            existing = target.read_text(encoding="utf-8")
            if existing and not existing.endswith("\n\n"):
                prefix = "\n" if existing.endswith("\n") else "\n\n"
        with target.open("a", encoding="utf-8") as f:
            f.write(prefix + content + "\n")
    except OSError as e:
        return f"Error appending to {path}: {e}"
    return f"Appended to: {target.relative_to(_vault_root())}"


def note_list(folder: str | None = None) -> str:
    """List notes in the vault, or in a subfolder. Returns vault-relative
    paths, one per line."""
    root = _vault_root()
    scan = root
    if folder:
        resolved = _resolve(folder.rstrip("/") + "/_")
        if resolved is None or not resolved.parent.is_dir():
            return f"Folder not found: {folder}"
        scan = resolved.parent
    notes = sorted(str(p.relative_to(root)) for p in _iter_notes(scan))
    if not notes:
        return "No notes found."
    return "\n".join(notes)


def note_search(query: str, folder: str | None = None) -> str:
    """Search note contents (case-insensitive regex). Returns matching
    vault-relative paths plus a snippet from the first hit in each."""
    root = _vault_root()
    scan = root
    if folder:
        scan = (root / folder.strip("/")).resolve()
        try:
            scan.relative_to(root.resolve())
        except ValueError:
            return f"Refused: '{folder}' escapes the vault."
        if not scan.is_dir():
            return f"Folder not found: {folder}"
    try:
        regex = re.compile(query, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex: {e}"

    hits: list[str] = []
    for fp in _iter_notes(scan):
        try:
            if fp.stat().st_size > MAX_FILE_BYTES:
                continue
            text = fp.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        match = regex.search(text)
        if not match:
            continue
        start = max(match.start() - 120, 0)
        end = min(match.end() + 280, len(text))
        snippet = text[start:end].replace("\n", " ").strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet += "..."
        hits.append(f"{fp.relative_to(root)}\n  {snippet}")
        if len(hits) >= MAX_RESULTS:
            break
    return "\n\n".join(hits) if hits else "No matches in vault."


def daily_note(content: str | None = None) -> str:
    """Open today's daily note. With `content`, append it first. Returns the
    note's full content either way."""
    target = _vault_root() / MIA_DAILY_DIR / f"{date.today().isoformat()}.md"
    rel = target.relative_to(_vault_root())
    if content:
        result = note_append(str(rel), content)
        if result.startswith("Error"):
            return result
    if not target.is_file():
        return f"{rel} (empty)"
    return f"{rel}\n\n{target.read_text(encoding='utf-8')}"


# ---------------------------------------------------------------------------
# Adapters + registration
# ---------------------------------------------------------------------------


def _read_adapter(args: dict, _cwd: str):
    return note_read(args["path"])


def _write_adapter(args: dict, _cwd: str):
    return note_write(args["path"], args["content"])


def _append_adapter(args: dict, _cwd: str):
    return note_append(args["path"], args["content"])


def _list_adapter(args: dict, _cwd: str):
    return note_list(args.get("folder"))


def _search_adapter(args: dict, _cwd: str):
    return note_search(args["query"], args.get("folder"))


def _daily_adapter(args: dict, _cwd: str):
    return daily_note(args.get("content"))


register(
    name="note_read",
    description=(
        "Read a note from the user's personal Obsidian vault. Accepts a bare "
        "name ('ideas'), a wikilink ('[[ideas]]'), or a subpath "
        "('notes/ideas.md'). Falls back to a vault-wide search by name."
    ),
    adapter=_read_adapter,
    properties={"path": {"type": "string", "description": "Note name, wikilink, or vault-relative path"}},
    required=["path"],
    read_only=True,
)

register(
    name="note_search",
    description=(
        "Search the contents of the user's personal Obsidian vault "
        "(case-insensitive regex). Returns matching note paths with snippets. "
        "Use this to find what the user has written before answering from memory."
    ),
    adapter=_search_adapter,
    properties={
        "query": {"type": "string", "description": "Search term or regex"},
        "folder": {"type": "string", "description": "Optional vault subfolder to limit the search to"},
    },
    required=["query"],
    read_only=True,
)

register(
    name="note_list",
    description="List notes in the user's personal Obsidian vault, optionally within a subfolder.",
    adapter=_list_adapter,
    properties={"folder": {"type": "string", "description": "Optional vault subfolder"}},
    read_only=True,
)

register(
    name="note_write",
    description=(
        "Create or overwrite a note in the user's personal Obsidian vault. "
        "Parent folders are created automatically. Overwrites the whole file — "
        "read it first if you intend to preserve existing content."
    ),
    adapter=_write_adapter,
    properties={
        "path": {"type": "string", "description": "Note name or vault-relative path"},
        "content": {"type": "string", "description": "Full note content (Markdown)"},
    },
    required=["path", "content"],
)

register(
    name="note_append",
    description=(
        "Append a block of Markdown to a note in the user's personal Obsidian "
        "vault, creating the note if it does not exist. Use for running logs, "
        "lists, and incremental notes."
    ),
    adapter=_append_adapter,
    properties={
        "path": {"type": "string", "description": "Note name or vault-relative path"},
        "content": {"type": "string", "description": "Markdown block to append"},
    },
    required=["path", "content"],
)

register(
    name="daily_note",
    description=(
        "Open today's daily note in the user's personal Obsidian vault. With "
        "`content`, append it to the note first. Returns the note's content."
    ),
    adapter=_daily_adapter,
    properties={"content": {"type": "string", "description": "Optional Markdown to append to today's note"}},
)
