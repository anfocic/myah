# tools/vault.py
"""Vault search — query the project's knowledge base for templates, patterns,
examples, and prior decisions.

The vault lives at `vault/` (sibling to `CLAUDE.md`). It is gitignored from
the main repo and maintained independently. The harness does not auto-load
vault content into context; the model must call `vault_search` when it needs
to check for existing patterns, templates, or documented decisions before
implementing new code.
"""

import re
from pathlib import Path

from tools.spec import register

VAULT_DIR = Path(__file__).resolve().parent.parent / "vault"
SKIP_DIRS = {".git", "raw", "__pycache__"}
MAX_FILE_BYTES = 500_000
MAX_RESULTS = 20


def _iter_vault_files(root: Path):
    if not root.exists():
        return
    for p in root.rglob("*.md"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(root).parts
        if any(part in SKIP_DIRS for part in rel_parts[:-1]):
            continue
        yield p


def vault_search(query: str, category: str | None = None):
    """Search the project vault for relevant knowledge.

    Looks for `query` (case-insensitive, regex-supported) across all markdown
    files under `vault/`. Returns matching file paths plus a short snippet from
    the first hit in each file.

    `category` optionally narrows the search to a vault subdirectory:
    - "code"      → vault/wiki/code/      (code subsystem reference)
    - "meta"      → vault/wiki/meta/      (decisions, patterns, gotchas)
    - "plans"     → vault/wiki/plans/     (engineering plans)
    - "templates" → vault/templates/      (page templates)
    """
    root = VAULT_DIR
    if category:
        category_map = {
            "code": "wiki/code",
            "meta": "wiki/meta",
            "plans": "wiki/plans",
            "templates": "templates",
        }
        mapped = category_map.get(category)
        if mapped:
            root = VAULT_DIR / mapped

    if not root.exists():
        return f"Vault path not found: {root}"

    try:
        regex = re.compile(query, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex: {e}"

    hits: list[str] = []
    for fp in _iter_vault_files(root):
        try:
            if fp.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue

        try:
            text = fp.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        match = regex.search(text)
        if not match:
            continue

        rel = fp.relative_to(VAULT_DIR)
        start = max(match.start() - 120, 0)
        end = min(match.end() + 280, len(text))
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        snippet = snippet.replace("\n", " ")

        hits.append(f"{rel}\n  {snippet.strip()}")
        if len(hits) >= MAX_RESULTS:
            break

    if not hits:
        return "No matches in vault."

    return "\n\n".join(hits)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def _vault_search_adapter(args: dict, cwd: str):
    return vault_search(
        args["query"],
        args.get("category"),
    )


register(
    name="vault_search",
    description=(
        "Search the project's knowledge vault for templates, patterns, examples, "
        "decisions, or prior documentation. Call this BEFORE implementing new "
        "features to check if a pattern, template, or plan already exists. "
        "Categories: 'code' (subsystem reference), 'meta' (decisions/patterns/gotchas), "
        "'plans' (engineering plans), 'templates' (page templates)."
    ),
    adapter=_vault_search_adapter,
    properties={
        "query": {
            "type": "string",
            "description": "Search term or regex to match against vault markdown files",
        },
        "category": {
            "type": "string",
            "description": "Optional category filter: 'code', 'meta', 'plans', or 'templates'",
        },
    },
    required=["query"],
    read_only=True,
)
