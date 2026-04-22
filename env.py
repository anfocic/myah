"""Tiny `.env` loader.

Pedagogical on purpose: small enough to read in one sitting, good enough for
the harness's local-development needs, and avoids pulling in another runtime
dependency just to populate `os.environ`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_VALID_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ROOT = Path(__file__).resolve().parent


def _default_dotenv_paths() -> list[Path]:
    candidates = [Path.cwd() / ".env", _ROOT / ".env"]
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _strip_inline_comment(value: str) -> str:
    for i, ch in enumerate(value):
        if ch == "#" and i > 0 and value[i - 1].isspace():
            return value[:i].rstrip()
    return value


def _parse_line(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[len("export ") :].lstrip()
    if "=" not in text:
        return None

    key, value = text.split("=", 1)
    key = key.strip()
    if not _VALID_KEY.match(key):
        return None

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    else:
        value = _strip_inline_comment(value)
    return key, value


def load_dotenv(path: str | os.PathLike[str] | None = None) -> None:
    """Populate missing env vars from a `.env` file.

    Existing process env wins over file values. If `path` is omitted, the
    loader checks `<cwd>/.env` first, then falls back to the repo-root `.env`.
    Missing files are ignored.
    """
    paths = [Path(path)] if path is not None else _default_dotenv_paths()
    for dotenv_path in paths:
        try:
            lines = dotenv_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            continue
        except OSError:
            continue

        for line in lines:
            parsed = _parse_line(line)
            if parsed is None:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)


__all__ = ["load_dotenv"]
