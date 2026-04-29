"""Layered configuration loader.

Reads three JSON config files, merges them with deep-merge, and caches
the result. The merge order (later wins):

  1. Hardcoded defaults
  2. ~/.myah/config.json              (user-level)
  3. .myah/config.json                (project-level, committed)
  4. .myah/config.local.json          (project-level, gitignored)

Environment variables and CLI flags are NOT handled here — they are
applied in config.py as the final override layer, preserving backward
compat with all existing `from config import X` sites.
"""
import json
from pathlib import Path

USER_CONFIG = Path.home() / ".myah" / "config.json"
PROJECT_CONFIG = Path(".myah") / "config.json"
PROJECT_LOCAL_CONFIG = Path(".myah") / "config.local.json"

# Flat defaults that mirror the current hardcoded values in config.py.
# Nested to support grouping; scalars for simple values.
DEFAULTS: dict = {
    "provider": {
        "default": "openai-compat",
        "models": {
            "ollama": "qwen/qwen3.5-9b",
            "openai-compat": "google/gemma-4-e4b",
        },
    },
    "context": {
        "num_ctx": 32768,
        "reserved_completion_tokens": 1024,
        "tool_result_max_bytes": 10_000,
    },
    "behavior": {
        "stream_delay_ms": 100,
        "max_completion_tokens": 4096,
        "max_iterations": 50,
        "confirm_mutating_tools": True,
    },
    "display": {
        "show_turn_header": True,
        "show_turn_footer": True,
        "tool_colors": True,
    },
    "paths": {
        "session_file": "~/.mia_session.json",
        "input_history": "~/.mia_input_history",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge. `override` wins on conflicts.
    Returns a new dict; neither input is mutated."""
    result = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_layer(path: Path) -> dict | None:
    """Read a single JSON config file. Returns None if missing or invalid."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _merge_layers() -> tuple[dict, dict[str, str]]:
    """Read all config layers and return (merged_config, provenance).

    `provenance` maps dot-path keys to the source label that set them:
    "default", "user", "project", or "project-local".
    """
    merged = dict(DEFAULTS)
    provenance: dict[str, str] = {}

    def _record_flat(src: dict, prefix: str = "", label: str = "default") -> None:
        for k, v in src.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _record_flat(v, path, label)
            else:
                provenance[path] = label

    # Record all leaf defaults
    _record_flat(DEFAULTS, label="default")

    layers = [
        (USER_CONFIG, "user"),
        (PROJECT_CONFIG, "project"),
        (PROJECT_LOCAL_CONFIG, "project-local"),
    ]

    for path, label in layers:
        layer = _load_layer(path)
        if layer is None:
            continue
        merged = _deep_merge(merged, layer)
        # Track provenance: every key in this layer overrides whatever was there.
        _record_flat(layer, label=label)

    return merged, provenance


_config_cache: dict | None = None
_provenance_cache: dict[str, str] | None = None


def load_config() -> dict:
    """Read all layers and return the merged config dict."""
    merged, prov = _merge_layers()
    global _config_cache, _provenance_cache
    _config_cache = merged
    _provenance_cache = prov
    return merged


def get_config() -> dict:
    """Cached access. Returns the same dict on repeated calls without
    re-reading files. Call `reload_config()` to invalidate."""
    if _config_cache is None:
        return load_config()
    return _config_cache


def get_provenance() -> dict[str, str]:
    """Return the provenance map for the current cached config."""
    if _provenance_cache is None:
        load_config()
    assert _provenance_cache is not None
    return _provenance_cache


def reload_config() -> dict:
    """Invalidate cache and re-read all config files."""
    global _config_cache, _provenance_cache
    _config_cache = None
    _provenance_cache = None
    return load_config()


def config_paths() -> dict[str, Path]:
    """Return a mapping of label → Path for all config file locations."""
    return {
        "user": USER_CONFIG,
        "project": PROJECT_CONFIG,
        "project-local": PROJECT_LOCAL_CONFIG,
    }
