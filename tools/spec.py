"""Tool registration and schema builder.

Every tool module calls `register()` at import time. `repl/tool_registry.py`
pulls the aggregated schemas and adapters from `get_registry()` so adding a
tool is a single-file change.

Adapters are small named functions (not lambdas) so test monkeypatching works.
The signature is `adapter(args: dict, cwd: str) -> str`.
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Tool:
    name: str
    schema: dict[str, Any]
    adapter: Callable[[dict, str], Any]
    read_only: bool = False


# Module-level registry populated at import time by each tool submodule.
_registry: dict[str, Tool] = {}


def _build_schema(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the OpenAI function-calling schema dict.

    Every tool schema has the same wrapper shape; only description, properties,
    and required vary. This helper keeps register() calls short and avoids
    copy-paste drift across modules."""
    params: dict[str, Any] = {"type": "object", "properties": properties or {}}
    if required:
        params["required"] = required
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params,
        },
    }


def register(
    name: str,
    description: str,
    adapter: Callable[[dict, str], Any],
    *,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
    read_only: bool = False,
) -> Callable[[dict, str], Any]:
    """Register a tool and return the adapter for convenient `register(...)`
    usage as a decorator or direct call.

    Raises ValueError on duplicate name so a copy-paste mistake surfaces
    immediately at import time instead of silently overwriting.
    """
    if name in _registry:
        raise ValueError(f"tool {name!r} already registered")
    schema = _build_schema(name, description, properties, required)
    _registry[name] = Tool(name=name, schema=schema, adapter=adapter, read_only=read_only)
    return adapter


def get_registry() -> dict[str, Tool]:
    """Return a shallow copy of the current registry."""
    return dict(_registry)
