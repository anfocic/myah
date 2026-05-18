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
    version: str = "1"


# Module-level registry populated at import time by each tool submodule.
_registry: dict[str, Tool] = {}


def _build_schema(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
    version: str = "1",
) -> dict[str, Any]:
    """Assemble the OpenAI function-calling schema dict.

    Every tool schema has the same wrapper shape; only description, properties,
    and required vary. This helper keeps register() calls short and avoids
    copy-paste drift across modules.

    The version is surfaced to the model as a `(vN)` suffix on the description
    rather than a separate schema field. Provider adapters validate the
    function-calling shape strictly, so piggy-backing on description keeps the
    schema portable across Ollama / OpenAI / Anthropic / DeepSeek without
    extension fields. The suffix is unconditional so behavior is uniform and
    introspectable from inside the model's prompt.
    """
    versioned_description = f"{description} (v{version})"
    params: dict[str, Any] = {"type": "object", "properties": properties or {}}
    if required:
        params["required"] = required
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": versioned_description,
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
    version: str = "1",
) -> Callable[[dict, str], Any]:
    """Register a tool and return the adapter for convenient `register(...)`
    usage as a decorator or direct call.

    Raises ValueError on duplicate name so a copy-paste mistake surfaces
    immediately at import time instead of silently overwriting.

    `version` lets a tool evolve its semantics without breaking session files
    that recorded a call by name. Defaults to "1" so existing tool modules
    don't need edits; bump when behavior changes meaningfully.
    """
    if name in _registry:
        raise ValueError(f"tool {name!r} already registered")
    schema = _build_schema(name, description, properties, required, version)
    _registry[name] = Tool(
        name=name,
        schema=schema,
        adapter=adapter,
        read_only=read_only,
        version=version,
    )
    return adapter


def get_registry() -> dict[str, Tool]:
    """Return a shallow copy of the current registry."""
    return dict(_registry)
