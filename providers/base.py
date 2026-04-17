"""Normalized shapes every provider adapter yields / accepts.

The point of this module is that nothing above `providers/` knows or cares
which backend it's talking to. Adapters translate their provider's native
streaming shape into `StreamChunk`s with the contract described below.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    name: str
    arguments: dict  # already json.loads'd by the adapter


@dataclass
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class StreamChunk:
    # Incremental text the UI renders as it arrives. "" (never None) so callers
    # can truthiness-check without guarding against None.
    content_delta: str = ""
    # Only ever populated when a *complete* tool call is available. Adapters
    # that receive tool calls as incremental deltas (OpenAI) buffer internally
    # and emit the finished call here in one chunk.
    tool_calls: list[ToolCall] = field(default_factory=list)
    # True exactly once, on the last chunk. `usage` is populated iff the
    # provider surfaced it (Ollama always, OpenAI-compat when the server
    # supports stream_options.include_usage).
    done: bool = False
    usage: Usage | None = None


class Provider(Protocol):
    """Adapter contract. Implementations live in sibling modules."""

    name: str
    model: str

    def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        num_ctx: int,
    ) -> Iterator[StreamChunk]: ...

    def chat(
        self,
        messages: list[dict],
        num_ctx: int,
    ) -> tuple[str, Usage]: ...


class ProviderError(Exception):
    """Anything the adapter couldn't translate: connect refused, auth failure,
    malformed SSE, timeout, etc. One class is enough for a learning harness —
    the message string carries the specifics."""
