"""Token math + result truncation. Kept separate from the loop because
both are utilities used by multiple layers (context management, the REPL
status line, the tool-result size cap)."""
import json

import tiktoken

from config import MODEL_NAME, TOOL_RESULT_MAX_BYTES

# OpenAI chat format overhead per message (role framing + separators).
# Matches the cookbook approximation used across the ecosystem.
PER_MESSAGE_TOKENS = 3
PER_REPLY_TOKENS = 3

_encoding_cache: dict[str, tiktoken.Encoding] = {}


def _get_encoding(model_name: str = MODEL_NAME) -> tiktoken.Encoding:
    """Return a cached tiktoken encoding for the given model. Falls back to
    cl100k_base when the model isn't in tiktoken's registry."""
    enc = _encoding_cache.get(model_name)
    if enc is not None:
        return enc
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    _encoding_cache[model_name] = enc
    return enc


def _estimate_tokens_char4(messages: list) -> int:
    """Rough token count: ~1 token per 4 characters of message content.
    Private fallback when tiktoken is unavailable or the caller only
    wants a cheap heuristic."""
    total = sum(len(m.get("content") or "") for m in messages)
    return total // 4


def count_message_tokens(msg: dict, encoding: tiktoken.Encoding | None = None) -> int:
    """Count tokens in a single message, including role framing overhead
    and serialized tool_calls if present."""
    if encoding is None:
        encoding = _get_encoding()
    total = PER_MESSAGE_TOKENS
    content = msg.get("content")
    if content:
        total += len(encoding.encode(content))
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        total += len(encoding.encode(json.dumps(tool_calls)))
    if msg.get("name"):
        total += 1
    return total


def count_tool_schema_tokens(tools: list | None, encoding: tiktoken.Encoding | None = None) -> int:
    """Count tokens contributed by the tool schema definitions."""
    if not tools:
        return 0
    if encoding is None:
        encoding = _get_encoding()
    return len(encoding.encode(json.dumps(tools)))


def count_tokens(
    messages: list,
    tools: list | None = None,
    provider=None,
) -> int:
    """Best-effort token count for `messages` + optional `tools`.

    Priority:
    1. Provider's exact tokenizer (if given and it succeeds).
    2. Local tiktoken with per-message overhead + tool schema counting.
    3. Char/4 fallback (if tiktoken somehow fails).

    Callers in the hot loop (e.g. `run_agent` iterating) should pass
    `provider=None` to avoid network latency from provider-side counting.
    Callers doing one-off reporting (e.g. `/context`) should pass the
    provider to get exact counts when the backend supports them locally.
    """
    if provider is not None:
        try:
            return provider.count_tokens(messages, tools)
        except Exception:
            pass

    model_name = MODEL_NAME if provider is None else provider.model
    try:
        encoding = _get_encoding(model_name)
    except Exception:
        return _estimate_tokens_char4(messages)

    total = PER_REPLY_TOKENS
    for msg in messages:
        total += count_message_tokens(msg, encoding)
    total += count_tool_schema_tokens(tools, encoding)
    return total


def estimate_tokens(messages: list) -> int:
    """Backward-compatible estimator. Uses tiktoken when available,
    otherwise char/4. Does not count tool schemas — callers that need
    tool-aware counting should use `count_tokens(messages, tools=…)`.
    """
    return count_tokens(messages)


def truncate_tool_result(result: str, max_bytes: int = TOOL_RESULT_MAX_BYTES) -> str:
    """Cap a tool result's size so one giant read_file can't blow the ctx
    window. Keeps head + tail (the useful bits are usually at the edges)."""
    if len(result) <= max_bytes:
        return result
    dropped = len(result) - max_bytes
    head = max_bytes // 2
    tail = max_bytes - head
    return (
        result[:head]
        + f"\n\n...[truncated {dropped} chars]...\n\n"
        + result[-tail:]
    )
