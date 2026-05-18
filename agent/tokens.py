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

# Flat per-image token charge — matches OpenAI's low-res tile heuristic.
# Vision-capable providers report the exact count in their Usage objects,
# so this only matters for trim/microcompact decisions before the call.
# Crucially it lets us avoid tokenizing the base64 image body itself, which
# would otherwise add ~250K tokens per megabyte of image data.
IMAGE_TOKEN_CHARGE = 85

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


def _content_text_chars(content) -> int:
    """Sum the character count of every text-bearing piece of `content`,
    skipping image base64 payloads. Used by the char/4 fallback and any
    other path that needs a quick size proxy without crashing on lists."""
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                total += len(block.get("text") or "")
        return total
    return 0


def _content_image_count(content) -> int:
    """Number of image blocks in `content`. Returns 0 for string content
    or any non-list shape."""
    if not isinstance(content, list):
        return 0
    return sum(1 for b in content if isinstance(b, dict) and b.get("type") == "image")


def _estimate_tokens_char4(messages: list) -> int:
    """Rough token count: ~1 token per 4 characters of message content.
    Private fallback when tiktoken is unavailable or the caller only
    wants a cheap heuristic. List content (image attachments) counts
    text chars only and charges IMAGE_TOKEN_CHARGE flat per image."""
    total = 0
    for m in messages:
        content = m.get("content")
        total += _content_text_chars(content)
    chars = total // 4
    images = sum(_content_image_count(m.get("content")) for m in messages)
    return chars + images * IMAGE_TOKEN_CHARGE


def count_message_tokens(msg: dict, encoding: tiktoken.Encoding | None = None) -> int:
    """Count tokens in a single message, including role framing overhead
    and serialized tool_calls if present. List content (image
    attachments) tokenizes only text blocks and charges
    IMAGE_TOKEN_CHARGE per image — the base64 image body is
    deliberately NOT tokenized (a 3MB image → ~1M phantom tokens
    otherwise, destroying trim decisions before any provider call)."""
    if encoding is None:
        encoding = _get_encoding()
    total = PER_MESSAGE_TOKENS
    content = msg.get("content")
    if isinstance(content, str) and content:
        total += len(encoding.encode(content))
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = block.get("text") or ""
                if text:
                    total += len(encoding.encode(text))
            elif block.get("type") == "image":
                total += IMAGE_TOKEN_CHARGE
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
    model_name: str | None = None,
) -> int:
    """Best-effort token count for `messages` + optional `tools`.

    Priority:
    1. Provider's exact tokenizer (if given and it succeeds).
    2. Local tiktoken with per-message overhead + tool schema counting.
    3. Char/4 fallback (if tiktoken somehow fails).

    Callers in the hot loop (e.g. `run_agent` iterating) should pass
    `provider=None` and an explicit `model_name` to avoid network latency
    from provider-side counting while keeping the encoding cache keyed on
    the live model (so `/model` swaps don't drift).
    Callers doing one-off reporting (e.g. `/context`) should pass the
    provider to get exact counts when the backend supports them locally.
    """
    if provider is not None:
        try:
            return provider.count_tokens(messages, tools)
        except Exception:
            pass

    if model_name is None:
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
