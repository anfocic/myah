"""Per-model context window lookup.

Hosted providers offer wildly different context sizes across their model
catalog: gpt-4o is 128K but gpt-3.5-turbo is 16K; claude-sonnet-4 is 1M
but claude-haiku-4 is 200K. A single per-provider literal lies as soon as
the user `/model swap`s within a family.

`lookup` matches the model name against substrings here; first hit wins,
so list more-specific prefixes before generic ones (`claude-sonnet-4` >
`claude-sonnet`). Unknown models fall back to the caller's `default`.
"""

# Substrings searched against `model.lower()`. Order matters: the first
# matching prefix wins, so put narrower keys before broader ones.
MODEL_CONTEXT_SIZES: dict[str, int] = {
    # Anthropic — Sonnet 4 has the 1M extended window; Opus + Haiku stay at 200K.
    "claude-sonnet-4": 1_000_000,
    "claude-opus-4": 200_000,
    "claude-haiku-4": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    # OpenAI — gpt-4o, gpt-4-turbo, gpt-4.1 = 128K; o1 = 200K; gpt-3.5 = 16K.
    "o1": 200_000,
    "gpt-4o": 128_000,
    "gpt-4.1": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 128_000,
    "gpt-3.5": 16_000,
    # DeepSeek — chat + reasoner both ship at 64K.
    "deepseek-reasoner": 64_000,
    "deepseek-chat": 64_000,
    # Gemini — 2.x family runs 1M; 1.5 family runs 1M; gemma local is 32K.
    "gemini-2": 1_000_000,
    "gemini-1.5": 1_000_000,
    "gemma": 32_000,
    # Moonshot AI — kimi-k2 is 128K.
    "kimi": 128_000,
    # Qwen3 — 32K standard, 128K for the long-context variant.
    "qwen3": 32_000,
}


def lookup(model: str, *, default: int) -> int:
    """Return the published context window for `model`, or `default` if
    no entry matches. Substring match against the lowercased model name;
    first hit wins, so the table is ordered narrowest-first."""
    needle = model.lower()
    for prefix, size in MODEL_CONTEXT_SIZES.items():
        if prefix in needle:
            return size
    return default
