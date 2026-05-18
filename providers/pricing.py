"""Per-model price catalog + cost computation.

Pricing is per-1-million-tokens in USD, separately for input (prompt)
and output (completion). Source: each provider's pricing page, current
as of the file's `LAST_UPDATED` constant. Numbers WILL drift; treat the
output as advisory and update when you notice it's wrong.

Lookup keys are `(provider_name, model_name)` where provider_name is
the same string each adapter reports as `Provider.name` and model_name
matches what the user typed into `/model`. Unknown combinations return
`None` so callers (loop, /stats) render an explicit "—" rather than
silently lying with a zero.

Local-only providers (ollama) are pinned to 0.0 across the board so the
"cost" surface still renders something useful (a $0 row signals the
spend is in electricity and disk, not API charges).
"""
from __future__ import annotations

from dataclasses import dataclass

LAST_UPDATED = "2026-05-18"


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens, split by direction."""

    input_per_mtok: float
    output_per_mtok: float


# Each provider gets its own table. Use the model strings we actually
# see in adapter `.model` fields, not marketing names.
_PRICES: dict[tuple[str, str], ModelPrice] = {
    # ── ollama ── local; no API cost
    # Catch-all via the provider-only fallback below; no per-model rows
    # needed.

    # ── OpenAI ── https://openai.com/api/pricing
    ("openai", "gpt-4o-mini"): ModelPrice(0.15, 0.60),
    ("openai", "gpt-4o"): ModelPrice(2.50, 10.00),
    ("openai", "gpt-4.1"): ModelPrice(2.00, 8.00),
    ("openai", "gpt-4.1-mini"): ModelPrice(0.40, 1.60),
    ("openai", "gpt-4.1-nano"): ModelPrice(0.10, 0.40),
    ("openai", "o3-mini"): ModelPrice(1.10, 4.40),
    ("openai", "o3"): ModelPrice(2.00, 8.00),

    # ── Anthropic ── https://www.anthropic.com/pricing
    ("anthropic", "claude-3-5-haiku-latest"): ModelPrice(0.80, 4.00),
    ("anthropic", "claude-3-5-sonnet-latest"): ModelPrice(3.00, 15.00),
    ("anthropic", "claude-haiku-4-5"): ModelPrice(1.00, 5.00),
    ("anthropic", "claude-sonnet-4-5"): ModelPrice(3.00, 15.00),
    ("anthropic", "claude-sonnet-4-6"): ModelPrice(3.00, 15.00),
    ("anthropic", "claude-opus-4-5"): ModelPrice(15.00, 75.00),
    ("anthropic", "claude-opus-4-6"): ModelPrice(15.00, 75.00),
    ("anthropic", "claude-opus-4-7"): ModelPrice(15.00, 75.00),

    # ── DeepSeek ── https://api-docs.deepseek.com/quick_start/pricing
    ("deepseek", "deepseek-chat"): ModelPrice(0.27, 1.10),
    ("deepseek", "deepseek-reasoner"): ModelPrice(0.55, 2.19),
}

# Providers whose models all cost the same per token, regardless of
# which specific model is loaded. ollama is local; treat as free.
_FLAT_PROVIDERS: dict[str, ModelPrice] = {
    "ollama": ModelPrice(0.0, 0.0),
}


def lookup_price(provider: str, model: str) -> ModelPrice | None:
    """Return the price for (provider, model), or None when unknown."""
    if price := _PRICES.get((provider, model)):
        return price
    return _FLAT_PROVIDERS.get(provider)


def compute_cost_usd(
    provider: str, model: str, prompt_tokens: int | None, completion_tokens: int | None,
) -> float | None:
    """Return the turn's USD cost or None when we can't price it.

    Returns None — not 0.0 — when the model is unknown so the UI can
    show "—" rather than a misleadingly cheap "$0.00". A real $0 cost
    (e.g. ollama) flows through as 0.0."""
    price = lookup_price(provider, model)
    if price is None:
        return None
    in_tok = prompt_tokens or 0
    out_tok = completion_tokens or 0
    return (in_tok * price.input_per_mtok + out_tok * price.output_per_mtok) / 1_000_000


def format_cost_usd(cost: float | None) -> str:
    """Human-readable cost. None → '—'. Values <$0.01 are rendered with
    enough precision to be useful (microcent-grained for a typical local
    or cheap-model turn)."""
    if cost is None:
        return "—"
    if cost == 0:
        return "$0"
    if cost < 0.01:
        return f"${cost:.5f}"
    if cost < 1:
        return f"${cost:.4f}"
    return f"${cost:.2f}"
