"""Tests for providers/pricing.py — lookup, cost math, formatting."""
from providers.pricing import (
    ModelPrice,
    compute_cost_usd,
    format_cost_usd,
    lookup_price,
)

# ── lookup_price ─────────────────────────────────────────────────────────────


def test_lookup_known_openai_model():
    price = lookup_price("openai", "gpt-4o-mini")
    assert isinstance(price, ModelPrice)
    assert price.input_per_mtok == 0.15
    assert price.output_per_mtok == 0.60


def test_lookup_known_anthropic_model():
    price = lookup_price("anthropic", "claude-opus-4-7")
    assert price is not None
    assert price.input_per_mtok == 15.0
    assert price.output_per_mtok == 75.0


def test_lookup_unknown_model_under_known_provider():
    assert lookup_price("openai", "no-such-model") is None


def test_lookup_unknown_provider():
    assert lookup_price("nopenai", "gpt-4o") is None


def test_lookup_ollama_is_free():
    price = lookup_price("ollama", "qwen2.5:7b-instruct")
    assert price is not None
    assert price.input_per_mtok == 0.0
    assert price.output_per_mtok == 0.0


# ── compute_cost_usd ─────────────────────────────────────────────────────────


def test_cost_known_model_math():
    # gpt-4o-mini: $0.15 in / $0.60 out per 1M tokens
    cost = compute_cost_usd("openai", "gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == 0.75


def test_cost_partial_tokens():
    # 100k input + 50k output of gpt-4o-mini = 0.015 + 0.030 = 0.045
    cost = compute_cost_usd("openai", "gpt-4o-mini", 100_000, 50_000)
    assert round(cost, 4) == 0.045


def test_cost_zero_tokens_returns_zero():
    cost = compute_cost_usd("openai", "gpt-4o-mini", 0, 0)
    assert cost == 0.0


def test_cost_none_token_counts_treated_as_zero():
    cost = compute_cost_usd("openai", "gpt-4o-mini", None, None)
    assert cost == 0.0


def test_cost_unknown_model_returns_none():
    cost = compute_cost_usd("openai", "no-such-model", 1_000_000, 1_000_000)
    assert cost is None


def test_cost_ollama_always_zero():
    cost = compute_cost_usd("ollama", "any-model", 1_000_000, 1_000_000)
    assert cost == 0.0


# ── format_cost_usd ──────────────────────────────────────────────────────────


def test_format_none_shows_dash():
    assert format_cost_usd(None) == "—"


def test_format_zero_shows_dollar_zero():
    assert format_cost_usd(0.0) == "$0"


def test_format_microcents_shows_5_decimals():
    assert format_cost_usd(0.00123) == "$0.00123"


def test_format_cents_shows_4_decimals():
    assert format_cost_usd(0.0456) == "$0.0456"


def test_format_dollars_shows_2_decimals():
    assert format_cost_usd(1.234) == "$1.23"
    assert format_cost_usd(42.5) == "$42.50"
