"""Ollama adapter — wraps `ollama.chat` into the Provider protocol.

Thin by design: the native shape (`chunk.message.content`,
`chunk.message.tool_calls`, `response.prompt_eval_count`, ...) gets
translated into the neutral dataclasses in `base.py`. Nothing above this
file sees Ollama-specific attributes."""

from collections.abc import Iterator

from .base import Provider, ProviderError, StreamChunk, ToolCall, Usage


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, model: str, base_url: str | None = None):
        try:
            import ollama
        except ImportError as e:  # pragma: no cover - guarded for the openai-compat-only install path
            raise ProviderError(
                "ollama package not installed; run `pip install ollama`"
            ) from e
        self._ollama = ollama
        self._client = ollama.Client(host=base_url) if base_url else ollama
        self.model = model

    def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        num_ctx: int,
    ) -> Iterator[StreamChunk]:
        try:
            gen = self._client.chat(
                model=self.model,
                messages=_strip_internal_tool_calls(messages),
                tools=tools,
                options={"num_ctx": num_ctx},
                stream=True,
            )
            final = None
            for chunk in gen:
                final = chunk
                msg = chunk.message
                # Ollama's SDK types arguments loosely (`Any | Mapping[str, Any]`);
                # coerce to a concrete dict so downstream code (incl. the
                # openai-compat adapter's JSON serialization) can rely on dict
                # semantics. dict(mapping) is a no-op for dicts and copies otherwise.
                tool_calls = [
                    ToolCall(name=tc.function.name, arguments=dict(tc.function.arguments or {}))
                    for tc in (msg.tool_calls or [])
                ]
                yield StreamChunk(
                    content_delta=msg.content or "",
                    tool_calls=tool_calls,
                )
            yield StreamChunk(
                done=True,
                usage=Usage(
                    prompt_tokens=getattr(final, "prompt_eval_count", None),
                    completion_tokens=getattr(final, "eval_count", None),
                ),
            )
        except ConnectionError as e:
            raise ProviderError(f"ollama unreachable: {e}") from e
        except Exception as e:
            # Anything ollama's httpx wrapper raises — connect errors, timeouts,
            # protocol errors — surface as ProviderError so the agent loop has
            # one catch clause.
            if e.__class__.__module__.startswith("httpx"):
                raise ProviderError(f"ollama HTTP error: {e}") from e
            raise

    def chat(self, messages: list[dict], num_ctx: int) -> tuple[str, Usage]:
        try:
            response = self._client.chat(
                model=self.model,
                messages=_strip_internal_tool_calls(messages),
                options={"num_ctx": num_ctx},
            )
        except ConnectionError as e:
            raise ProviderError(f"ollama unreachable: {e}") from e
        return (
            (response.message.content or "").strip(),
            Usage(
                prompt_tokens=getattr(response, "prompt_eval_count", None),
                completion_tokens=getattr(response, "eval_count", None),
            ),
        )


def _strip_internal_tool_calls(messages: list[dict]) -> list[dict]:
    """agent.py stores tool_calls on assistant messages as `{"name", "arguments"}`
    so the openai-compat adapter can synthesize IDs. Ollama's pydantic validator
    requires the nested `{"function": {"name", "arguments"}}` shape — and the
    Ollama wire protocol doesn't actually need tool_calls on replayed assistant
    messages anyway (the tool role messages that follow carry the information).
    Cheapest fix: drop the field before sending."""
    out = []
    for m in messages:
        if m.get("role") == "assistant" and "tool_calls" in m:
            out.append({k: v for k, v in m.items() if k != "tool_calls"})
        else:
            out.append(m)
    return out
