"""Direct Anthropic Messages API adapter.

Why a first-party adapter (vs. routing through openai-compat):

- **System prompt is a top-level field**, not a message. The adapter has
  to lift any `{"role": "system", ...}` entries out of the message list
  into the request's `system` key.
- **Tool calls are typed content blocks**, not a sibling `tool_calls`
  array. An assistant turn that used a tool becomes
  `{"role": "assistant", "content": [{"type": "text", ...},
  {"type": "tool_use", ...}]}`. The reply is a user turn with
  `tool_result` blocks.
- **Streaming is event-tagged SSE**. Each event has a named type
  (`message_start`, `content_block_delta`, `message_delta`, ...) plus a
  JSON payload. Tool arguments stream as `input_json_delta` fragments
  that must be buffered and JSON-parsed at block close.
- **Tool schema shape** is `{name, description, input_schema}`, not
  `{type, function: {...}}`. We translate on the way out.

`AnthropicProvider` translates between Myah's Ollama-shaped internal
format and the Anthropic wire format, and normalizes streaming events
down to the same `StreamChunk` every other adapter yields. Everything
above `providers/` stays Anthropic-agnostic.

Not implemented yet: prompt caching (`cache_control` breakpoints),
extended thinking blocks (`thinking`), the Batches API. All fit cleanly
on top of this shape when a use case appears.
"""

import json
from collections.abc import Iterator

import httpx

from .base import Provider, ProviderError, StreamChunk, ToolCall, Usage

ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.anthropic.com/v1",
        max_tokens: int = 4096,
        context_size: int = 200_000,
    ):
        if not api_key:
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set — export it or use another provider"
            )
        self.model = model
        self.context_size = context_size
        self._base = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
        # Long read timeout so streaming doesn't trip mid-response.
        self._client = httpx.Client(timeout=httpx.Timeout(30.0, read=None))

    def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        num_ctx: int,  # noqa: ARG002 — Anthropic manages context server-side
    ) -> Iterator[StreamChunk]:
        system, translated = _translate_messages(messages)
        payload: dict = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": translated,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = _translate_tools(tools)

        try:
            with self._client.stream(
                "POST",
                f"{self._base}/messages",
                json=payload,
                headers=self._headers,
            ) as r:
                if r.status_code >= 400:
                    body = r.read().decode(errors="replace")
                    raise ProviderError(f"HTTP {r.status_code}: {body[:500]}")
                yield from _parse_sse(r.iter_lines())
        except httpx.ConnectError as e:
            raise ProviderError(
                f"anthropic unreachable at {self._base}: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise ProviderError(f"anthropic timeout: {e}") from e

    def count_tokens(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> int:
        """Anthropic exposes POST /v1/messages/count_tokens — same request
        body as /v1/messages minus max_tokens, returns {input_tokens: N}.
        Free (not billed), GA since 2024. One network round-trip."""
        system, translated = _translate_messages(messages)
        payload: dict = {
            "model": self.model,
            "messages": translated,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = _translate_tools(tools)

        try:
            r = self._client.post(
                f"{self._base}/messages/count_tokens",
                json=payload,
                headers=self._headers,
            )
        except httpx.ConnectError as e:
            raise ProviderError(f"anthropic unreachable: {e}") from e
        if r.status_code >= 400:
            raise ProviderError(f"HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        tokens = data.get("input_tokens")
        if tokens is None:
            raise ProviderError(
                f"anthropic count_tokens missing input_tokens: {r.text[:200]}"
            )
        return int(tokens)

    def chat(self, messages: list[dict], num_ctx: int) -> tuple[str, Usage]:
        """Non-streaming one-shot. Used by the summarize-dropped-turns path
        in agent/context; everywhere else goes through stream_chat."""
        system, translated = _translate_messages(messages)
        payload: dict = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": translated,
        }
        if system:
            payload["system"] = system
        try:
            r = self._client.post(
                f"{self._base}/messages",
                json=payload,
                headers=self._headers,
            )
        except httpx.ConnectError as e:
            raise ProviderError(f"anthropic unreachable: {e}") from e
        if r.status_code >= 400:
            raise ProviderError(f"HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        # content is an array of blocks. Summarization paths only want text;
        # concat text blocks and ignore tool_use (we didn't ask for tools).
        text_parts = [
            b.get("text", "")
            for b in data.get("content", [])
            if b.get("type") == "text"
        ]
        content = "".join(text_parts).strip()
        usage = data.get("usage") or {}
        return content, Usage(
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
        )


def _parse_sse(lines: Iterator[str]) -> Iterator[StreamChunk]:
    """Parse Anthropic's event-tagged SSE stream into StreamChunks.

    Event sequence (per Anthropic docs):
      message_start         — first event, carries input_tokens in usage
      content_block_start   — begins a text or tool_use block
      content_block_delta   — text_delta or input_json_delta
      content_block_stop    — block complete; finalize tool_use here
      message_delta         — carries output_tokens updates
      message_stop          — final event
      ping                  — keepalive, skipped

    Tool arguments stream as partial-JSON string fragments (one per
    input_json_delta event). We buffer by block index and parse the
    accumulated string on content_block_stop, then emit a single
    StreamChunk with the completed ToolCall — same contract the
    openai-compat adapter follows, just with a different provider-native
    shape to reconcile."""
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_buf: dict[int, dict] = {}
    current_event: str | None = None

    for line in lines:
        if not line:
            continue
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
            continue
        if not line.startswith("data:"):
            continue
        data_str = line[len("data:"):].strip()
        try:
            evt = json.loads(data_str)
        except json.JSONDecodeError as e:
            raise ProviderError(f"bad SSE JSON: {e}: {data_str[:200]}") from e

        # `type` may be on the payload (Anthropic always includes it) or
        # fall back to the most recent `event:` line. Defensive against
        # proxies that strip one or the other.
        t = evt.get("type") or current_event

        if t == "message_start":
            usage = (evt.get("message") or {}).get("usage") or {}
            input_tokens = usage.get("input_tokens")
        elif t == "content_block_start":
            idx = evt.get("index", 0)
            block = evt.get("content_block") or {}
            if block.get("type") == "tool_use":
                tool_buf[idx] = {
                    "id": block.get("id"),
                    "name": block.get("name", ""),
                    "arg_parts": [],
                }
        elif t == "content_block_delta":
            idx = evt.get("index", 0)
            delta = evt.get("delta") or {}
            dt = delta.get("type")
            if dt == "text_delta":
                text = delta.get("text") or ""
                if text:
                    yield StreamChunk(content_delta=text)
            elif dt == "input_json_delta":
                partial = delta.get("partial_json") or ""
                if idx in tool_buf:
                    tool_buf[idx]["arg_parts"].append(partial)
        elif t == "content_block_stop":
            idx = evt.get("index", 0)
            if idx in tool_buf:
                slot = tool_buf.pop(idx)
                raw = "".join(slot["arg_parts"]) or "{}"
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    # Empty-input tool calls arrive as `""` (not `"{}"`);
                    # also defend against any server glitch in partial JSON.
                    args = {}
                yield StreamChunk(tool_calls=[
                    ToolCall(name=slot["name"], arguments=args)
                ])
        elif t == "message_delta":
            usage = evt.get("usage") or {}
            if "output_tokens" in usage:
                output_tokens = usage["output_tokens"]
        elif t == "message_stop":
            break
        # ping + anything we don't recognize is ignored on purpose

    yield StreamChunk(
        done=True,
        usage=Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        ),
    )


def _translate_tools(tools: list[dict]) -> list[dict]:
    """OpenAI tool schema → Anthropic tool schema.

    OpenAI: {type: "function", function: {name, description, parameters}}
    Anthropic: {name, description, input_schema}

    `parameters` and `input_schema` are both JSON Schema so no inner
    translation is needed."""
    out = []
    for t in tools:
        fn = t.get("function") or {}
        out.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {
                "type": "object",
                "properties": {},
            },
        })
    return out


def _translate_messages(
    messages: list[dict],
) -> tuple[str | None, list[dict]]:
    """Translate Myah's internal (Ollama-shaped) message list to Anthropic's.

    Internal shape (produced by agent/loop.py):
        {"role": "system", "content": "..."}
        {"role": "user", "content": "..."}
        {"role": "assistant", "content": "...", "tool_calls": [
            {"name": "...", "arguments": {...}}, ...]}
        {"role": "tool", "content": "..."}   # matches tool_calls[0]
        {"role": "tool", "content": "..."}   # matches tool_calls[1]
        {"role": "assistant", "content": "..."}

    Anthropic shape:
        system="..." (separate field, returned as the first tuple element)
        messages=[
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "..."},
                {"type": "tool_use", "id": "toolu_0", "name": "...",
                 "input": {...}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_0",
                 "content": "..."},
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": "..."},
            ]},
            {"role": "assistant", "content": "..."},
        ]

    Two subtleties:

    1. Anthropic requires consecutive tool_result blocks to live in a
       single user message — two separate user messages with one
       tool_result each is rejected. We coalesce them.
    2. Summaries injected by the §7 summarize-dropped path arrive as
       additional system-role messages in the middle of history; those
       get concatenated onto the leading system prompt with blank-line
       separators so Anthropic still sees a single system string.
    """
    system_parts: list[str] = []
    out: list[dict] = []
    pending_ids: list[str] = []
    counter = 0

    for msg in messages:
        role = msg.get("role")

        if role == "system":
            text = msg.get("content") or ""
            if text:
                system_parts.append(text)
            continue

        if role == "assistant" and msg.get("tool_calls"):
            content_blocks: list[dict] = []
            text = msg.get("content") or ""
            if text:
                content_blocks.append({"type": "text", "text": text})
            ids_for_this: list[str] = []
            for tc in msg["tool_calls"]:
                tid = f"toolu_{counter}"
                counter += 1
                ids_for_this.append(tid)
                content_blocks.append({
                    "type": "tool_use",
                    "id": tid,
                    "name": tc["name"],
                    "input": tc["arguments"] or {},
                })
            out.append({"role": "assistant", "content": content_blocks})
            pending_ids = ids_for_this
            continue

        if role == "tool":
            if pending_ids:
                tid = pending_ids.pop(0)
            else:
                # Defensive: should not happen under run_agent's contract,
                # but a truncated session could deliver a tool message
                # without a preceding tool_use. Synthesize an id so the
                # payload shape stays valid.
                tid = f"toolu_{counter}"
                counter += 1
            block = {
                "type": "tool_result",
                "tool_use_id": tid,
                "content": msg.get("content") or "",
            }
            # Coalesce consecutive tool_result blocks into one user turn.
            # Anthropic rejects adjacent user turns where both carry only
            # tool_result content.
            if (
                out
                and out[-1]["role"] == "user"
                and isinstance(out[-1]["content"], list)
                and all(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in out[-1]["content"]
                )
            ):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
            continue

        # Plain user / assistant text messages.
        out.append({"role": role, "content": msg.get("content") or ""})

    system = "\n\n".join(system_parts) if system_parts else None
    return system, out
