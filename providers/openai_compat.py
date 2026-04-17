"""Generic OpenAI-compatible HTTP adapter.

One adapter wraps dozens of servers: llama.cpp, LM Studio, vLLM, Groq,
Together, Fireworks, DeepInfra, OpenRouter, OpenAI itself. All speak the
same `/v1/chat/completions` endpoint with SSE streaming.

Two tricks live in here that Ollama's SDK hid for free:

1. **Tool-call delta accumulation.** OpenAI streams tool calls incrementally
   across many SSE events — name in one delta, arguments as partial JSON
   chunks across N more. The adapter buffers by `index` and only emits a
   complete `ToolCall` on `finish_reason == "tool_calls"`.

2. **`tool_call_id` synthesis.** The OpenAI spec requires every tool-result
   message to carry the `tool_call_id` of the assistant's preceding call,
   and the assistant message itself to declare its `tool_calls` inline.
   `agent.py` stores tool calls on the assistant message in our internal
   format (see §Message translation in the plan); `_translate_messages`
   walks the list once per send and synthesizes IDs (`call_0`, `call_1`,
   ...) so each subsequent `role:tool` lines up. IDs are opaque to the
   server — it just needs them to round-trip within a single request.
"""

import json
from typing import Iterator

import httpx

from .base import Provider, ProviderError, StreamChunk, ToolCall, Usage


class OpenAICompatProvider(Provider):
    name = "openai-compat"

    def __init__(self, model: str, base_url: str, api_key: str = ""):
        self.model = model
        self._base = base_url.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        # Long read timeout so streaming doesn't time out mid-response.
        self._client = httpx.Client(timeout=httpx.Timeout(30.0, read=None))

    def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        num_ctx: int,  # noqa: ARG002 — OpenAI has no standard num_ctx field
    ) -> Iterator[StreamChunk]:
        payload: dict = {
            "model": self.model,
            "messages": _translate_messages(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools  # OpenAI schema happens to match Ollama's

        try:
            with self._client.stream(
                "POST",
                f"{self._base}/chat/completions",
                json=payload,
                headers=self._headers,
            ) as r:
                if r.status_code >= 400:
                    body = r.read().decode(errors="replace")
                    raise ProviderError(f"HTTP {r.status_code}: {body[:500]}")
                yield from _parse_sse(r.iter_lines())
        except httpx.ConnectError as e:
            raise ProviderError(f"openai-compat unreachable at {self._base}: {e}") from e
        except httpx.TimeoutException as e:
            raise ProviderError(f"openai-compat timeout: {e}") from e

    def chat(self, messages: list[dict], num_ctx: int) -> tuple[str, Usage]:
        payload = {
            "model": self.model,
            "messages": _translate_messages(messages),
            "stream": False,
        }
        try:
            r = self._client.post(
                f"{self._base}/chat/completions",
                json=payload,
                headers=self._headers,
            )
        except httpx.ConnectError as e:
            raise ProviderError(f"openai-compat unreachable at {self._base}: {e}") from e
        if r.status_code >= 400:
            raise ProviderError(f"HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        content = (data["choices"][0]["message"].get("content") or "").strip()
        usage = data.get("usage") or {}
        return content, Usage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


def _parse_sse(lines: Iterator[str]) -> Iterator[StreamChunk]:
    """Consume an SSE line iterator, yield normalized StreamChunks.

    Pulled out so it can be unit-tested against a fake byte stream without
    spinning up an HTTP server."""
    tool_buf: dict[int, dict] = {}
    usage: dict | None = None

    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            evt = json.loads(data)
        except json.JSONDecodeError as e:
            raise ProviderError(f"bad SSE JSON: {e}: {data[:200]}") from e

        if evt.get("usage"):
            usage = evt["usage"]

        choices = evt.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}

        if delta.get("content"):
            yield StreamChunk(content_delta=delta["content"])

        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = tool_buf.setdefault(idx, {"id": None, "name": "", "arg_parts": []})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            if fn.get("arguments") is not None:
                slot["arg_parts"].append(fn["arguments"])

        if choice.get("finish_reason") == "tool_calls" and tool_buf:
            yield StreamChunk(
                tool_calls=[
                    ToolCall(
                        name=slot["name"],
                        arguments=json.loads("".join(slot["arg_parts"]) or "{}"),
                    )
                    for _, slot in sorted(tool_buf.items())
                ]
            )
            tool_buf.clear()

    yield StreamChunk(
        done=True,
        usage=Usage(
            prompt_tokens=(usage or {}).get("prompt_tokens"),
            completion_tokens=(usage or {}).get("completion_tokens"),
        ),
    )


def _translate_messages(messages: list[dict]) -> list[dict]:
    """Translate from the internal (Ollama-shaped) message list to the
    OpenAI-compat shape. The only real work is synthesizing tool_call_ids
    so each assistant's tool_calls pair with the following tool messages.

    Internal shape (produced by agent.py):
        {"role": "system", "content": "..."}
        {"role": "user", "content": "..."}
        {"role": "assistant", "content": "...", "tool_calls": [
            {"name": "...", "arguments": {...}}, ...]}
        {"role": "tool", "content": "..."}    # matches tool_calls[0]
        {"role": "tool", "content": "..."}    # matches tool_calls[1]
        {"role": "assistant", "content": "..."}

    OpenAI shape:
        assistant.tool_calls = [{"id": "call_0", "type": "function",
                                 "function": {"name": ..., "arguments": json_str}}]
        tool.tool_call_id = "call_0"
    """
    out: list[dict] = []
    pending_ids: list[str] = []  # queue of IDs emitted by the last assistant
    counter = 0

    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            ids_for_this = []
            oai_tool_calls = []
            for tc in msg["tool_calls"]:
                tid = f"call_{counter}"
                counter += 1
                ids_for_this.append(tid)
                oai_tool_calls.append({
                    "id": tid,
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                })
            out.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": oai_tool_calls,
            })
            pending_ids = ids_for_this
        elif role == "tool":
            tid = pending_ids.pop(0) if pending_ids else f"call_{counter}"
            if not pending_ids and tid.startswith("call_"):
                counter += 1  # only bump if we minted a fresh one
            out.append({
                "role": "tool",
                "tool_call_id": tid,
                "content": msg.get("content") or "",
            })
        else:
            out.append({k: v for k, v in msg.items() if k in ("role", "content", "name")})

    return out
