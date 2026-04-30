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
import os
import shutil
import subprocess
from collections.abc import Iterator

import httpx

from config import MAX_COMPLETION_TOKENS

from .base import Provider, ProviderError, StreamChunk, ToolCall, Usage

# Per-message framing overhead for OpenAI chat models (from the OpenAI
# cookbook). 3 tokens wrap each message (<|im_start|>role<|im_sep|>content
# <|im_end|>); the assistant reply is primed with another 3. Exact for
# GPT-4/GPT-4o/GPT-3.5-turbo; a close approximation for non-OpenAI models
# (DeepSeek, Qwen) that still ride the chatml-ish structure.
_PER_MESSAGE_TOKENS = 3
_PER_REPLY_TOKENS = 3


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
            "max_tokens": MAX_COMPLETION_TOKENS,
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

    def count_tokens(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> int:
        """Local tiktoken count. Exact for OpenAI chat models; very close
        for other BPE-based servers (DeepSeek, Qwen via llama.cpp) that
        share cl100k_base-ish vocabularies. No network call."""
        try:
            import tiktoken
        except ImportError as e:
            raise ProviderError("tiktoken not installed; run `pip install tiktoken`") from e
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        translated = _translate_messages(messages)
        total = 0
        for msg in translated:
            total += _PER_MESSAGE_TOKENS
            for key, value in msg.items():
                if value is None:
                    continue
                # tool_calls on assistant msgs is a list of dicts; serialize.
                text = value if isinstance(value, str) else json.dumps(value)
                total += len(enc.encode(text))
                if key == "name":
                    # Named messages pay a small extra overhead per cookbook.
                    total += 1
        total += _PER_REPLY_TOKENS

        if tools:
            # Tools get folded into the system prompt server-side in an
            # opaque format; JSON-encoding the schema is the canonical
            # approximation used across the ecosystem.
            total += len(enc.encode(json.dumps(tools)))
        return total

    def chat(self, messages: list[dict], num_ctx: int) -> tuple[str, Usage]:
        payload = {
            "model": self.model,
            "messages": _translate_messages(messages),
            "stream": False,
            "max_tokens": MAX_COMPLETION_TOKENS,
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

    def ensure_exclusive(self) -> None:
        """Unload every other model resident on LM Studio, if this base URL
        is pointed at one. Generic OpenAI-compat servers (vLLM, llama.cpp,
        OpenRouter, first-party OpenAI) don't expose a portable unload, so
        the call is a no-op there.

        LM Studio's `lms` CLI is the supported programmatic path: `lms ps
        --json` lists loaded models, `lms unload <identifier>` evicts one.
        Best-effort — missing CLI, non-zero exit, or parse failure all fall
        through silently. The point of this method is to keep two large
        models from sitting in VRAM at the same time when the user swaps
        via /model or when /eval pins a different provider.
        """
        if not self._is_lm_studio():
            return
        lms = _find_lms_binary()
        if lms is None:
            return
        try:
            proc = subprocess.run(
                [lms, "ps", "--json"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return
        if proc.returncode != 0 or not proc.stdout.strip():
            return
        try:
            loaded = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return
        if not isinstance(loaded, list):
            return
        for entry in loaded:
            if not isinstance(entry, dict):
                continue
            # `identifier` is the stable field across LM Studio versions;
            # `modelKey` / `path` are fallbacks in case a future release
            # renames the top-level key.
            ident = entry.get("identifier") or entry.get("modelKey") or entry.get("path")
            if not ident or ident == self.model:
                continue
            try:
                subprocess.run(
                    [lms, "unload", ident],
                    capture_output=True,
                    timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue

    def _is_lm_studio(self) -> bool:
        # LM Studio defaults to port 1234 on localhost; no other server in
        # the OpenAI-compat zoo uses that port by convention. Overridable via
        # env so a remote LM Studio install can opt in explicitly.
        if os.environ.get("MYAH_OPENAI_COMPAT_IS_LM_STUDIO") == "1":
            return True
        base = self._base.lower()
        return ":1234" in base and ("localhost" in base or "127.0.0.1" in base)


def _find_lms_binary() -> str | None:
    # Prefer whatever `lms` is on PATH; fall back to LM Studio's default
    # install path so the feature works even from a shell that hasn't
    # sourced the lmstudio shim.
    found = shutil.which("lms")
    if found:
        return found
    default = os.path.expanduser("~/.lmstudio/bin/lms")
    return default if os.path.isfile(default) else None


def _parse_sse(lines: Iterator[str]) -> Iterator[StreamChunk]:
    """Consume an SSE line iterator, yield normalized StreamChunks.

    Pulled out so it can be unit-tested against a fake byte stream without
    spinning up an HTTP server."""
    tool_buf: dict[int, dict] = {}
    usage: dict | None = None

    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
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

        # LM Studio (qwen3), DeepSeek R1, and other reasoning-capable
        # servers surface chain-of-thought on a separate delta key instead
        # of inline `<think>...</think>` tags. If we don't read it we
        # silently discard the model's actual output: for qwen3 in
        # thinking mode, `content` can be empty while `reasoning_content`
        # holds hundreds of tokens.
        if delta.get("reasoning_content"):
            yield StreamChunk(reasoning_delta=delta["reasoning_content"])

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
            parsed_tool_calls: list[ToolCall] = []
            for _, slot in sorted(tool_buf.items()):
                raw = "".join(slot["arg_parts"]) or "{}"
                try:
                    arguments = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise ProviderError(f"bad streamed tool-call JSON: {e}: {raw[:200]}") from e
                if not isinstance(arguments, dict):
                    raise ProviderError("bad streamed tool-call JSON: expected an object")
                parsed_tool_calls.append(ToolCall(name=slot["name"], arguments=arguments))
            yield StreamChunk(tool_calls=parsed_tool_calls)
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
            # Orphan guard: if the previous assistant emitted tool_calls that
            # never got matching tool messages (truncated session, mid-turn
            # compaction, etc.), OpenAI rejects the payload. Stub placeholders
            # for each leftover so the contract "every tool_call has a tool
            # result" holds before we append the new assistant block.
            for orphan_id in pending_ids:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": orphan_id,
                        "content": "[no tool result — turn truncated]",
                    }
                )
            ids_for_this = []
            oai_tool_calls = []
            for tc in msg["tool_calls"]:
                tid = f"call_{counter}"
                counter += 1
                ids_for_this.append(tid)
                oai_tool_calls.append(
                    {
                        "id": tid,
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                )
            assistant_msg = {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": oai_tool_calls,
                # Moonshot AI (kimi-k2.6) requires reasoning_content on every
                # assistant message with tool_calls when thinking is enabled.
                # Empty string is fine; missing key triggers a 400.
                "reasoning_content": msg.get("reasoning_content") or "",
            }
            out.append(assistant_msg)
            pending_ids = ids_for_this
        elif role == "tool":
            if pending_ids:
                tid = pending_ids.pop(0)
            else:
                tid = f"call_{counter}"
                counter += 1
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": msg.get("content") or "",
                }
            )
        else:
            out.append(
                {
                    k: v
                    for k, v in msg.items()
                    if k in ("role", "content", "name", "reasoning_content")
                }
            )

    return out
