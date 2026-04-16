# Concepts — Agent Harness

Running notes on every concept introduced while building this harness. Read top-to-bottom to follow the build chronologically.

---

## 1. The agentic loop

An agent harness is a `while True` around a chat call. Each turn:

1. Send `messages` (system + history + new user) to the model
2. Model responds with either plain content **or** a list of `tool_calls`
3. If tool calls: execute each tool locally, append results as `role:"tool"` messages, **loop back to step 1**
4. If no tool calls: return the assistant's content and exit the loop

The "agentic" part is step 3 — the model can chain tool calls across many inner iterations before producing a final answer for the user.

See: `agent.py:run_agent`

## 2. Tool calling (OpenAI function-calling format)

Tools are declared to the model as JSON schemas:

```python
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "...",
    "parameters": {"type": "object", "properties": {...}, "required": [...]}
  }
}
```

The model returns `tool_calls[*].function.name` + `arguments`. The harness is responsible for actually executing them — the model just *describes* the call.

Ollama, OpenAI, and Anthropic all use slight variants of this same schema shape.

## 3. History vs messages

Two lists, easily confused:

- **`history`**: persistent across turns. Holds only final `user`/`assistant` pairs. Lives in `main.py`.
- **`messages`**: built fresh each `run_agent` call = `[system] + history + [new_user]`. Grows *within* the turn as tool calls + tool results are appended.

Tool messages are deliberately **not** kept in history — they're intermediate work, not conversation.

## 4. Context window

The model has a fixed token budget (`NUM_CTX`, default 4096). Everything you send — system prompt, history, tool schemas, tool results — counts against it. Overflow = silent truncation in Ollama.

## 5. Token counting: estimate vs real

Two ways to measure:

- **Estimate**: `len(content) // 4`. Cheap, no model call. Ignores tool schemas + role metadata, so it under-counts.
- **Real**: `response.prompt_eval_count` — Ollama reports the exact prompt tokens after each call. Authoritative but only available *after* a call.

We use real when available, fall back to the estimate (e.g. for pre-call trim decisions).

See: `agent.py:estimate_tokens`

## 6. Context pressure → trim

When `ctx_used > 80% * NUM_CTX`, drop oldest user/assistant pairs from history until back under 50%. Two thresholds (`high` / `target`) create hysteresis — you don't trim-one-pair every single turn.

See: `agent.py:trim_history`

## 7. Summarize dropped turns

Naive trim loses information. The fix: before dropping, ask the model to compress the dropped messages into 2-3 sentences, then inject the summary back as a synthetic `role:"system"` note at the start of history.

Cost: one extra model call per trim event. Benefit: you keep the gist of old context instead of amnesia.

See: `agent.py:summarize_dropped`

## 8. TUI with `rich`

`rich.Console` gives colored output, BBCode-style markup (`[bold cyan]...[/bold cyan]`), and styled input prompts. Zero terminal escape-code wrangling.

See: `main.py:ctx_tag`

## 9. Debug logging

Every `ollama.chat` call is appended as one JSON line to `logs/agent.jsonl`. Fields captured:

- `prompt_eval_count` / `eval_count` — real input/output token counts
- `eval_duration_ms` / `total_duration_ms` — model vs. round-trip latency
- `content` + `tool_calls` — what the model actually returned
- `messages_in_prompt` — how big the prompt was

JSONL (one JSON object per line) is ideal here: append-only, greppable with `jq`, and unlike a single JSON array it doesn't require rewriting the whole file on each write.

See: `agent.py:log_response`

## 10. Spinner feedback during blocking calls

`ollama.chat` is synchronous and can sit for 5-30s on a tool-heavy turn. Without feedback the terminal looks dead. `rich.console.Console.status()` opens a context manager that shows a spinner with a live-updating text line, refreshed in a background thread while the main thread blocks on the model.

Pattern: pass the `status` handle into the agent loop so it can report *what* it's currently doing — `Thinking...`, `Running read_file (1/2)`, `Summarizing dropped turns...` — along with a cumulative token count and elapsed seconds.

See: `agent.py:status_line`, `main.py` REPL loop

## 11. Models don't know what they are

Ask a local `qwen2.5` model "who are you?" and it will confidently answer "I'm built by Anthropic" (or OpenAI, depending on which corpus leaked hardest into its training data). The model has no introspection — it just pattern-matches on text it's seen.

**Identity lives in the system prompt.** If you want the model to correctly say "I'm Mia, running on qwen2.5 via Ollama," you have to tell it that, and you usually have to explicitly negate the false answers ("You were NOT built by OpenAI or Anthropic") because the training-data priors are strong.

See: system prompt in `agent.py:run_agent`

## 12. Streaming responses + TTFT

Calling `ollama.chat(..., stream=True)` returns an iterator of partial chunks instead of one fat response. Each chunk is a `ChatResponse` with `message.content` containing whatever new text was generated since the previous chunk. The final chunk carries `prompt_eval_count`, `eval_count`, and durations.

Two UX wins:
1. **No more dead terminal.** The first content token appears in 0.5-2s; the model then streams into the display. Subjectively the 7b feels 3× faster even though the total duration is identical.
2. **TTFT (time-to-first-token) becomes visible** — distinct from total duration. TTFT tells you "how long until the user sees *something*." On a tool-calling turn, TTFT of the final answer includes the tool round-trip, which is a useful latency attribution.

Implementation notes:
- Inside `run_agent`, accumulate chunk content into a buffer while streaming to `console.print(chunk, end="")` for live rendering
- Stop the spinner on the first content token and print the `Mia ›` prefix once
- Tool calls arrive together (not token-by-token), typically with empty content — so tool-only turns bypass streaming naturally
- `ttft_ms` logged per call; it's `null` for tool-only turns since no content ever appeared

See: `agent.py:run_agent`

## 13. Small models hallucinate library APIs

Asked qwen2.5:7b to critique this project. It suggested "improvements" using:

- `response['usage']['total_tokens']` — that's **OpenAI's** response shape. Ollama returns `prompt_eval_count` as a top-level attribute.
- `tool_call.schema`, `tool_call.function_name` — invented. Ollama uses `tool_call.function.name` and `tool_call.function.arguments`.
- A "new" `manage_context_window` helper that reimplements the `trim_history` + `summarize_dropped` it had literally just been shown.

The model sounded confident and structurally coherent, but the code was wrong in ways that would only be obvious to someone who's read the Ollama docs.

**Lesson:** small local models are fine as the *subject* of a harness — they'll happily play agent, call tools, and let you build around them. They are **not** a reliable *code reviewer* for the stack they run on. Their training data is dominated by OpenAI's API shape, so anything provider-specific gets confabulated.

Use a bigger model (`qwen2.5-coder:14b`, Claude, GPT-4) when you want code advice about the harness itself.

---

## To cover next

- [ ] System prompt as configuration, not hardcode
- [ ] Multi-provider abstraction (OpenAI / Anthropic / Ollama)
- [ ] Tool-call error handling (model calls a tool that raises)
- [ ] Parallel tool calls (model returns 2+ calls in one turn)
- [ ] Persisting history across sessions
- [ ] Cost/latency tracking per turn
