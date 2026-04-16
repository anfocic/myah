# Building an Agent Harness from Scratch

> A working draft. Updated as the project evolves.

## Why

<!-- TODO (you): personal hook. Why build one when Claude Code, Cursor, Aider, etc. all exist? -->

I wanted to understand — by building — what's actually happening inside the tools I use every day. What is a "harness," exactly? How does the model decide to call a tool? Who's responsible for the loop? How does context get managed when the conversation grows?

So I wrote one. Python, local models via [Ollama](https://ollama.com), no frameworks. A few hundred lines I can read end-to-end.

## The stack

- **Python** — easy to iterate, great REPL story
- **Ollama** running `qwen2.5:7b-instruct` — local, free, good enough at tool calling for a learning toy
- **`rich`** for the terminal UI — colored prompts, context bar, etc.
- **No agent framework** — the whole point is to see the mechanism

## What "agentic" actually means

The core insight I didn't have before starting: an agent loop is just a `while True` around a chat call.

```python
while True:
    response = ollama.chat(messages=messages, tools=tools, ...)
    if response.message.tool_calls:
        # model wants to use a tool — execute it, append result, loop
        ...
    else:
        # model is done — return its answer
        return response.message.content
```

That's it. Everything else — memory, trimming, summarization, streaming, cost tracking — is decoration on top of that loop.

## Milestone 1: the loop + tool calling

<!-- TODO: expand with code walkthrough and what surprised you -->

Tools are declared as JSON schemas in the OpenAI function-calling format. You pass them to the model; the model returns a `tool_calls` list describing *which* function to run with *which* arguments; **your code** does the actual running and appends the result back as a `role: "tool"` message.

The model never runs anything. It just plays dispatcher.

## Milestone 2: context window tracking

<!-- TODO: screenshot of the colored ctx tag in the TUI -->

Every message in `messages` burns tokens. Overflow and Ollama silently truncates. I added:

1. A **character-based estimate** (`len(content) // 4`) for cheap pre-call guesses.
2. **Real counts** from `response.prompt_eval_count` — Ollama returns exact prompt tokens on every call.
3. A **color-coded tag** next to each reply: `Mia [42%] › ...` that goes green → yellow → red as the window fills.

The surprise: the estimate under-counts noticeably because it ignores tool schemas and role metadata, both of which are surprisingly heavy.

## Milestone 3: trim + summarize

When `ctx_used > 80%`, the harness drops oldest user/assistant pairs until usage is back under 50%. But naive trimming causes amnesia — so before dropping, I call the model a second time and ask it to compress the dropped turns into 2-3 sentences, then inject the summary back as a synthetic `role: "system"` note.

<!-- TODO: include a real before/after transcript -->

One extra round-trip per trim event. Feels like a fair trade vs. losing context entirely.

## Milestone 4: debug logging

<!-- TODO: write once PR #1 is merged -->

Every `ollama.chat` response appended as one JSON line to `logs/agent.jsonl`. Fields: `prompt_eval_count`, `eval_count`, `eval_duration_ms`, content, tool calls. JSONL so I can append without rewriting and grep with `jq`.

This is the single most useful thing I've added for *learning*. Being able to inspect what the model actually returned — including the raw tool-call arguments — turned debugging from guesswork into reading.

## What's next

- [ ] Streaming responses token-by-token
- [ ] Multi-provider abstraction (Ollama / OpenAI / Anthropic behind one interface)
- [ ] Tool-call error handling (what happens when a tool raises?)
- [ ] Parallel tool calls
- [ ] Persisting history across sessions
- [ ] Swap out the summarizer for a smaller/faster model

## Things I'd tell past-me

<!-- TODO: fill in as you notice patterns -->

- The model is stateless; **you** maintain the conversation.
- "Tool use" is a data protocol, not magic. The model says "call this"; you do the calling.
- Token counting seems boring until your 10th message mysteriously gets truncated.

---

*Source: [github.com/anfocic/mia](https://github.com/anfocic/mia)*
