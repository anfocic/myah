# Agent Harness — Project Guide

## Purpose

A hand-rolled agentic loop built for **learning how agent harnesses work**. The goal is to understand — by building — what Claude Code, Cursor, etc. do internally: tool calling, the chat/tool/chat cycle, history management, context budgeting.

This is not meant to be production software. When explaining or suggesting changes, prefer clarity and pedagogy over cleverness. Show the mechanism, don't hide it behind abstractions.

**Stack:** Python · Ollama (`qwen2.5:7b-instruct`) · `rich` for TUI · local tools (file I/O, time)

## Project Structure

```
main.py        — REPL, tool registry, tool dispatcher, context-bar renderer
agent.py       — Core agentic loop (ollama.chat + tool execution + token estimate)
config.py      — Model/provider/context config
tools/
  files.py     — read_file, write_file
  utils.py     — get_current_time
  search.py    — (reserved)
```

## The Agent Loop

`run_agent()` in `agent.py`:

1. Build `messages` = system prompt + prior history + new user message
2. Call `ollama.chat()` with `messages`, `tools`, and `options={"num_ctx": NUM_CTX}`
3. If the model returns `tool_calls`:
   - Append the assistant message
   - Execute each tool via the caller-supplied `execute_tool(name, args)`
   - Append each result as a `{"role": "tool", "content": ...}` message
   - Loop back to step 2
4. If no tool calls: append the final user/assistant pair to `history`, return `(content, history, ctx_used)`

`history` lives in `main.py`, is passed into `run_agent()` each turn, and grows unbounded.

## Context Window Tracking (implemented)

`estimate_tokens()` in `agent.py` sums the character length of every message's content and divides by 4 (`~1 token ≈ 4 chars`). It's called at the end of `run_agent()` and returned to `main.py`.

`print_context_bar()` in `main.py` renders the usage each turn as a green/yellow/red bar against `NUM_CTX` (thresholds: 50% / 80%).

**Known limits of the estimate:**
- Ignores tool-call schemas, function names, and role/metadata tokens — real usage is higher
- Character-count is rough; real tokenization varies by model
- To get exact usage, call Ollama with `options={"num_predict": 0}` or inspect `response.prompt_eval_count` if exposed

**Next things worth trying:**
- Print `response.prompt_eval_count` (Ollama returns actual prompt tokens on the response object) and compare to the estimate
- Trim history when `ctx_used > 0.8 * NUM_CTX`: keep the system prompt + last N turns
- Summarize dropped turns into a single synthetic `{"role": "system", "content": "Earlier: ..."}` message

## Configuration (`config.py`)

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PROVIDER` | `"ollama"` | Backend provider |
| `MODEL_NAME` | `"qwen2.5:7b-instruct"` | Ollama model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `NUM_CTX` | `4096` | Context window budget, passed to `ollama.chat` |

Larger models (`qwen2.5:14b`) support wider context windows but are slower. If you bump `NUM_CTX`, make sure the model/Ollama build actually supports it — Ollama will silently truncate otherwise.

## Adding a Tool

1. Write the function in `tools/`
2. Add its OpenAI-style schema to the `tools` list in `main.py`
3. Add a branch for it in `execute_tool()` in `main.py`
4. Import it at the top of `main.py`

The schema shape matters — Ollama follows the OpenAI function-calling format: `{"type": "function", "function": {"name", "description", "parameters": {JSON Schema}}}`.

## Running

```bash
python main.py
```

Requires Ollama running locally with the configured model pulled:

```bash
ollama pull qwen2.5:7b-instruct
```

## Debugging

No debug prints currently. If a turn misbehaves, inspect the `response` object from `ollama.chat()` — `response.message.tool_calls`, `response.message.content`, and the raw eval counts are the useful fields.
