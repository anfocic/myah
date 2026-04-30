# Agent Harness — Project Guide

## Purpose

A hand-rolled agent harness built for **learning how agent harnesses work**. The goal is to understand — by building — what Claude Code, Codex, Cursor, etc. do internally: tool calling, the chat/tool/chat cycle, history management, permissioning, and context budgeting.

Do NOT use emojis in responses.

This is not meant to be production software. When explaining or suggesting changes, prefer clarity and pedagogy over cleverness. Show the mechanism, don't hide it behind abstractions.

**Stack:** Python · `prompt_toolkit` + `rich` · provider adapters (Ollama by default; also OpenAI-compatible, OpenAI, Anthropic, DeepSeek) · local tools (`read_file`, `edit_file`, `glob`, `grep`, `bash`, `git_checkout`, `spawn_subagent`, etc.)

## Project Structure

```text
main.py         — REPL entry point: prompt loop, slash commands, session restore/save, post-turn stats
config.py       — provider/model defaults, context budget, stream pacing, tool-result cap
display.py      — tool start/end rendering
permissions.py  — user approval gate for mutating / shell tools
security.py     — cwd guard + prompt-injection annotation

agent/
  loop.py       — core chat/tool/chat loop + parallel tool execution
  context.py    — trim_history, /compact support, microcompact, summaries
  system_prompt.py — persona + env block + memory-file loading + mode-specific rules
  tokens.py     — token estimate + tool-result truncation
  status.py     — status text + JSONL response logging

repl/
  commands.py   — slash commands (/help, /context, /plan, /debug, /retry, /compact, /rewind, /model, /profile, /eval)
  tool_registry.py — tool schemas + dispatcher factory
  persistence.py — session save/load
  state.py      — typed REPL state
  ui.py         — prompt + context tag + prompt_toolkit setup

providers/
  base.py       — Provider / StreamChunk / Usage contract
  ollama_adapter.py
  openai_compat.py
  anthropic_adapter.py

tools/
  files.py      — read_file, write_file, edit_file
  search.py     — glob, grep
  bash.py       — guarded shell execution
  git.py        — narrow git helpers
  harness.py    — harness_info snapshot
  subagent.py   — nested run_agent delegation
  utils.py      — get_current_time

tests/          — loop, provider, permission, security, slash-command, and subagent coverage
```

## The Agent Loop

`run_agent()` lives in `agent/loop.py`:

1. Build `messages` = system prompt + prior persisted history + new user message
2. Call the active provider via `provider.stream_chat(messages, tools, NUM_CTX)`
3. Stream text to the UI while collecting completed `tool_calls` and any surfaced usage
4. If tool calls arrive:
   - Append the assistant message plus its `tool_calls`
   - Permission-gate each call serially, then run approved calls in parallel
   - Annotate suspicious tool output, truncate oversized results, append each as a `{"role": "tool", "content": ...}` message
   - Loop back to step 2 with the expanded message list
5. If no tool calls arrive:
   - Append only the final user/assistant pair to persisted `history`
   - Return `(content, history, ctx_used, stats)`

`history` lives in REPL state and stores only durable conversation turns. Tool messages are intra-turn scratch space, not long-term history.

## Context Management

Current harness has four layers of context management:

- `estimate_tokens()` in `agent/tokens.py` uses the char/4 heuristic as a fallback
- If a provider returns exact prompt usage, `run_agent()` prefers that count
- `trim_history()` drops oldest turns after a high-water mark and targets a lower steady-state window
- `apply_summary()` summarizes dropped turns into a synthetic system note
- `microcompact()` elides older tool results inside the current turn when tool-heavy loops start filling context
- `/compact` lets the user proactively keep the last 2 turns and summarize the rest

The REPL now prints a per-turn context tag rather than the old context bar.

**Known limits:**
- The char/4 estimate still ignores role, schema, and metadata overhead
- Exact usage depends on provider support; not every backend surfaces it on every streamed turn
- Summarization is another model call and can fail closed, leaving only the retained recent turns

## Configuration (`config.py`)

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PROVIDER` | `"openai-compat"` | Active provider at startup |
| `OLLAMA_MODEL` | `"qwen/qwen3.5-9b"` | Default Ollama model |
| `OPENAI_COMPAT_MODEL` | `"google/gemma-4-e4b"` | Default model for generic OpenAI-compatible servers |
| `MODEL_NAME` | derived from provider | Active model label shown by harness surfaces |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `NUM_CTX` | `32768` | Context budget passed to providers |
| `TOOL_RESULT_MAX_BYTES` | `10000` | Harness-level cap before tool output is truncated |

Hosted providers read their own env vars (`OPENAI_*`, `ANTHROPIC_*`, `DEEPSEEK_*`) via the provider factory.

## Control Surfaces

**Slash commands**
- `/help`
- `/clear`
- `/context`
- `/plan`
- `/debug`
- `/retry`
- `/compact`
- `/rewind [N]`
- `/model [provider:]name`

**Built-in tools exposed to the model**
- `get_current_time`
- `read_file`, `write_file`, `edit_file`
- `glob`, `grep`
- `bash`
- `harness_info`
- `git_checkout`
- `spawn_subagent`

In plan mode, only read-only tools run; mutating tools are short-circuited so the model can inspect but not change state.

## Adding a Tool

1. Implement the function in `tools/`
2. Add its OpenAI-style schema to `repl/tool_registry.py`
3. Add a branch in `make_execute_tool()`
4. Decide whether it should be read-only or permission-gated / plan-mode-blocked
5. Add or extend tests under `tests/`

The schema shape follows OpenAI function calling:
`{"type": "function", "function": {"name", "description", "parameters": {JSON Schema}}}`.

## Running

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
ollama pull qwen2.5:7b-instruct
python main.py
```

Resume the last saved session:

```bash
python main.py --resume
```

Start on a hosted provider instead of Ollama:

```bash
MYAH_PROVIDER=openai OPENAI_API_KEY=... python main.py
```

## Debugging

- `/debug` prints the exact `messages` array before each provider call
- `logs/agent.jsonl` records per-turn traces and surfaced usage
- `tests/test_integration.py` uses a scripted fake provider to exercise the full loop without a live backend
- For provider-specific failures, inspect the adapter in `providers/` before assuming the loop is at fault
