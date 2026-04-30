# Roadmap

Merged from `agentic_harness_must_haves.md` and `agentic_harness_borrowed_ideas.md`. Status as of 2026-04-30.

## Loop Control

| Feature | Status | Where |
|---|---|---|
| Max iteration limit with graceful exit | Done | `agent/loop.py:287-296`, cap 50 (`config.py:100`) |
| Spin detection (same tool + same args repeated) | Done | `agent/loop.py:363-381`, window 3 (`config.py:104`) |
| Distinguish "done" from "stuck" | Done | Model emits no tool calls → turn ends. Spin guard catches stuck loops. |

## Context Management

| Feature | Status | Where |
|---|---|---|
| Auto-trim on context pressure | Done | `agent/context.py:19-43`, fires at 80%, targets 50% |
| Manual compact (`/compact`) | Done | `repl/commands.py:220-240`, keeps last 2 turns |
| Intra-turn microcompact | Done | `agent/context.py:88-112`, fires at 60% |
| LLM summarization of dropped turns | Done | `agent/context.py:139-161`, extractive fallback |
| Tool result truncation | Done | `agent/tokens.py:114-126`, cap 10KB (`config.py:93`) |
| Per-line truncation in file reads | Done | `tools/files.py:22`, `MAX_LINE_CHARS = 500` |

## Permission & Safety

| Feature | Status | Where |
|---|---|---|
| User approval for destructive tools | Done | `permissions.py:146-174`, `prompt_toolkit` y/n/a prompt |
| Session-scoped "always allow" | Done | `permissions.py:18`, keyed by `(name, args)` |
| Plan mode blocks all mutating tools | Done | `agent/loop.py:520-528` |
| CWD security boundary | Done | `security.py:19-20`, anchored to startup cwd |
| Prompt-injection annotation | Done | `security.py:114-123`, warns model without destroying data |
| Structured error recovery hints | Partial | Errors surfaced but no "try alternative approach" guidance in system prompt |

## Tool Infrastructure

| Feature | Status | Where |
|---|---|---|
| Tool registry with schemas | Done | `tools/spec.py:23` — `register()` + `Tool` dataclass |
| Named narrow tools over generic shell | Done | `tools/git.py`, `tools/vault.py`, `tools/web_search.py` |
| CWD-aware path resolution | Done | `tools/cd.py:63-75` — `resolve_against` bridge |
| Sub-agent spawning | Done | `tools/subagent.py:26-96`, depth 1, thread-locked |
| Vault knowledge base search | Done | `tools/vault.py:37-105` |
| Tool registry versioning | Not done | Registry exists but no version field or swap-by-version |
| Step-level retries with backoff | Not done | `/retry` retries the whole turn, not individual tool calls |
| Idempotency keys on tool calls | Not done | Spin guard halts rather than deduplicates |

## State & Persistence

| Feature | Status | Where |
|---|---|---|
| Session save/load to disk | Done | `repl/persistence.py:18-52`, `~/.mia_session.json` |
| Resume from last session (`--resume`) | Done | `main.py:59-61` |
| Turn-level snapshots for rewind | Done | `main.py:120`, `/rewind` command, max 20 deep |
| Conversation variables (named key-value) | Not done | No `set_var`/`get_var` tool. Model state is only history + cwd. |

## Execution Model

| Feature | Status | Where |
|---|---|---|
| Freeform chat/tool loop | Done | `agent/loop.py:286` — `while True:` |
| Explicit state machine / graph | Not done | No nodes, edges, or conditional routing separate from model |
| Ephemeral execution per tool | Not done | All tools run in same process. No container/sandbox per call. |
| Stateful filesystem across calls | Partial | Shared FS + cwd tracking works, but no sandbox. Same as host process. |
| Event-driven resumption | Not done | No webhook or async job support. `permissions.py` blocks synchronously. |

## Working Memory & Planning

| Feature | Status | Where |
|---|---|---|
| Todo list as working memory | Not done | Model has no structured task tracker beyond flat history |

## Priority Queue

Ordered by impact-to-effort ratio:

1. **Error recovery hints** (Partial → Done) — add "if a tool fails, try a different approach" to system prompt. Trivial.
2. **Todo list as working memory** — new tool `set_todo`/`get_todo` or inject into system prompt as structured block. High impact, low effort.
3. **Tool registry versioning** — add `version: str` to `Tool` dataclass. Low effort.
4. **Conversation variables** — `set_var(name, value)` / `get_var(name)` tools. Medium effort, enables persistent cross-turn state without history pollution.
5. **Step-level retries with backoff** — wrap tool execution in retry loop. Medium effort.
6. **Idempotency keys** — deduplicate by `(name, canonical_args)` within a turn. Medium effort.
7. **Explicit state machine** — model loop as graph with nodes/edges. High effort, structural change.
8. **Ephemeral execution / sandboxing** — container per tool call. High effort, infrastructure change.
9. **Event-driven resumption** — async pause/resume via webhooks. High effort, requires external queue.
