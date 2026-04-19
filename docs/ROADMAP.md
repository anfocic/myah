# Roadmap — Mia vs Claude Code

Gap analysis: what this harness is missing to feel like Claude Code, ordered by pedagogy / payoff ratio.

Each item is a self-contained learning milestone. Pick one, ship a PR, update `CONCEPTS.md`.

---

## Tier 1 — Biggest "feel" gaps

These are the differences you notice in the first 5 minutes of using Claude Code vs Mia.

| Feature | Why it matters | Effort | Status |
|---|---|---|---|
| Real tool suite (`Edit`, `Grep`, `Glob`, `Bash`) | `write_file` nuking whole files is the biggest toy-tell. Diff-style `Edit` + targeted `Grep` change what the model can actually do on real codebases | Medium — each tool is small, collectively large | Shipped |
| Tool permissioning | Ask before destructive writes / bash commands. Teaches the trust model — Claude Code's whole UX revolves around this | Small | Shipped |
| Parallel tool execution | When the model emits multiple `tool_calls`, run them concurrently via `asyncio` or threads. Teaches harness concurrency | Small-medium | Shipped |
| Graceful Ctrl+C | Abort mid-stream without crashing the REPL or corrupting `history` | Small | Shipped |
| Tool result truncation | `read_file` on a 5MB log currently blasts the ctx window. Needs a max-bytes cap + `...truncated` marker | Small | Shipped |

**Suggested first session:** `Edit` + `Grep` + tool permissioning, bundled. Those three together move Mia from "chatbot with two tools" to "actually useful coding assistant" and teach the safety layer in the same PR.

## Tier 2 — Architecture / interface design

Teaches how real harnesses are structured.

| Feature | Why it matters | Effort | Status |
|---|---|---|---|
| Multi-provider abstraction | Claude Code runs on Anthropic; Mia only on Ollama. Abstracting teaches protocol design | Medium | Shipped |
| Persistent history + `CLAUDE.md` loading | Resume across sessions; project-level memory file injected as system context | Small | Shipped |
| Slash commands (`/clear`, `/context`, `/help`) | The control-plane vs. data-plane split — commands the harness handles vs. text the model sees | Small | Shipped |
| `harness_info` tool | Model-side introspection: model name, ctx budget, cwd, git branch, date, tool list. Complement of `/context` on the data plane | Small | Shipped |
| Manual compact (`/compact`) | User-initiated context reset — keep last 2 turns, summarize the rest. Complements the reactive auto-trim | Small | Shipped |
| Rewind (`/rewind N`) | In-memory snapshot stack → pop-back undo for conversation state. Different job than `/retry` (undo, not re-run) | Small | Shipped |
| Microcompact | Elide stale tool results mid-turn when a single turn racks up many tool calls. Intra-turn counterpart to `trim_history` | Small | Shipped |
| Subagents | Spawn a nested `run_agent` with isolated history for a delegated task. Teaches hierarchical agents | Medium | Shipped |
| Hooks | Pre/post tool-call hooks the user can configure. Teaches extensibility | Small | |

## Tier 3 — Polish / power features

Nice to have; smaller pedagogical payoff per unit of effort.

| Feature | Why it matters | Effort | Status |
|---|---|---|---|
| Persistent input history (`~/.mia_history`, arrow keys) | Re-run previous prompts via readline, across sessions | Trivial | Shipped |
| Env context injection | Auto-inject cwd, git branch, OS, date into the system prompt | Small | Shipped |
| Plan mode | Non-executing mode where the model proposes before doing. Claude Code's `ExitPlanMode` tool | Small-medium | Shipped |
| Streaming tool args | Watch `tool_calls` assemble token-by-token (Ollama supports this in newer versions) | Medium | |
| Rendered markdown output | Code blocks, tables, headings via `rich.markdown.Markdown` | Small | Shipped, then reverted (see Tier 4 — incompatible with `patch_stdout` after TUI refactor) |
| MCP-style plugin tools | Dynamically load tools from external processes. Big arch lift | Large | |

## Tier 4 — Push toward 9/10

These are the things keeping Mia from feeling like a "complete" pedagogical harness. Captured here so we don't lose them; each one earns a real chunk of score when shipped.

| Feature | Why it matters | Effort | Status |
|---|---|---|---|
| Direct Anthropic provider adapter | Current coverage: Ollama + OpenAI-compat. Without a real Anthropic adapter, the harness can't teach the features that only exist there (prompt caching, thinking blocks, server-side tool use, batch API). Biggest single addition for "what's an agent harness for a frontier API actually like" | Medium | Shipped (native Messages API adapter; caching / thinking blocks still TODO — see §44) |
| Direct OpenAI provider adapter | OpenAI-compat works against local endpoints but papers over real OpenAI features (responses API, structured outputs, reasoning effort). Shipping a first-party adapter forces the Protocol to hold up under a second real-world target | Medium | Shipped as a factory preset (same wire format as openai-compat; Responses API / reasoning_effort would motivate a dedicated class) |
| Test coverage → 70%+ floor | Current floor: 40% (coverage gate). Untested: provider adapters (need network mocking via `respx` / `httpx_mock`), tool implementations (need filesystem fixtures + subprocess stubs). Ratchet the `fail_under` up as modules get mocked coverage | Medium | |
| Observability — per-turn trace + cost | `logs/agent.jsonl` captures usage already, but there's no viewer, no cost accounting, no cross-turn diff. A `/trace` slash command or a tiny `scripts/trace-viewer.py` that replays a session with token breakdowns would teach the operational side of running an agent | Small-medium | |
| Restore streaming markdown rendering | The prompt_toolkit refactor (§42) dropped `rich.live.Live`. Streaming is now plain text. Two ways to fix: (a) promote the REPL to a full `prompt_toolkit.Application` with a managed log region; (b) render markdown only at end-of-stream (lose the "feels alive" token trickle). Decide, document, ship. The current state is a regression | Medium-large | |
| Diagrams in CONCEPTS.md | Zero diagrams today. At least three would pay for themselves: (1) agent loop sequence (user → build_messages → stream → if tool_calls → execute → loop → else return), (2) context lifecycle (trim → microcompact → compact → rewind), (3) control plane / data plane split. Use mermaid blocks so they render on GitHub | Small | |
| Subagents | Already in Tier 2. Duplicated here because it's load-bearing for the 9/10 narrative: a harness that can't spawn itself isn't teaching hierarchical agents | Medium | Shipped |
| Hooks | Already in Tier 2. User-configurable pre/post tool hooks teach extensibility and mirror Claude Code's hooks config | Small | |

---

## How to use this file

- Pick an item, create a branch, ship a PR. Update the checkbox here when it merges.
- If a feature reveals a new concept while you build it, add it to `CONCEPTS.md`.
- If a feature produces a "surprise I want to tell someone about" moment, mine it for `BLOG.md`.
