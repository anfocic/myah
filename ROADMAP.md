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
| Subagents | Spawn a nested `run_agent` with isolated history for a delegated task. Teaches hierarchical agents | Medium | |
| Hooks | Pre/post tool-call hooks the user can configure. Teaches extensibility | Small | |

## Tier 3 — Polish / power features

Nice to have; smaller pedagogical payoff per unit of effort.

| Feature | Why it matters | Effort | Status |
|---|---|---|---|
| Persistent input history (`~/.mia_history`, arrow keys) | Re-run previous prompts via readline, across sessions | Trivial | Shipped |
| Env context injection | Auto-inject cwd, git branch, OS, date into the system prompt | Small | Shipped |
| Plan mode | Non-executing mode where the model proposes before doing. Claude Code's `ExitPlanMode` tool | Small-medium | Shipped |
| Streaming tool args | Watch `tool_calls` assemble token-by-token (Ollama supports this in newer versions) | Medium | |
| Rendered markdown output | Code blocks, tables, headings via `rich.markdown.Markdown` | Small | Shipped |
| MCP-style plugin tools | Dynamically load tools from external processes. Big arch lift | Large | |

---

## How to use this file

- Pick an item, create a branch, ship a PR. Update the checkbox here when it merges.
- If a feature reveals a new concept while you build it, add it to `CONCEPTS.md`.
- If a feature produces a "surprise I want to tell someone about" moment, mine it for `BLOG.md`.
