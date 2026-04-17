# Mia — a hand-rolled agent harness

> A minimal Claude-Code-style agent loop, built for learning how agent harnesses work under the hood.

Mia is a ~2000-line Python REPL that runs a local LLM (Ollama) or any OpenAI-compatible HTTP endpoint, lets it call tools (read/write files, grep, glob, bash, git), and wraps the whole thing in the ergonomics you'd expect: slash commands, streaming markdown, permission prompts, session persistence, context management.

The goal isn't to compete with Claude Code or Cursor. It's to **build one the hard way**, so the mechanics stop being magic.

## Why this exists

Large agent harnesses look mysterious from the outside — there's a chat loop, tools magically execute, context somehow stays under budget, and the UX feels effortless. Mia exists to make every one of those moves visible and editable in ~30 files of Python.

Each feature in the codebase has a corresponding concept writeup in [`docs/CONCEPTS.md`](docs/CONCEPTS.md) explaining *why* it's there and *what tradeoff* it represents — 37 sections and counting, chronological. Read top-to-bottom and you've traced the full build.

## Features

- **Tool-calling loop** — OpenAI function-calling schemas, Ollama & OpenAI-compat backends
- **Real tool suite** — `read_file` / `write_file` / `edit_file` (surgical) / `glob` / `grep` / `bash` / `git_checkout` / `harness_info`
- **Permission gate** — destructive tools prompt before executing (Claude Code's trust model)
- **Plan mode** — read-only investigation gate; mutating tools short-circuited
- **Streaming markdown** — live-rendered assistant replies via `rich`
- **Context management** — auto-trim on pressure, manual `/compact`, intra-turn microcompact, `/rewind` snapshot undo
- **Session persistence** — conversations resume across runs via `~/.mia_session.json`
- **Runtime model swap** — `/model qwen2.5-coder:14b` without restart
- **Parallel tool execution** — multiple tool calls in one turn fire concurrently
- **Project context injection** — `CLAUDE.md` in the cwd is injected into every system prompt

## Install

Requires Python 3.11+ and either Ollama running locally or an OpenAI-compatible endpoint.

```bash
git clone https://github.com/anfocic/mia
cd mia
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
ollama pull qwen2.5:7b-instruct   # or bring your own
```

## Run

```bash
python main.py
```

Type `/help` in the REPL for the full command list. Highlights:

| Command | Effect |
|---|---|
| `/help` | list all commands |
| `/model` | list local models + swap active one |
| `/plan` | toggle plan mode (read-only; describe before acting) |
| `/compact` | summarize older turns, keep the last 2 |
| `/rewind [N]` | undo N turns via in-memory snapshots |
| `/context` | show current model / ctx usage / history depth |
| `/retry` | re-run the last turn (pop + resubmit) |
| `/clear` | wipe history + saved session |

## Configuration

Environment variables read at startup (see `config.py`):

| Var | Default | Purpose |
|---|---|---|
| `MIA_PROVIDER` | `ollama` | `ollama` or `openai-compat` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama daemon URL |
| `OPENAI_COMPAT_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compat endpoint |
| `OPENAI_COMPAT_MODEL` | `gpt-4o-mini` | Model name sent to the endpoint |
| `OPENAI_COMPAT_API_KEY` | — | Auth header (optional for local servers) |

Runtime swap via `/model <name>` — no restart.

## Architecture

```
main.py        REPL, slash commands, tool registry, display callbacks
agent.py       Agentic loop, context management, system prompt
config.py      Startup constants (env-derived)
permissions.py User-approval gate for destructive tools
display.py     Rich renderers (diffs, file previews)
providers/     Protocol + Ollama / OpenAI-compat adapters
tools/         Tool implementations (files, search, bash, git, harness)
tests/         pytest suite
docs/
  CONCEPTS.md  37 concepts — chronological, pedagogical
  ROADMAP.md   What's shipped, what's next
```

The **control plane / data plane** split is the load-bearing idea: slash commands mutate REPL state directly without the model in the loop; tools go through the model. See [CONCEPTS §22](docs/CONCEPTS.md#22-control-plane-vs-data-plane--slash-commands).

## Testing

```bash
pytest tests/ -q
```

Tests cover the hairy invariants: context-compaction shape, snapshot-stack semantics, openai-compat tool_call_id orphan handling, permission fail-closed behavior, session-file validation.

## Pedagogical spine

Each feature has an entry in [`docs/CONCEPTS.md`](docs/CONCEPTS.md) explaining the design choice in 200–400 words. Notable ones:

- [§6–7](docs/CONCEPTS.md#6-context-pressure--trim) — Context pressure & summarize-dropped-turns
- [§14](docs/CONCEPTS.md#14-surgical-editing-vs-full-file-writes) — Why `edit_file` exists instead of `write_file`
- [§16](docs/CONCEPTS.md#16-tool-permissioning--the-trust-model) — Tool permissioning / the trust model
- [§22](docs/CONCEPTS.md#22-control-plane-vs-data-plane--slash-commands) — Control plane vs data plane
- [§25](docs/CONCEPTS.md#25-parallel-tool-execution--serial-gate-parallel-body) — Parallel tool execution
- [§30](docs/CONCEPTS.md#30-multi-provider-abstraction--two-protocol-families-one-contract) — Multi-provider abstraction
- [§33–35](docs/CONCEPTS.md#33-manual-compact--proactive-context-control) — Context preservation (compact / rewind / microcompact)
- [§37](docs/CONCEPTS.md#37-startup-constants-vs-runtime-state--model-switching) — Startup constants vs runtime state

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md). Gap analysis against Claude Code, ordered by pedagogy / payoff.

## License

[MIT](LICENSE)
