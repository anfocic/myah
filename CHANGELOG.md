# Changelog

All notable changes to this project are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning intentional: this is a learning harness, not a released library —
entries are grouped by merged PR date, not SemVer releases.

## [Unreleased]

### Added

- **Subagents** (`spawn_subagent` tool) — delegate a self-contained
  subtask to a nested `run_agent` with isolated history. The subagent
  shares the parent's `execute_tool` and permission gate but runs with
  a fresh context window, so its tool chatter doesn't eat the parent's
  budget. One level deep (schema filter + module-level depth counter).
  CONCEPTS §43 and `tests/test_subagent.py` cover the design.
- **CONCEPTS §42** — pinned prompt: swapping readline + `rich.Live` for
  `prompt_toolkit`. Input now sticks to the bottom of the terminal while
  output scrolls above it (Claude-Code-style).
- **`--resume` flag** on `python main.py` — session persistence is now
  fully opt-in on both load *and* save, so a forgotten flag can't
  silently overwrite the prior session.
- `tests/test_repl_ui.py` — pure-function tests for `SlashCompleter`
  and `build_prompt` (the parts that don't need a real TTY).
- Coverage reporting via `pytest-cov` + Codecov upload from CI.
- `mypy` type-check step in CI with pragmatic defaults.
- Pre-commit hook config (`ruff`, whitespace, YAML check).
- Dependabot config for pip + GitHub Actions updates.
- CI status + coverage badges in README.
- `CHANGELOG.md` (this file).

### Changed

- **TUI engine**: `readline` + `rich.Console.input()` → `prompt_toolkit`
  `PromptSession` under `patch_stdout`. Tab completion of slash
  commands now lives in a `SlashCompleter(Completer)` subclass.
  Input history moved from `~/.mia_history` (readline format) to
  `~/.mia_input_history` (prompt_toolkit `FileHistory`) — formats
  aren't compatible, so arrow-key recall resets on first launch.
- `rich.live.Live` streaming canvas removed; tokens now stream as
  plain text because `Live` + `patch_stdout` fight over the cursor.
  Restoring streaming markdown rendering is tracked in ROADMAP Tier 4.
- `rich.Console.status()` spinner removed; status lines now print to
  scrollback (`Thinking…`, `Running tools (2/3)`) under the pinned
  prompt. `run_agent`, `_run_tools_parallel`, and `check_permission`
  all dropped their `status=` parameter.
- README trimmed to a one-screen landing page — the detailed tour
  lives in `docs/CONCEPTS.md`.

### Fixed

- `tools/__init__.py` was missing — `tools/` was implicitly a namespace
  package, which confused mypy. Now a proper package.

## 2026-04 — Maturity pass

### Added

- **CONCEPTS §38–40**: typed state (TypedDict), module boundaries,
  testing the loop via Protocol substitution.
- **GitHub Actions CI** (pytest + ruff on Python 3.11/3.12/3.13).
- **Integration test** (`tests/test_integration.py`) — `FakeProvider`
  replays scripted turns to exercise the full `run_agent` loop
  deterministically.
- **`State` TypedDict** for the REPL state dict — catches typos under
  mypy, zero runtime cost.
- **Runtime model switching** via `/model <name>` — no restart needed.
  `get_active_provider()` / `set_active_provider()` registry replaces
  the module-level `_provider` singleton.
- **Context preservation**: `/compact`, `/rewind [N]`, intra-turn
  `microcompact`.
- **`git_checkout` tool** (+ dash-prefix guard to reject names like
  `-f` that git would parse as flags).
- **Open-source packaging**: README, LICENSE (MIT), `pyproject.toml`,
  `tests/` with 29 unit tests covering compact/rewind/permission
  gate/openai-compat adapter/session load.

### Changed

- `main.py` split from 807 lines into a `repl/` package (7 modules)
  plus a thin 119-line entry point.
- `agent.py` split from 598 lines into an `agent/` package (6 modules).
- Tool-call display callbacks (`on_tool_start`, `on_tool_end`) moved
  into `display.py` — colocated with the renderers they call.
- Ruff cleanup: `typing.Iterator`/`Mapping`/`Callable` → `collections.abc`
  imports; ambiguous `l` renamed.

### Fixed

Six bug sweep (PR #27):

- Mutable default arg in `run_agent` (`history: list = []` → `None`
  sentinel + per-call init).
- Missing `permission_check` now fails closed (read-only tools only)
  instead of silently allowing sensitive ones.
- `git_checkout` rejects dash-prefix branch names before passing to
  `git` (which would parse them as flags).
- OpenAI-compat adapter orphan `tool_call_id` — when tool-result count
  is less than preceding assistant's `tool_calls` count, stub entries
  are now flushed so the payload remains valid.
- Session-file load validates each entry's shape; corrupt entries are
  dropped with a user note instead of crashing later.
- `tools/bash`: narrowed `except Exception` to `OSError` so real
  failures surface instead of being collapsed to a generic message.

## 2026-03 — Foundation

### Added

- Core agentic loop (`run_agent`) with streaming + tool-calling.
- Tool suite: `read_file`, `write_file`, `edit_file` (surgical), `glob`,
  `grep`, `bash`, `harness_info`.
- Permission gate for destructive tools (`SENSITIVE_TOOLS`).
- Plan mode: read-only investigation gate.
- Rendered markdown output via `rich.Live`.
- Parallel tool execution (ThreadPoolExecutor).
- Session persistence (`~/.mia_session.json`, atomic writes).
- Project context injection (`CLAUDE.md` in cwd).
- Env block (cwd / platform / git state) in system prompt.
- Multi-provider abstraction (Ollama + OpenAI-compatible HTTP).
- REPL QoL pass: tool-call display, prompt chrome (branch/mode badges),
  diff rendering for `edit_file`, file-preview for `read_file`,
  tok/s stats.
- Slash commands: `/help`, `/clear`, `/context`, `/plan`, `/debug`,
  `/retry`.
- Chronological `docs/CONCEPTS.md` — 37 entries at start of April,
  40 by mid-April.
