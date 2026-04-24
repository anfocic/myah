# Worklog

## 2026-04-24 — Eval harness v1

**Done**
- Added `evals/` package: `runner.py` (task loader, fixture-copy into tempdir, per-task provider swap via `build_provider`/`set_active_provider`, trace capture through `on_tool_start`/`on_tool_end`, thread-based `wall_timeout_s`, JSONL + rich-table output) and `checks.py` (dispatch over `tool_trace`, `content_regex`, `content_substr`, `fs_file_equals`, `fs_file_contains`, `python`).
- Task modules are plain Python files under `evals/tasks/` — no YAML dep, inline callables supported for the `python` check type.
- Two seed tasks shipped: `find_string` (grep a literal and mention the two files it appears in) and `edit_rename` (rename a function inside a tempdir copy of a fixture).
- CLI wrapper at `scripts/run_evals.py`. `--task`, `--provider`, `--model`, `--list`. Exit code 0 iff all tasks pass, so it's CI-gateable.
- `tests/test_evals_runner.py`: unit tests for every check type plus two end-to-end tests against the existing `FakeProvider` from `tests/test_integration.py` (the happy path and a forbidden-tool regression). Full suite: 94 passing.
- Documented the architecture as CONCEPTS §45 and marked the roadmap row Shipped.

**Discovered**
- The explore-agent report described `meta` dicts on `on_tool_start`/`on_tool_end` that only exist on another branch. Main's loop passes plain `(name, args)` and `(name, args, result, ok)`. Trace stitching fell back to name-based matching of end → start. Moral: verify surfaces against the actual branch you're building on, not a summary report.
- `run_agent`'s `permission_check` arg is the only real seam for "non-interactive eval" — a lambda that returns True/False without prompting. Three callers now (REPL, subagent, eval runner), three different lambdas, zero special cases in the loop. That's the abstraction earning its keep.
- `daemon=True` thread + `join(timeout)` is good enough for v1 but leaks the thread on timeout. Noted; production would need real cancellation.

**Next**
- Expand seed suite: `read_specific_line`, `refuse_rm_rf`, `plan_mode_respect`, `multi_step`, `subagent_delegate`.
- LLM-judge check type for free-form answers (opt-in; keeps determinism as the default).
- Per-task cost accounting from the JSONL — a trivial aggregator script can pair with this.

## 2026-04-16 — CLAUDE.md reframe

**Done**
- Rewrote `CLAUDE.md` to frame the repo as a learning project (hand-rolled agent harness for understanding tool-calling loops), not production software.
- Updated config table to include `NUM_CTX`.
- Replaced the "add this token-tracking code" section with an accurate description of what's already implemented: `estimate_tokens()` in `agent.py:7`, `print_context_bar()` in `main.py:70`, returned `ctx_used` from `run_agent()`.
- Documented the estimate's known limits (ignores tool schemas, role/metadata tokens, char/4 is rough).
- Dropped stale "Debug Output" section — `agent.py` no longer has DEBUG prints.

**Discovered**
- User thought token tracking needed building; it was already wired end-to-end. CLAUDE.md was out of sync with the code.

**Next**
- Swap (or supplement) the char/4 estimate with Ollama's real `response.prompt_eval_count` and compare the two on-screen.
- History trimming when `ctx_used > 0.8 * NUM_CTX`: keep system prompt + last N turns.
- Summarization of dropped turns into a synthetic context message.
