# Worklog

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
