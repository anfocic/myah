# Concepts — Agent Harness

Running notes on every concept introduced while building this harness. Read top-to-bottom to follow the build chronologically.

---

## 1. The agentic loop

An agent harness is a `while True` around a chat call. Each turn:

1. Send `messages` (system + history + new user) to the model
2. Model responds with either plain content **or** a list of `tool_calls`
3. If tool calls: execute each tool locally, append results as `role:"tool"` messages, **loop back to step 1**
4. If no tool calls: return the assistant's content and exit the loop

The "agentic" part is step 3 — the model can chain tool calls across many inner iterations before producing a final answer for the user.

See: `agent.py:run_agent`

## 2. Tool calling (OpenAI function-calling format)

Tools are declared to the model as JSON schemas:

```python
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "...",
    "parameters": {"type": "object", "properties": {...}, "required": [...]}
  }
}
```

The model returns `tool_calls[*].function.name` + `arguments`. The harness is responsible for actually executing them — the model just *describes* the call.

Ollama, OpenAI, and Anthropic all use slight variants of this same schema shape.

## 3. History vs messages

Two lists, easily confused:

- **`history`**: persistent across turns. Holds only final `user`/`assistant` pairs. Lives in `main.py`.
- **`messages`**: built fresh each `run_agent` call = `[system] + history + [new_user]`. Grows *within* the turn as tool calls + tool results are appended.

Tool messages are deliberately **not** kept in history — they're intermediate work, not conversation.

## 4. Context window

The model has a fixed token budget (`NUM_CTX`, default 4096). Everything you send — system prompt, history, tool schemas, tool results — counts against it. Overflow = silent truncation in Ollama.

## 5. Token counting: estimate vs real

Two ways to measure:

- **Estimate**: `len(content) // 4`. Cheap, no model call. Ignores tool schemas + role metadata, so it under-counts.
- **Real**: `response.prompt_eval_count` — Ollama reports the exact prompt tokens after each call. Authoritative but only available *after* a call.

We use real when available, fall back to the estimate (e.g. for pre-call trim decisions).

See: `agent.py:estimate_tokens`

## 6. Context pressure → trim

When `ctx_used > 80% * NUM_CTX`, drop oldest user/assistant pairs from history until back under 50%. Two thresholds (`high` / `target`) create hysteresis — you don't trim-one-pair every single turn.

See: `agent.py:trim_history`

## 7. Summarize dropped turns

Naive trim loses information. The fix: before dropping, ask the model to compress the dropped messages into 2-3 sentences, then inject the summary back as a synthetic `role:"system"` note at the start of history.

Cost: one extra model call per trim event. Benefit: you keep the gist of old context instead of amnesia.

See: `agent.py:summarize_dropped`

## 8. TUI with `rich`

`rich.Console` gives colored output, BBCode-style markup (`[bold cyan]...[/bold cyan]`), and styled input prompts. Zero terminal escape-code wrangling.

See: `main.py:ctx_tag`

## 9. Debug logging

Every `ollama.chat` call is appended as one JSON line to `logs/agent.jsonl`. Fields captured:

- `prompt_eval_count` / `eval_count` — real input/output token counts
- `eval_duration_ms` / `total_duration_ms` — model vs. round-trip latency
- `content` + `tool_calls` — what the model actually returned
- `messages_in_prompt` — how big the prompt was

JSONL (one JSON object per line) is ideal here: append-only, greppable with `jq`, and unlike a single JSON array it doesn't require rewriting the whole file on each write.

See: `agent.py:log_response`

## 10. Spinner feedback during blocking calls

`ollama.chat` is synchronous and can sit for 5-30s on a tool-heavy turn. Without feedback the terminal looks dead. `rich.console.Console.status()` opens a context manager that shows a spinner with a live-updating text line, refreshed in a background thread while the main thread blocks on the model.

Pattern: pass the `status` handle into the agent loop so it can report *what* it's currently doing — `Thinking...`, `Running read_file (1/2)`, `Summarizing dropped turns...` — along with a cumulative token count and elapsed seconds.

See: `agent.py:status_line`, `main.py` REPL loop

## 11. Models don't know what they are

Ask a local `qwen2.5` model "who are you?" and it will confidently answer "I'm built by Anthropic" (or OpenAI, depending on which corpus leaked hardest into its training data). The model has no introspection — it just pattern-matches on text it's seen.

**Identity lives in the system prompt.** If you want the model to correctly say "I'm Mia, running on qwen2.5 via Ollama," you have to tell it that, and you usually have to explicitly negate the false answers ("You were NOT built by OpenAI or Anthropic") because the training-data priors are strong.

See: system prompt in `agent.py:run_agent`

## 12. Streaming responses + TTFT

Calling `ollama.chat(..., stream=True)` returns an iterator of partial chunks instead of one fat response. Each chunk is a `ChatResponse` with `message.content` containing whatever new text was generated since the previous chunk. The final chunk carries `prompt_eval_count`, `eval_count`, and durations.

Two UX wins:
1. **No more dead terminal.** The first content token appears in 0.5-2s; the model then streams into the display. Subjectively the 7b feels 3× faster even though the total duration is identical.
2. **TTFT (time-to-first-token) becomes visible** — distinct from total duration. TTFT tells you "how long until the user sees *something*." On a tool-calling turn, TTFT of the final answer includes the tool round-trip, which is a useful latency attribution.

Implementation notes:
- Inside `run_agent`, accumulate chunk content into a buffer while streaming to `console.print(chunk, end="")` for live rendering
- Stop the spinner on the first content token and print the `Mia ›` prefix once
- Tool calls arrive together (not token-by-token), typically with empty content — so tool-only turns bypass streaming naturally
- `ttft_ms` logged per call; it's `null` for tool-only turns since no content ever appeared

See: `agent.py:run_agent`

## 13. Small models hallucinate library APIs

Asked qwen2.5:7b to critique this project. It suggested "improvements" using:

- `response['usage']['total_tokens']` — that's **OpenAI's** response shape. Ollama returns `prompt_eval_count` as a top-level attribute.
- `tool_call.schema`, `tool_call.function_name` — invented. Ollama uses `tool_call.function.name` and `tool_call.function.arguments`.
- A "new" `manage_context_window` helper that reimplements the `trim_history` + `summarize_dropped` it had literally just been shown.

The model sounded confident and structurally coherent, but the code was wrong in ways that would only be obvious to someone who's read the Ollama docs.

**Lesson:** small local models are fine as the *subject* of a harness — they'll happily play agent, call tools, and let you build around them. They are **not** a reliable *code reviewer* for the stack they run on. Their training data is dominated by OpenAI's API shape, so anything provider-specific gets confabulated.

Use a bigger model (`qwen2.5-coder:14b`, Claude, GPT-4) when you want code advice about the harness itself.

## 14. Surgical editing vs full-file writes

`write_file` overwrites the entire file. That's fine for new files; catastrophic for modifying existing code. One wrong token from the model and 500 lines turn into 30.

`edit_file(path, old_string, new_string, replace_all=False)` solves this by forcing the model to produce an `old_string` that *uniquely* identifies the target. If the string appears zero times → reject. If it appears more than once without `replace_all` → reject as ambiguous. Otherwise replace.

Why this works:
- The uniqueness constraint forces the model to include enough surrounding context to disambiguate, which is the same context a human would scan to confirm the edit location.
- Failure modes become loud: "old_string appears 3 times, be more specific" is a recoverable error the model can retry on. A full-file-overwrite failure is silent data loss.
- Tool result size stays tiny regardless of file size — only the diff-like delta travels back through the context window.

This is why Claude Code's `Edit` tool looks the way it does; we copied the shape deliberately.

See: `tools/files.py:edit_file`

## 15. Regex search as a tool

`grep(pattern, path, glob, output_mode)` mirrors ripgrep's core shape but in stdlib `re`. Two output modes:

- `"files_with_matches"` (default) — just the list of files containing a hit. Cheap to read.
- `"content"` — `path:line:text` per match, like `grep -n`. Richer, but hungrier on the ctx window.

Defaulting to files-only is a deliberate context-budget choice: the model can follow up with `read_file` if it wants the details. Dumping every line of every match by default would blow NUM_CTX on any non-trivial search.

Other sanity caps: skip files > 1 MB, skip binaries (anything that fails UTF-8 decode), skip dotdirs / `venv` / `__pycache__` / `node_modules` / `logs`. Hard cap at 50 results with a `... (truncated)` marker. These exist not for correctness but for **context-window discipline** — a single unbounded tool result can destroy a turn.

See: `tools/search.py:grep`

## 16. Tool permissioning / the trust model

The model is a *proposer*, not an executor. For destructive tools (`write_file`, `edit_file`), the harness pauses and asks the human to approve the specific call before it runs.

UX:
- Sensitive tools listed in `SENSITIVE_TOOLS`. Non-sensitive tools (`read_file`, `grep`, `get_current_time`) never prompt.
- Prompt shows the tool name and the actual arguments — so the user sees exactly what the model wants to do, not a generic "allow tool?" dialog.
- Three options: `[y]es` (allow once), `[n]o` (deny, and the model is told it was denied), `[a]lways` (allow this tool for the rest of the process).

Two pedagogical points worth noticing:

1. **Denial is a tool result, not an exception.** When the user says no, the harness appends `{"role": "tool", "content": "User denied this tool call."}` so the next model turn can adapt gracefully ("understood, I won't do that"). If denial raised or hung, the loop would desync and the assistant would produce a confused follow-up. This is the same reason errors from tools are returned as tool messages instead of thrown.

2. **The spinner must yield the terminal.** The `rich.Status` spinner redraws in a background thread; if you call `console.input()` while it's running, the spinner scribbles over the prompt. So `check_permission` does `status.stop()` → prompt → `status.start()`. Same pattern as streaming content in `run_agent`.

Claude Code's whole UX is built on this trust model — every destructive action is a confirmation. It's worth seeing the minimal version to understand why that pattern exists.

See: `permissions.py`, `agent.py:run_agent` (the callback wiring), `main.py` (closure per turn)

---

## 17. Line-numbered reads + the "what's on line N" failure mode

A `read_file` that returns raw text looks fine in isolation and collapses the moment the user asks "what's on line 34?". Observed session:

- User: *"what's in search.py line 34?"*
- Model called `grep('(?m)^\\s*34\\s*', './search.py')` — treating `grep` as a line-number lookup tool. Wrong.
- Next turn, model called `grep('^.*$', 'tools/search.py', output_mode='content')` — dumping the whole file through `grep` because *that* tool returns `path:lineno:text`. Model cherry-picked the `:34:` line from the grep result and presented it as "line 34 contents." Wrong but plausible-looking to the user.
- Next turn, *"read lines 40-60"* → model invented `grep('^\\s*\\d{1,2}\\s+.*')` and reported nothing matched.

Root cause: small models are bad at counting newlines in a raw text blob. When the tool they *do* have (`read_file`) can't answer line-indexed questions, they reach for the nearest tool that *has* line numbers in its output — even when that tool is wrong for the task.

Fix: `read_file` now returns one line per output line, prefixed with `"{lineno:>6}\t"`, same format Claude Code uses. It also takes `offset` (1-indexed start line) and `limit` (default 1000) for pagination, and per-line truncation at 500 chars so one pathological log line can't blow the ctx window.

Two broader lessons:

1. **Tool output shape is part of the tool's contract.** Adding line numbers isn't a cosmetic choice; it enables a whole class of queries that are otherwise impossible. A tool is only as useful as the questions its output can answer.
2. **Watch what the model reaches for.** When it calls the wrong tool for a task, that's signal about what it wished the right tool exposed. The grep-as-line-lookup hallucination directly told us `read_file` was missing line numbers.

See: `tools/files.py:read_file`

## 18. Confident-plausible regressions (§13, second flavor)

Asked qwen2.5:7b to review `tools/search.py`. The review itself was generic ("add docstrings", "improve error handling", "follow PEP 8") — no actual bugs caught. Then it offered a "refined" rewrite. Nine concrete regressions in ~50 lines:

| # | Regression | Consequence |
|---|---|---|
| 1 | Dropped `import re` | `NameError` on first call |
| 2 | `files[:MAX_RESULTS]` caps files, not hits | Misses matches past the 50th file |
| 3 | Catches only `IOError, OSError` around read | Crashes on first binary file (`UnicodeDecodeError`) |
| 4 | `f.read(MAX_FILE_BYTES)` instead of size-skip | Silently truncates 5 MB logs at 1 MB |
| 5 | `return` on file error instead of `continue` | One unreadable file aborts the whole search |
| 6 | `output_mode` branches collapsed into one | `files_with_matches` mode broken |
| 7 | Format changed from `path:lineno:text` to `path:line N:text` | Breaks ripgrep-compatible output the model itself parses |
| 8 | `truncated` set after-the-fact against file-cap | Flag meaningless |
| 9 | Long generic docstrings | Contradicts project CLAUDE.md ("no docstrings unless non-obvious"), which the model read minutes before |

§13 was about **API hallucination** — `response['usage']['total_tokens']` when the actual field is `prompt_eval_count`. §18 is **semantic regression** — the code compiles, looks cleaner (fewer branches, nicer docstrings), and is strictly worse. The structure passes the eyeball test. Only tracing behavior case-by-case surfaces the breakage.

Two lessons:

1. **"Looks like a refactor" ≠ "is a refactor."** A rewrite that removes branches often removes the branches that were handling the edge cases. Review by *enumerating which inputs behave differently*, not by reading the new code for style.
2. **Projects need a self-defense principle.** The model had CLAUDE.md in its context — "Don't add comments/docstrings unless the logic is non-obvious" — and violated it within the same turn. Style rules don't bind the model unless the harness refuses to accept violations. Code review by a stronger model, pre-commit checks, or lint configuration are the enforcement; the system prompt alone is not.

See: also §13. Review transcript archived in `logs/agent.jsonl`.

## 19. Shell access + why the permission layer matters here

`bash(command, cwd, timeout)` is the first tool where the blast radius escapes the process. `write_file` can clobber a file you own; `bash` can `rm -rf`, `curl | sh`, or `git push --force`. Claude Code's entire reputation depends on the user-approval prompt that appears before every command.

Three design choices that make shell-out safe enough to teach with:

1. **`shell=True` is acceptable *because of* permissioning, not despite it.** Normally `shell=True` is a command-injection footgun — the "attacker" (any user input) can escape arg boundaries and build arbitrary pipelines. In this harness the "attacker" is the LLM, and every command it produces passes through `check_permission()` where the human sees the exact string before it runs. The defense isn't syntactic escaping; it's the human eye reading the command. This means the permission prompt *must* show the command in full — truncating it would erase the defense. `NEVER_TRUNCATE_KEYS` in `permissions.py` now covers both `path` and `command` for this reason.

2. **Output capture has to be bounded.** `ls -R /`, `cat bigfile.log`, `find / -type f` all produce megabytes of text that would blast the context window in a single tool call. `MAX_OUTPUT_BYTES = 50_000` per stream, with a truncation marker the model can see and decide whether to narrow its query. Same philosophy as the `grep` 50-result cap and `read_file` per-line cap: **a tool's context cost should be bounded by the tool, not by hoping the model picks short-output commands**.

3. **Timeout is a liveness guarantee, not a convenience.** Without `timeout`, `bash("sleep infinity")` or a hung network call freezes the REPL. Default 30s, configurable per call, failure returned as a tool-result string the model can recover from (same pattern as permission denial — errors are data, not exceptions).

Exit codes, stdout, and stderr are all surfaced separately: the model sees `[stderr]` blocks and `exit: N` footers, so it can distinguish "test failed" from "test ran and passed." Dumping everything into one stream would collapse that distinction.

See: `tools/bash.py`, `permissions.py:NEVER_TRUNCATE_KEYS`

## 20. Generic tool-result cap (defence in depth)

Per-tool caps (`read_file`'s line limit, `grep`'s 50-result ceiling, `bash`'s 50KB stream cap) all assumed each tool author would remember to bound its own output. Works until someone adds a tool and forgets — one `curl` wrapper returning a 5MB JSON blob lands straight in `messages` and the next `ollama.chat` silently drops half the prompt.

`truncate_tool_result()` in `agent.py` is the harness-level safety net: applied in the tool-dispatch path in `run_agent`, *after* `execute_tool` returns but *before* the result is appended to `messages`. Default cap is `TOOL_RESULT_MAX_BYTES = 10_000` chars. Strategy is head-and-tail preservation (errors + summaries live at the edges; the middle of a huge dump is usually the least informative part), joined by a `...[truncated N chars]...` marker the model can see.

Pedagogy: per-tool caps are a *contract* with tool authors; the harness cap is an *invariant*. Contracts get forgotten; invariants don't. Two layers, so one forgotten ceiling doesn't blow the ctx window.

See: `agent.py:truncate_tool_result`, `config.py:TOOL_RESULT_MAX_BYTES`

## 21. Interrupts and input history — cheap REPL ergonomics

Two small changes, one theme: a REPL that doesn't punish the user for bad habits.

**Ctrl+C while the model is streaming.** Without handling, `KeyboardInterrupt` propagates out of the `for chunk in ollama.chat(...)` generator, skips the `history.append` calls at the bottom of `run_agent`, and crashes the REPL entirely. Two places catch it:

1. `agent.py:run_agent` wraps the streaming loop in `try/except KeyboardInterrupt`, stops the spinner, prints a closing newline to salvage the half-streamed line, and re-raises. It does *not* attempt to save partial content to `history` — the invariant is "history only contains complete turns." An aborted turn leaves no trace, which means the next turn doesn't see a truncated assistant reply that the model would then try to "continue."
2. `main.py` catches `KeyboardInterrupt` around the whole `run_agent` + trim/summarize block and returns to the prompt with `↳ aborted — history unchanged`. The same handler at the input call exits the program on Ctrl+C/Ctrl+D at an empty prompt, since readline already gives you in-line editing cancellation before the exception fires.

The pedagogy: **side effects belong at commit points, not mid-stream.** `run_agent` mutates `history` only after a clean completion. That single discipline is what makes interrupts free — no rollback logic, no "undo the partial append," just a skipped commit.

**Arrow-key input history.** `import readline` in `main.py` is enough to give `input()` (which `console.input` wraps) line editing, history navigation, and Ctrl+R search for the session. Persisting across sessions = `readline.read_history_file(~/.mia_history)` on startup + `atexit.register(readline.write_history_file, ...)`. Two lines of setup, infinite re-runs of the same "grep for X" prompt.

The surprise: the *import* is the feature. Python's `input()` silently upgrades its behavior if `readline` is importable. No API calls needed to get arrow keys working — just the side-effect of importing. Classic Python.

See: `agent.py:run_agent` (try/except block), `main.py:_load_input_history`

## 22. Control plane vs data plane — slash commands

Three inputs hit the REPL: `hello`, `/clear`, `exit`. Two of them never reach the model. That split — what the harness handles vs. what the LLM sees — is the **control plane / data plane** split, and it's one of the most important structural ideas in an agent harness.

- **Data plane**: tokens flowing to/from the model. User text, tool calls, tool results, assistant replies. Every byte costs latency + tokens.
- **Control plane**: harness-local operations with zero model involvement. `/clear`, `/help`, `/context`, `exit`, the permission prompt's y/n. Instant, free, deterministic.

`main.py:handle_slash` dispatches any input starting with `/` against a `SLASH_COMMANDS` dict: `{name: (handler, description)}`. If it matches, the handler mutates the REPL's `state` dict (history list, last `ctx_used`) and `continue`s the loop — the model never saw a turn happen. `/help` reads its own registry to render the list, so adding a new command only requires touching the dict.

Two design choices worth naming:

1. **`state` as a single dict, not separate locals.** `trim_history` rebinds its input list (`history = history[2:]`), so if `main.py` kept a local `history` variable *and* stashed a reference in `state`, the two would silently drift after the first trim — `/clear` would clear the stale one while the REPL kept using the fresh one. Making `state["history"]` the single source of truth eliminates the bug class entirely. The cost is `state["history"]` everywhere instead of `history` — cheap.

2. **Slash ≠ tool.** The model could call a hypothetical `clear_history` tool and get the same effect, but that would be bad design: `/clear` is a *user* action, not a *reasoning* action. Separating them means the model can't erase context on itself mid-plan, and the user never has to wait for a model round-trip to reset. The tool layer is for things the model needs to reason *with*; the slash layer is for things the user does *to* the harness. (Separately, a `harness_info` tool for mid-turn introspection is still useful — that's data-plane. Different job.)

See: `main.py:SLASH_COMMANDS`, `main.py:handle_slash`

## 23. Harness introspection — tool closes over state

`/context` answers the user. `harness_info` answers the model. Same information, different audience.

The tool returns `{model, provider, num_ctx, ctx_used, history_turns, cwd, git_branch, date, tools}` as a plain string. The model calls it when a prompt asks "what harness am I running in?" or — more interestingly — when it's mid-reasoning and wants to decide whether to summarize ("am I near the ctx ceiling?") or whether a tool it wants to use actually exists ("do I have `grep`?").

The implementation choice worth noting is **`make_execute_tool(state)` instead of a global**. The tool needs live access to the REPL's `ctx_used` and `history`, both of which mutate every turn. Three ways to wire that:

1. **Module-level globals** — `harness_info` reads `main.ctx_used`. Works, but now agent/tool separation is broken and testing the tool means importing `main`.
2. **Pass state through `run_agent`** — `run_agent(..., state=state)`, then to `execute_tool`. Works, but leaks REPL state into `agent.py`'s signature; `run_agent` has no business knowing about `history` as a mutable state dict.
3. **Factory closure** — `make_execute_tool(state)` returns a dispatcher that captures `state` in its closure. `agent.py` still sees `execute_tool(name, args)` as a plain callable.

Chose (3). `agent.py` stays state-ignorant; `main.py` owns the state; the tool sees fresh values without polling. This is the same pattern Python decorators use — lexical capture as a substitute for passing dependencies through signatures they don't belong in.

The caveat written into the output string: `ctx_used` is the **previous turn's settled value**, not "right now." There is no "right now" — the model is calling the tool from *inside* the current turn, and the harness doesn't know the final prompt token count until the response lands. Saying so in the output prevents the model from reasoning about stale data as if it were live.

See: `tools/harness.py`, `main.py:make_execute_tool`

---

## To cover next

- [ ] System prompt as configuration, not hardcode
- [ ] Multi-provider abstraction (OpenAI / Anthropic / Ollama)
- [ ] Tool-call error handling (model calls a tool that raises)
- [ ] Parallel tool calls (model returns 2+ calls in one turn)
- [ ] Persisting history across sessions
- [ ] Cost/latency tracking per turn
- [x] Tool result truncation (per-tool caps + harness-level `truncate_tool_result`)
