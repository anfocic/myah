# Handbook — Mia Agent Harness

Stable guide to the main ideas in Mia as it exists today.

Use this file to study the architecture. Use [`BUILD_NOTES.md`](BUILD_NOTES.md) for the chronological build story, tradeoffs, and implementation surprises.

## Study Path

If you want the shortest route through the project, read these first:

1. Agent loop
2. Tools and tool results
3. History and context management
4. Trust model and permissions
5. Provider abstraction
6. State, slash commands, and persistence

## 1. The Agent Loop

Mia is an agent harness because it runs a chat/tool/chat loop rather than a single model call.

Per turn:

1. Build `messages` from the system prompt, prior history, and the new user message.
2. Send them to the active provider.
3. Stream content as it arrives.
4. If the model emits tool calls, execute them locally, append their results as `role: "tool"` messages, and loop back to the provider.
5. If no tool calls arrive, commit the final user/assistant pair to persistent history and return to the REPL.

Core code:
- [`agent/loop.py`](../agent/loop.py)
- [`main.py`](../main.py)

Why it matters:
- This is the core pattern behind coding agents.
- Everything else in the harness exists to make this loop usable, safe, and observable.

## 2. Messages vs History

Mia keeps two related but different data structures:

- `messages`: the full prompt sent to the model for the current inner loop iteration
- `history`: the durable conversation transcript carried across user turns

Tool messages live in `messages` but not in long-term `history`. That keeps the persistent conversation compact and prevents tool chatter from dominating future turns.

Core code:
- [`agent/loop.py`](../agent/loop.py)
- [`agent/context.py`](../agent/context.py)

Why it matters:
- Many harness bugs come from mixing scratchpad state with durable state.
- This split is what makes context management and rewind manageable.

## 3. Tools Are Contracts, Not Magic

The model does not execute code. It emits structured requests for named tools. The harness owns the actual execution.

Mia exposes tools using OpenAI-style function schemas and dispatches them through a single registry.

Current tool surface includes:
- file reads and edits
- search
- shell access
- git branch switching
- harness introspection
- subagent delegation

Core code:
- [`repl/tool_registry.py`](../repl/tool_registry.py)
- [`tools/`](../tools)

Why it matters:
- The model can only do what the harness explicitly makes possible.
- Tool naming and schema design shape model behavior just as much as prompts do.

## 4. Context Management Is Layered

Mia does not rely on one single “context window fix.” It uses several layers:

- rough or provider-reported token counting
- reactive history trimming
- summary insertion for dropped turns
- intra-turn elision of stale tool results
- manual compaction commands

Core code:
- [`agent/tokens.py`](../agent/tokens.py)
- [`agent/context.py`](../agent/context.py)
- [`main.py`](../main.py)

Why it matters:
- Different context failures happen at different layers.
- A long conversation, a tool-heavy turn, and a large persistent system prompt are separate problems.

### 4a. Loop Guards

The agentic `while True` is bounded by two harness-owned checks that stop the model from running forever or silently spinning:

- **Iteration cap** (`MAX_AGENT_ITERATIONS`, default 50). Hard ceiling on provider iterations inside a single `run_agent` call. On hit, the loop returns a synthetic assistant message (`[halted: iter_cap] ...`) and `stats["halt_reason"] = "iter_cap"`.
- **Spinning detection** (`SPIN_WINDOW`, default 3). A sliding window of the most recent `(tool_name, args)` tuples; if they're all equal, the loop halts before executing again. A legitimate re-read of the same file later in a trajectory does not trigger — only *consecutive* identical calls do.

Both reset per-call, so a fresh user turn starts clean. Return shape matches the normal non-error path, so eval runners, subagents, and the REPL need no special handling — they read `stats["halt_reason"]` when they care.

Core code:
- [`agent/loop.py`](../agent/loop.py)
- [`config.py`](../config.py)

Why it matters:
- Small local models are particularly prone to stuck-tool loops; without a cap the only stop is Ctrl-C or a context-window OOM.
- Returning a named halt reason keeps this observable in eval reports and logs rather than looking like a silent hang.

## 5. Provider Usage Beats Estimates

Mia starts with a cheap token heuristic, then prefers exact prompt usage reported by the provider when available.

That means:
- before a call, the harness may only have an estimate
- after a call, surfaced provider usage is the best ground truth

Core code:
- [`agent/tokens.py`](../agent/tokens.py)
- [`agent/loop.py`](../agent/loop.py)
- [`providers/base.py`](../providers/base.py)

Why it matters:
- Exact token accounting is provider- and payload-dependent.
- This keeps the harness simple while still improving accuracy whenever the backend exposes real counts.

## 6. The Trust Model

Mia treats the model as a proposer, not an autonomous executor.

Sensitive tools require user approval. The harness shows the exact call and waits for a decision before running it. Plan mode adds an additional guard by allowing investigation while blocking mutation.

Core code:
- [`permissions.py`](../permissions.py)
- [`agent/loop.py`](../agent/loop.py)
- [`repl/commands.py`](../repl/commands.py)

Why it matters:
- This is the operational safety model behind coding agents.
- The harness must make it obvious when the model is describing an action versus actually performing it.

## 7. Tool Design Should Match Model Weaknesses

Mia’s tools are intentionally narrow.

Examples:
- `edit_file` does surgical replacement instead of whole-file rewrite
- `read_file` returns numbered lines
- `glob` resolves bare filenames before reads or edits
- `git_checkout` exists as a named action instead of forcing the model through generic shell

Core code:
- [`tools/files.py`](../tools/files.py)
- [`tools/search.py`](../tools/search.py)
- [`tools/git.py`](../tools/git.py)

Why it matters:
- Small and medium models are bad at long diffs, precise line counting, and generic shell planning.
- Good tools compensate for model weaknesses.

## 8. Control Plane vs Data Plane

Mia has two ways to change behavior:

- slash commands the REPL handles directly
- tools the model sees and can call

Slash commands like `/context`, `/model`, `/compact`, and `/rewind` are harness controls. They are not part of the model-facing tool surface.

Core code:
- [`repl/commands.py`](../repl/commands.py)
- [`repl/ui.py`](../repl/ui.py)

Why it matters:
- It keeps operator controls separate from model behavior.
- This is a useful pattern in any interactive harness.

## 9. The System Prompt Supplies Environment Context

Mia rebuilds its system prompt every turn. It includes:

- a role/persona block
- environment facts like cwd, platform, date, and git state
- project instructions from `CLAUDE.md`
- extra rules when plan mode or subagent mode is active

Core code:
- [`agent/system_prompt.py`](../agent/system_prompt.py)

Why it matters:
- The model gets basic runtime context without spending tool calls.
- Rebuilding each turn lets prompt-affecting state change live.

## 10. Provider Abstraction Happens at the Stream Level

Mia supports multiple providers by normalizing them into a shared protocol:

- `stream_chat(...)`
- `chat(...)`
- `StreamChunk`
- `ToolCall`
- `Usage`
- `ensure_exclusive()` — optional; evicts every other model resident on the same backend so two large local models never sit in VRAM at once. Implemented for Ollama (daemon-side `ps` + `keep_alive=0` generate) and for LM Studio (the `lms` CLI). Hosted providers don't need it. `set_active_provider` calls it on every swap, including the lazy-init one at startup. The openai-compat adapter detects LM Studio by its default `:1234` on localhost; override with `MIA_OPENAI_COMPAT_IS_LM_STUDIO=1` for a remote LM Studio install.

Provider adapters translate their own wire formats into this common shape.

Core code:
- [`providers/base.py`](../providers/base.py)
- [`providers/__init__.py`](../providers/__init__.py)
- [`providers/ollama_adapter.py`](../providers/ollama_adapter.py)
- [`providers/openai_compat.py`](../providers/openai_compat.py)
- [`providers/anthropic_adapter.py`](../providers/anthropic_adapter.py)

Why it matters:
- The loop stays provider-agnostic.
- All protocol weirdness is pushed to the adapter boundary.

## 11. Display Logic Is Separate From Agent Logic

The loop does not print tool UI directly. It emits callbacks, and the display layer renders them.

That keeps the agent core focused on state transitions while the REPL handles formatting, previews, and visual affordances.

Core code:
- [`display.py`](../display.py)
- [`agent/loop.py`](../agent/loop.py)

Why it matters:
- Printing inside the core loop couples logic to one interface.
- Callback boundaries make refactors easier and support alternate UIs.

## 12. State Is Explicit

Mia keeps its mutable REPL state in a typed dict with a small, explicit shape:

- history
- context usage
- plan/debug mode flags
- rewind snapshots
- retry input

Core code:
- [`repl/state.py`](../repl/state.py)
- [`main.py`](../main.py)

Why it matters:
- Typed state catches drift while keeping runtime behavior simple.
- This is a good midpoint between “everything global” and over-abstracted classes.

## 13. Persistence Should Be Small And Intentional

Mia persists:

- conversation history
- input history

It does not persist tool messages or broad runtime internals.

Core code:
- [`repl/persistence.py`](../repl/persistence.py)
- [`repl/ui.py`](../repl/ui.py)

Why it matters:
- Persist the durable conversation, not every temporary artifact.
- Smaller persisted state is easier to validate and recover.

## 14. Testing Works Best At Boundaries

Mia’s strongest tests focus on seams:

- loop behavior with scripted providers
- adapter translation functions
- permission-gate behavior
- state transitions in slash commands
- security/path-guard behavior

Core code:
- [`tests/test_integration.py`](../tests/test_integration.py)
- [`tests/test_openai_compat.py`](../tests/test_openai_compat.py)
- [`tests/test_anthropic_adapter.py`](../tests/test_anthropic_adapter.py)
- [`tests/test_permission_gate.py`](../tests/test_permission_gate.py)
- [`tests/test_security.py`](../tests/test_security.py)

Why it matters:
- The hardest bugs in agent harnesses usually sit at boundaries, not in isolated helper functions.

## 15. Security Is Defense In Depth, Not a Sandbox

Mia includes several defensive layers:

- cwd path scoping for file/search tools
- prompt-injection annotation on tool output
- permission gating for sensitive tools
- size caps on tool results

Core code:
- [`security.py`](../security.py)
- [`permissions.py`](../permissions.py)
- [`agent/tokens.py`](../agent/tokens.py)

Why it matters:
- These controls reduce accidental damage and obvious exploitation paths.
- They do not replace real sandboxing.

## 16. Subagents Are Isolated Nested Loops

Mia can delegate a bounded subtask to a nested `run_agent` call with fresh history.

The child:
- gets a focused task
- runs with the same provider and tool execution path
- does not inherit the parent’s prior conversation
- cannot recurse indefinitely

Core code:
- [`tools/subagent.py`](../tools/subagent.py)
- [`agent/loop.py`](../agent/loop.py)
- [`tests/test_subagent.py`](../tests/test_subagent.py)

Why it matters:
- This is the simplest version of hierarchical agents.
- It teaches delegation without requiring a distributed system.

## 17. Module Boundaries Matter Once the Script Stops Teaching

Mia started as a smaller, more monolithic harness. It is now split by concern:

- loop
- context
- tokens
- prompt assembly
- provider adapters
- tool registry
- REPL controls and UI

Core code:
- [`agent/`](../agent)
- [`providers/`](../providers)
- [`repl/`](../repl)
- [`tools/`](../tools)

Why it matters:
- The right level of modularity preserves legibility.
- Past a certain size, a single-file pedagogical script stops being pedagogical.

## 18. How To Use This Repo

Use the docs in this order:

1. [`README.md`](../README.md) for setup and entry points
2. this handbook for the stable architecture
3. [`BUILD_NOTES.md`](BUILD_NOTES.md) for the chronological design trail
4. [`ROADMAP.md`](ROADMAP.md) for what still feels missing

If you are changing behavior:
- update this handbook when the stable architecture changes
- update [`BUILD_NOTES.md`](BUILD_NOTES.md) when the interesting part is the story, surprise, or tradeoff
